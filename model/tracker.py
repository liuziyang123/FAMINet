import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
from torch.distributions.normal import Normal

from lib.image import imwrite_indexed
from lib.utils import AverageMeter
from .augmenter import ImageAugmenter
from .discriminator import Discriminator
from .seg_network import SegNetwork

from .jointup import PacJointUpsample

from time import time

import cv2
from collections import OrderedDict
import random

from .raft import RAFT
from lib.tensorlist import TensorList
from line_profiler import LineProfiler
import profile

from lib.selftune import *

import os
import sys
root_path = os.getcwd()
sys.path.insert(0, '/data2/liuziyang/frtm-vos-master/')


Count = 0
dir = './results/viz/'
def save_tensor(input):
    global Count
    # input = (input - input.min()) / (input.max() - input.min())
    input[input<0] = 0.
    input = input / input.max()
    input = input.squeeze().cpu().numpy()
    heatmap = cv2.applyColorMap(np.uint8(255 * input), cv2.COLORMAP_HOT)

    cv2.imwrite(dir + str(Count) + '.png', heatmap)
    Count += 1


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


class TargetObject:

    def __init__(self, obj_id, disc_params, **kwargs):

        self.object_id = obj_id
        self.discriminator = Discriminator(**disc_params)
        self.disc_layer = disc_params.layer
        self.start_frame = None
        self.start_mask = None
        self.index = -1

        for key, val in kwargs.items():
            setattr(self, key, val)

    def initialize(self, ft, mask):
        self.discriminator.init(ft[self.disc_layer], mask)

    def classify(self, ft):
        return self.discriminator.apply(ft)

    def classify_(self, ft):
        return self.discriminator.apply_(ft)


class Tracker(nn.Module):

    def __init__(self, augmenter: ImageAugmenter, feature_extractor, feature_extractor_flow, disc_params, refiner: SegNetwork, device):

        super().__init__()

        self.augmenter = augmenter
        self.augment = augmenter.augment_first_frame
        self.disc_params = disc_params
        self.feature_extractor = feature_extractor
        self.feature_extractor_flow = feature_extractor_flow

        self.refiner = refiner
        for m in self.refiner.parameters():
            m.requires_grad_(False)
        self.refiner.eval()

        self.device = device

        self.first_frames = []
        self.current_frame = 0
        self.current_masks = None
        self.num_objects = 0

        self.psnr = 0
        self.count = 0

    def clear(self):
        self.first_frames = []
        self.current_frame = 0
        self.current_masks = None
        self.num_objects = 0
        torch.cuda.empty_cache()            #

    def run_dataset(self, dataset, out_path, speedrun=False, restart=None):
        """
        :param dataset:   Dataset to work with (See datasets.py)
        :param out_path:  Root path for storing label images. Sequences of label pngs will be created in subdirectories.
        :param speedrun:  [Optional] Whether or not to warm up Pytorch when measuring the run time. Default: False
        :param restart:   [Optional] Name of sequence to restart from. Useful for debugging. Default: None
        """

        out_path.mkdir(exist_ok=True, parents=True)

        self.project = nn.ModuleDict()
        self.filter = nn.ModuleDict()
        self.flowkey = []
        self.project_weight = []
        self.filter_weight = []
        self.project[self.flowkey[i]] = nn.Conv2d(256, 96, 1, bias=False).cuda(self.device)
        self.filter[self.flowkey[i]] = nn.Conv2d(96, 2, 3, bias=False).cuda(self.device)

        dset_fps = AverageMeter()
        print('Evaluating', dataset.name)

        restarted = False
        for sequence in dataset:
            if restart is not None and not restarted:
                if sequence.name != restart:
                    continue
                restarted = True

            # We preload data as we cannot both read from disk and upload to the GPU in the background,
            # which would be a reasonable thing to do. However, in PyTorch, it is hard or impossible
            # to upload images to the GPU in a data loader running as a separate process.
            sequence.preload(self.device)
            self.clear()  # Mitigate out-of-memory that may occur on some YouTubeVOS sequences on 11GB devices.

            for i in range(len(self.flowkey)):
                self.project[self.flowkey[i]].weight.data = self.project_weight[i].clone()
                self.filter[self.flowkey[i]].weight.data = self.filter_weight[i].clone()

            outputs, seq_fps = self.run_sequence(sequence, speedrun)
            dset_fps.update(seq_fps)

            dst = out_path / sequence.name
            dst.mkdir(exist_ok=True)
            for lb, f in zip(outputs, sequence.frame_names):
                imwrite_indexed(dst / (f + ".png"), lb)

        print("Average frame rate: %.2f fps" % dset_fps.avg)

    def run_sequence(self, sequence, speedrun=False):
        """
        :param sequence:  FileSequence to run.
        :param speedrun:  Only for DAVIS 2016: If True, let pytorch initialize its buffers in advance
                          to not incorrectly measure the memory allocation time in the first frame.
        :return:
        """

        self.eval()
        self.object_ids = sequence.obj_ids
        self.current_frame = 0
        self.targets = dict()

        self.psnr = 0
        self.count = 0
        self.output_mask = []
        self.prev_outpout = []

        N = 0

        object_ids = torch.tensor([0] + sequence.obj_ids, dtype=torch.uint8, device=self.device)  # Mask -> labels LUT

        self.outputs = []
        t0 = time()

        for i, (image, labels, new_objects) in tqdm.tqdm(enumerate(sequence), desc=sequence.name, total=len(sequence), unit='frames'):

            old_objects = set(self.targets.keys())

            image = image.to(self.device)
            if len(new_objects) > 0:
                labels = labels.to(self.device)
                self.initialize(image, labels, new_objects)

            if len(old_objects) > 0:

                # FAMINet-inter
                inter = 2
                images = []
                labels = []
                if i >= inter - 1:
                    for res in range(inter):
                        im, _, _ = sequence.__getitem__(i - (inter - 1 - res))
                        images.append(im)
                else:
                    for res in range(i):
                        im, _, _ = sequence.__getitem__(res)
                        images.append(im)
                    for res in range(inter - i):
                        im, _, _ = sequence.__getitem__(i)
                        images.append(im)

                self.track(images, self.current_masks)

                masks = self.current_masks
                if len(sequence.obj_ids) == 1:
                    labels = object_ids[(masks[1:2] > 0.5).long()]
                else:
                    masks = torch.clamp(masks, 1e-7, 1 - 1e-7)
                    masks[0:1] = torch.min((1 - masks[1:]), dim=0, keepdim=True)[0]  # background activation
                    segs = F.softmax(masks / (1 - masks), dim=0)  # s = one-hot encoded object activations
                    labels = object_ids[segs.argmax(dim=0)]

                for p_ids in range(len(self.prev_outpout)):
                    self.prev_outpout[p_ids].append((labels == (p_ids + 1)).float().unsqueeze(0).unsqueeze(0))

            if isinstance(labels, list) and len(labels) == 0:  # No objects yet
                labels = image.new_zeros(1, *image.shape[-2:])

            self.outputs.append(labels)
            self.current_frame += 1
            N += 1

        torch.cuda.synchronize()
        T = time() - t0
        fps = N / T

        # print(self.psnr / self.count)

        return self.outputs, fps

    def initialize(self, image, labels, new_objects):

        self.current_masks = torch.zeros((len(self.targets) + len(new_objects) + 1, *image.shape[-2:]), device=self.device)

        for obj_id in new_objects:

            # Create target

            mask = (labels == obj_id).byte()
            target = TargetObject(obj_id=obj_id, index=len(self.targets)+1, disc_params=self.disc_params,
                                  start_frame=self.current_frame, start_mask=mask)
            self.targets[obj_id] = target

            # HACK for debugging
            torch.random.manual_seed(0)
            np.random.seed(0)

            # Augment first image and extract features

            im, msk = self.augment(image, mask)
            with torch.no_grad():
                ft = self.feature_extractor(im, [target.disc_layer])
            target.initialize(ft, msk)

            self.current_masks[target.index] = mask

            self.prev_outpout.append([])

        return self.current_masks

    def track(self, images, masks):

        self.padder = InputPadder(images[0].shape)
        features = []
        features_flow = []
        im_size = images[0].shape[-2:]
        warp = SpatialTransformer(images[0].shape[-2:]).cuda(images[0].device)
        for i in range(len(images)):
            images[i] = images[i].unsqueeze(0).float()
            masks[i] = masks[i].unsqueeze(0).cuda(images[0].device)
            # masks[i] = self.padder.pad(masks[i])[0]
            features.append(self.feature_extractor(images[i]))
            images[i] = self.padder.pad(images[i])[0]
            features_flow.append(self.feature_extractor(images[i]))

        # Classify

        for obj_id, target in self.targets.items():
            if target.start_frame < self.current_frame:
                # with torch.autograd.profiler.profile(enabled=True) as prof:
                flows = []
                # compute mask according to the segmentations
                mask = (masks[-1].clone() == obj_id).float()
                mask_ref = (masks[-2].clone() == obj_id).float()
                robust_mask = torch.zeros(mask.shape).cuda(mask.device)
                robust_mask_ref = torch.zeros(mask.shape).cuda(mask.device)
                center = masks_to_bboxes(mask.squeeze())
                # center = masks_to_bboxes_gauss(mask.squeeze())
                robust_mask[:, :,
                int(torch.round(center[1] - center[3] / 2)):int(torch.round(center[1] + center[3] / 2)),
                int(torch.round(center[0] - center[2] / 2)):int(torch.round(center[0] + center[2] / 2))] = 1

                center = masks_to_bboxes(mask_ref.squeeze())
                robust_mask_ref[:, :,
                int(torch.round(center[1] - center[3] / 2)):int(torch.round(center[1] + center[3] / 2)),
                int(torch.round(center[0] - center[2] / 2)):int(torch.round(center[0] + center[2] / 2))] = 1

                if torch.sum(mask) == 0:
                    robust_mask[...] = 1
                if torch.sum(mask_ref) == 0:
                    robust_mask_ref[...] = 1

                features_flow = []
                features_flow.append(self.feature_extractor(images[0] * robust_mask_ref))
                features_flow.append(self.feature_extractor(images[1] * robust_mask))

                # perform Relaxed Steepest Descent to compute optical flow
                for m in self.project[self.flowkey[obj_id]].parameters():
                    m.requires_grad = True
                for m in self.filter[self.flowkey[obj_id]].parameters():
                    m.requires_grad = True
                self.project[self.flowkey[obj_id]].train()
                self.filter[self.flowkey[obj_id]].train()
                for i in range(1, len(features)):
                    mask = (masks[i].clone() == obj_id).float()
                    robust_mask = torch.zeros(mask.shape).cuda(mask.device)
                    center = masks_to_bboxes(mask.squeeze())
                    robust_mask[:, :,
                    int(torch.round(center[1] - center[3] / 2)):int(torch.round(center[1] + center[3] / 2)),
                    int(torch.round(center[0] - center[2] / 2)):int(torch.round(center[0] + center[2] / 2))] = 1
                    if torch.sum(mask) == 0:
                        robust_mask[...] = 1

                    if torch.sum(mask) == 0: #or torch.sum(mask_ref) == 0:
                    # if True:
                        fmap = torch.cat((features_flow[i]['layer3'], features_flow[i - 1]['layer3']), 1)
                        flow = self.filter[self.flowkey[obj_id]](self.project[self.flowkey[obj_id]](fmap.detach()))
                        flow = F.interpolate(flow, size=im_size, mode='bilinear')
                        # robust_mask[...] = 1
                    else:
                        for j in range(2):
                            parameters = TensorList([self.project[self.flowkey[obj_id]].weight,
                                                     self.filter[self.flowkey[obj_id]].weight])
                            # end_time = time()
                            fmap = torch.cat((features_flow[i]['layer3'], features_flow[i - 1]['layer3']), 1)
                            flow = self.filter[self.flowkey[obj_id]](
                                self.project[self.flowkey[obj_id]](fmap.detach()))
                            flow = F.interpolate(flow, size=im_size, mode='bilinear')
                            warp_im = warp(images[i - 1] / 255., flow)
                            residuals = (images[i] / 255. - warp_im).abs() * robust_mask
                            # whether to use SSIM loss
                            # ssim = SSIM().cuda(images[i].device)
                            # ssim_value = ssim(images[i] / 255., warp_im) * robust_mask
                            # residuals = residuals.sum() * 0.15 + ssim_value.sum() * 0.8
                            residuals = residuals.sum() * 0.15 * 0.8
                            residuals = TensorList([residuals])
                            parameters.requires_grad_(True)
                            lzy_g = TensorList(torch.autograd.grad(residuals, parameters, create_graph=True))
                            lzy_g_2 = 0
                            for k in range(len(lzy_g)):
                                lzy_g_2 += (lzy_g[k] * lzy_g[k]).sum()
                            lzy_alpha = residuals[0] / lzy_g_2 * 0.8 * 0.15
                            # commonly used SGD algorithm
                            # lzy_alpha = torch.tensor(5e-4).cuda(images[0].device)
                            step = lzy_g.apply(
                                lambda e: lzy_alpha.reshape([-1 if d == 0 else 1 for d in range(e.dim())]) * e)
                            if not torch.isnan(lzy_alpha):
                                self.project[self.flowkey[obj_id]].weight.data = parameters[0] - step[0]
                                self.filter[self.flowkey[obj_id]].weight.data = parameters[1] - step[1]
                            else:
                                print(1)
                    flows.append(flow)
                # print(prof.key_averages().table(sort_by="self_cpu_time_total"))

                with torch.no_grad():
                    s = []
                    for i in range(len(images)):
                        # s.append(target.classify(features[i][target.disc_layer]))
                        if i == len(images) - 1:
                            s.append(target.classify(features[i][target.disc_layer]))
                        else:
                            s.append(target.classify_(features[i][target.disc_layer]))
                    y, flows = self.refiner(s, features, features_flow, images, im_size, flows, padder=self.padder)
                    # y = self.padder.unpad(y)
                    y = torch.sigmoid(y)
                    self.current_masks[target.index] = y

        # self.output_mask.append(flows[-1])
        # images[-1] = self.padder.unpad(images[-1])
        # images[-2] = self.padder.unpad(images[-2])
        # warp = SpatialTransformer(images[-1].shape[-2:]).cuda(images[-1].device)
        # image_pre = warp(images[-2] / 255., flows[-1])
        # image_pre = (image_pre * 255.0).round()
        # psnr = calculate_psnr(image_pre, images[-1].float())
        # self.psnr += psnr.cpu()
        # self.count += 1

        # Update

        for obj_id, t1 in self.targets.items():
            if t1.start_frame < self.current_frame:
                for obj_id2, t2 in self.targets.items():
                    if obj_id != obj_id2 and t2.start_frame == self.current_frame:
                        self.current_masks[t1.index] *= (1 - t2.start_mask.squeeze(0)).float()

        p = torch.clamp(self.current_masks, 1e-7, 1 - 1e-7)                 # normalize to [min, max]
        p[0:1] = torch.min((1 - p[1:]), dim=0, keepdim=True)[0]
        segs = F.softmax(p / (1 - p), dim=0)
        inds = segs.argmax(dim=0)

        # self.out_buffer = segs * F.one_hot(inds, segs.shape[0]).permute(2, 0, 1)
        for i in range(self.current_masks.shape[0]):
            self.current_masks[i] = segs[i] * (inds == i).float()

        for obj_id, target in self.targets.items():
            if target.start_frame < self.current_frame and self.disc_params.update_filters:
                target.discriminator.update(self.current_masks[target.index].unsqueeze(0).unsqueeze(0))

        return self.current_masks

    def dilation(self, input_):

        input = input_.squeeze().unsqueeze(-1)
        input = input.cpu().numpy()

        kernel = np.ones((10, 10), np.uint8)
        input = cv2.dilate(input, kernel, iterations=1)
        input = torch.from_numpy(input)
        input = input.squeeze().float().unsqueeze(0).unsqueeze(0).cuda(input_.device)

        return input

    def finetune(self, sequence, batchsize=9, epoch=50):
        length = len(sequence)
        optimizer = torch.optim.Adam(self.refiner.flow.parameters(), lr=0.00001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        for m in self.refiner.flow.parameters():
            m.requires_grad = True
        self.refiner.flow.train()
        iters = length // batchsize
        for j in tqdm.tqdm(range(epoch)):
            index_list = list(range(length))
            index_list = index_list[1:]
            random.shuffle(index_list)
            for iter in range(iters):
                im1 = []
                im2 = []
                index = iter * batchsize
                for i in range(batchsize):
                    image1, _, _ = sequence.__getitem__(index_list[index + i])
                    image2, _, _ = sequence.__getitem__(index_list[index + i] - 1)
                    image1 = image1.unsqueeze(0).float()
                    image2 = image2.unsqueeze(0).float()
                    im1.append(image1)
                    im2.append(image2)
                im1 = torch.cat(im1, 0)
                im2 = torch.cat(im2, 0)
                self.padder = InputPadder(im1.shape)
                im1, im2, = self.padder.pad(im1, im2)
                features1 = self.feature_extractor(im1)
                features2 = self.feature_extractor(im2)

                optimizer.zero_grad()
                fmap1 = features1['layer3']
                fmap2 = features2['layer3']
                flows12 = self.refiner.flow(fmap1, fmap2, im1 / 255.)
                self.flow.cuda(fmap1.device)
                flows_ref = self.flow(fmap1, fmap2, im1 / 255.)
                loss = davis_loss(im1 / 255., im2 / 255., flows12, flows_ref=flows_ref)
                loss.backward()
                optimizer.step()
                scheduler.step(loss.item())

        for m in self.refiner.flow.parameters():
            m.requires_grad = False
        self.refiner.flow.eval()



