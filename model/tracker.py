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

from time import time

import cv2
from collections import OrderedDict
import random

from .raft import RAFT
from lib.tensorlist import TensorList

import os
import sys
root_path = os.getcwd()
sys.path.insert(0, '/data2/liuziyang/frtm-vos-master/')

Count = 0
dir = './results/viz/'
def save_tensor(input):
    global Count
    input[input<0] = 0.
    input = input / input.max()
    input = input.squeeze().cpu().numpy()
    heatmap = cv2.applyColorMap(np.uint8(255 * input), cv2.COLORMAP_HOT)

    cv2.imwrite(dir + str(Count) + '.png', heatmap)
    Count += 1


def masks_to_bboxes(mask, fmt='c'):

    """ Convert a mask tensor to one or more bounding boxes.
    Note: This function is a bit new, make sure it does what it says.  /Andreas
    :param mask: Tensor of masks, shape = (..., H, W)
    :param fmt: bbox layout. 'c' => "center + size" or (x_center, y_center, width, height)
                             't' => "top left + size" or (x_left, y_top, width, height)
                             'v' => "vertices" or (x_left, y_top, x_right, y_bottom)
    :return: tensor containing a batch of bounding boxes, shape = (..., 4)
    """
    batch_shape = mask.shape[:-2]
    mask = mask.reshape((-1, *mask.shape[-2:]))
    bboxes = []

    H, W = mask.shape[-2:]
    p_size = 0

    for m in mask:
        # mx = m.sum(dim=-2).nonzero()
        # my = m.sum(dim=-1).nonzero()
        mx = torch.nonzero(m.sum(dim=-2), as_tuple=False)
        my = torch.nonzero(m.sum(dim=-1), as_tuple=False)
        if (len(mx) > 0 and len(my) > 0):
            if mx.min() - p_size > 0:
                a = mx.min() - p_size
            else:
                a = 0
            if mx.max() + p_size < W:
                b = mx.max() + p_size
            else:
                b = W
            if my.min() - p_size > 0:
                c = my.min() - p_size
            else:
                c = 0
            if my.max() + p_size < H:
                d = my.max() + p_size
            else:
                d = H
            bb = [a, c, b, d]
        else:
            bb = [0, 0, 0, 0]

        # bb = [mx.min(), my.min(), mx.max(), my.max()] if (len(mx) > 0 and len(my) > 0) else [0, 0, 0, 0]
        # bb = [a, c, b, d] if (len(mx) > 0 and len(my) > 0) else [0, 0, 0, 0]
        bboxes.append(bb)

    bboxes = torch.tensor(bboxes, dtype=torch.float32, device=mask.device)
    bboxes = bboxes.reshape(batch_shape + (4,))

    if fmt == 'v':
        return bboxes

    x1 = bboxes[..., :2]
    s = bboxes[..., 2:] - x1 + 1

    if fmt == 'c':
        return torch.cat((x1 + 0.5 * s, s), dim=-1)
    elif fmt == 't':
        return torch.cat((x1, s), dim=-1)

    raise ValueError("Undefined bounding box layout '%s'" % fmt)


class SpatialTransformer(nn.Module):

    def __init__(self, size, mode='bilinear'):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer, self).__init__()

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids) # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)
        self.mode = mode

    def forward(self, src, flow):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        new_locs = self.grid + flow
        # new_locs = self.grid[:, [1,0], ...] + flow
        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2*(new_locs[:, i, ...]/(shape[i]-1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode, padding_mode="border", align_corners=False)


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

        for i in range(10):
            self.flowkey.append(str(i))
        for i in range(len(self.flowkey)):
            self.project[self.flowkey[i]] = nn.Conv2d(256, 96, 1, bias=False).cuda(self.device)
            self.project_weight.append(self.project[self.flowkey[i]].weight.data.clone())
            self.filter[self.flowkey[i]] = nn.Conv2d(96, 2, 3, bias=False).cuda(self.device)
            self.filter_weight.append(self.filter[self.flowkey[i]].weight.data.clone())

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

        im_l = []
        lb_l = []
        if speedrun:
            image, labels, obj_ids = sequence[0]
            image = image.to(self.device)
            labels = labels.to(self.device)
            im_l.append(image)
            im_l.append(image)
            lb_l.append(labels)
            lb_l.append(labels)
            self.initialize(image, labels, sequence.obj_ids)  # Assume DAVIS 2016
            self.track(im_l, lb_l)
            torch.cuda.synchronize()
            self.targets = dict()

        self.outputs = []
        t0 = time()

        for i, (image, labels, new_objects) in tqdm.tqdm(enumerate(sequence), desc=sequence.name, total=len(sequence), unit='frames'):

            old_objects = set(self.targets.keys())

            image = image.to(self.device)
            if len(new_objects) > 0:
                labels = labels.to(self.device)
                self.initialize(image, labels, new_objects)

            if len(old_objects) > 0:
                inter = 2

                images = []
                labels = []
                if i >= inter - 1:
                    for res in range(inter):
                        im, lb, _ = sequence.__getitem__(i - (inter - 1 - res))
                        images.append(im)
                        labels.append(lb)
                else:
                    for res in range(i):
                        im, lb, _ = sequence.__getitem__(res)
                        images.append(im)
                        labels.append(lb)
                    for res in range(inter - i):
                        im, lb, _ = sequence.__getitem__(i)
                        images.append(im)
                        labels.append(lb)

                self.track(images, labels)

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
        images_pad = []
        for i in range(len(images)):
            images[i] = images[i].unsqueeze(0).float()
            masks[i] = masks[i].unsqueeze(0).cuda(images[0].device)
            features.append(self.feature_extractor(images[i]))
            images_pad.append(self.padder.pad(images[i])[0])
            features_flow.append(self.feature_extractor(images_pad[i]))
        
        # Classify
        for obj_id, target in self.targets.items():
            if target.start_frame < self.current_frame:
                flows = []
                # obj_id = 0
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
                
                    if torch.sum(mask) == 0:
                        fmap = torch.cat((features_flow[i]['layer3'], features_flow[i - 1]['layer3']), 1)
                        flow = self.filter[self.flowkey[obj_id]](self.project[self.flowkey[obj_id]](fmap.detach()))
                        flow = F.interpolate(flow, size=im_size, mode='bilinear')
                        # robust_mask[...] = 1
                    else:
                        for j in range(4):
                            parameters = TensorList([self.project[self.flowkey[obj_id]].weight,
                                                     self.filter[self.flowkey[obj_id]].weight])
                            fmap = torch.cat((features_flow[i]['layer3'], features_flow[i - 1]['layer3']), 1)
                            flow = self.filter[self.flowkey[obj_id]](
                                self.project[self.flowkey[obj_id]](fmap.detach()))
                            flow = F.interpolate(flow, size=im_size, mode='bilinear')
                            warp_im = warp(images[i - 1] / 255., flow)
                            residuals = (images[i] / 255. - warp_im).abs() * robust_mask
                            residuals = residuals.sum() * 0.15 * 0.8
                            residuals = TensorList([residuals])
                            parameters.requires_grad_(True)
                            rsd_g = TensorList(torch.autograd.grad(residuals, parameters, create_graph=True))
                            rsd_g_2 = 0
                            for k in range(len(rsd_g)):
                                rsd_g_2 += (rsd_g[k] * rsd_g[k]).sum()
                            rsd_alpha = residuals[0] / rsd_g_2 * 0.8 * 0.15
                            step = rsd_g.apply(
                                lambda e: rsd_alpha.reshape([-1 if d == 0 else 1 for d in range(e.dim())]) * e)
                            self.project[self.flowkey[obj_id]].weight.data = parameters[0] - step[0]
                            self.filter[self.flowkey[obj_id]].weight.data = parameters[1] - step[1]
                    flows.append(flow)

                with torch.no_grad():
                    s = []
                    for i in range(len(images)):
                        if i == len(images) - 1:
                            s.append(target.classify(features[i][target.disc_layer]))
                        else:
                            s.append(target.classify_(features[i][target.disc_layer]))
                    y, flows = self.refiner(s, features, features_flow, images, im_size, flows=flows, padder=None,
                                            mask1=masks[-1], mask2=masks[-1], current_output=self.output_mask)
                    y = torch.sigmoid(y)
                    self.current_masks[target.index] = y

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

        for i in range(self.current_masks.shape[0]):
            self.current_masks[i] = segs[i] * (inds == i).float()

        for obj_id, target in self.targets.items():
            if target.start_frame < self.current_frame and self.disc_params.update_filters:
                target.discriminator.update(self.current_masks[target.index].unsqueeze(0).unsqueeze(0))

        return self.current_masks


