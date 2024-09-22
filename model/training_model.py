import torch
import torch.nn as nn

from lib.utils import AverageMeter, interpolate
from model.tracker import SpatialTransformer
from lib.training_datasets import SampleSpec
from .augmenter import ImageAugmenter
from .discriminator import Discriminator

import torch.nn.functional as F


def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(255.0 / torch.sqrt(mse))


class TargetObject:

    def __init__(self, disc_params, **kwargs):

        self.discriminator = Discriminator(**disc_params)
        for key, val in kwargs.items():
            setattr(self, key, val)

    def initialize(self, ft, mask):
        self.discriminator.init(ft[self.discriminator.layer], mask)

    def initialize_pretrained(self, state_dict):

        self.discriminator.load_state_dict(state_dict)
        for p in self.discriminator.parameters():
            p.requires_grad_(False)
        self.discriminator.eval()

    def get_state_dict(self):
        return self.discriminator.state_dict()

    def classify(self, ft):
        return self.discriminator.apply(ft)


class TrainerModel(nn.Module):

    def __init__(self, augmenter, feature_extractor, feature_extractor_flow, disc_params, seg_network, batch_size=0,
                 tmodel_cache=None, device=None):

        super().__init__()

        self.augmenter = augmenter
        self.augment = augmenter.augment_first_frame
        self.tmodels = [TargetObject(disc_params) for _ in range(batch_size)]
        self.tmodels_ref = [TargetObject(disc_params) for _ in range(batch_size)]
        self.feature_extractor = feature_extractor
        self.feature_extractor_flow = feature_extractor_flow
        self.refiner = seg_network
        self.tmodel_cache = tmodel_cache
        self.device = device

        self.compute_loss = nn.BCELoss()
        self.compute_accuracy = self.intersection_over_union

        self.scores = None
        self.ft_channels = None

        self.warp = SpatialTransformer(size=[480, 864])

    def load_state_dict(self, state_dict):

        prefix = "refiner."
        sd2 = dict()
        for k, v in state_dict.items():
            assert k.startswith(prefix)
            k = k[len(prefix):]
            sd2[k] = v

        self.refiner.load_state_dict(sd2)

    def state_dict(self):
        return self.refiner.state_dict(prefix="refiner.")

    def intersection_over_union(self, pred, gt):

        pred = (pred > 0.5).float()
        gt = (gt > 0.5).float()

        intersection = pred * gt
        i = intersection.sum(dim=-2).sum(dim=-1)
        union = ((pred + gt) > 0.5).float()
        u = union.sum(dim=-2).sum(dim=-1)
        iou = i / u

        iou[torch.isinf(iou)] = 0.0
        iou[torch.isnan(iou)] = 1.0

        return iou

    def forward(self, images, labels, meta):

        specs = SampleSpec.from_encoded(meta)

        seglosses = AverageMeter()
        flowlosses = AverageMeter()
        iter_acc = 0
        n = 0

        cache_hits = self._initialize(images[0], labels[0], specs)

        s, flows = self._forward(images)

        y = labels[-1].to(self.device).float()
        acc = self.compute_accuracy(s.detach(), y)
        segloss = self.compute_loss(s, y)

        loss = segloss # + flowloss
        loss.backward()

        seglosses.update(loss.item())
        iter_acc += acc.mean().cpu().numpy()
        n += 1

        stats = dict()
        stats['stats/segloss'] = seglosses.avg
        stats['stats/flowloss'] = flowlosses.avg
        stats['stats/accuracy'] = iter_acc / n
        stats['stats/fcache_hits'] = cache_hits

        return stats

    def _initialize(self, first_image, first_labels, specs):

        cache_hits = 0

        # Augment first image and extract features
        L = self.tmodels[0].discriminator.layer

        N = first_image.shape[0]  # Batch size
        for i in range(N):

            state_dict = None
            if self.tmodel_cache.enable:
                state_dict = self.load_target_model(specs[i], L)

            have_pretrained = (state_dict is not None)

            if not have_pretrained:
                im, lb = self.augment(first_image[i].to(self.device), first_labels[i].to(self.device))
                ft = self.feature_extractor.no_grad_forward(im, output_layers=[L], chunk_size=4)
                self.tmodels[i].initialize(ft, lb)

                if self.tmodel_cache.enable and not self.tmodel_cache.read_only:
                    self.save_target_model(specs[i], L, self.tmodels[i].get_state_dict())

            else:
                if self.ft_channels is None:
                    self.ft_channels = self.feature_extractor.get_out_channels()[L]
                self.tmodels[i].initialize_pretrained(state_dict)
                cache_hits += 1

        return cache_hits

    def _forward(self, image):
        image = image[1:]
        features = []
        features_flow = []
        for i in range(len(image)):
            image[i] = image[i].to(self.device)
            features.append(self.feature_extractor(image[i]))
            features_flow.append(self.feature_extractor_flow(image[i]))

        batch_size = image[0].shape[0]

        scores_all = []
        for j in range(len(image)):
            ft = features[j][self.tmodels[0].discriminator.layer]
            scores = []
            for i, tmdl in zip(range(batch_size), self.tmodels):
                x = ft[i, None]
                s = tmdl.classify(x)
                scores.append(s)
            scores = torch.cat(scores, dim=0)
            scores_all.append(scores)

        y, flow = self.refiner(scores_all, features, features_flow, image, image[0].shape)

        for i in range(len(y)):
            y[i] = torch.sigmoid(interpolate(y[i], image[0].shape[-2:]))

        return y, flow

    def tmodel_filename(self, spec, layer_name):
        return self.tmodel_cache.path / spec.seq_name / ("%05d.%d.%s.pth" % (spec.frame0_id, spec.obj_id, layer_name))

    def load_target_model(self, spec, layer_name):
        fname = self.tmodel_filename(spec, layer_name)
        try:
            state_dict = torch.load(fname, map_location=self.device) if fname.exists() else None
        except Exception as e:
            print("Could not read %s: %s" % (fname, e))
            state_dict = None
        return state_dict

    def save_target_model(self, spec: SampleSpec, layer_name, state_dict):
        fname = self.tmodel_filename(spec, layer_name)
        fname.parent.mkdir(exist_ok=True, parents=True)
        torch.save(state_dict, fname)
