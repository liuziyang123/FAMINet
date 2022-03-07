import torch
from torch import nn as nn
from torch.nn import functional as F
from lib.utils import conv, relu, interpolate, adaptive_cat
from lib.selftune import SpatialTransformer
from collections import OrderedDict
from feature_extractor import ResnetFeatureExtractor

from torchvision.models import resnet18, resnet34, resnet50, resnet101
from lib.utils import get_out_channels

from .stgru import stgru, gru, davis_loss, stgru_raw, stgru_lzy, gru_vos
import cv2


Count = 0
dir = './results/viz/'
def save_tensor(input):
    global Count
    input = input.squeeze().cpu().numpy()
    cv2.imwrite(dir + str(Count) + '.png', input * 255)
    Count += 1


def norm_flow(flow):
    B, _, H, W = flow.shape
    normflow = torch.sqrt(flow[:, 0]**2 + flow[:, 1]**2)
    maxflow, _ = torch.max(normflow.view(B, H*W), dim=-1)
    minflow, _ = torch.min(normflow.view(B, H*W), dim=-1)
    normflow = (normflow - minflow.view(B, 1, 1)) / maxflow.view(B, 1, 1)

    return normflow.unsqueeze(1)


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


class TSE(nn.Module):

    def __init__(self, fc, ic, oc):
        super().__init__()

        nc = ic + oc
        self.reduce = nn.Sequential(conv(fc, oc, 1), relu(), conv(oc, oc, 1))
        self.transform = nn.Sequential(conv(nc, nc, 3), relu(), conv(nc, nc, 3), relu(), conv(nc, oc, 3), relu())

    def forward(self, ft, score, x=None):
        h = self.reduce(ft)
        hpool = F.adaptive_avg_pool2d(h, (1, 1)) if x is None else x
        h = adaptive_cat((h, score), dim=1, ref_tensor=0)
        h = self.transform(h)
        return h, hpool


class CAB(nn.Module):

    def __init__(self, oc, deepest):
        super().__init__()

        self.convreluconv = nn.Sequential(conv(2 * oc, oc, 1), relu(), conv(oc, oc, 1))
        self.deepest = deepest

    def forward(self, deeper, shallower, att_vec=None):

        shallow_pool = F.adaptive_avg_pool2d(shallower, (1, 1))
        deeper_pool = deeper if self.deepest else F.adaptive_avg_pool2d(deeper, (1, 1))
        if att_vec is not None:
            global_pool = torch.cat([shallow_pool, deeper_pool, att_vec], dim=1)
        else:
            global_pool = torch.cat((shallow_pool, deeper_pool), dim=1)
        conv_1x1 = self.convreluconv(global_pool)
        inputs = shallower * torch.sigmoid(conv_1x1)
        out = inputs + interpolate(deeper, inputs.shape[-2:])

        return out


class RRB(nn.Module):

    def __init__(self, oc, use_bn=False):
        super().__init__()
        self.conv1x1 = conv(oc, oc, 1)
        if use_bn:
            self.bblock = nn.Sequential(conv(oc, oc, 3), nn.BatchNorm2d(oc), relu(), conv(oc, oc, 3, bias=False))
        else:
            self.bblock = nn.Sequential(conv(oc, oc, 3), relu(), conv(oc, oc, 3, bias=False))  # Basic block

    def forward(self, x):
        h = self.conv1x1(x)
        return F.relu(h + self.bblock(h))


class Upsampler(nn.Module):

    def __init__(self, in_channels=64):
        super().__init__()

        self.conv1 = conv(in_channels, in_channels // 2, 3)
        self.conv2 = conv(in_channels // 2, 1, 3)

    def forward(self, x, image_size):
        print(x.shape)
        x = F.interpolate(x, (2 * x.shape[-2], 2 * x.shape[-1]), mode='bicubic', align_corners=False)
        x = F.relu(self.conv1(x))
        x = F.interpolate(x, image_size[-2:], mode='bicubic', align_corners=False)
        x = self.conv2(x)
        return x


class PyrUpBicubic2d(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.channels = channels

        def kernel(d):
            x = d + torch.arange(-1, 3, dtype=torch.float32)
            x = torch.abs(x)
            a = -0.75
            f = (x < 1).float() * ((a + 2) * x * x * x - (a + 3) * x * x + 1) + \
                ((x >= 1) * (x < 2)).float() * (a * x * x * x - 5 * a * x * x + 8 * a * x - 4 * a)
            W = f.reshape(1, 1, 1, len(x)).float()
            Wt = W.permute(0, 1, 3, 2)
            return W, Wt

        We, We_t = kernel(-0.25)
        Wo, Wo_t = kernel(-0.25 - 0.5)

        self.W00 = (We_t @ We).expand(channels, 1, 4, 4).contiguous()           # matrix multiply
        self.W01 = (We_t @ Wo).expand(channels, 1, 4, 4).contiguous()
        self.W10 = (Wo_t @ We).expand(channels, 1, 4, 4).contiguous()
        self.W11 = (Wo_t @ Wo).expand(channels, 1, 4, 4).contiguous()

    def forward(self, input):

        if input.device != self.W00.device:
            self.W00 = self.W00.to(input.device)
            self.W01 = self.W01.to(input.device)
            self.W10 = self.W10.to(input.device)
            self.W11 = self.W11.to(input.device)

        a = F.pad(input, (2, 2, 2, 2), 'replicate')

        I00 = F.conv2d(a, self.W00, groups=self.channels)
        I01 = F.conv2d(a, self.W01, groups=self.channels)
        I10 = F.conv2d(a, self.W10, groups=self.channels)
        I11 = F.conv2d(a, self.W11, groups=self.channels)

        n, c, h, w = I11.shape

        J0 = torch.stack((I00, I01), dim=-1).view(n, c, h, 2 * w)
        J1 = torch.stack((I10, I11), dim=-1).view(n, c, h, 2 * w)
        out = torch.stack((J0, J1), dim=-2).view(n, c, 2 * h, 2 * w)

        out = F.pad(out, (-1, -1, -1, -1))
        return out


class SegNetwork(nn.Module):

    def __init__(self, in_channels=1, out_channels=32, ft_channels=None, use_bn=False):

        super().__init__()

        assert ft_channels is not None
        self.ft_channels = ft_channels

        self.TSE = nn.ModuleDict() 
        self.RRB1 = nn.ModuleDict()
        self.CAB = nn.ModuleDict()
        self.RRB2 = nn.ModuleDict()

        ic = in_channels
        oc = out_channels

        for L, fc in self.ft_channels.items():
            self.TSE[L] = TSE(fc, ic, oc)
            self.RRB1[L] = RRB(oc, use_bn=use_bn)
            self.CAB[L] = CAB(oc, L == 'layer5')
            self.RRB2[L] = RRB(oc, use_bn=use_bn)

        #if torch.__version__ == '1.0.1'
        self.project = BackwardCompatibleUpsampler(out_channels)
        #self.project = Upsampler(out_channels)

        self.gru = stgru()
        # self.gru = gru()
        # self.gru = gru_vos()
        # self.gru = stgru_raw()

    def forward(self, scores, features, features_flow, images, image_size, flows, padder=None, mask1=None, mask2=None, current_output=None):

        num_targets = scores[0].shape[0]
        num_fmaps = features[0][next(iter(self.ft_channels))].shape[0]
        if num_targets > num_fmaps:
            multi_targets = True
        else:
            multi_targets = False

        unary_input = []
        for score, feature in zip(scores, features):
            x = None
            for i, L in enumerate(self.ft_channels):
                ft = feature[L]
                s = interpolate(score, ft.shape[-2:])  # Resample scores to match features size

                if multi_targets:
                    h, hpool = self.TSE[L](ft.repeat(num_targets, 1, 1, 1), s, x)
                else:
                    h, hpool = self.TSE[L](ft, s, x)

                h = self.RRB1[L](h)
                h = self.CAB[L](hpool, h)
                x = self.RRB2[L](h)

            unary_input.append(self.project(x, image_size))

        images_raw = []
        for i in range(len(images)):
            if padder == None:
                images_raw.append(images[i])
            else:
                images_raw.append(padder.unpad(images[i]))
        mask = self.gru(images_raw, flows, unary_input, current_output=current_output)
        return mask, flows
