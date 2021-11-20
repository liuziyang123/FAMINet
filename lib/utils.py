from collections import OrderedDict as odict

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F


def text_bargraph(values):                                   # convert a scalar of 0-1 into histogram
    blocks = np.array(('u', ' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█', 'o'))
    nsteps = len(blocks) - 2 - 1
    hstep = 1 / (2 * nsteps)
    values = np.array(values)
    nans = np.isnan(values)
    values[nans] = 0  # '░'
    indices = ((values + hstep) * nsteps + 1).astype(np.int)
    indices[values < 0] = 0
    indices[values > 1] = len(blocks) - 1
    graph = blocks[indices]
    graph[nans] = '░'
    graph = str.join('', graph)
    return graph


def conv(ic, oc, ksize, bias=True, dilation=1, stride=1):
    return nn.Conv2d(ic, oc, ksize, padding=ksize // 2, bias=bias, dilation=dilation, stride=stride)


def relu(negative_slope=0.0, inplace=False):
    return nn.LeakyReLU(negative_slope, inplace=inplace)


def interpolate(t, sz):
    sz = sz.tolist() if torch.is_tensor(sz) else sz         # tensor -> list
    return F.interpolate(t, sz, mode='bilinear', align_corners=False) if t.shape[-2:] != sz else t


def adaptive_cat(seq, dim=0, ref_tensor=0):                 # NxCxHxW -> NCxHxW
    sz = seq[ref_tensor].shape[-2:]
    t = torch.cat([interpolate(t, sz) for t in seq], dim=dim)
    return t


def get_out_channels(layer):
    if hasattr(layer, 'out_channels'):                      # 判断一个实例对象中是否包含某个属性
        oc = layer.out_channels
    elif hasattr(layer, '_modules'):
        oc = get_out_channels(layer._modules)
    else:
        ocs = []
        for key in reversed(layer):                         # 将列表中的序列值反转，然后返回一个迭代器
            ocs.append(get_out_channels(layer[key]))

        oc = 0
        for elem in ocs:
            if elem:
                return elem

    return oc


def is_finite(t):
    return (torch.isnan(t) + torch.isinf(t)) == 0


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.seq_avg = []

    def reset(self):
        self.__init__()

    def update(self, val, n=1):
        if not np.isnan(val):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

    def update_multi(self, val):
        val = np.array(val)                 # val可以为一个list
        v = val[~np.isnan(val)]             # 若为scalar， 则scalar -> 1d array；若为nan（非数），则清空，在list中删去该元素
        n = len(v)
        self.val = val
        self.sum += np.nansum(v)            # 对array或list中的非nan元素进行求和
        self.count += n
        self.avg = self.sum / self.count

