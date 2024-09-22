import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def convert_to_mask(inp):
    input = []

    ids = torch.arange(0, 2)
    inp = torch.sigmoid(inp)
    back_g = 1 - inp

    input.append(back_g)
    input.append(inp)

    mask = input[0].squeeze().unsqueeze(0)
    for tensor in input:
        mask = torch.cat((mask, tensor.squeeze().unsqueeze(0)), dim=0)
    mask = mask[1:]

    masks = torch.clamp(mask, 1e-7, 1 - 1e-7)
    masks[0:1] = torch.min((1 - masks[1:]), dim=0, keepdim=True)[0]  # background activation
    segs = F.softmax(masks / (1 - masks), dim=0)  # s = one-hot encoded object activations
    labels = ids[segs.argmax(dim=0)]

    return labels.unsqueeze(0).unsqueeze(0).float().cuda(inp.device)


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


class stgru(nn.Module):

    def __init__(self, input_dim=1, kernel_size=3):
        super(stgru, self).__init__()

        identity = torch.zeros((input_dim, input_dim, kernel_size, kernel_size))
        for k in range(input_dim):
            identity[k, k, kernel_size // 2, kernel_size // 2] = 1.

        self.W_ir = nn.ModuleDict()
        self.W_hh = nn.ModuleDict()
        self.W_xh = nn.ModuleDict()

        self.str = ['1', '2', '3']

        for i in range(3):
            self.W_ir[self.str[i]] = nn.Conv2d(3, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            self.W_hh[self.str[i]] = nn.Conv2d(input_dim, input_dim, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            self.W_xh[self.str[i]] = nn.Conv2d(input_dim, input_dim, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

        shape = self.W_ir['1'].weight.shape
        for i in range(3):
            self.W_ir[self.str[i]].weight = torch.nn.Parameter(torch.normal(mean=torch.full(shape, 0.0), std=torch.full(shape, 0.001)))
        shape = self.W_hh['1'].weight.shape
        for i in range(3):
            self.W_xh[self.str[i]].weight = torch.nn.Parameter(6. * identity + torch.normal(mean=torch.full(shape, -0.5), std=torch.full(shape, 0.01)))
            self.W_hh[self.str[i]].weight = torch.nn.Parameter(6. * identity + torch.normal(mean=torch.full(shape, -0.5), std=torch.full(shape, 0.01)))

    def forward(self, images, flows, unary_inputs, current_output=None):

        warp = SpatialTransformer(images[0].shape[-2:]).cuda(images[0].device)
        for i in range(len(images) - 1):
            I_warp = images[i] / 255.
            I_warp = warp(I_warp, flows[i])
            unary_warp = warp(unary_inputs[i], flows[i])
            I_diff = (images[-1] / 255. - I_warp).abs()
            r = 1. - torch.sigmoid(self.W_ir[self.str[i]](I_diff))
            h_tlide = self.W_xh[self.str[i]](unary_inputs[i+1]) + self.W_hh[self.str[i]](unary_warp) * r
            unary_inputs[i + 1] = h_tlide

        return unary_inputs[-1]