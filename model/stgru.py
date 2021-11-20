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

        # self.W_xh = nn.Conv2d(input_dim, input_dim, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

        shape = self.W_ir['1'].weight.shape
        for i in range(3):
            self.W_ir[self.str[i]].weight = torch.nn.Parameter(torch.normal(mean=torch.full(shape, 0.0), std=torch.full(shape, 0.001)))
        shape = self.W_hh['1'].weight.shape
        # self.W_xh.weight = torch.nn.Parameter(6. * identity + torch.normal(mean=torch.full(shape, -0.5), std=torch.full(shape, 0.01)))
        for i in range(3):
            self.W_xh[self.str[i]].weight = torch.nn.Parameter(6. * identity + torch.normal(mean=torch.full(shape, -0.5), std=torch.full(shape, 0.01)))
            self.W_hh[self.str[i]].weight = torch.nn.Parameter(6. * identity + torch.normal(mean=torch.full(shape, -0.5), std=torch.full(shape, 0.01)))

    def forward(self, images, flows, unary_inputs, current_output=None):

        warp = SpatialTransformer(images[0].shape[-2:]).cuda(images[0].device)

        # for i in range(len(unary_inputs)):
        #     unary_inputs[i] = torch.sigmoid(unary_inputs[i]) - 0.5

        # h_tlide = 0
        for i in range(len(images) - 1):
            I_warp = images[i] / 255.
            I_warp = warp(I_warp, flows[i])
            unary_warp = warp(unary_inputs[i], flows[i])
            I_diff = (images[-1] / 255. - I_warp).abs()
            r = 1. - torch.sigmoid(self.W_ir[self.str[i]](I_diff))
            h_tlide = self.W_xh[self.str[i]](unary_inputs[i+1]) + self.W_hh[self.str[i]](unary_warp) * r
            unary_inputs[i + 1] = h_tlide

        return unary_inputs[-1]


class stgru_lzy(nn.Module):

    def __init__(self, input_dim=1, kernel_size=3):
        super(stgru_lzy, self).__init__()

        identity = torch.zeros((input_dim, input_dim, kernel_size, kernel_size))
        for k in range(input_dim):
            identity[k, k, kernel_size // 2, kernel_size // 2] = 1.

        self.W_ir = nn.Conv2d(3, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.W_xh = nn.Conv2d(input_dim, input_dim, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.W_hh = nn.Conv2d(input_dim, input_dim, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

        shape = self.W_ir.weight.shape
        self.W_ir.weight = torch.nn.Parameter(torch.normal(mean=torch.full(shape, 0.0), std=torch.full(shape, 0.001)))
        shape = self.W_xh.weight.shape
        self.W_xh.weight = torch.nn.Parameter(6. * identity + torch.normal(mean=torch.full(shape, -0.5), std=torch.full(shape, 0.01)))
        self.W_hh.weight = torch.nn.Parameter(6. * identity + torch.normal(mean=torch.full(shape, -0.5), std=torch.full(shape, 0.01)))

        stds = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float, requires_grad=False).reshape(1, 3, 1, 1)
        means = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float, requires_grad=False).reshape(1, 3, 1, 1)
        self.norm_weight = (1 / stds)
        self.norm_bias = (-means / stds)

        # self.ssim = SSIM()

    def forward(self, image, image_ref, flow, h_prev, unary_input, current_output=None):

        self.norm_weight = self.norm_weight.cuda(image.device)
        self.norm_bias = self.norm_bias.cuda(image.device)

        image = image * self.norm_weight + self.norm_bias
        image_ref = image_ref * self.norm_weight + self.norm_bias

        warp = SpatialTransformer(image.shape[-2:]).cuda(image.device)

        I_diff = image - warp(image_ref, flow)
        I_diff = torch.abs(I_diff)
        h_prev_warped = warp(h_prev, flow)
        r = 1. - torch.sigmoid(self.W_ir(I_diff))
        h_tlide = self.W_xh(unary_input) + self.W_hh(h_prev_warped) * r

        return h_tlide


class stgru_raw(nn.Module):

    def __init__(self, input_dim=1, kernel_size=3):
        super(stgru_raw, self).__init__()

        identity = torch.zeros((input_dim, input_dim, kernel_size, kernel_size))
        for k in range(input_dim):
            identity[k, k, kernel_size // 2, kernel_size // 2] = 1.

        self.W_ir = nn.Conv2d(3, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.W_xh = nn.Conv2d(input_dim, input_dim, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.W_hh = nn.Conv2d(input_dim, input_dim, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.W_xz = nn.Conv2d(input_dim, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.W_hz = nn.Conv2d(input_dim, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

        self.lamb = nn.Parameter(torch.ones(1))
        self.bias_r = nn.Parameter(torch.zeros(1))
        self.bias_z = nn.Parameter(torch.zeros(1))

        shape = self.W_ir.weight.shape
        self.W_ir.weight = torch.nn.Parameter(torch.normal(mean=torch.full(shape, 0.0), std=torch.full(shape, 0.001)))
        shape = self.W_xh.weight.shape
        self.W_xh.weight = torch.nn.Parameter(6. * identity + torch.normal(mean=torch.full(shape, 0.0), std=torch.full(shape, 0.01)))
        self.W_hh.weight = torch.nn.Parameter(6. * identity + torch.normal(mean=torch.full(shape, 0.0), std=torch.full(shape, 0.01)))
        shape = self.W_xz.weight.shape
        self.W_xz.weight = torch.nn.Parameter(torch.normal(mean=torch.full(shape, 0.0), std=torch.full(shape, 0.01)))
        self.W_hz.weight = torch.nn.Parameter(torch.normal(mean=torch.full(shape, 0.0), std=torch.full(shape, 0.01)))

        stds = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float, requires_grad=False).reshape(1, 3, 1, 1)
        means = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float, requires_grad=False).reshape(1, 3, 1, 1)
        self.norm_weight = (1 / stds)
        self.norm_bias = (-means / stds)

    def forward(self, image, image_ref, flow, h_prev, unary_input, current_output=None):

        self.norm_weight = self.norm_weight.cuda(image.device)
        self.norm_bias = self.norm_bias.cuda(image.device)

        image = image * self.norm_weight + self.norm_bias
        image_ref = image_ref * self.norm_weight + self.norm_bias

        warp = SpatialTransformer(image.shape[-2:]).cuda(image.device)

        h_prev = torch.sigmoid(h_prev) - 0.5
        unary_input = torch.sigmoid(unary_input) - 0.5

        I_diff = image - warp(image_ref, flow)
        h_prev_warped = warp(h_prev, flow)

        # I_diff_ref = torch.zeros(I_diff.shape).cuda(I_diff.device)
        r = 1. - torch.tanh(torch.abs(self.W_ir(I_diff) + self.bias_r))
        h_prev_reset = h_prev_warped * r
        h_tlide = self.W_xh(unary_input) + self.W_hh(h_prev_reset)
        z = torch.sigmoid(self.W_xz(unary_input) + self.W_hz(h_prev_reset) + self.bias_z)
        h = self.lamb * (1 - z) * h_prev_reset + z * h_tlide

        return h


class stgru_modify(nn.Module):

    def __init__(self, input_dim=1, kernel_size=3):
        super(stgru_modify, self).__init__()

        identity = torch.zeros((input_dim, input_dim, kernel_size, kernel_size))
        for k in range(input_dim):
            identity[k, k, kernel_size // 2, kernel_size // 2] = 1.

        self.str = ['1', '2', '3', '4']

        self.W_ir = nn.Conv2d(3, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.W_xh = nn.Conv2d(input_dim, input_dim, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.W_hh = nn.Conv2d(input_dim, input_dim, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.W_xz = nn.Conv2d(input_dim, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.W_hz = nn.Conv2d(input_dim, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

        self.lamb = nn.Parameter(torch.ones(1))
        self.bias_r = nn.Parameter(torch.zeros(1))
        self.bias_z = nn.Parameter(torch.zeros(1))

        shape = self.W_ir.weight.shape
        self.W_ir.weight = torch.nn.Parameter(torch.normal(mean=torch.full(shape, 0.0), std=torch.full(shape, 0.001)))
        shape = self.W_xh.weight.shape
        self.W_xh.weight = torch.nn.Parameter(6. * identity + torch.normal(mean=torch.full(shape, 0.0), std=torch.full(shape, 0.01)))
        self.W_hh.weight = torch.nn.Parameter(6. * identity + torch.normal(mean=torch.full(shape, 0.0), std=torch.full(shape, 0.01)))
        shape = self.W_xz.weight.shape
        self.W_xz.weight = torch.nn.Parameter(torch.normal(mean=torch.full(shape, 0.0), std=torch.full(shape, 0.01)))
        self.W_hz.weight = torch.nn.Parameter(torch.normal(mean=torch.full(shape, 0.0), std=torch.full(shape, 0.01)))

        stds = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float, requires_grad=False).reshape(1, 3, 1, 1)
        means = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float, requires_grad=False).reshape(1, 3, 1, 1)
        self.norm_weight = (1 / stds)
        self.norm_bias = (-means / stds)

    def forward(self, image, image_ref, flow, h_prev, unary_input, current_output=None):

        self.norm_weight = self.norm_weight.cuda(image.device)
        self.norm_bias = self.norm_bias.cuda(image.device)

        image = image * self.norm_weight + self.norm_bias
        image_ref = image_ref * self.norm_weight + self.norm_bias

        warp = SpatialTransformer(image.shape[-2:]).cuda(image.device)

        h_prev = torch.sigmoid(h_prev) - 0.5
        unary_input = torch.sigmoid(unary_input) - 0.5

        I_diff = image - warp(image_ref, flow)
        h_prev_warped = warp(h_prev, flow)

        # I_diff_ref = torch.zeros(I_diff.shape).cuda(I_diff.device)
        r = 1. - torch.tanh(torch.abs(self.W_ir(I_diff) + self.bias_r))
        h_prev_reset = h_prev_warped * r
        h_tlide = self.W_xh(unary_input) + self.W_hh(h_prev_reset)
        z = torch.sigmoid(self.W_xz(unary_input) + self.W_hz(h_prev_reset) + self.bias_z)
        h = self.lamb * (1 - z) * h_prev_reset + z * h_tlide

        return h


class gru(nn.Module):

    def __init__(self, hidden_dim=1, input_dim=1):
        super(gru, self).__init__()
        self.convz = nn.Conv2d(1, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(1, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(4, hidden_dim, 3, padding=1)

    def forward(self, image, image_ref, flow, h_prev, unary_input):

        hx = h_prev
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * hx, flow, unary_input], dim=1)))
        # h = (1 - z) * h_prev_warped + z * q
        h = (1 - z) * hx + z * q

        return h


class gru_vos(nn.Module):

    def __init__(self, hidden_dim=1, input_dim=1):
        super(gru_vos, self).__init__()

        self.conxz = nn.Conv2d(1, 1, 3, padding=1)
        self.conhz = nn.Conv2d(1, 1, 3, padding=1)

        self.conxr = nn.Conv2d(1, 1, 3, padding=1)
        self.conhr = nn.Conv2d(1, 1, 3, padding=1)

        self.conxh = nn.Conv2d(1, 1, 3, padding=1)
        self.conhh = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, image, image_ref, flow, h_prev, unary_input):

        warp = SpatialTransformer(image.shape[-2:]).cuda(image.device)

        h = warp(h_prev, flow)
        x = unary_input

        z = torch.sigmoid(self.conxz(x) + self.conhz(h))
        r = torch.sigmoid(self.conxr(x) + self.conhr(h))

        h_ = torch.tanh(self.conxh(x) + r * self.conhh(h))
        h__ = (1 - z) * h + z * h_

        return h__
