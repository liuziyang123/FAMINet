import torch
import torch.nn as nn
import torch.nn.functional as F


def mesh_grid(B, H, W):
    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)  # BHW
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2)  # BHW

    base_grid = torch.stack([x_base, y_base], 1)  # B2HW
    return base_grid


def norm_grid(v_grid):
    _, _, H, W = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (W - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (H - 1) - 1.0
    return v_grid_norm.permute(0, 2, 3, 1)  # BHW2


def get_corresponding_map(data):
    """

    :param data: unnormalized coordinates Bx2xHxW
    :return: Bx1xHxW
    """
    B, _, H, W = data.size()

    # x = data[:, 0, :, :].view(B, -1).clamp(0, W - 1)  # BxN (N=H*W)
    # y = data[:, 1, :, :].view(B, -1).clamp(0, H - 1)

    x = data[:, 0, :, :].view(B, -1)  # BxN (N=H*W)
    y = data[:, 1, :, :].view(B, -1)

    # invalid = (x < 0) | (x > W - 1) | (y < 0) | (y > H - 1)   # BxN
    # invalid = invalid.repeat([1, 4])

    x1 = torch.floor(x)
    x_floor = x1.clamp(0, W - 1)
    y1 = torch.floor(y)
    y_floor = y1.clamp(0, H - 1)
    x0 = x1 + 1
    x_ceil = x0.clamp(0, W - 1)
    y0 = y1 + 1
    y_ceil = y0.clamp(0, H - 1)

    x_ceil_out = x0 != x_ceil
    y_ceil_out = y0 != y_ceil
    x_floor_out = x1 != x_floor
    y_floor_out = y1 != y_floor
    invalid = torch.cat([x_ceil_out | y_ceil_out,
                         x_ceil_out | y_floor_out,
                         x_floor_out | y_ceil_out,
                         x_floor_out | y_floor_out], dim=1)

    # encode coordinates, since the scatter function can only index along one axis
    corresponding_map = torch.zeros(B, H * W).type_as(data)
    indices = torch.cat([x_ceil + y_ceil * W,
                         x_ceil + y_floor * W,
                         x_floor + y_ceil * W,
                         x_floor + y_floor * W], 1).long()  # BxN   (N=4*H*W)
    values = torch.cat([(1 - torch.abs(x - x_ceil)) * (1 - torch.abs(y - y_ceil)),
                        (1 - torch.abs(x - x_ceil)) * (1 - torch.abs(y - y_floor)),
                        (1 - torch.abs(x - x_floor)) * (1 - torch.abs(y - y_ceil)),
                        (1 - torch.abs(x - x_floor)) * (1 - torch.abs(y - y_floor))],
                       1)
    # values = torch.ones_like(values)

    values[invalid] = 0

    corresponding_map.scatter_add_(1, indices, values)
    # decode coordinates
    corresponding_map = corresponding_map.view(B, H, W)

    return corresponding_map.unsqueeze(1)


def get_occu_mask_backward(flow21, th=0.2):
    B, _, H, W = flow21.size()
    base_grid = mesh_grid(B, H, W).type_as(flow21)  # B2HW

    corr_map = get_corresponding_map(base_grid + flow21)  # BHW
    occu_mask = corr_map.clamp(min=0., max=1.) < th
    return occu_mask.float()


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


def masks_to_bboxes_gauss(mask, fmt='c', psize=None):

    """ Convert a mask tensor to one or more bounding boxes.
    Note: This function is a bit new, make sure it does what it says.  /Andreas
    :param mask: Tensor of masks, shape = (..., H, W)
    :param fmt: bbox layout. 'c' => "center + size" or (x_center, y_center, width, height)
                             't' => "top left + size" or (x_left, y_top, width, height)
                             'v' => "vertices" or (x_left, y_top, x_right, y_bottom)
    :return: tensor containing a batch of bounding boxes, shape = (..., 4)
    """
    binary_image = mask.cpu().numpy()
    center, std = compute_robust_moments(binary_image)
    batch_shape = mask.shape[:-2]
    mask = mask.reshape((-1, *mask.shape[-2:]))
    bboxes = []

    H, W = mask.shape[-2:]
    if psize == None:
        p_size = 0
    else:
        p_size = psize

    for m in mask:
        # mx = m.sum(dim=-2).nonzero()
        # my = m.sum(dim=-1).nonzero()
        mx = torch.nonzero(m.sum(dim=-2), as_tuple=False)
        my = torch.nonzero(m.sum(dim=-1), as_tuple=False)
        if (len(mx) > 0 and len(my) > 0):
            if round(center[0] - 2 * std[0] - p_size) > 0:
                a = round(center[0] - 2 * std[0] - p_size)
            else:
                a = 0
            if round(center[0] + 2 * std[0] + p_size) < W:
                b = round(center[0] + 2 * std[0] + p_size)
            else:
                b = W
            if round(center[1] - 2 * std[1] - p_size) > 0:
                c = round(center[1] - 2 * std[1] - p_size)
            else:
                c = 0
            if round(center[1] + 2 * std[1] + p_size) < H:
                d = round(center[1] + 2 * std[1] + p_size)
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


def compute_robust_moments(binary_image, isotropic=False):
  index = np.nonzero(binary_image)
  points = np.asarray(index).astype(np.float32)
  if points.shape[1] == 0:
    return np.array([-1.0,-1.0],dtype=np.float32), \
        np.array([-1.0,-1.0],dtype=np.float32)
  points = np.transpose(points)
  points[:,[0,1]] = points[:,[1,0]]
  center = np.median(points, axis=0)
  if isotropic:
    diff = np.linalg.norm(points - center, axis=1)
    mad = np.median(diff)
    mad = np.array([mad,mad])
  else:
    diff = np.absolute(points - center)
    mad = np.median(diff, axis=0)
  std_dev = 1.4826*mad
  std_dev = np.maximum(std_dev, [5.0, 5.0])
  return center, std_dev


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

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        # x = (x + 1.0) / 2.0
        # y = (y + 1.0) / 2.0

        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    # img = (img + 1.0) / 2.0

    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()

def davis_loss(image1, image2, flow_preds, gamma=0.8):
    """ Loss function defined over sequence of flow predictions """

    warp = SpatialTransformer(image1.shape[-2:]).cuda(image1.device)
    ssim = SSIM()

    # # when training frtm
    # flow_preds = flow_preds.unsqueeze(0)
    # flow_preds = list(flow_preds)

    n_predictions = len(flow_preds)
    flow_loss = 0.0
    mae_ = 0.0
    ssim_ = 0.0
    smooth_ = 0.0

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        image2_pre = warp(image1, flow_preds[i])
        mae_loss = (image2_pre - image2).abs()
        ssim_loss = ssim(image2_pre, image2)
        smooth_loss = get_smooth_loss(flow_preds[i], image2)

        mae_ += mae_loss
        ssim_ += ssim_loss
        smooth_ += smooth_loss
        flow_loss += i_weight * mae_loss.mean() # * 0.15 + \
                     # i_weight * ssim_loss.mean() * 0.85 + i_weight * smooth_loss * 0.1

    flow_loss_item = flow_loss.item()

    metrics = {
        'l1_loss': mae_.mean().item(),
        'ssim_loss': ssim_.mean().item(),
        'smooth_loss': smooth_.item(),
        'flow_loss': flow_loss_item,
    }

    return flow_loss, metrics


def selftune(model, fmap1, fmap2, image1, image2):
    # self fine-tune
    optimizer = torch.optim.Adam(model.flow.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    for m in model.flow.parameters():
        m.requires_grad = True
    model.train()

    fmap_1 = fmap1.detach()
    fmap_2 = fmap2.detach()
    image = image1.detach()

    for i in range(20):
        for itr in range(1):
            flows = model(fmap_1, fmap_2, image, iters=1, test_mode=False)
        optimizer.zero_grad()
        loss, _ = davis_loss(image1, image2, flows)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())

    for m in model.flow.parameters():
        m.requires_grad = False
    model.eval()

    return model
