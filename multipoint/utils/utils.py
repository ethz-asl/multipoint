import collections
import numpy as np
import torch
from torchvision.ops import nms
from torchvision.ops.boxes import batched_nms

import torch.nn as nn
import math

def dict_update(d, u):
    """
    Update for nested dictionaries.

    Arguments:
        d: The dictionary to be updated.
        u: The update dictionary.

    Returns:
        The updated dictionary.
    """
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def data_to_device(data, device):
    for key in data.keys():
        if type(data[key]) is torch.Tensor:
            data[key] = data[key].to(device)
        elif type(data[key]) is dict:
            data[key] = data_to_device(data[key], device)
    return data

def tensors_to_dtype(data, dtype):
    for key in data.keys():
        if type(data[key]) is torch.Tensor:
            data[key] = data[key].to(dtype)
        elif type(data[key]) is dict:
            data[key] = data_to_device(data[key], dtype)
    return data

def data_unsqueeze(data, dim):
    for key in data.keys():
        if type(data[key]) is torch.Tensor:
            data[key] = data[key].unsqueeze(dim)
        elif type(data[key]) is dict:
            data[key] = data_unsqueeze(data[key], dim)
    return data

def parse_primitives(names, all_primitives):
    p = all_primitives if (names == 'all') \
            else (names if isinstance(names, list) else [names])
    assert set(p) <= set(all_primitives)
    return p

def generate_keypoint_map(keypoints, image_shape):    
    tmp = keypoints.astype(np.int64)
    keypoint_map = np.zeros(image_shape, dtype=np.bool)
    keypoint_map[tmp[:,0],tmp[:,1]] = True
    return keypoint_map

def depth_to_space(x, block_size):
    N, C, H, W = x.size()
    x = x.view(N, block_size, block_size, C // (block_size ** 2), H, W)  # (N, bs, bs, C//bs^2, H, W)
    x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
    x = x.view(N, C // (block_size ** 2), H * block_size, W * block_size)  # (N, C//bs^2, H * bs, W * bs)
    return x

def space_to_depth(x, block_size):
    N, C, H, W = x.size()
    x = x.view(N, C, H // block_size, block_size, W // block_size, block_size)  # (N, C, H//bs, bs, W//bs, bs)
    x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
    x = x.view(N, C * (block_size ** 2), H // block_size, W // block_size)  # (N, C*bs^2, H//bs, W//bs)
    return x

def box_nms(prob, size, min_prob, iou=0.1, keep_top_k=0, on_cpu=False):
    """Performs non maximum suppression on the heatmap by considering hypothetical
    bounding boxes centered at each pixel's location (e.g. corresponding to the receptive
    field). Optionally only keeps the top k detections.

    Arguments:
        prob: the probability heatmap, with shape `[(B, 1) ,H, W]`
        size: a scalar, the size of the bounding boxes.
        iou: a scalar, the IoU overlap threshold.
        min_prob: a threshold under which all probabilities are discarded before NMS.
        keep_top_k: an integer, the number of top scores to keep.
    """
    if not (len(prob.shape) == 2 or len(prob.shape) == 4):
        raise ValueError('The probability must be either 2D (H,W), or 4D (B, 1, H, W)')

    device = prob.device
    if on_cpu:
        prob = prob.cpu()

    points = (prob > min_prob).nonzero()
    scores = prob[points.split(1, dim=1)]
    batched = len(prob.shape) > 2
    if batched:
        boxes = torch.cat([ points[:,2:]-size * 0.5,  points[:,2:]+size * 0.5], dim=1)
        idxs = points[:,0]
        indices = batched_nms(boxes, scores[:,0], idxs, iou)
    else:
        boxes = torch.cat([points-size * 0.5, points+size * 0.5], dim=1)
        indices = nms(boxes, scores[:,0], iou)

    # nms already returns the sorted indices so just keep the first k elements
    if keep_top_k > 0:
        if batched:
            tmp = torch.ones((0), dtype=indices.dtype, device=indices.device)
            for i in range(prob.shape[0]):
                tmp = torch.cat((tmp, indices[idxs[indices] == i][:keep_top_k]), 0)
            indices = tmp
        else:
            indices = indices[:keep_top_k]

    # start with a zero tensor and fill in the probabilities where they are kept
    prob_nms = torch.zeros_like(prob)
    prob_nms[points[indices].split(1, dim=1)] = scores[indices]

    return prob_nms.to(device)

def get_gaussian_filter(kernel_size, sigma=None, channels = 1):
    if sigma is None:
        sigma = 0.3*((kernel_size-1)*0.5 - 1) + 0.8

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter

def interpolate_descriptors(keypoints, descriptors_lowres, H, W):
    keypoints = keypoints.float()

    keypoints[:,0] = (keypoints[:,0] / (float(H) * 0.5)) - 1.0
    keypoints[:,1] = (keypoints[:,1] / (float(W) * 0.5)) - 1.0
    keypoints = torch.flip(keypoints.view(1, 1, -1, 2), [3])

    desc = torch.nn.functional.grid_sample(descriptors_lowres.unsqueeze(0), keypoints, align_corners=True)[0,:,0,:].transpose(0,1)
    return torch.nn.functional.normalize(desc, p=2, dim=1)

def fix_model_weigth_keys(weights):
    new_weights = collections.OrderedDict()
    for key, value in weights.items():
        renamed = key.split('__')[-1]
        new_weights[renamed] = value

    return new_weights
