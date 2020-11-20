import cv2
from math import pi
import numpy as np
import torch
try:
    import kornia
    from kornia.geometry.warp.homography_warper import homography_warp
    kornia_available = True
except:
    kornia_available = False

from .utils import dict_update, get_gaussian_filter

import torch.nn as nn


homography_adaptation_default_config = {
        'num': 100,
        'aggregation': 'prod',
        'homographies': {
            'translation': True,
            'rotation': True,
            'scaling': True,
            'perspective': True,
            'scaling_amplitude': 0.15,
            'perspective_amplitude_x': 0.15,
            'perspective_amplitude_y': 0.15,
            'patch_ratio': 0.9,
            'max_angle': pi,
            'allow_artifacts': True,
        },
        'erosion_radius': 5,
        'mask_border': True,
        'min_count': 2,
        'filter_size': 0,
}

def homographic_adaptation_multispectral(data, net, homographic_adaptation_config = {}):
    device = data['optical']['image'].device
    config = dict_update(homography_adaptation_default_config, homographic_adaptation_config)

    if config['num'] < 1:
        raise ValueError('num must be larger than 0 for the homographic adaptation')

    if config['filter_size'] % 2 == 0 and config['filter_size'] != 0:
        raise ValueError('The filter_size must be uneven')

    image_shape = data['optical']['image'].shape

    # process the original images
    out_optical = net(data['optical'])
    out_thermal = net(data['thermal'])

    if config['filter_size'] > 0:
        filter = get_gaussian_filter(config['filter_size']).to(device)
        pad = nn.ReflectionPad2d(int((config['filter_size'] - 1) / 2))
        out_optical['prob'] = filter(pad(out_optical['prob']))
        out_thermal['prob'] = filter(pad(out_thermal['prob']))
    else:
        filter = None

    count = torch.ones(image_shape).to(device)
    if config['aggregation'] == 'prod':
        prob = out_optical['prob'] * out_thermal['prob']
    elif config['aggregation'] == 'sum':
        prob = out_optical['prob'] + out_thermal['prob']
    else:
        raise ValueError('Unknown aggregation: ' + config['aggregation'])

    # TODO decide if also a mask should be included for the original prob with the eroded border
    # create warping module and move it to multigpu if available
    warper = WarpingModule()

    if torch.cuda.device_count() > 1:
        warper = torch.nn.DataParallel(warper)

    for i in range(1,config['num']):
        # sample a homography and build the valid mask
        homography = sample_homography(np.array(image_shape[2:]), **config['homographies'])
        valid_mask = compute_valid_mask(tuple(image_shape[2:]), homography, config['erosion_radius'], config['mask_border'])
        valid_mask = torch.from_numpy(valid_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).repeat(image_shape[0],1,1,1).to(device)

        homography = torch.from_numpy(homography.astype(np.float32)).unsqueeze(0).repeat(image_shape[0],1,1).to(device)

        # predict the probabilites
        warped_images = warper(torch.cat([data['optical']['image'], data['thermal']['image']]), torch.cat([homography, homography]), image_shape[2:], 'bilinear', 'reflection')
        input = {
            'image': warped_images[:image_shape[0]],
            'is_optical': data['optical']['is_optical'],
        }
        out_optical = net(input)

        input = {
            'image': warped_images[image_shape[0]:],
            'is_optical': data['thermal']['is_optical'],
        }

        out_thermal = net(input)

        if filter is not None:
            out_optical['prob'] = filter(pad(out_optical['prob']))
            out_thermal['prob'] = filter(pad(out_thermal['prob']))

        # aggregate the probabilities from the two spectra
        if config['aggregation'] == 'prod':
            prob_warped = out_optical['prob'] * out_thermal['prob']
        elif config['aggregation'] == 'sum':
            prob_warped = out_optical['prob'] + out_thermal['prob']
        else:
            raise ValueError('Unknown aggregation: ' + config['aggregation'])

        count_sample = warper(valid_mask, torch.inverse(homography), image_shape[2:], 'nearest')
        count += count_sample
        prob += warper(prob_warped, torch.inverse(homography), image_shape[2:], 'bilinear') * count_sample

    out = prob / count

    if config['aggregation'] == 'prod':
        out = out.sqrt()
    elif config['aggregation'] == 'sum':
        out *= 0.5
    else:
        raise ValueError('Unknown aggregation: ' + config['aggregation'])

    if config['min_count'] > 0:
        out[count < config['min_count']] = 0.0

    return out

def homographic_adaptation(data, net, homographic_adaptation_config = {}):
    device = data['image'].device
    config = dict_update(homography_adaptation_default_config, homographic_adaptation_config)

    if config['num'] < 1:
        raise ValueError('num must be larger than 0 for the homographic adaptation')

    if config['filter_size'] % 2 == 0 and config['filter_size'] != 0:
        raise ValueError('The filter_size must be uneven')

    image_shape = data['image'].shape

    # process the original images
    out = net(data)

    if config['filter_size'] > 0:
        filter = get_gaussian_filter(config['filter_size']).to(device)
        pad = nn.ReflectionPad2d(int((config['filter_size'] - 1) / 2))

        out['prob'] = filter(pad(out['prob']))
    else:
        filter = None

    count = torch.ones(image_shape).to(device)
    prob = out['prob']

    # create warping module and move it to multigpu if available
    warper = WarpingModule()

    if torch.cuda.device_count() > 1:
        warper = torch.nn.DataParallel(warper)

    for i in range(1,config['num']):
        # sample a homography and build the valid mask
        homography = sample_homography(np.array(image_shape[2:]), **config['homographies'])
        valid_mask = compute_valid_mask(tuple(image_shape[2:]), homography, config['erosion_radius'], config['mask_border'])
        valid_mask = torch.from_numpy(valid_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).repeat(image_shape[0],1,1,1).to(device)

        homography = torch.from_numpy(homography.astype(np.float32)).unsqueeze(0).repeat(image_shape[0],1,1).to(device)

        # predict the probabilites
        warped_images = warper(data['image'], homography, image_shape[2:], 'bilinear', 'reflection')
        input = {
            'image': warped_images,
        }
        out = net(input)

        if filter is not None:
            out['prob'] = filter(pad(out['prob']))

        count_sample = warper(valid_mask, torch.inverse(homography), image_shape[2:], 'nearest')
        count += count_sample
        prob += warper(out['prob'], torch.inverse(homography), image_shape[2:], 'bilinear') * count_sample

    out = prob / count

    if config['min_count'] > 0:
        out[count < config['min_count']] = 0.0

    return out

def sample_homography(image_shape, perspective=True, scaling=True, rotation=True, translation=True,
                      n_scales=10, n_angles=25, scaling_amplitude=0.2, perspective_amplitude_x=0.1,
                      perspective_amplitude_y=0.1, patch_ratio=0.8, max_angle=pi/2,
                      allow_artifacts=True, translation_overflow=0.1):
    """
    Sample a random valid homography.

    Arguments:
        image_shape: The shape of the image
        perspective: A boolean that enables the perspective and affine transformations.
        scaling: A boolean that enables the random scaling of the patch.
        rotation: A boolean that enables the random rotation of the patch.
        translation: A boolean that enables the random translation of the patch.
        n_scales: The number of tentative scales that are sampled when scaling.
        n_angles: The number of tentatives angles that are sampled when rotating.
        scaling_amplitude: Controls the amount of scale.
        perspective_amplitude_x: Controls the perspective effect in x direction.
        perspective_amplitude_y: Controls the perspective effect in y direction.
        patch_ratio: Controls the size of the patches used to create the homography.
        max_angle: Maximum angle used in rotations.
        allow_artifacts: A boolean that enables artifacts when applying the homography.
        translation_overflow: Amount of border artifacts caused by translation.

    Returns:
        A numpy array containing the homographic transformation matrix
    """

    def transform_perspective(points):
        t_min, t_max = -points.min(axis=0), 1.0-points.max(axis=0)
        t_max[1] = min(abs(t_min[1]), abs(t_max[1]))
        t_min[1] = -t_max[1]
        if not allow_artifacts:
            perspective_amplitude_min = np.maximum(np.array([-perspective_amplitude_x,-perspective_amplitude_y]), t_min)
            perspective_amplitude_max = np.minimum(np.array([perspective_amplitude_x,perspective_amplitude_y]), t_max)
        else:
            perspective_amplitude_min = np.array([-perspective_amplitude_x,-perspective_amplitude_y])
            perspective_amplitude_max = np.array([perspective_amplitude_x,perspective_amplitude_y])

        perspective_displacement = np.random.uniform(perspective_amplitude_min[1], perspective_amplitude_max[1])
        h_displacement_left = np.random.uniform(perspective_amplitude_min[0], perspective_amplitude_max[0])
        h_displacement_right = np.random.uniform(perspective_amplitude_min[0], perspective_amplitude_max[0])

        tmp = points.copy()
        points += np.array([[h_displacement_left,   perspective_displacement],
                          [h_displacement_left,  -perspective_displacement],
                          [h_displacement_right,  perspective_displacement],
                          [h_displacement_right, -perspective_displacement]])

        return points

    def transform_scale(points):
        scales = np.random.uniform(-scaling_amplitude, scaling_amplitude, n_scales) + 1.0
        center = points.mean(axis=0)
        scaled = np.expand_dims(points - center, 0) * np.expand_dims(np.expand_dims(scales, 1), 1) + center

        if allow_artifacts:
            valid = np.arange(n_scales)  # all scales are valid except scale=1
        else:
            valid = []
            for i in range(n_scales):
                if scaled[i,...].max() < 1.0 and scaled[i,...].min() >= 0.0:
                    valid.append(i)

        if valid is not None:
            idx = np.random.choice(valid)
            points = scaled[idx]
        else:
            print('sample_homography: No valid scale found')

        return points

    def transform_translation(points):
        t_min, t_max = -points.min(axis=0), 1.0-points.max(axis=0)
        if allow_artifacts:
            t_min -= translation_overflow
            t_max += translation_overflow
        points += np.array([np.random.uniform(t_min[0], t_max[0]),
                          np.random.uniform(t_min[1], t_max[1])])

        return points

    def transform_rotation(points):
        angles = np.random.uniform(-max_angle, max_angle, n_angles)
        angles = np.append(angles, 0)  # in case no rotation is valid
        center = points.mean(axis=0)
        rot_mat = np.reshape(np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)], axis=1), [-1, 2, 2])
        rotated = np.matmul(np.tile(np.expand_dims(points - center, axis=0), [n_angles+1, 1, 1]), rot_mat) + center
        if allow_artifacts:
            valid = np.arange(n_angles)  # all angles are valid, except angle=0
        else:
            valid = []
            for i in range(len(angles)):
                if rotated[i,...].max() < 1.0 and rotated[i,...].min() >= 0.0:
                    valid.append(i)

        idx = np.random.choice(valid)
        points = rotated[idx]

        return points

    # Corners of the input image
    pts1 = np.array([[0., 0.], [0., 1.], [1., 1.], [1., 0.]])

    # Corners of the output patch
    margin = (1 - patch_ratio) * 0.5
    pts2 = margin + patch_ratio * pts1

    # Random perspective and affine perturbations
    functions = []
    if perspective:
        functions.append(transform_perspective)

    # Random scaling
    if scaling:
        functions.append(transform_scale)

    # Random translation
    if translation:
        functions.append(transform_translation)

    # Random rotation
    # sample several rotations, check collision with borders, randomly pick a valid one
    if rotation:
        functions.append(transform_rotation)

    indices = np.arange(len(functions))
    np.random.shuffle(indices)

    for i in range(len(functions)):
            idx = indices[i]
            pts2 = functions[idx](pts2)

    # Rescale to actual size
    shape = image_shape[::-1]  # different convention [y, x]
    pts1 *= shape
    pts2 *= shape

    homography = cv2.getPerspectiveTransform(pts1.astype(np.float32), pts2.astype(np.float32))
    return homography

def warp_keypoints(keypoints, homography, return_type=np.int):
    """
    Warp the keypoints based on the specified homographic transformation matrix

    Arguments:
        keypoints: Array containing the keypoints, shape: [N,2]
        homography: 3x3 transformation matrix

    Returns: Array containing the warped keypoints, shape: [N,2]
    """
    if len(keypoints) > 0:
        warped_points = cv2.perspectiveTransform(np.array([keypoints[:,::-1]], dtype=np.float64), homography)
        return warped_points[0,:,::-1].astype(return_type)
    else:
        # no keypoints available so return the empty array
        return keypoints

def warp_points_pytorch(points, homography):
    # get the points to the homogeneous format
    warped_points = torch.cat([points.flip(-1), torch.ones([points.shape[0], points.shape[1], 1], dtype=torch.float32, device=points.device)], -1)

    # apply homography
    warped_points = torch.bmm(homography, warped_points.permute([0,2,1])).permute([0,2,1])
    warped_points = warped_points[:, :, :2] / warped_points[:, :, 2:]

    return warped_points.flip(-1)

def filter_points(points, shape):
    """
    Filter points which would be outside the image frame

    Arguments:
        points: Array containing the keypoints, shape: [N,2]
        shape: Image shape

    Returns: Array containing the filtered keypoints, shape: [M,2]
    """
    points = points[points[:,0] >= 0]
    points = points[points[:,1] >= 0]
    points = points[points[:,0] < shape[0]]
    points = points[points[:,1] < shape[1]]

    return points

def compute_valid_mask(image_shape, homography, erosion_radius=0, mask_border = False):
    """
    Compute a boolean mask of the valid pixels resulting from an homography applied to
    an image of a given shape. Pixels that are False correspond to bordering artifacts.
    A margin can be discarded using erosion.
 
    Arguments:
        input_shape: Array of rank 2 representing the image shape, i.e. `[H, W]`.
        homography: Array of shape (3, 3)
        erosion_radius: radius of the margin to be discarded.
        mask_border: Boolean indicating if the border is used to erode the valid region 
 
    Returns: Array of shape (H, W).
    """
    mask = cv2.warpPerspective(np.ones(image_shape), homography, image_shape[::-1], flags=cv2.INTER_NEAREST)

    if erosion_radius > 0:
        if mask_border:
            tmp = np.zeros((image_shape[0]+2, image_shape[1]+2))
            tmp[1:-1,1:-1] = mask
            mask = tmp
        kernel = np.ones((erosion_radius * 2 + 1,erosion_radius * 2 + 1),np.float32)
        mask = cv2.erode(mask,kernel,iterations = 1)

        if mask_border:
            mask = mask[1:-1,1:-1]

    return mask

def warp_perspective_tensor(src, M, dsize, mode='bilinear', padding_mode='zeros'):
    '''
    A copy of kornia.warp_perspective where the additional variables are properly passed.
    '''
    if not kornia_available:
        raise RuntimeError('Kornia not imported but required for warp_perspective_tensor')

    if not torch.is_tensor(src):
        raise TypeError("Input src type is not a torch.Tensor. Got {}"
                        .format(type(src)))
    if not torch.is_tensor(M):
        raise TypeError("Input M type is not a torch.Tensor. Got {}"
                        .format(type(M)))
    if not len(src.shape) == 4:
        raise ValueError("Input src must be a BxCxHxW tensor. Got {}"
                         .format(src.shape))
    if not (len(M.shape) == 3 or M.shape[-2:] == (3, 3)):
        raise ValueError("Input M must be a Bx3x3 tensor. Got {}"
                         .format(src.shape))
    # launches the warper
    M_norm = kornia.geometry.transform.imgwarp.dst_norm_to_dst_norm(M, (src.shape[-2:]), dsize)
    return homography_warp(src, torch.inverse(M_norm), dsize, mode, padding_mode)

class WarpingModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, src, M, dsize, mode='bilinear', padding_mode='zeros'):
        return warp_perspective_tensor(src, M, dsize, mode, padding_mode)

