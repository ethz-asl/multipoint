import cv2
import numpy as np
import random

from multipoint.datasets.augmentation import photometric_augmentation as photoaug
import multipoint.utils as utils

def photometric_augmentation(image, **config):
    '''
    Augment the image with various photometric functions according the input configuration.
    '''
    primitives = utils.parse_primitives(config['primitives'], photoaug.augmentations)
    primitives_configs = [config['params'].get( p, {}) for p in primitives]

    indices = np.arange(len(primitives))
    if config['random_order']:
        random.shuffle(indices)

    for i in range(len(primitives)):
        idx = indices[i]
        image = getattr(photoaug, primitives[idx])(image, **primitives_configs[idx])

    return image

def homographic_augmentation(image, keypoints = None, return_homography=False, **config):
    '''
    Augment the image by warping it with a random homography according to the config params.
    If keypoints are warped with the same transformation as the image if keypoints are present.
    '''
    image_shape = image.shape
    homography = utils.sample_homography(image_shape, **config['params'])

    if config['border_reflect']:
        warped_image = cv2.warpPerspective(image, homography, image.shape[::-1], borderMode=cv2.BORDER_REFLECT_101)
    else:
        warped_image = cv2.warpPerspective(image, homography, image.shape[::-1], borderMode=cv2.BORDER_CONSTANT)

    valid_mask = utils.compute_valid_mask(image_shape, homography,
                                          config['valid_border_margin'] * 2,
                                          config['mask_border'])

    if keypoints is not None:
        if keypoints.size > 0:
            warped_points = utils.warp_keypoints(keypoints, homography)
            warped_points = utils.filter_points(warped_points, image_shape)
        else:
            # no keypoints present so just copy the empty array
            warped_points = keypoints
    else:
        warped_points = None

    if return_homography:
        return warped_image, warped_points, valid_mask, homography
    else:
        return warped_image, warped_points, valid_mask

def dummy_valid_mask(image_shape):
    '''
    Returns a mask where all pixels are valid with the shape of image_shape.
    '''
    return np.ones(image_shape)
