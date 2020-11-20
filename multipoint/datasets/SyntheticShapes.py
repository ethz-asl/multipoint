from __future__ import print_function

import cv2
import numpy as np
import random
import torch
from torch.utils.data.dataset import Dataset
import h5py

import multipoint.utils as utils
import multipoint.utils.draw_primitives as draw_primitives

from .augmentation import augmentation

class SyntheticShapes(Dataset):
    '''
    Implementation of the synthetic dataset according to
    "SuperPoint: Self-Supervised Interest Point Detection and Description"
    and the tensorflow implementation of Remi Pautrat (https://github.com/rpautrat/SuperPoint)
    '''

    default_config = {
        'length': 1000,
        'primitives': 'all',
        'on-the-fly': True,
        'hdf5-file': None,
        'generation_size': [960, 1280],
        'image_size': [240, 320],
        'keypoints_as_map': True,
        'generation': {
            'min_contrast' : 0.1,
            'generate_background': {
                'min_kernel_size': 150, 'max_kernel_size': 500,
                'min_rad_ratio': 0.02, 'max_rad_ratio': 0.031},
            'draw_lines': {'nb_lines': 10},
            'draw_polygons': {'max_sides': 8},
            'draw_stripes': {'transform_params': (0.1, 0.1)},
            'draw_multiple_polygons': {'kernel_boundaries': (50, 100)}
        },
        'processing': {
            'blur_size': 21,
            'additional_ir_blur': True,
            'additional_ir_blur_size': 51,
        },
        'augmentation': {
            'photometric': {
                'enable': True,
                'primitives': 'all',
                'params': {},
                'random_order': True,
            },
            'homographic': {
                'enable': True,
                'params': {},
                'border_reflect': True,
                'valid_border_margin': 0,
                'mask_border': True,
            },
        }
    }

    all_primitives = [
        'draw_lines',
        'draw_polygon',
        'draw_multiple_polygons',
        'draw_ellipses',
        'draw_star',
        'draw_checkerboard',
        'draw_stripes',
        'draw_cube',
        'gaussian_noise'
    ]

    def __init__(self, config = None):
        """Constructor for SyntheticShapes dataset
        :param:
            config: (string) name of config yaml file
        """

        if config:
            self.config = utils.dict_update(self.default_config, config)
        else:
            self.config = self.default_config


        self.primitives = utils.parse_primitives(self.config['primitives'], self.all_primitives)

        if self.config['on-the-fly'] is False:
            # Will raise IOError if not found
            try:
                h5_file = h5py.File(self.config['hdf5-file'], 'r')
            except IOError as e:
                print("Config is set to load data from hdf5 file (on-the-fly False),")
                print("but file {} not found or invalid.".format(self.config['hdf5-file']))
                raise(e)
            # extract info from the h5 file
            self.config['length'] = len(h5_file.keys())
            self.memberslist = list(h5_file.keys())
            h5_file.close()

    def generate_synthetic_image(self, index):
        """Generate a SyntheticShapes image
        :param:
            index: index value (not used for random generation)
        :return:
            image: Image as a numpy array of float32
            keypoints: 2xn numpy array of interest points in pixel coords
            is_optical: bool flag indicating optical image or not
        """

        # sample if it is an optical image
        is_optical = bool(random.randint(0, 1))

        # create a random background with blobs
        image = draw_primitives.generate_background(
            shape=self.config['generation_size'],
            **self.config['generation']['generate_background'])

        # draw the primitive
        primitive = np.random.choice(self.primitives)
        keypoints = getattr(draw_primitives, primitive)(image,
                                                        min_contrast=
                                                        self.config[
                                                            'generation'][
                                                            'min_contrast'],
                                                        **self.config[
                                                            'generation'].get(
                                                            primitive, {}))
        keypoints = np.flip(keypoints, 1)

        # blur the image
        image = cv2.GaussianBlur(image, (
        self.config['processing']['blur_size'],
        self.config['processing']['blur_size']), 0)
        if (not is_optical and self.config['processing'][
            'additional_ir_blur']):
            image = cv2.GaussianBlur(image, (
            self.config['processing']['additional_ir_blur_size'],
            self.config['processing']['additional_ir_blur_size']), 0)

        # resize the image if required
        if self.config['generation_size'] != self.config['image_size']:
            image = cv2.resize(image, tuple(self.config['image_size'][::-1]),
                               interpolation=cv2.INTER_LINEAR)
            keypoints = (
                        np.array(self.config['image_size']).astype(np.float) /
                        np.array(self.config[
                                     'generation_size']) * keypoints).round().astype(
                np.int)

        return image, keypoints, is_optical

    def get_hdf5_image(self, index):
        """Get a SyntheticShapes image from hdf5 dataset
        :param:
            index: index value
        :return:
            image: Image as a numpy array of float32
            keypoints: 2xn numpy array of interest points in pixel coords
            is_optical: bool flag indicating optical image or not
        """

        h5_file = h5py.File(self.config['hdf5-file'], 'r', swmr=True)
        sample = h5_file[self.memberslist[index]]
        image = np.asarray(sample['image'], dtype=np.float32)/(2.0**8 - 1)
        keypoints = np.asarray(sample['points'], dtype=np.float32)
        is_optical = True
        return image, keypoints, is_optical

    def apply_augmentation(self, image, keypoints, is_optical):
        """Apply augmentation to a SyntheticShapes image, return as a dictionary
         of torch data ready to be used as data instance
        :param:
            image: Image as a numpy float32 array
            keypoints: 2xn numpy array of interest points in pixel coords
            is_optical: bool flag indicating optical image or not
        :return:
            data_dict: Dictionary for data record containing:
                'image': torch image array
                'keypoints': torch array either bool (if map) or float
                'valid_mask': torch boolean array mask for valid image locations
                'is_optical': torch BoolTensor
        """

        # boundary check, the round can cause a keypoint to be outside the valid image part
        keypoints[keypoints[:,0] >= self.config['image_size'][0], 0] = self.config['image_size'][0] - 1
        keypoints[keypoints[:,1] >= self.config['image_size'][1], 1] = self.config['image_size'][1] - 1

        # augmentation
        if self.config['augmentation']['photometric']['enable']:
            image = augmentation.photometric_augmentation(image, **
            self.config['augmentation']['photometric'])

        if self.config['augmentation']['homographic']['enable']:
            image, keypoints, valid_mask = augmentation.homographic_augmentation(
                image, keypoints,
                **self.config['augmentation']['homographic'])
        else:
            valid_mask = augmentation.dummy_valid_mask(image.shape)

        if self.config['keypoints_as_map']:
            keypoints = utils.generate_keypoint_map(keypoints, image.shape)
            keypoint_dtype = np.bool
        else:
            keypoint_dtype = np.float32

        # add channel information to image and mask
        image = np.expand_dims(image, 0)
        valid_mask = np.expand_dims(valid_mask, 0)

        return {'image': torch.from_numpy(image.astype(np.float32)),
                'keypoints': torch.from_numpy(keypoints.astype(keypoint_dtype)),
                'valid_mask': torch.from_numpy(valid_mask.astype(np.bool)),
                'is_optical': torch.BoolTensor([is_optical])}

    def __getitem__(self, index):
        if self.config['on-the-fly']:
            im, kp, isopt = self.generate_synthetic_image(index)
        else:
            im, kp, isopt = self.get_hdf5_image(index)
        return self.apply_augmentation(im, kp, isopt)

    def returns_pair(self):
        return False

    def __len__(self):
        return self.config['length']
