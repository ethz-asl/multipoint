from __future__ import print_function

import numpy as np
import random
import sys
import torch
from torch.utils.data.dataset import Dataset
import h5py

import multipoint.utils as utils
from .augmentation import augmentation

class ImagePairDataset(Dataset):
    '''
    Class to load a sample from a given hdf5 file.
    '''
    default_config = {
        'filename': None,
        'keypoints_filename': None,
        'height': -1,
        'width': -1,
        'raw_thermal': False,
        'single_image': True,
        'random_pairs': False,
        'return_name' : True,
        'augmentation': {
            'photometric': {
                'enable': False,
                'primitives': 'all',
                'params': {},
                'random_order': True,
            },
            'homographic': {
                'enable': False,
                'params': {},
                'border_reflect': True,
                'valid_border_margin': 0,
                'mask_border': True,
            },
        }
    }

    def __init__(self, config):
        if config:
            import copy
            self.config = utils.dict_update(copy.copy(self.default_config), config)
        else:
            self.config = self.default_config

        if self.config['filename'] is None:
            raise ValueError('ImagePairDataset: The dataset filename needs to be present in the config file')

        try:
            h5_file = h5py.File(self.config['filename'], 'r')
        except IOError as e:
            print('I/O error({0}): {1}: {2}'.format(e.errno, e.strerror, filename))
            sys.exit()

        if self.config['single_image'] and self.config['random_pairs']:
            print('INFO: random_pairs has no influence if single_image is true')

        # extract info from the h5 file
        self.num_files = len(h5_file.keys())
        self.memberslist = list(h5_file.keys())
        h5_file.close()

        # process keypoints if filename is present
        if self.config['keypoints_filename'] is not None:
            # check if file exists
            try:
                keypoints_file = h5py.File(self.config['filename'], 'r')
            except IOError as e:
                print('I/O error({0}): {1}: {2}'.format(e.errno, e.strerror, filename))
                sys.exit()

            # check that for every sample there are keypoints
            keypoint_members = list(keypoints_file.keys())
            missing_labels = []
            for item in self.memberslist:
                if item not in keypoint_members:
                    missing_labels.append(item)

            if len(missing_labels) > 0:
                raise IndexError('Labels for the following samples not available: {}'.format(missing_labels))

        print('The dataset ' + self.config['filename'] + ' contains {} samples'.format(self.num_files))

    def __getitem__(self, index):
        h5_file = h5py.File(self.config['filename'], 'r', swmr=True)
        sample = h5_file[self.memberslist[index]]

        optical = sample['optical'][...]
        if self.config['raw_thermal']:
            thermal = sample['thermal_raw'][...]
        else:
            thermal = sample['thermal'][...]

        if thermal.shape != optical.shape:
            raise ValueError('ImagePairDataset: The optical and thermal image must have the same shape')

        if self.config['keypoints_filename'] is not None:
            with h5py.File(self.config['keypoints_filename'], 'r', swmr=True) as keypoints_file:
                keypoints = np.array(keypoints_file[self.memberslist[index]]['keypoints'])
        else:
            keypoints = None

        # subsample images if requested
        if self.config['height'] > 0 or self.config['width'] > 0:
            if self.config['height'] > 0:
                h = self.config['height']
            else:
                h = thermal.shape[0]

            if self.config['width'] > 0:
                w = self.config['width']
            else:
                w = thermal.shape[1]

            if w > thermal.shape[1] or h > thermal.shape[0]:
                raise ValueError('ImagePairDataset: Requested height/width exceeds original image size')

            # subsample the image
            i_h = random.randint(0, thermal.shape[0]-h)
            i_w = random.randint(0, thermal.shape[1]-w)

            optical = optical[i_h:i_h+h, i_w:i_w+w]
            thermal = thermal[i_h:i_h+h, i_w:i_w+w]

            if keypoints is not None:
                # shift keypoints
                keypoints = keypoints - np.array([[i_h,i_w]])

                # filter out bad ones
                keypoints = keypoints[np.logical_and(
                                      np.logical_and(keypoints[:,0] >=0,keypoints[:,0] < h),
                                      np.logical_and(keypoints[:,1] >=0,keypoints[:,1] < w))]

        else:
            h = thermal.shape[0]
            w = thermal.shape[1]

        out = {}

        if self.config['single_image']:
            is_optical = bool(random.randint(0,1))

            if is_optical:
                image = optical
            else:
                image = thermal

            # augmentation
            if self.config['augmentation']['photometric']['enable']:
                image = augmentation.photometric_augmentation(image, **self.config['augmentation']['photometric'])

            if self.config['augmentation']['homographic']['enable']:
                image, keypoints, valid_mask = augmentation.homographic_augmentation(image, keypoints, **self.config['augmentation']['homographic'])
            else:
                valid_mask = augmentation.dummy_valid_mask(image.shape)

            # add channel information to image and mask
            image = np.expand_dims(image, 0)
            valid_mask = np.expand_dims(valid_mask, 0)

            # add to output dict
            out['image'] = torch.from_numpy(image.astype(np.float32))
            out['valid_mask'] = torch.from_numpy(valid_mask.astype(np.bool))
            out['is_optical'] = torch.BoolTensor([is_optical])
            if keypoints is not None:
                keypoints = utils.generate_keypoint_map(keypoints, (h,w))
                out['keypoints'] = torch.from_numpy(keypoints.astype(np.bool))

        else:
            # initialize the images
            out['optical'] = {}
            out['thermal'] = {}

            optical_is_optical = True
            thermal_is_optical = False
            if self.config['random_pairs']:
                tmp_optical = optical
                tmp_thermal = thermal
                if bool(random.randint(0,1)):
                    optical = tmp_thermal
                    optical_is_optical = False
                if bool(random.randint(0,1)):
                    thermal = tmp_optical
                    thermal_is_optical = True

            # augmentation
            if self.config['augmentation']['photometric']['enable']:
                optical = augmentation.photometric_augmentation(optical, **self.config['augmentation']['photometric'])
                thermal = augmentation.photometric_augmentation(thermal, **self.config['augmentation']['photometric'])

            if self.config['augmentation']['homographic']['enable']:
                # randomly pick one image to warp
                if bool(random.randint(0,1)):
                    valid_mask_thermal = augmentation.dummy_valid_mask(thermal.shape)
                    keypoints_thermal = keypoints
                    optical, keypoints_optical, valid_mask_optical, H = augmentation.homographic_augmentation(optical,
                                                                                                              keypoints,
                                                                                                              return_homography = True,
                                                                                                              **self.config['augmentation']['homographic'])
                    out['optical']['homography'] = torch.from_numpy(H.astype(np.float32))
                    out['thermal']['homography'] = torch.eye(3, dtype=torch.float32)
                else:
                    valid_mask_optical = augmentation.dummy_valid_mask(optical.shape)
                    keypoints_optical = keypoints
                    thermal, keypoints_thermal, valid_mask_thermal, H = augmentation.homographic_augmentation(thermal,
                                                                                                              keypoints,
                                                                                                              return_homography = True,
                                                                                                              **self.config['augmentation']['homographic'])
                    out['thermal']['homography'] = torch.from_numpy(H.astype(np.float32))
                    out['optical']['homography'] = torch.eye(3, dtype=torch.float32)
            else:
                keypoints_optical = keypoints
                keypoints_thermal = keypoints
                valid_mask_optical = valid_mask_thermal = augmentation.dummy_valid_mask(optical.shape)

            # add channel information to image and mask
            optical = np.expand_dims(optical, 0)
            thermal = np.expand_dims(thermal, 0)
            valid_mask_optical = np.expand_dims(valid_mask_optical, 0)
            valid_mask_thermal = np.expand_dims(valid_mask_thermal, 0)

            out['optical']['image'] = torch.from_numpy(optical.astype(np.float32))
            out['optical']['valid_mask'] = torch.from_numpy(valid_mask_optical.astype(np.bool))
            out['optical']['is_optical'] = torch.BoolTensor([optical_is_optical])
            if keypoints_optical is not None:
                keypoints_optical = utils.generate_keypoint_map(keypoints_optical, (h,w))
                out['optical']['keypoints'] = torch.from_numpy(keypoints_optical.astype(np.bool))

            out['thermal']['image'] = torch.from_numpy(thermal.astype(np.float32))
            out['thermal']['valid_mask'] = torch.from_numpy(valid_mask_thermal.astype(np.bool))
            out['thermal']['is_optical'] = torch.BoolTensor([thermal_is_optical])
            if keypoints_optical is not None:
                keypoints_thermal = utils.generate_keypoint_map(keypoints_thermal, (h,w))
                out['thermal']['keypoints'] = torch.from_numpy(keypoints_thermal.astype(np.bool))

        if self.config['return_name']:
            out['name'] = self.memberslist[index]

        return out

    def get_name(self, index):
        return self.memberslist[index]

    def returns_pair(self):
        return not self.config['single_image']

    def __len__(self):
        return self.num_files
