import argparse
import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from tqdm import tqdm

import multipoint.datasets as data
import multipoint.utils as utils

parser = argparse.ArgumentParser(description='Show a sample of the dataset')
parser.add_argument('-d', '--dataset-file', required=True, help='Input dataset file')
parser.add_argument('-k', '--keypoint-file', required=True, help='Keypoint dataset file')
parser.add_argument('-n', dest='sample_number', type=int, default=0, help='Sample to show')
parser.add_argument('-r', '--radius', default=4, type=int, help='Radius of the keypoint circle')

args = parser.parse_args()

config = {
    'filename': args.dataset_file,
    'height': -1,
    'width': -1,
    'raw_thermal': False,
    'single_image': False,
    'augmentation': {
        'photometric': {
            'enable': False,
        },
        'homographic': {
            'enable': False,
        },
    }
}

dataset = data.ImagePairDataset(config)
try:
    keypoint_file = h5py.File(args.keypoint_file, 'r', swmr=True)
except IOError as e:
    print('I/O error({0}): {1}: {2}'.format(e.errno, e.strerror, filename))
    sys.exit()

# display an individual sample
# get the data
sample = dataset[args.sample_number]
name = dataset.get_name(args.sample_number)
labels = keypoint_file[name]['keypoints'][...]

out_thermal = cv2.cvtColor((np.clip(sample['thermal']['image'].squeeze().numpy(), 0.0, 1.0) * 255.0).astype(np.uint8),cv2.COLOR_GRAY2RGB)
out_optical = cv2.cvtColor((np.clip(sample['optical']['image'].squeeze().numpy(), 0.0, 1.0) * 255.0).astype(np.uint8),cv2.COLOR_GRAY2RGB)

print("Number of keypoints: {}".format(labels.shape[0]))
predictions = [cv2.KeyPoint(c[1], c[0], args.radius) for c in labels.astype(np.float32)]

# draw predictions and ground truth on image
out_optical = cv2.drawKeypoints(out_optical,
                                predictions,
                                outImage=np.array([]),
                                color=(0, 255, 0),
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
out_thermal = cv2.drawKeypoints(out_thermal,
                                predictions,
                                outImage=np.array([]),
                                color=(0, 255, 0),
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

mask_optical = np.repeat(np.expand_dims(sample['optical']['valid_mask'].squeeze().numpy(), axis=2), 3, axis=2)
mask_thermal = np.repeat(np.expand_dims(sample['thermal']['valid_mask'].squeeze().numpy(), axis=2), 3, axis=2)
cv2.imshow('thermal', out_thermal)
cv2.imshow('optical', out_optical)
cv2.imshow('thermal masked', out_thermal * mask_thermal)
cv2.imshow('optical masked', out_optical * mask_optical)
cv2.waitKey(0)
