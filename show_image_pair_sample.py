import argparse
import cv2
import numpy as np
import torch

import multipoint.datasets as data

def main():
    parser = argparse.ArgumentParser(description='Show a sample of the dataset')
    parser.add_argument('-i', '--input-file', default='/tmp/test.hdf5', help='Input dataset file')
    parser.add_argument('-k', '--keypoint-file', help='Keypoint dataset file')
    parser.add_argument('-n', dest='sample_number', type=int, default=0, help='Sample to show')
    parser.add_argument('-r', '--radius', default=4, type=int, help='Radius of the keypoint circle')

    args = parser.parse_args()

    config = {
        'filename': args.input_file,
        'keypoints_filename': args.keypoint_file,
        'height': -1,
        'width': -1,
        'raw_thermal': False,
        'single_image': True,
    }

    dataset = data.ImagePairDataset(config)

    sample = dataset[args.sample_number]

    out_image = cv2.cvtColor((np.clip(sample['image'].squeeze().numpy(), 0.0, 1.0) * 255.0).astype(np.uint8),cv2.COLOR_GRAY2RGB)
    if 'keypoints' in sample.keys():
        pred = torch.nonzero(sample['keypoints'].squeeze())
        predictions = [cv2.KeyPoint(c[1], c[0], args.radius) for c in pred.numpy().astype(np.float32)]

        for kp in predictions:
            pt = tuple([int(kp.pt[0]), int(kp.pt[1])])
            out_image = cv2.circle(out_image, pt, args.radius, (0, 0, 255), 3)

    mask =  np.repeat(np.expand_dims(sample['valid_mask'].squeeze().numpy(), axis=2), 3, axis=2)
    cv2.imshow('single image', out_image)
    cv2.imshow('single image masked', out_image * mask)

    config = {
        'filename': args.input_file,
        'keypoints_filename': args.keypoint_file,
        'height': -1,
        'width': -1,
        'raw_thermal': False,
        'single_image': False,
    }

    dataset = data.ImagePairDataset(config)

    sample = dataset[args.sample_number]

    out_thermal = cv2.cvtColor((np.clip(sample['thermal']['image'].squeeze().numpy(), 0.0, 1.0) * 255.0).astype(np.uint8),cv2.COLOR_GRAY2RGB)
    out_optical = cv2.cvtColor((np.clip(sample['optical']['image'].squeeze().numpy(), 0.0, 1.0) * 255.0).astype(np.uint8),cv2.COLOR_GRAY2RGB)

    if 'keypoints' in sample['thermal'].keys():
        pred = torch.nonzero(sample['optical']['keypoints'].squeeze())
        predictions = [cv2.KeyPoint(c[1], c[0], args.radius) for c in pred.numpy().astype(np.float32)]

        for kp in predictions:
            pt = tuple([int(kp.pt[0]), int(kp.pt[1])])
            out_optical = cv2.circle(out_optical, pt, args.radius, (0, 0, 255), 5)

        pred = torch.nonzero(sample['thermal']['keypoints'].squeeze())
        predictions = [cv2.KeyPoint(c[1], c[0], args.radius) for c in pred.numpy().astype(np.float32)]

        for kp in predictions:
            pt = tuple([int(kp.pt[0]), int(kp.pt[1])])
            out_thermal = cv2.circle(out_thermal, pt, args.radius, (0, 0, 255), 5)

    mask_optical = np.repeat(np.expand_dims(sample['optical']['valid_mask'].squeeze().numpy(), axis=2), 3, axis=2)
    mask_thermal = np.repeat(np.expand_dims(sample['thermal']['valid_mask'].squeeze().numpy(), axis=2), 3, axis=2)
    cv2.imshow('thermal', out_thermal)
    cv2.imshow('optical', out_optical)
    cv2.imwrite('/tmp/optical.png', out_optical)
    cv2.imwrite('/tmp/thermal.png', out_thermal)
    cv2.imshow('thermal masked', out_thermal * mask_thermal)
    cv2.imshow('optical masked', out_optical * mask_optical)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
