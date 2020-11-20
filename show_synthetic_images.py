import argparse
import cv2
import numpy as np
import torch
import yaml

from multipoint.datasets import SyntheticShapes

def main():
    parser = argparse.ArgumentParser(description='Show a samples of the synthetic dataset')
    parser.add_argument('-n', dest='sample_number', type=int, default=1, help='Number of sample to show')
    parser.add_argument('-r', dest='radius', type=int, default=1, help='Radius of the circle to indicate a keypoint')
    parser.add_argument('-y', dest='yaml', help='Config file')
    parser.add_argument('-m', dest='show_mask', action='store_true', help='Show the masked pixels with a different color')

    args = parser.parse_args()

    if args.yaml:
        with open(args.yaml, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config = config['dataset']
    else:
        config = None
    dataset = SyntheticShapes(config)

    for i in range(args.sample_number):
        data = dataset[i]
        
        image = data['image'][0]
        mask = data['valid_mask'][0]
        kp = data['keypoints']

        if kp.shape == image.shape:
            kp = torch.nonzero(kp)

        keypoints = [cv2.KeyPoint(c[1], c[0], args.radius) for c in kp.numpy().astype(np.float32)]

        image = np.clip(image.numpy(), 0.0, 1.0)
        color_image = cv2.cvtColor((image * 255.0).astype(np.uint8),cv2.COLOR_GRAY2RGB)

        if args.show_mask:
            color_image[:,:,0] = (mask.numpy() * 255.0).astype(np.uint8)

        out_image = cv2.drawKeypoints(color_image,
                                      keypoints,
                                      outImage=np.array([]),
                                      color=(0, 0, 255),
                                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        mask = np.repeat(np.expand_dims(data['valid_mask'].squeeze().numpy(), axis=2), 3, axis=2)
        cv2.imshow(str(i) + ' raw', out_image)
        cv2.imshow(str(i) + ' masked', out_image * mask)

    cv2.waitKey(0)

if __name__ == "__main__":
    main()
