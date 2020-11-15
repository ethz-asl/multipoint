import cv2
import numpy as np
import torch
import torch.nn as nn

import multipoint.utils as utils

class ClassicDetectors(nn.Module):
    default_config = {
        'method': 'SURF',
        'prob_smoothing': False,
        'smoothing_kernel_size': 5,
        'min_keypoints': 100,
        'image_H': 512,
        'image_W': 640,
    }

    def __init__(self, config = None):
        super(ClassicDetectors, self).__init__()

        if config:
            self.config = utils.dict_update(self.default_config, config)
        else:
            self.config = self.default_config

        if self.config['method'] == 'SURF':
            self.method = cv2.xfeatures2d.SURF_create(1500)
            self.method_2 = cv2.xfeatures2d.SURF_create(300)
        elif self.config['method'] == 'SIFT':
            self.method = cv2.xfeatures2d.SIFT_create(1000)
            self.method_2 = cv2.xfeatures2d.SIFT_create(1500)
        elif self.config['method'] == 'LGHD':
            self.method = LGHD(self.config['image_H'], self.config['image_W'])
            self.method_2 = LGHD(self.config['image_H'], self.config['image_W'])
        else:
            raise ValueError('Unknown alignment method: ' + self.config['method'])

        self.filter = None
        if self.config['prob_smoothing']:
            if self.config['smoothing_kernel_size'] % 2 == 0:
                raise ValueError('smoothing_kernel_size needs to be uneven')

            self.filter = utils.get_gaussian_filter(self.config['smoothing_kernel_size'])
            val = int((self.config['smoothing_kernel_size'] - 1) / 2)
            self.padding = (val, val, val, val)

    def forward(self, data):
        assert data['image'].shape[0] == 1
        assert len(data['image'].shape) == 4

        device = data['image'].device

        # convert image and get detections and descriptors
        image_np = (data['image'].squeeze().cpu().numpy() * 255.0).astype(np.uint8)

        keypoints, descriptors = self.method.detectAndCompute(image_np,None)

        if len(keypoints) < self.config['min_keypoints']:
            keypoints, descriptors = self.method_2.detectAndCompute(image_np,None)

        # convert ot pytorch
        prob = torch.zeros_like(data['image']).to(device)
        if len(keypoints) > 0:
            desc = torch.zeros([1, descriptors.shape[1], data['image'].shape[2], data['image'].shape[3]]).to(device)
            for kp, des in zip(keypoints, descriptors):
                idx = np.array(kp.pt[::-1]).round().astype(int)
                prob[0,0, idx[0], idx[1]] = 1.0
                desc[0, :, idx[0], idx[1]] = torch.from_numpy(des)

            if self.filter is not None:
                prob = self.filter(torch.nn.functional.pad(prob, self.padding))
        else:
            desc = torch.zeros([1, 1, data['image'].shape[2], data['image'].shape[3]]).to(device)

        out = {'prob': prob,
               'desc': desc,
        }

        return out

class LGHD():
    def __init__(self, H, W, patch_size = 40, n_scales = 4, n_angles = 6, min_wavelength = 3, multiplier = 1.6, sigma_onf = 0.75, k = 1, cutoff = 0.5, g = 3):
        # create filter bank
        self.filter_bank = self.create_filter_bank(H, W, n_scales, n_angles, min_wavelength, multiplier, sigma_onf, k, cutoff, g)

        self.patch_size_half = int(np.floor(patch_size * 0.5))
        self.patch_size_fourth = int(np.floor(patch_size * 0.25))
        self.n_scales = n_scales
        self.n_angles = n_angles

        if patch_size / 4 != int(patch_size / 4):
            raise ValueError('The patch size must be a multiple of 4')

    def detectAndCompute(self, image, mask):
        # fft transformation of image
        image_fft = cv2.dft(np.float32(image),flags = cv2.DFT_COMPLEX_OUTPUT)

        eo = np.zeros((self.filter_bank.shape[0], self.filter_bank.shape[1], self.filter_bank.shape[2], 2))
        for i, filter in enumerate(self.filter_bank):
            eo[i] = cv2.idft(np.multiply(np.expand_dims(filter, -1), image_fft))

        # keypoint detection
        fast = cv2.FastFeatureDetector_create()
        keypoints = fast.detect(image,None)

        eo_magnitude = cv2.magnitude(eo[:,:,:,0],eo[:,:,:,1])
        valid_keypoints = np.ones(len(keypoints)).astype(np.bool)
        descriptors = np.zeros((len(keypoints), 16 * self.n_scales * self.n_angles))
        for i, kp in enumerate(keypoints):
            pos =  np.array(kp.pt[::-1]).round().astype(np.int)

            lower_bound = pos - self.patch_size_half
            upper_bound = pos + self.patch_size_half

            if np.any(lower_bound < 0) or np.any(upper_bound > image.shape):
                # invalid since the patch is not fully inside the image
                valid_keypoints[i] = False
            else:
                # compute the descriptor for the valid keypoint
                patch = eo_magnitude[:, lower_bound[0]:upper_bound[0], lower_bound[1]:upper_bound[1]]
                desc = np.zeros((self.n_scales, 4, 4, self.n_angles))
                for s in range(self.n_scales):
                    patch_scale = patch[s*self.n_angles:(s+1)*self.n_angles]
                    max_idx = np.argmax(patch_scale, axis=0)
                    for j in range(4):
                        for k in range(4):
                            minipatch = max_idx[j*self.patch_size_fourth:(j+1)*self.patch_size_fourth, k*self.patch_size_fourth:(k+1)*self.patch_size_fourth]
#                             desc[s, j, k], edges = np.histogram(minipatch, [0,1,2,3,4,5,6])
                            desc[s, j, k] = np.bincount(minipatch.ravel(), minlength=self.n_angles)

                descriptors[i] = desc.ravel()

        keypoints = [kp for kp, valid in zip(keypoints, valid_keypoints) if valid]
        descriptors = descriptors[valid_keypoints]
        return keypoints, descriptors
        import pdb
        pdb.set_trace()

        # test plotting
        image_back = cv2.idft(image_fft)
        im = cv2.magnitude(image_back[:,:,0],image_back[:,:,1])
        im = cv2.normalize(im,None, alpha=0,beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.imshow('back', im)
        cv2.imshow('image', image)

        cv2.waitKey()


        import pdb
        pdb.set_trace()

    def create_filter_bank(self, H, W, n_scales, n_angles, min_wavelength, multiplier, sigma_onf, k, cutoff, g):
        xrange = np.linspace(-0.5, 0.5, W)
        yrange = np.linspace(-0.5, 0.5, H)
        x, y = np.meshgrid(xrange, yrange)

        radius = np.sqrt(x**2 + y**2)
        theta = np.arctan2(-y,x)

        radius = np.fft.ifftshift(radius)
        theta = np.fft.ifftshift(theta)

        sintheta = np.sin(theta)
        costheta = np.cos(theta)

        lp_filter = self.lowpassfilter(H, W, 0.45, 15)

        log_gabor_filters = np.zeros((n_scales, H, W))
        for sc in range(n_scales):
            wavelength = min_wavelength * multiplier**sc
            log_gabor_filters[sc] = np.exp((-(np.log(radius*wavelength))**2) / (2*np.log(sigma_onf)**2)) * lp_filter

        spreads = np.zeros((n_angles, H, W))
        for o in range(n_angles):
            angle = o * np.pi / n_angles
            ds = sintheta * np.cos(angle) - costheta * np.sin(angle)
            dc = costheta * np.cos(angle) + sintheta * np.sin(angle)
            dtheta = abs(np.arctan2(ds,dc))
            dtheta = np.minimum(dtheta*n_angles*0.5, np.pi)
            spreads[o] = (np.cos(dtheta)+1)/2


        filter_bank = np.zeros((n_scales*n_angles, H, W))
        for sc in range(n_scales):
            for o in range(n_angles):
                filter_bank[sc * n_angles + o] = log_gabor_filters[sc] * spreads[o]

        return filter_bank

    def lowpassfilter(self, H, W, cutoff, n):
        if cutoff < 0 or cutoff > 0.5:
            raise ValueError('the cutoff frequency needs to be between 0 and 0.5')

        if not n == int(n) or n < 1.0:
            raise ValueError('n must be an integer >= 1')

        xrange = np.linspace(-0.5, 0.5, W)
        yrange = np.linspace(-0.5, 0.5, H)

        x, y = np.meshgrid(xrange, yrange)
        radius = np.sqrt(x**2 + y**2)
        radius = np.fft.ifftshift(radius)
        return 1.0 / (1.0 + (radius / cutoff)**(2*n))
