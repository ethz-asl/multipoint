import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import multipoint.utils as utils

class MultiPoint(nn.Module):
    default_config = {
        'multispectral': True,
        'descriptor_head': True,
        'intepolation_mode': 'bilinear',
        'descriptor_size': 256,
        'normalize_descriptors': True,
        'final_batchnorm': True,
        'reflection_pad': True,
        'bn_first': False,
        'double_convolution': True,
        'channel_version': 0,
        'verbose': False,
        'mixed_precision': False,
        'force_return_logits': False
    }

    def __init__(self, config = None):
        super(MultiPoint, self).__init__()

        if config:
            self.config = utils.dict_update(copy.deepcopy(self.default_config), config)
        else:
            self.config = self.default_config

        if self.config['reflection_pad']:
            self.pad_method = nn.ReflectionPad2d
        else:
            self.pad_method = nn.ZeroPad2d

        if self.config['channel_version'] == 0:
            self.n_channels = [1, 64, 64, 128, 128]
            self.head_channels = 256

        elif self.config['channel_version'] == 1:
            self.n_channels = [1, 32, 64, 96, 128]
            self.head_channels = self.config['descriptor_size']

        elif self.config['channel_version'] == 2:
            self.n_channels = [1, 8, 16, 32, 64]
            self.head_channels = self.config['descriptor_size']

        else:
            print('Unknown channel_version: ', self.config['channel_version'])
            self.n_channels = [1, 64, 64, 128, 128]
            self.head_channels = 256

        if self.config['multispectral']:
            self.encoder_thermal = self.generate_encoder()
            self.encoder_optical = self.generate_encoder()
        else:
            self.encoder = self.generate_encoder()

        # detector head
        self.detector_head_convolutions = [
            self.pad_method(1),
            nn.Conv2d(self.n_channels[4], self.head_channels, 3),
            *self.getNonlinearity(self.head_channels),
            nn.Conv2d(self.head_channels, 65, 1),
        ]

        if self.config['final_batchnorm']:
            self.detector_head_convolutions.append(nn.BatchNorm2d(65))

        self.detector_head_convolutions = nn.Sequential(*self.detector_head_convolutions)

        self.softmax = nn.Softmax2d()
        self.shuffle = nn.PixelShuffle(8)

        if self.config['descriptor_head']:
            self.descriptor_head_convolutions = [
                self.pad_method(1),
                nn.Conv2d(self.n_channels[4], self.head_channels, 3),
                *self.getNonlinearity(self.head_channels),
                nn.Conv2d(self.head_channels, self.config['descriptor_size'], 1),
            ]

            if self.config['final_batchnorm']:
                self.descriptor_head_convolutions.append(nn.BatchNorm2d(self.config['descriptor_size']))

            self.descriptor_head_convolutions = nn.Sequential(*self.descriptor_head_convolutions)

        if self.config['verbose']:
            print('MultiPoint number of trainable parameter: ' + str(sum([p.numel() for p in self.parameters()])))

    def set_force_return_logits(self, value):
        if not isinstance(value, bool):
            raise ValueError('set_force_return_logits: The input value needs to be a bool')

        self.config['force_return_logits']= value

    def forward(self, data):
        if self.config['mixed_precision']:
            with torch.cuda.amp.autocast():
                return self.forward_impl(data)
        else:
            return self.forward_impl(data)

    def forward_impl(self, data):
        if self.config['multispectral']:
            # create a tensor with the output shape of the encoder
            shape = data['image'].shape
            tensor_dtype = torch.float
            if self.config['mixed_precision']:
                tensor_dtype = torch.half

            x = torch.zeros((shape[0], self.n_channels[4], int(shape[2]/8), int(shape[3]/8)), dtype = tensor_dtype).to(data['image'].device)

            # check if there is at least one optical image in the batch
            num_optical = data['is_optical'][:,0].sum()
            num_thermal = shape[0] - num_optical
            if num_optical > 0.0:
                x[data['is_optical'][:,0],:] = self.encoder_optical(data['image'][data['is_optical'][:,0]])
            if num_thermal > 0.0:
                x[~data['is_optical'][:,0],:] = self.encoder_thermal(data['image'][~data['is_optical'][:,0],:])
        else:
            x = self.encoder(data['image'])

        prob, logits = self.detector_head(x)
        out = {'prob': prob,
               'logits': logits,
        }

        if self.config['descriptor_head']:
            desc = self.descriptor_head(x)
            out['desc'] = desc

        return out

    def getNonlinearity(self, N):
        if self.config['bn_first']:
            return nn.BatchNorm2d(N), nn.ReLU(True)
        else:
            return nn.ReLU(True), nn.BatchNorm2d(N)

    def getConvolutionBlock(self, N_in, N_out):
        if self.config['double_convolution']:
            return (self.pad_method(1), nn.Conv2d(N_in, N_out, 3), *self.getNonlinearity(N_out),
                    self.pad_method(1), nn.Conv2d(N_out, N_out, 3), *self.getNonlinearity(N_out))
        else:
            return (self.pad_method(1), nn.Conv2d(N_in, N_out, 3), *self.getNonlinearity(N_out))

    def detector_head(self, x):
        logits = self.detector_head_convolutions(x).to(torch.float)

        if self.training or self.config['force_return_logits']:
            return None, logits
        else:
            prob = self.softmax(logits)
            prob = self.shuffle(prob[:,:-1])
            return prob, None

    def descriptor_head(self, x):
        x = self.descriptor_head_convolutions(x).to(torch.float)

        if self.config['normalize_descriptors']:
            x = torch.nn.functional.normalize(x, p=2, dim=1)

        return x

    def generate_encoder(self):
        encoder = nn.Sequential(
            *self.getConvolutionBlock(self.n_channels[0], self.n_channels[1]),

            nn.MaxPool2d(2,2),

            *self.getConvolutionBlock(self.n_channels[1], self.n_channels[2]),

            nn.MaxPool2d(2,2),

            *self.getConvolutionBlock(self.n_channels[2], self.n_channels[3]),

            nn.MaxPool2d(2,2),

            *self.getConvolutionBlock(self.n_channels[3], self.n_channels[4]),
        )

        return encoder
