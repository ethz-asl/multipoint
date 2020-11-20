import numpy as np
import torch
from torch.nn import Module

from .utils import space_to_depth, dict_update, tensors_to_dtype
from .homographies import warp_points_pytorch

class SuperPointLoss(Module):
    '''
    Loss to train the SuperPoint model according to:
    "SuperPoint: Self-Supervised Interest Point Detection and Description"
    '''
    default_config = {
        'detector_loss': True,
        'detector_use_cross_entropy': True,
        'descriptor_loss': True,
        'descriptor_loss_threshold': 8.0,
        'sparse_descriptor_loss': False,
        'sparse_descriptor_loss_num_cell_divisor': 64,
        'descriptor_loss_use_mask': True,
        'positive_margin': 1.0,
        'negative_margin': 0.2,
        'lambda_d': 250,
        'lambda': 0.0001,
    }

    def __init__(self, config = None):
        super(SuperPointLoss, self).__init__()

        if config:
            self.config = dict_update(self.default_config, config)
        else:
            self.config = self.default_config

    def forward(self, pred, data, pred2 = None, data2 = None):
        if ((pred2 is None and data2 is not None) or
            (pred2 is not None and data2 is None)):
            raise ValueError('The data and the label must be present to compute the loss')

        pred = tensors_to_dtype(pred, torch.float)

        loss_components = {}
        loss = torch.zeros([1]).to(data['keypoints'].device)
        if self.config['detector_loss']:
            detector_loss1 = self.detector_loss(pred['logits'], data['keypoints'], data['valid_mask'])
            loss += detector_loss1
            loss_components['detector_loss1'] = detector_loss1.item()

            if pred2 is not None:
                pred2 = tensors_to_dtype(pred2, torch.float)

                detector_loss2 = self.detector_loss(pred2['logits'], data2['keypoints'], data2['valid_mask'])
                loss += detector_loss2
                loss_components['detector_loss2'] = detector_loss2.item()

        if self.config['descriptor_loss']:
            if pred2 is None:
                raise ValueError('The descriptor loss requires predictions from two images')

            if 'homography' in data.keys():
                homography = data['homography']
            else:
                homography = None

            if 'homography' in data2.keys():
                homography2 = data2['homography']
            else:
                homography2 = None

            descriptor_loss, positive_dist, negative_dist =  self.descriptor_loss(pred['desc'],
                                                                                  pred2['desc'],
                                                                                  homography,
                                                                                  homography2,
                                                                                  data['valid_mask'],
                                                                                  data2['valid_mask'])

            loss_components['descriptor_loss'] = descriptor_loss.item()
            loss_components['positive_dist'] = positive_dist.item()
            loss_components['negative_dist'] = negative_dist.item()

            loss += self.config['lambda'] * descriptor_loss

        return loss, loss_components

    def detector_loss(self, logits, keypoint_map, valid_mask = None):
        # convert the labels into the encoded space
        labels = space_to_depth(keypoint_map.unsqueeze(1), 8)
        shape = list(labels.shape)
        shape[1] = 1

        # convert the valid mask to mask the bins instead of pixels
        # if any pixel in the bin is invalid the bin is invalid
        valid_mask = torch.ones_like(keypoint_map).bool().unsqueeze(1).to(labels.device) if valid_mask is None else valid_mask
        valid_mask = space_to_depth(valid_mask, 8)
        valid_mask = torch.prod(valid_mask, 1)

        if self.config['detector_use_cross_entropy']:
            # add random values to the labels to randomly pick one keypoint if there are
            # more than one keypoint in one bin
            labels = 3.0*labels + torch.rand(labels.shape).to(labels.device)

            # add channel for the no interest point bin
            labels = torch.cat([labels, 2.0 * torch.ones(shape).to(labels.device)], dim=1)
            labels = torch.argmax(labels, 1)

            # compute the cross entropy loss and mask it, then return the average
            loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none')

        else:
            # add the dustbin channel and set it to 1 if no label was present in the cell
            labels = torch.cat([labels.float(), torch.zeros(shape).to(labels.device)], dim=1)
            labels[:,-1] = 1 - labels.sum(1).clamp(max=1.0)

            # normalize along the channel dimension
            labels = labels / labels.sum(dim=1).unsqueeze(1)

            # compute the loss
            loss = torch.nn.functional.binary_cross_entropy(torch.nn.functional.softmax(logits, dim=1), labels, reduction='none').sum(1)

        loss *= valid_mask

        return (loss.sum(-1).sum(-1) / valid_mask.sum(-1).sum(-1)).mean()

    def descriptor_loss(self, descriptor1, descriptor2, homography1, homography2, valid_mask1 = None, valid_mask2 = None):
        # input check
        assert descriptor1.shape == descriptor2.shape

        if homography1 is not None and homography2 is not None:
            assert homography1.shape == homography2.shape
            assert descriptor1.shape[0] == homography1.shape[0]

        # Compute the position of the center pixel of every cell in the image
        batch_size = descriptor1.shape[0]
        desc_size = descriptor1.shape[1]
        Hc = descriptor1.shape[2]
        Wc = descriptor1.shape[3]

        if self.config['sparse_descriptor_loss']:
            # select random indices
            num_cells = int(np.floor(Hc*Wc/self.config['sparse_descriptor_loss_num_cell_divisor']))
            coord_cells = torch.stack((torch.randint(Hc, (num_cells,)), torch.randint(Wc, (num_cells,))), dim = -1)

            # extend the cell coordinates in the batch dimension
            coord_cells = coord_cells.unsqueeze(0).expand([batch_size, -1, -1]).clone().to(descriptor1.device)

            # warp the coordinates into the common frame
            if homography1 is not None:
                warped_cells_1 = warp_points_pytorch(coord_cells.float(), homography1)

            if homography2 is not None:
                warped_cells_2 = warp_points_pytorch(coord_cells.float(), homography2)

            # compute the correspondance
            # do it this way instead of setting the identity matrix since we could sample the same cell twice
            dist = (coord_cells.unsqueeze(1).float() - coord_cells.unsqueeze(-2).float()).norm(dim=-1)
            correspondance = (dist <= np.sqrt(0.5)).float()

            # create a valid mask based on which cells are visible in both images
            valid = (((warped_cells_1[:,:,0] > - 0.5).float() *
                     (warped_cells_1[:,:,0] < Hc - 0.5).float()).unsqueeze(1) *
                     ((warped_cells_2[:,:,0] > - 0.5).float() *
                     (warped_cells_2[:,:,0] < Wc - 0.5).float()).unsqueeze(-1))

            # make sure the indexes are within the image frame
            idx_1 = warped_cells_1.round().int()
            idx_1[:,:,0].clamp_(0,Hc-1)
            idx_1[:,:,1].clamp_(0,Wc-1)
            idx_2 = warped_cells_2.round().int()
            idx_2[:,:,0].clamp_(0,Hc-1)
            idx_2[:,:,1].clamp_(0,Wc-1)

            # memory cleanup
            del coord_cells
            del warped_cells_1
            del warped_cells_2
            del dist

            # extract the descriptors
            desc_1 = torch.zeros((batch_size, desc_size, num_cells)).to(descriptor1.device)
            desc_2 = torch.zeros((batch_size, desc_size, num_cells)).to(descriptor1.device)

            for i, idx in enumerate(idx_1):
                desc_1[i] = descriptor1[i,:,idx[:,0].long(), idx[:,1].long()]

            for i, idx in enumerate(idx_2):
                desc_2[i] = descriptor2[i,:,idx[:,0].long(), idx[:,1].long()]

            # compute the dot product
            dot_product_desc = torch.matmul(desc_2.permute([0,2,1]), desc_1)

            # Compute the loss
            positive_dist = self.config['lambda_d'] * correspondance *  torch.max(torch.zeros(1, device=descriptor1.device), self.config['positive_margin'] - dot_product_desc)
            negative_dist = (1 - correspondance) * torch.max(torch.zeros(1, device=descriptor1.device), dot_product_desc - self.config['negative_margin'])

            # apply the valid mask
            positive_dist *= valid
            negative_dist *= valid

            loss = negative_dist + positive_dist

            normalization = valid.sum(-1).sum(-1)

            loss = (loss.sum(-1).sum(-1) / normalization).mean()
            positive_dist = (positive_dist.sum(-1).sum(-1) / normalization).mean()
            negative_dist = (negative_dist.sum(-1).sum(-1) / normalization).mean()

        else:
            coord_cells = torch.stack(torch.meshgrid(torch.arange(Hc), torch.arange(Wc)), dim=-1)
            coord_cells = coord_cells * 8.0 + 4.0
            coord_cells = coord_cells.unsqueeze(0).expand([batch_size, -1, -1, -1]).clone().to(descriptor1.device)

            # warp the pixel centers
            shape = coord_cells.shape
            if homography1 is not None:
                warped_cells_1 = warp_points_pytorch(coord_cells.reshape(batch_size, -1,2), homography1.inverse()).reshape(shape)
            else:
                warped_cells_1 = coord_cells

            if homography2 is not None:
                warped_cells_2 = warp_points_pytorch(coord_cells.reshape(batch_size, -1,2), homography2.inverse()).reshape(shape)
            else:
                warped_cells_2 = coord_cells

            # compute the pair wise distance
            dist = (warped_cells_1.unsqueeze(1).unsqueeze(1) - warped_cells_2.unsqueeze(-2).unsqueeze(-2)).norm(dim=-1)
            correspondance = (dist <= self.config['descriptor_loss_threshold']).float()

            # memory cleanup
            del coord_cells
            del warped_cells_1
            del warped_cells_2
            del dist

            #dot_product_desc2 = (descriptor1.unsqueeze(2).unsqueeze(2) * descriptor2.unsqueeze(-1).unsqueeze(-1)).sum(1) # uses too much memory
            dot_product_desc = torch.matmul(descriptor2.view(batch_size, desc_size, -1).permute([0,2,1]),
                                            descriptor1.view(batch_size, desc_size, -1)).view(batch_size, Hc, Wc, Hc, Wc)

            # Compute the loss
            positive_dist = self.config['lambda_d'] * correspondance *  torch.max(torch.zeros(1, device=descriptor1.device), self.config['positive_margin'] - dot_product_desc)
            negative_dist = (1 - correspondance) * torch.max(torch.zeros(1, device=descriptor1.device), dot_product_desc - self.config['negative_margin'])
            del dot_product_desc
            loss = positive_dist + negative_dist

            if self.config['descriptor_loss_use_mask']:
                # get the valid mask
                if valid_mask1 is None:
                    valid_mask1 = torch.ones(batch_size, 1, Hc, Wc).to(descriptor1.device)
                else:
                    valid_mask1 = space_to_depth(valid_mask1, 8)
                    valid_mask1 = torch.prod(valid_mask1, 1)

                if valid_mask2 is None:
                    valid_mask2 = torch.ones(batch_size, 1, Hc, Wc).to(descriptor1.device)
                else:
                    valid_mask2 = space_to_depth(valid_mask2, 8)
                    valid_mask2 = torch.prod(valid_mask2, 1)

                valid_mask = torch.matmul(valid_mask2.view(batch_size, -1, 1).float(),
                                          valid_mask1.view(batch_size, 1, -1).float()).view(batch_size, Hc, Wc, Hc, Wc)

                loss *= valid_mask
                positive_dist *= valid_mask
                negative_dist *= valid_mask
                normalization = valid_mask.sum(-1).sum(-1).sum(-1).sum(-1)
            else:
                normalization = Hc * Wc *  Hc * Wc

            loss = (loss.sum(-1).sum(-1).sum(-1).sum(-1) / normalization).mean()
            positive_dist = (positive_dist.sum(-1).sum(-1).sum(-1).sum(-1) / normalization).mean()
            negative_dist = (negative_dist.sum(-1).sum(-1).sum(-1).sum(-1) / normalization).mean()

        return loss, positive_dist, negative_dist
