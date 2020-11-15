import cv2
import numpy as np
import torch

class SuperPointMagicLeap(torch.nn.Module):
    """
    Pytorch definition of SuperPoint Network. Copied and slightly adapted
    from the upstream magic leap repo (https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork)
    """
    def __init__(self, config = None):
        super(SuperPointMagicLeap, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

    def forward(self, data):
        """
        Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x H x W.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # Shared Encoder.
        x = self.relu(self.conv1a(data['image']))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        # Detector Head.
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)
        # Descriptor Head.
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
        output = {}
        output['logits'] = semi
        output['desc'] = desc

        output['prob'] = self.generate_heatmap(semi, data['image'].shape)
        return output

    def generate_heatmap(self, semi, shape):
        out = torch.zeros(shape, device = semi.device)
        for i, sample in enumerate(semi):
            # --- Process points.
            dense = np.exp(sample.cpu().numpy()) # Softmax.
            dense = dense / (np.sum(dense, axis=0)+.00001) # Should sum to 1.
            # Remove dustbin.
            nodust = dense[:-1, :, :]
            # Reshape to get full resolution heatmap.
            Hc = int(out.shape[-2] / 8)
            Wc = int(out.shape[-1] / 8)
            nodust = nodust.transpose(1, 2, 0)
            heatmap = np.reshape(nodust, [Hc, Wc, 8, 8])
            heatmap = np.transpose(heatmap, [0, 2, 1, 3])
            heatmap = np.reshape(heatmap, [Hc*8, Wc*8])
            out[i,0] = torch.from_numpy(heatmap)

        return out
