import torch
import torch.nn as nn

import torch.nn.functional as F


class STN(nn.Module):
    """
    Follow implement in https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
    """

    def __init__(self, img_size=384, n_channel=3):
        super(STN, self).__init__()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(n_channel, 8, kernel_size=7),  # 3 or 1 channel
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.nx = ((img_size - 7 + 1) // 2 - 5 + 1) // 2
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * self.nx * self.nx, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * self.nx * self.nx)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)
        return x