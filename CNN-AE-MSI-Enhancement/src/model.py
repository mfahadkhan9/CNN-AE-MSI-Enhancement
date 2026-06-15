"""CNN-AE model for multispectral transmission image enhancement.

The architecture follows the manuscript description: three Conv2D encoder
blocks, a 32 x 32 x 256 bottleneck for 256 x 256 RGB inputs, and three
ConvTranspose2D decoder stages with sigmoid output normalization.
"""

from __future__ import annotations

import torch
from torch import nn


class CNNAutoEncoder(nn.Module):
    """Convolutional autoencoder for registration-oriented image enhancement.

    Parameters
    ----------
    in_channels:
        Number of input image channels. Use 3 for RGB multispectral frame
        representations.
    base_channels:
        Number of feature channels in the first encoder block.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c3, c2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c2, c1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c1, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output


# Backward-compatible alias for older scripts.
DenoisingAutoencoder = CNNAutoEncoder
