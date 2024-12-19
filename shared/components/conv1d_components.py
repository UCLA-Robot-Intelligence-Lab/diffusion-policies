import torch
import torch.nn as nn
import torch.nn.functional as F


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels=dim, out_channels=dim, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, num_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=inp_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)
