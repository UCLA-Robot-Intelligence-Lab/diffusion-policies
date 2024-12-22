import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange

"""
Dimension key:

B: batch size
S: sequence length / horizon
I: input channels / dimensions
O: output channels/ dimensions
C: conditioning (observation) dimension
D: diffusion step embedding dimension
G: global conditioning dimension
L: local conditioning dimension

Taken from Noam Shazeer's shape suffixes post on Medium

TODO: Update dimension key and update other components with dimension key and annotationso
TOOD: Shape testing
"""


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


class ConditionalResBlock1d(nn.Module):
    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        num_groups: int = 8,
        cond_predict_scale: bool = False,
    ):
        super().__init__()

        # cond_predict_scale should be set if you want the conditional input
        # to modulate the scale and bias, or just be an additive bias

        self.blocks = nn.ModuleList(
            [
                # x : [ B, S, I ] -> [ B, S, O ]
                Conv1dBlock(
                    inp_channels=inp_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    num_groups=num_groups,
                ),
                # x : [ B, S, O ] -> [ B, S, O ]
                Conv1dBlock(
                    inp_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    num_groups=num_groups,
                ),
            ]
        )

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # If cond_predict_scale is True, predict both scale and bias (2*O channels)
        # Otherwise, predict just bias (O channels)
        cond_channels = out_channels * 2 if cond_predict_scale else out_channels

        # cond : [ B, C ] -> embed : [ B, O, 1]
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(in_features=cond_dim, out_features=cond_channels),
            Rearrange(
                "b t -> b t 1"
            ),  # we re-arrange to get a final singleton dimension for broadcasting
        )

        # Residual connection: adjust channels if inp_channels != out_channels
        # x : [ B, S, I ] -> [ B, S, O ]
        self.residual_conv = (
            nn.Conv1d(
                in_channels=inp_channels, out_channels=out_channels, kernel_size=1
            )
            if inp_channels != out_channels
            else nn.Identity()
        )

        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels

    def forward(self, x_BSI: torch.Tensor, cond_BC: torch.Tensor) -> torch.Tensor:
        """
        Applies conditional residual block with FiLM conditioning.

        args:
            x_BSI: Input tensor [ B, S, I ]
            cond_BC: Conditioning tensor [ B, C ]

        returns:
            out_BSO: Output tensor [ B, S, O ]
        """
        out_BSO = self.blocks[0](x_BSI)
        embed_BO1 = self.cond_encoder(cond_BC)

        # this is FiLM conditioning:
        if self.cond_predict_scale:
            # Reshape conditioning into scale and bias
            # embed : [ B, O, 1 ] -> [ B, 2, O, 1 ]
            embed_B2O1 = embed_BO1.reshape(embed_BO1.shape[0], 2, self.out_channels, 1)

            # Split into scale & bias, each [ B, O, 1 ]
            scale_BO1 = embed_B2O1[:, 0, ...]
            bias_BO1 = embed_B2O1[:, 1, ...]

            # Apply scale and bias; broadcasting over S dimension
            out_BSO = scale_BO1 * out_BSO + bias_BO1
        else:
            # Apply just bias; broadcasting over S dimension
            out_BSO = out_BSO + embed_BO1

        out_BSO = self.blocks[1](out_BSO)
        out_BSO = out_BSO + self.residual_conv(x_BSI)

        return out_BSO
