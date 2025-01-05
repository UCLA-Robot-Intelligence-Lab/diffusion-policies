import torch
import torch.nn as nn

from einops.layers.torch import Rearrange

"""
Selected Dimension Keys:

B: batch size
T: prediction horizon
C: conditioning dimension
I: (conv) input channel dimension
O: (conv) output channel dimension
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
    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        kernel_size: int,
        num_groups: int = 8,
    ):
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


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        condition_dim: int,
        kernel_size: int = 3,
        num_groups: int = 8,
        film_modulation_scale: bool = False,
    ):
        super().__init__()
        """
        args:
            inp_channels : I number of input channels
            out_channels : O desired number of output channels
            condition_dim : C dimension of the conditioning vector used in FiLM
            kernel_size : desired kernel size used in Conv1d blocks
            num_groups : number of groups for the GroupNorm used in Conv1d blocks
            film_modulation_scale : set to True if you want FiLM conditioning to also
                                    modulate the scale (gamma). By default we always
                                    modulate with a bias (beta). Read more below.
        """

        self.blocks = nn.ModuleList(
            [
                # x : [ B, I, T ] -> [ B, O, T ]
                Conv1dBlock(
                    inp_channels=inp_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    num_groups=num_groups,
                ),
                # x : [ B, O, T ] -> [ B, O, T ]
                Conv1dBlock(
                    inp_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    num_groups=num_groups,
                ),
            ]
        )

        """
        FiLM (Feature-wise Linear Modulation) modulation as defined in [1].

        FiLM learns a function f that outputs (gamma, beta) vectors.
        For convnets, this modulates the per-feature-map distribution of
        activations based on input x_i, agnost to spatial location.
        For a networks activation F_i,c referring to the ith input's cth
        feature map, we have this feature-wise affine transformation.
        For each feature channel c in the convolutional layer:

        FiLM(F_c | gamma_c, beta_c) = gamma_c * F_c + beta_c

        This function (FiLM generator) gives our network the ability
        to manipulate the feature maps of a target. Each feature
        map is conditioned independently, i.e., predicted per-channel.

        References:
            [1]: FiLM: Visual Reasoning with a General Conditioning Layer
            https://arxiv.org/pdf/1709.07871
        """

        # If film_modulation_scale is True, we need 2*O channels to predict both scale and bias
        condition_channels = out_channels * 2 if film_modulation_scale else out_channels

        # condition_dim : [ B, C ] -> embed : [ B, O, 1 ]
        self.film_generator = nn.Sequential(
            nn.Mish(),
            nn.Linear(condition_dim, condition_channels),
            Rearrange("batch t -> batch t 1"),  # For broadcasting later.
        )

        # x : [ B, T, I ] -> [ B, T, O ]
        self.conv_residual = (
            nn.Conv1d(
                in_channels=inp_channels, out_channels=out_channels, kernel_size=1
            )
            if inp_channels != out_channels
            else nn.Identity()
        )

        self.film_modulation_scale = film_modulation_scale
        self.out_channels = out_channels

    def forward(self, x_BIT: torch.Tensor, condition_BC: torch.Tensor) -> torch.Tensor:
        """
        args:
            x_BIT : [ B, I, T ] input tensor
            condition_BC : [ B C ] conditioning tensor, C is condition_dim

        returns:
            out_BOT : [ B, O, T ] output tensor, FiLM conditioning is either
                                  fully or partially (just bias) applied
        """
        blocks = self.blocks
        out_channels = self.out_channels
        film_modulation_scale = self.film_modulation_scale
        film_generator = self.film_generator
        conv_residual = self.conv_residual

        out_BOT = blocks[0](x_BIT)
        embed_BO1 = film_generator(condition_BC)

        # FiLM conditioning:
        if film_modulation_scale:
            # Reshape conditioning embedding into learned scale and bias
            # embed : [ B, O, 1 ] -> [ B, 2, O, 1 ]
            embed_B2O1 = embed_BO1.reshape(embed_BO1.shape[0], 2, out_channels, 1)

            # Split into scale & bias, each [ B, O, 1 ]
            scale_BO1 = embed_B2O1[:, 0, ...]
            bias_BO1 = embed_B2O1[:, 1, ...]

            # Apply scale and bias; broadcasting over T dimension
            out_BOT = scale_BO1 * out_BOT + bias_BO1
        else:
            # Apply just bias; broadcasting over T dimension
            out_BOT = out_BOT + embed_BO1

        out_BOT = blocks[1](out_BOT)
        out_BOT = out_BOT + conv_residual(x_BIT)

        return out_BOT
