import torch
import torch.nn as nn
from typing import Union
from einops.layers.torch import Rearrange
from shared.components.conv1d_components import Upsample1d, Downsample1d, Conv1dBlock
from shared.components.positional_embed import SinusoidalPosEmb

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
"""


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
            if in_channels != out_channels
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


class ConditionalUnet1d(nn.Module):
    def __init__(
        self,
        input_dim: int,
        local_cond_dim: int = None,
        global_cond_dim: int = None,
        diffusion_step_embed_dim: int = 256,
        down_dims: Union[int, list] = [256, 512, 1024],
        kernel_size: int = 3,
        num_groups: int = 8,
        cond_predict_scale: bool = False,
    ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        # Diffusion step: [ B, ] -> [ B, D ]
        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dim=dsed),
            nn.Linear(in_features=dsed, out_features=dsed * 4),
            nn.Mish(),
            nn.Linear(in_features=dsed * 4, out_features=dsed),
        )

        # Combined conditioning: [ B, D ] + ([ B, G ] if global_cond else [ ]) -> [ B, C ]
        cond_dim = dsed + global_cond_dim if global_cond_dim is not None else dsed

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        # Local conditioning: [ B, S, L ] -> [ B, O, S ]
        local_cond_encoder = None
        if local_cond_dim is not None:
            _, out_dim = in_out[0]
            inp_dim = local_cond_dim
            local_cond_encoder = nn.ModuleList(
                [
                    ConditionalResBlock1d(
                        inp_channels=inp_dim,
                        out_channels=out_dim,
                        cond_dim=cond_dim,
                        kernel_size=kernel_size,
                        num_groups=num_groups,
                        cond_predict_scale=cond_predict_scale,
                    ),
                    ConditionalResBlock1d(
                        inp_channels=inp_dim,
                        out_channels=out_dim,
                        cond_dim=cond_dim,
                        kernel_size=kernel_size,
                        num_groups=num_groups,
                        cond_predict_scale=cond_predict_scale,
                    ),
                ]
            )

        # Bottleneck / middle blocks: [ B, S, L ] -> [ B, O, S ]
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResBlock1d(
                    inp_channels=mid_dim,
                    out_channels=mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    num_groups=num_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
                ConditionalResBlock1d(
                    inp_channels=mid_dim,
                    out_channels=mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    num_groups=num_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
            ]
        )

        # Downsampling path: [ B, I, S ] -> [B, O, S/2 ] for each step
        down_modules = nn.ModuleList([])
        for ind, (inp_dim, out_dim) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResBlock1d(
                            inp_channels=inp_dim,
                            out_channels=out_dim,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            num_groups=num_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        ConditionalResBlock1d(
                            inp_channels=out_dim,
                            out_channels=out_dim,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            num_groups=num_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        Downsample1d(dim=out_dim) if not is_last else nn.Identity(),
                    ]
                )
            )

        # Upsampling path: [ B, O*2, S ] -> [B, I, S*2 ] for each step
        up_modules = nn.ModuleList([])
        for ind, (inp_dim, out_dim) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResBlock1d(
                            inp_channels=out_dim * 2,
                            out_channels=inp_dim,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            num_groups=num_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        ConditionalResBlock1d(
                            inp_channels=inp_dim,
                            out_channels=inp_dim,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            num_groups=num_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        Upsample1d(dim=inp_dim) if not is_last else nn.Identity(),
                    ]
                )
            )

        # Final projection: [ B, O, S ] -> [ B, I, S ]
        final_conv = nn.Sequential(
            Conv1dBlock(
                inp_channels=start_dim, out_channels=start_dim, kernel_size=kernel_size
            ),
            nn.Conv1d(in_channels=start_dim, out_channels=input_dim, kernel_size=1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

    def forward(
        self,
        sample_BSI: torch.Tensor,
        timestep_B: Union[torch.Tensor, float, int],
        local_cond_BSL=None,
        global_cond_BG=None,
        **kwargs,
    ):
        """
        args:
            sample_BSI : [ B, S, I ] Input tensor
            timestep_B : [ B, ] Diffusion timestep
            local_cond_BSL : [ B, S, L]  Local conditioning (optional)
            global_cond_BG : [ B, G ] Global conditioning (optional)

        returns:
            out_BSI : [ B, S, I ] Output tensor
        """
        # Rearrange input; x : [ B S I ] -> [ B I S ] for 1d convs
        sample_BIS = einops.rearrange(sample_BSI, "b s i -> b i s")

        # 1. Process timesteps
        timesteps = timestep_B
        if not torch.is_tensor(timesteps):
            # Convert scalar to tensor (requires CPU-GPU sync, prefer passing tensor)
            timesteps = torch([timesteps], dtype=torch.long, device=sample_BIS.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample_BIS.device)

        # Broadcast to batch dimensions
        timesteps_B = timesteps(sample_BIS.shape[0])

        # 2. Encode timestep and global conditioning
        global_feat_BD = self.diffusion_step_encoder(timesteps_B)

        if global_cond_BG is not None:
            global_feat_BC = torch.cat([global_feat_BD, global_cond_BG], dim=-1)
        else:
            global_feat_BC = global_feat_BD

        # 3. Encode local conditioning if provided
        h_local_BOS = []  # List of local feature maps
        if local_cond_BSL is not None:
            local_cond_BLS = einops.rearrange(local_cond_BSL, "b s l -> b l s")
            visual_encoder1, visual_encoder2 = self.local_cond_encoder

            x_BOS = visual_encoder1(local_cond_BLS, global_feat_BC)
            h_local_BOS.append(x_BOS)

            x_BOS = visual_encoder2(local_cond_BLS, global_feat_BC)
            h_local_BOS.append(x_BOS)

        # 4. Downsampling path
        x_BOS = sample_BIS
        h_BOS = []
        for idx, (visual_encoder1, visual_encoder2, downsample) in enumerate(
            self.down_modules
        ):
            x_BOS = visual_encoder1(x_BOS, global_feat_BC)

            # Add first local features at initial resolution
            if idx == 0 and len(h_local_BOS) > 0:
                x_BOS = x_BOS + h_local_BOS[0]

            x_BOS = visual_encoder2(x_BOS, global_feat_BC)
            h_BOS.append(x_BOS)
            x_BOS = downsample(x_BOS)

        # 5. Middle / bottleneck blocks
        for mid_module in self.mid_modules:
            x_BOS = mid_module(x_BOS, global_feat_BC)

        # 6. Upsampling path
        for idx, (visual_encoder1, visual_encoder2, upsample) in enumerate(
            self.up_modules
        ):
            x_BOS = torch.cat((x_BOS, h_BOS.pop()), dim=1)
            x_BOS = visual_encoder1(x_BOS, global_feat_BC)

            if idx == (len(self.up_modules) - 1) and len(h_local_BOS) > 0:
                x_BOS = x_BOS + h_local_BOS[1]

            x_BOS = visual_encoder2(x_BOS, global_feat_BC)
            x_BOS = upsample(x_BOS)

        # 7. Final projection / convolution
        x_BIS = self.final_conv(x_BOS)

        output_BSI = einops.rearrange(x_BIS, "b i s -> b s i")

        return output_BSI
