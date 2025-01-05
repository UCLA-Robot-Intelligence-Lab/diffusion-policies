import logging
import torch
import torch.nn as nn
import einops

from typing import Union, Optional
from shared.models.components.conv1d_components import (
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
    ConditionalResidualBlock1D,
)
from shared.models.components.positional_embedding import SinusoidalPosEmb

"""
Selected Dimension Keys

B: batch size
T: prediction horizon
F: feature dimension
G: global conditioning dimension
L: local conditioning dimension

I: input channel dimension (generic, not specific)
O: output channel dimension (generic, not specific)
O and I are defined by down_dims
"""


logger = logging.getLogger(__name__)


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        local_cond_L: int = None,
        global_cond_G: int = None,
        diffusion_embed_D: int = 256,
        down_dims: list = [256, 512, 1024],
        kernel_size: int = 3,
        num_groups: int = 8,
        film_modulation_scale: bool = False,
    ):
        super().__init__()
        layer_dims = [input_dim] + list(down_dims)
        base_feat_dim = down_dims[0]

        # Diffusion step: [ B, ] -> [ B, D ]
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_embed_D),
            nn.Linear(diffusion_embed_D, diffusion_embed_D * 4),
            nn.Mish(),
            nn.Linear(diffusion_embed_D * 4, diffusion_embed_D),
        )

        # Combined conditioning:
        # [ B, D ] + ([ B, G ] if global_cond_G else [ ]) -> [ B, C ]
        cond_dim = (
            diffusion_embed_D + global_cond_G
            if global_cond_G is not None
            else diffusion_embed_D
        )

        # Pairs of consecutive dims from layer_dims
        layer_dim_pairs = list(zip(layer_dims[:-1], layer_dims[1:]))

        # Local conditioning: [ B, T, L ] -> [ B, O, T ]
        local_cond_encoder = None
        if local_cond_L is not None:
            _, dim_out = layer_dim_pairs[0]
            dim_in = local_cond_L
            local_cond_encoder = nn.ModuleList(
                [
                    # down encoder
                    ConditionalResidualBlock1D(
                        dim_in,
                        dim_out,
                        condition_dim=cond_dim,
                        kernel_size=kernel_size,
                        num_groups=num_groups,
                        film_modulation_scale=film_modulation_scale,
                    ),
                    # up encoder
                    ConditionalResidualBlock1D(
                        dim_in,
                        dim_out,
                        condition_dim=cond_dim,
                        kernel_size=kernel_size,
                        num_groups=num_groups,
                        film_modulation_scale=film_modulation_scale,
                    ),
                ]
            )

        # Bottleneck / middle blocks: [ B, T, L ] -> [ B, O, T ]
        mid_dim = layer_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    condition_dim=cond_dim,
                    kernel_size=kernel_size,
                    num_groups=num_groups,
                    film_modulation_scale=film_modulation_scale,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    condition_dim=cond_dim,
                    kernel_size=kernel_size,
                    num_groups=num_groups,
                    film_modulation_scale=film_modulation_scale,
                ),
            ]
        )

        # Downsampling path: [ B, I, T ] -> [ B, O, T/2 ] for each iteration
        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(layer_dim_pairs):
            is_last = ind >= (len(layer_dim_pairs) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            condition_dim=cond_dim,
                            kernel_size=kernel_size,
                            num_groups=num_groups,
                            film_modulation_scale=film_modulation_scale,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            condition_dim=cond_dim,
                            kernel_size=kernel_size,
                            num_groups=num_groups,
                            film_modulation_scale=film_modulation_scale,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        # Upsampling path: [ B, O*2, T ] -> [ B, I, T*2 ] for each iteration
        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(layer_dim_pairs[1:])):
            is_last = ind >= (len(layer_dim_pairs) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            condition_dim=cond_dim,
                            kernel_size=kernel_size,
                            num_groups=num_groups,
                            film_modulation_scale=film_modulation_scale,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            condition_dim=cond_dim,
                            kernel_size=kernel_size,
                            num_groups=num_groups,
                            film_modulation_scale=film_modulation_scale,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        # Final projection: [ B, O, T ] -> [ B, F, T ]
        final_conv = nn.Sequential(
            Conv1dBlock(base_feat_dim, base_feat_dim, kernel_size=kernel_size),
            nn.Conv1d(base_feat_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(
        self,
        x_BTF: torch.Tensor,
        timesteps_B: Union[torch.Tensor, float, int],
        local_cond_BTL: Optional[torch.Tensor] = None,
        global_cond_BG: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        args:
            x_BTF : [ B, T, F ] Input tensor, typically a trajectory that
                                has already been conditioned and noised.
                                A "sample" from the noisy data distribution.
                                We predict a denoised trajectory.
            timesteps_B : [ B, ] Diffusion step(s) corresponding to each sample
                                 in the batch. This is to enable the model to
                                 know, for each sample in the batch, how noisy
                                 the input is and how much noise to remove in
                                 order to reconstruct the original data.
            local_cond_BTL : [ B, T, L ] Temporal context: (local) conditioning
                                         specific to each timestep.
            global_cond_BG : [ B, G ] Global context: (global) conditioning, in
                                      this case, encoded observations O_t.
        returns:
            out_BTF : [ B, T, F ] Output tensor, predicted denoised sample.

        Although the input dimension is specifically F, we will use I and O
        since the dimension will vary as we move through the forward pass.

        Notes:
        - We collect feature maps (local_feat_maps) from our local conditional
        data encoder (defined earlier, consisting of two resnet blocks). Each
        of these feature maps is a tensor [ B, L, T ]. We encode our local
        conditioning data local_cond_BTL and collect the appropriate feature
        maps. This allows the model to directly inject relevant local
        conditioning information into intermediate layers, acting like an
        auxiliary set of skip connections.
        - We also collect feature maps (feat_maps) to be our skip connections
        between the downsampling and upsampling paths. We collect these feature
        maps during the downsampling path and apply them in our upsampling
        path. Each feature map x_BFT is added to our list after passing through
        two conditional residual blocks, and are applied in between these
        residual blocks during the upsampling path.
        """
        diffusion_step_encoder = self.diffusion_step_encoder
        local_cond_encoder = self.local_cond_encoder
        down_modules = self.down_modules
        mid_modules = self.mid_modules
        up_modules = self.up_modules
        final_conv = self.final_conv

        # Rearrange input; x : [ B F T ] -> [ B F T ] for convolutional layers.
        x_BFT = einops.rearrange(x_BTF, "B T F -> B F T")

        # 1. Process timesteps.
        if not torch.is_tensor(timesteps_B):
            timesteps_B = torch.tensor(
                [timesteps_B], dtype=torch.long, device=x_BFT.device
            )
        elif torch.is_tensor(timesteps_B) and len(timesteps_B.shape) == 0:
            timesteps_B = timesteps_B[None].to(x_BFT.device)

        # Broadcast to batch dimension by adding a view; compatible with ONNX
        timesteps_B = timesteps_B.expand(x_BFT.shape[0])

        # 2. Encode timestep and global conditioning
        global_feat_BD = diffusion_step_encoder(timesteps_B)
        print("global_feat_BF shape 1: ", global_feat_BD.shape)

        if global_cond_BG is not None:
            global_feat_BF = torch.cat([global_feat_BD, global_cond_BG], axis=-1)
        print("global_cond_BG shape: ", global_cond_BG.shape)
        print("global_feat_BF shape 2: ", global_feat_BF.shape)

        # 3. Encode local conditioning if provided
        # Local feature maps for downsampling-upsampling skip connections
        local_feat_maps = []
        if local_cond_BTL is not None:
            local_cond_BLT = einops.rearrange(local_cond_BTL, "B T L -> B L T")
            res_block1, res_block2 = local_cond_encoder

            x_BOT = res_block1(local_cond_BLT, global_feat_BF)
            local_feat_maps.append(x_BOT)  # [0] Used at first downsample step

            x_BOT = res_block2(local_cond_BLT, global_feat_BF)
            local_feat_maps.append(x_BOT)  # [1] Used at final upsample step

        # 4. Downsampling path
        # Feature maps for downsampling-upsampling skip connections
        feat_maps = []
        print("x_BFT shape before downsampling: ", x_BFT.shape)
        for idx, (res_block1, res_block2, downsample) in enumerate(down_modules):
            print("Entering downsample path on idx: ", idx)
            x_BFT = res_block1(x_BFT, global_feat_BF)
            print("x_BFT: ", x_BFT.shape)
            # Add first set of local feature maps at initial resolution
            if idx == 0 and len(local_feat_maps) > 0:
                x_BFT = x_BFT + local_feat_maps[0]

            x_BFT = res_block2(x_BFT, global_feat_BF)
            print("x_BFT: ", x_BFT.shape)
            feat_maps.append(x_BFT)

            x_BFT = downsample(x_BFT)
            print("x_BFT: ", x_BFT.shape)

        # 5. Middle / bottlenecks blocks
        for mid_module in mid_modules:
            x_BFT = mid_module(x_BFT, global_feat_BF)

        # 6. Upsampling path
        for idx, (res_block1, res_block2, upsample) in enumerate(up_modules):
            x_BFT = torch.cat((x_BFT, feat_maps.pop()), dim=1)
            x_BFT = res_block1(x_BFT, global_feat_BF)

            # Prev.len(up_modules). According to original repo, this is correct
            if idx == (len(up_modules) - 1) and len(local_feat_maps) > 0:
                x_BFT = x_BFT + local_feat_maps[1]

            x_BFT = res_block2(x_BFT, global_feat_BF)
            x_BFT = upsample(x_BFT)

        # 7. Final projection / convolution
        x_BFT = final_conv(x_BFT)

        out_BTF = einops.rearrange(x_BFT, "B F T -> B T F")

        return out_BTF
