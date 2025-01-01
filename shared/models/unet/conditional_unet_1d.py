import torch
import torch.nn as nn
import einops

from typing import Union, Optional
from einops.layers.torch import Rearrange
from shared.components.conv1d_components import (
    Upsample1d,
    Downsample1d,
    Conv1dBlock,
    ConditionalResBlock1d,
)
from shared.components.positional_embeddings import SinusoidalPosEmb

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
        self.has_global_cond = global_cond_dim is not None
        if self.has_global_cond:
            self.global_cond_proj = nn.Sequential(
                nn.Linear(in_features=global_cond_dim, out_features=dsed),
                nn.Mish(),
            )
            cond_dim = dsed * 2  # Combined dim after concat
        else:
            self.global_cond_proj = None
            cond_dim = dsed

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        # Local conditioning: [ B, S, L ] -> [ B, O, S ]
        self.has_local_cond = local_cond_dim is not None
        local_cond_encoder = None
        if self.has_local_cond:
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
        local_cond_BSL: Optional[torch.Tensor] = None,
        global_cond_BG: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
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
        if not torch.is_tensor(timestep_B):
            timestep_B = torch.tensor([timestep_B], device=sample_BIS.device)
        elif len(timestep_B.shape) == 0:
            timestep_B = timestep_B[None].to(sample_BIS.device)

        # Broadcast to batch dimensions
        timesteps_B = timestep_B.expand(sample_BIS.shape[0])

        # 2. Encode timestep and global conditioning
        global_feat_BD = self.diffusion_step_encoder(timesteps_B)

        if self.has_global_cond:
            if global_cond_BG is None:
                global_cond_BG = torch.zeros(
                    (sample_BIS.shape[0], self.global_cond_proj[0].in_features),
                    device=sample_BIS.device,
                )
            global_cond_BG = self.global_cond_proj(global_cond_BG)
            global_feat_BC = torch.cat([global_feat_BD, global_cond_BG], dim=-1)
        else:
            global_feat_BC = global_cond_BD

        # 3. Encode local conditioning if provided
        h_local_BOS = []  # List of local feature maps
        if self.has_local_cond:
            if local_cond_BSL is None:
                local_cond_BSL = torch.zeros(
                    (
                        sample_BIS.shape[0],
                        sample_BIS.shape[-1],
                        self.local_cond_encoder[0].blocks[0].block[0].in_channels,
                    ),
                    device=sample_BIS.device,
                )
            local_cond_BLS = einops.rearrange(local_cond_BSL, "b s l -> b l s")
            for encoder in self.local_cond_encoder:
                h_local_BOS.append(encoder(local_cond_BLS, global_feat_BC))

        # 4. Downsampling path
        x_BOS = sample_BIS
        h_BOS = []
        # encoders are "resnets" in the og impl, but they are just residual blocks here
        for idx, (encoder1, encoder2, downsample) in enumerate(self.down_modules):
            x_BOS = encoder1(x_BOS, global_feat_BC)

            # Add first local features at initial resolution
            if idx == 0 and len(h_local_BOS) > 0:
                x_BOS = x_BOS + h_local_BOS[0]

            x_BOS = encoder2(x_BOS, global_feat_BC)
            h_BOS.append(x_BOS)
            x_BOS = downsample(x_BOS)

        # 5. Middle / bottleneck blocks
        for mid_module in self.mid_modules:
            x_BOS = mid_module(x_BOS, global_feat_BC)

        # 6. Upsampling path
        for idx, (encoder1, encoder2, upsample) in enumerate(self.up_modules):
            x_BOS = torch.cat((x_BOS, h_BOS.pop()), dim=1)
            x_BOS = encoder1(x_BOS, global_feat_BC)

            if idx == (len(self.up_modules)) and len(h_local_BOS) > 0:
                x_BOS = x_BOS + h_local_BOS[1]

            x_BOS = encoder2(x_BOS, global_feat_BC)
            x_BOS = upsample(x_BOS)

        # 7. Final projection / convolution
        x_BIS = self.final_conv(x_BOS)

        output_BSI = einops.rearrange(x_BIS, "b i s -> b s i")

        return output_BSI


# ====== TEST FUNCTION ======
# The following function does some shape testing on the forward pass
def main():
    batch_size = 4
    seq_length = 32
    input_dim = 16
    local_cond_dim = 8
    global_cond_dim = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ConditionalUnet1d(
        input_dim=input_dim,
        local_cond_dim=local_cond_dim,
        global_cond_dim=global_cond_dim,
    ).to(device)

    sample = torch.randn(batch_size, seq_length, input_dim).to(device)
    timestep = torch.ones(batch_size).to(device)
    local_cond = torch.randn(batch_size, seq_length, local_cond_dim).to(device)
    global_cond = torch.randn(batch_size, global_cond_dim).to(device)

    print("Testing ConditionalUnet1d shapes...")
    try:
        output = model(
            sample_BSI=sample,
            timestep_B=timestep,
            local_cond_BSL=local_cond,
            global_cond_BG=global_cond,
        )

        assert (
            output.shape == sample.shape
        ), f"Shape mismatch: expected {sample.shape}, got {output.shape}"

        print(f"Input shape: {sample.shape}")
        print(f"Output shape: {output.shape}")
        print("Model forward pass works with conditioning")

        output_no_cond = model(sample_BSI=sample, timestep_B=timestep)
        assert (
            output_no_cond.shape == sample.shape
        ), "Shape mismatch with no conditioning"
        print("Model forward pass works without conditioning")

    except Exception as e:
        print(f"Test failed: {str(e)}")


if __name__ == "__main__":
    main()
