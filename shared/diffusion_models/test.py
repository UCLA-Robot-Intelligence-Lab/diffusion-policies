import torch
import torch.nn as nn
import einops
from typing import Union, Optional
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

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(
                    inp_channels=inp_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    num_groups=num_groups,
                ),
                Conv1dBlock(
                    inp_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    num_groups=num_groups,
                ),
            ]
        )

        # FiLM modulation
        cond_channels = out_channels * 2 if cond_predict_scale else out_channels

        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(in_features=cond_dim, out_features=cond_channels),
            Rearrange("b t -> b t 1"),
        )

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
        out_BSO = self.blocks[0](x_BSI)
        embed_BO1 = self.cond_encoder(cond_BC)

        if self.cond_predict_scale:
            embed_B2O1 = embed_BO1.reshape(embed_BO1.shape[0], 2, self.out_channels, 1)
            scale_BO1 = embed_B2O1[:, 0, ...]
            bias_BO1 = embed_B2O1[:, 1, ...]
            out_BSO = scale_BO1 * out_BSO + bias_BO1
        else:
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

        # Diffusion step embedding
        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dim=dsed),
            nn.Linear(in_features=dsed, out_features=dsed * 4),
            nn.Mish(),
            nn.Linear(in_features=dsed * 4, out_features=dsed),
        )

        # Calculate combined conditioning dimension
        self.has_global_cond = global_cond_dim is not None
        if self.has_global_cond:
            self.global_cond_proj = nn.Sequential(
                nn.Linear(in_features=global_cond_dim, out_features=dsed),
                nn.Mish(),
            )
            cond_dim = dsed * 2  # Combined dimension after concatenation
        else:
            self.global_cond_proj = None
            cond_dim = dsed

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        # Local conditioning
        self.has_local_cond = local_cond_dim is not None
        local_cond_encoder = None
        if self.has_local_cond:
            _, out_dim = in_out[0]
            local_cond_encoder = nn.ModuleList(
                [
                    ConditionalResBlock1d(
                        inp_channels=local_cond_dim,
                        out_channels=out_dim,
                        cond_dim=cond_dim,
                        kernel_size=kernel_size,
                        num_groups=num_groups,
                        cond_predict_scale=cond_predict_scale,
                    ),
                    ConditionalResBlock1d(
                        inp_channels=local_cond_dim,
                        out_channels=out_dim,
                        cond_dim=cond_dim,
                        kernel_size=kernel_size,
                        num_groups=num_groups,
                        cond_predict_scale=cond_predict_scale,
                    ),
                ]
            )

        # Middle blocks
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

        # Downsampling path
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

        # Upsampling path
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

        # Final projection
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
        # Rearrange input
        sample_BIS = einops.rearrange(sample_BSI, "b s i -> b i s")

        # Process timesteps
        if not torch.is_tensor(timestep_B):
            timestep_B = torch.tensor([timestep_B], device=sample_BIS.device)
        elif len(timestep_B.shape) == 0:
            timestep_B = timestep_B[None]

        # Broadcast timesteps and get diffusion embedding
        timestep_B = timestep_B.expand(sample_BIS.shape[0])
        diff_emb = self.diffusion_step_encoder(timestep_B)

        # Process global conditioning
        if self.has_global_cond:
            if global_cond_BG is None:
                global_cond_BG = torch.zeros(
                    (sample_BIS.shape[0], self.global_cond_proj[0].in_features),
                    device=sample_BIS.device,
                )
            global_feat = self.global_cond_proj(global_cond_BG)
            global_feat_BC = torch.cat([diff_emb, global_feat], dim=-1)
        else:
            global_feat_BC = diff_emb

        # Process local conditioning
        h_local_BOS = []
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

        # Downsampling
        x_BOS = sample_BIS
        h_BOS = []
        for idx, (visual_encoder1, visual_encoder2, downsample) in enumerate(
            self.down_modules
        ):
            x_BOS = visual_encoder1(x_BOS, global_feat_BC)
            if idx == 0 and len(h_local_BOS) > 0:
                x_BOS = x_BOS + h_local_BOS[0]
            x_BOS = visual_encoder2(x_BOS, global_feat_BC)
            h_BOS.append(x_BOS)
            x_BOS = downsample(x_BOS)

        # Middle
        for mid_module in self.mid_modules:
            x_BOS = mid_module(x_BOS, global_feat_BC)

        # Upsampling
        for idx, (visual_encoder1, visual_encoder2, upsample) in enumerate(
            self.up_modules
        ):
            x_BOS = torch.cat((x_BOS, h_BOS.pop()), dim=1)
            x_BOS = visual_encoder1(x_BOS, global_feat_BC)
            if idx == (len(self.up_modules)) and len(h_local_BOS) > 0:
                x_BOS = x_BOS + h_local_BOS[1]
            x_BOS = visual_encoder2(x_BOS, global_feat_BC)
            x_BOS = upsample(x_BOS)

        # Final projection
        x_BIS = self.final_conv(x_BOS)
        output_BSI = einops.rearrange(x_BIS, "b i s -> b s i")

        return output_BSI


def test_shapes():
    """
    Test the shapes of tensors through the ConditionalUnet1d pipeline
    """
    # Test parameters
    batch_size = 4
    seq_length = 32
    input_dim = 16
    local_cond_dim = 8
    global_cond_dim = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize model
    model = ConditionalUnet1d(
        input_dim=input_dim,
        local_cond_dim=local_cond_dim,
        global_cond_dim=global_cond_dim,
    ).to(device)

    # Create dummy inputs
    sample = torch.randn(batch_size, seq_length, input_dim).to(device)
    timestep = torch.ones(batch_size).to(device)
    local_cond = torch.randn(batch_size, seq_length, local_cond_dim).to(device)
    global_cond = torch.randn(batch_size, global_cond_dim).to(device)

    # Test forward pass
    print("Testing ConditionalUnet1d shapes...")
    try:
        output = model(
            sample_BSI=sample,
            timestep_B=timestep,
            local_cond_BSL=local_cond,
            global_cond_BG=global_cond,
        )

        # Verify output shape matches input shape
        assert (
            output.shape == sample.shape
        ), f"Shape mismatch: expected {sample.shape}, got {output.shape}"

        print(f"✓ Input shape: {sample.shape}")
        print(f"✓ Output shape: {output.shape}")
        print("✓ All shapes match!")

        # Test without conditioning
        output_no_cond = model(sample_BSI=sample, timestep_B=timestep)
        assert (
            output_no_cond.shape == sample.shape
        ), "Shape mismatch with no conditioning"
        print("✓ Model works without conditioning")

    except Exception as e:
        print(f"✗ Test failed: {str(e)}")


if __name__ == "__main__":
    test_shapes()
