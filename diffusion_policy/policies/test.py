import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from typing import Dict, Tuple, Optional

from shared.encoders.obs_encoder import ObsEncoder
from shared.models.unet.conditional_unet_1d import ConditionalUnet1d
from shared.models.common.mask_generator import LowdimMaskGenerator
from shared.models.common.normalizer import LinearNormalizer
from shared.utils.pytorch_util import dict_apply


class DiffusionUnetImagePolicy(nn.Module):
    """
    Implements a diffusion-based policy using Conditional UNet and a DDPM Scheduler.
    """

    def __init__(
        self,
        shape_meta: Dict[str, Dict[str, Tuple[int, ...]]],
        noise_scheduler: DDPMScheduler,
        obs_encoder: ObsEncoder,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_inference_steps: Optional[int] = None,
        obs_as_global_cond: bool = True,
        diffusion_step_embed_dim: int = 256,
        down_dims: Tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        **kwargs,
    ):
        super().__init__()

        # Parse shapes
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1, "Action shape must be 1-dimensional."
        self.action_dim = action_shape[0]
        self.obs_feature_dim = obs_encoder.output_shape()[0]

        # Configure diffusion model dimensions
        input_dim = self.action_dim + self.obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = self.action_dim
            global_cond_dim = self.obs_feature_dim * n_obs_steps

        self.model = ConditionalUnet1d(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            num_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        self.obs_encoder = obs_encoder
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=self.action_dim,
            obs_dim=0 if obs_as_global_cond else self.obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.mask_generator.device = next(self.model.parameters()).device
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        self.num_inference_steps = (
            num_inference_steps or noise_scheduler.config.num_train_timesteps
        )

    def conditional_sample(
        self,
        condition_data: torch.Tensor,  # [B, T, D]
        condition_mask: torch.Tensor,  # [B, T, D]
        local_cond: Optional[torch.Tensor] = None,  # [B, T, C]
        global_cond: Optional[torch.Tensor] = None,  # [B, C]
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> torch.Tensor:
        print("--- Entering conditional_sample ---")
        trajectory = torch.randn(
            condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )
        print("Initial trajectory shape:", trajectory.shape)

        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = self.model(
                trajectory, t, local_cond=local_cond, global_cond=global_cond
            )
            trajectory = self.noise_scheduler.step(
                model_output, t, trajectory, generator=generator, **kwargs
            ).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        print("--- Exiting conditional_sample ---")
        return trajectory

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        print("--- Entering predict_action ---")
        nobs = self.normalizer.normalize(obs_dict)
        B, To = list(nobs.values())[0].shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype

        flat_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(flat_nobs).reshape(B, To, -1)

        if self.obs_as_global_cond:
            global_cond = nobs_features.reshape(B, -1)
            cond_data = torch.zeros((B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            cond_data = torch.zeros((B, T, Da + Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

        nsample = self.conditional_sample(cond_data, cond_mask, global_cond=global_cond)
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer.unnormalize({"action": naction_pred})["action"]

        start, end = To - 1, To - 1 + self.n_action_steps
        action = action_pred[:, start:end]
        print("--- Exiting predict_action ---")
        return {"action": action, "action_pred": action_pred}

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Computes the diffusion loss for training.

        Args:
        batch: Batch containing observations and actions.

        Returns:
        torch.Tensor: The computed loss.
        """
        # Normalize observations and actions
        normalized_batch = self.normalizer.normalize(batch)
        nobs = {
            key: normalized_batch[key] for key in self.obs_encoder.rgb_keys
        }  # Extract image data
        nactions = normalized_batch["action"]  # Extract normalized actions

        B, T = nactions.shape[:2]
        print(f"[DEBUG] Batch size: {B}, Horizon: {T}")

        # Prepare `nobs` as a dictionary for obs_encoder
        flat_nobs = {
            key: value.reshape(-1, *value.shape[2:]) for key, value in nobs.items()
        }
        nobs_features = self.obs_encoder(flat_nobs)  # [B*T, Do]
        nobs_features = nobs_features.reshape(B, T, -1)  # [B, T, Do]
        print(f"[DEBUG] nobs_features shape: {nobs_features.shape}")

        global_cond = nobs_features.reshape(B, -1) if self.obs_as_global_cond else None
        condition_mask = self.mask_generator.forward(nactions.shape).to(nactions.device)
        print(f"[DEBUG] Condition mask shape: {condition_mask.shape}")

        noise = torch.randn_like(nactions)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (B,),
            device=nactions.device,
        )
        noisy_trajectory = self.noise_scheduler.add_noise(nactions, noise, timesteps)
        print(f"[DEBUG] Noisy trajectory shape: {noisy_trajectory.shape}")

        noisy_trajectory[condition_mask] = nactions[condition_mask]
        pred = self.model(noisy_trajectory, timesteps, global_cond=global_cond)

        target = (
            noise
            if self.noise_scheduler.config.prediction_type == "epsilon"
            else nactions
        )
        loss = F.mse_loss(pred, target, reduction="none")
        unmasked_loss = loss.clone()
        print(f"[DEBUG] Unmasked loss mean: {unmasked_loss.mean().item()}")
        print("CONDITION MASK: ", condition_mask)
        loss = loss.masked_fill(~condition_mask, 0).mean()
        print(f"[DEBUG] Final masked loss: {loss.item()}")
        return loss


# ====== TEST FUNCTIONS ======
def main():
    test_conditional_sample()
    test_predict_action()
    test_compute_loss()
    print("All tests passed!")


def test_conditional_sample():
    """Test the conditional_sample function."""
    B, T, C, H, W = 4, 32, 3, 224, 224  # Image-based observations
    D = 16  # Action dimensions

    condition_data = torch.randn(B, T, D)
    condition_mask = torch.zeros(B, T, D, dtype=torch.bool)
    condition_mask[:, :5, :] = True

    shape_meta = {
        "action": {"shape": (D,), "type": "low_dim"},  # Action metadata
        "obs": {"shape": (C, H, W), "type": "rgb"},  # Image metadata
    }

    policy = DiffusionUnetImagePolicy(
        shape_meta=shape_meta,
        noise_scheduler=DDPMScheduler(num_train_timesteps=1000),
        obs_encoder=ObsEncoder(shape_meta, vision_backbone=None),
        horizon=T,
        n_action_steps=5,
        n_obs_steps=2,
    )

    output = policy.conditional_sample(condition_data, condition_mask)
    assert (
        output.shape == condition_data.shape
    ), "Shape mismatch in conditional_sample output."
    print("OUTPUT: ", output)
    print("test_conditional_sample passed!")


def test_predict_action():
    """Test the predict_action function."""
    B, To, C, H, W = 4, 12, 3, 224, 224  # Image-based observations
    D = 16  # Action dimensions
    Ta = 4  # Action time steps

    obs_dict = {
        "obs": torch.randn(B, To, C, H, W),  # Image-based observations
    }

    shape_meta = {
        "action": {"shape": (D,), "type": "low_dim"},  # Action metadata
        "obs": {"shape": (C, H, W), "type": "rgb"},  # Image metadata
    }

    # Initialize policy
    policy = DiffusionUnetImagePolicy(
        shape_meta=shape_meta,
        noise_scheduler=DDPMScheduler(num_train_timesteps=1000),
        obs_encoder=ObsEncoder(shape_meta, vision_backbone=None),
        horizon=To + Ta,  # Horizon includes observation and action time steps
        n_action_steps=Ta,
        n_obs_steps=To,
    )

    # Fit the normalizer with example data
    example_data = {
        "obs": torch.randn(B, To, C, H, W),
        "action": torch.randn(B, Ta, D),  # Match action time steps
    }
    policy.normalizer.fit(example_data)

    # Run predict_action
    output = policy.predict_action(obs_dict)
    assert (
        "action" in output and "action_pred" in output
    ), "Missing keys in predict_action output."
    assert output["action"].shape == (B, Ta, D), "Incorrect shape for 'action'."
    assert output["action_pred"].shape == (
        B,
        policy.horizon,
        D,
    ), "Incorrect shape for 'action_pred'."
    print("OUTPUT: ", output)
    print("test_predict_action passed!")


def test_compute_loss():
    """Test the compute_loss function."""
    B, T, C, H, W = 4, 32, 3, 224, 224  # Image-based observations
    D = 16  # Action dimensions

    # Example input batch
    batch = {
        "rgb": torch.randn(B, T, C, H, W),  # Image-based observations
        "action": torch.randn(B, T, D),  # Action data
    }

    # Flattened shape_meta
    shape_meta = {
        "rgb": {"shape": (C, H, W), "type": "rgb"},  # Flattened key for image metadata
        "action": {"shape": (D,), "type": "low_dim"},  # Action metadata
    }

    # Initialize policy
    policy = DiffusionUnetImagePolicy(
        shape_meta=shape_meta,
        noise_scheduler=DDPMScheduler(num_train_timesteps=1000),
        obs_encoder=ObsEncoder(shape_meta, vision_backbone=None),
        horizon=T,
        n_action_steps=5,
        n_obs_steps=2,
    )

    # Fit the normalizer with example data
    example_data = {
        "rgb": torch.randn(B, T, C, H, W),
        "action": torch.randn(B, T, D),
    }
    policy.normalizer.fit(example_data)

    # Compute loss
    loss = policy.compute_loss(batch)
    print("Loss: ", loss.item())
    assert isinstance(loss, torch.Tensor), "Loss is not a torch.Tensor."
    print("test_compute_loss passed!")


def test_integration():
    """Integration test to validate all core methods together."""
    print("--- Running integration test ---")
    B, T, C, H, W = 4, 32, 3, 224, 224
    D = 16

    batch = {
        "rgb": torch.randn(B, T, C, H, W),
        "action": torch.randn(B, T, D),
    }

    shape_meta = {
        "rgb": {"shape": (C, H, W), "type": "rgb"},
        "action": {"shape": (D,), "type": "low_dim"},
    }

    policy = DiffusionUnetImagePolicy(
        shape_meta=shape_meta,
        noise_scheduler=DDPMScheduler(num_train_timesteps=1000),
        obs_encoder=ObsEncoder(shape_meta, vision_backbone=None),
        horizon=T,
        n_action_steps=5,
        n_obs_steps=2,
    )

    example_data = {
        "rgb": torch.randn(B, T, C, H, W),
        "action": torch.randn(B, T, D),
    }
    policy.normalizer.fit(example_data)

    obs_dict = {"rgb": torch.randn(B, T, C, H, W)}
    predict_output = policy.predict_action(obs_dict)
    loss = policy.compute_loss(batch)

    print("Integration test predict_action output:", predict_output)
    print("Integration test loss:", loss.item())
    print("--- Integration test passed ---")


if __name__ == "__main__":
    main()
    test_integration()
