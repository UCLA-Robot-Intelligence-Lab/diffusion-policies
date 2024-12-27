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

"""
Dimension key:

B: batch size
T: time steps (horizon)
D: action feature dimensions
O: observation feature dimensions
C: conditioning feature dimensions
"""


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
        """
        Performs conditional sampling using the diffusion model.

        Args:
            condition_data: The initial trajectory to condition on.
            condition_mask: A mask specifying which parts are conditioned.
            local_cond: Optional local conditioning tensor.
            global_cond: Optional global conditioning tensor.
            generator: Random generator for reproducibility.

        Returns:
            torch.Tensor: The generated trajectory.
        """
        trajectory = torch.randn(
            condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        # Set reverse diffusion timesteps
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
        return trajectory

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Encodes observations, samples trajectories, and predicts actions.

        Args:
            obs_dict: Input observations dictionary.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing predicted actions.
        """
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype

        # Prepare conditioning inputs
        if self.obs_as_global_cond:
            nobs_features = self.obs_encoder(dict_apply(nobs, lambda x: x[:, :To, ...]))
            global_cond = nobs_features.reshape(B, -1)
            cond_data = torch.zeros((B, T, self.action_dim), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            nobs_features = self.obs_encoder(dict_apply(nobs, lambda x: x[:, :To, ...]))
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(
                (B, T, self.action_dim + self.obs_feature_dim),
                device=device,
                dtype=dtype,
            )
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, self.action_dim :] = nobs_features
            cond_mask[:, :To, self.action_dim :] = True

        nsample = self.conditional_sample(cond_data, cond_mask, global_cond=global_cond)

        naction_pred = nsample[..., : self.action_dim]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        start, end = To - 1, To - 1 + self.n_action_steps
        action = action_pred[:, start:end]

        return {"action": action, "action_pred": action_pred}

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Computes the diffusion loss for training.

        Args:
            batch: Batch containing observations and actions.

        Returns:
            torch.Tensor: The computed loss.
        """
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        B, T = nactions.shape[:2]

        if self.obs_as_global_cond:
            nobs_features = self.obs_encoder(
                dict_apply(nobs, lambda x: x[:, : self.n_obs_steps, ...])
            )
            global_cond = nobs_features.reshape(B, -1)
        else:
            nobs_features = self.obs_encoder(dict_apply(nobs, lambda x: x))
            nobs_features = nobs_features.reshape(B, T, -1)

        condition_mask = self.mask_generator(nactions.shape)
        noise = torch.randn_like(nactions)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (B,),
            device=nactions.device,
        )
        noisy_trajectory = self.noise_scheduler.add_noise(nactions, noise, timesteps)

        noisy_trajectory[condition_mask] = nactions[condition_mask]
        pred = self.model(noisy_trajectory, timesteps, global_cond=global_cond)

        target = (
            noise
            if self.noise_scheduler.config.prediction_type == "epsilon"
            else nactions
        )
        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss.masked_fill(~condition_mask, 0).mean()

        return loss

    def set_normalizer(self, normalizer: LinearNormalizer):
        """
        Sets the normalizer state.

        Args:
            normalizer: A pre-initialized LinearNormalizer.
        """
        self.normalizer.load_state_dict(normalizer.state_dict())

    def output_shape(self) -> Tuple[int, ...]:
        """
        Returns the output shape of the policy based on the model and encoder.

        Returns:
            Tuple[int, ...]: Output feature dimensions.
        """
        example_obs = {
            key: torch.zeros(
                1, *meta["shape"], device=next(self.model.parameters()).device
            )
            for key, meta in self.normalizer.get_input_stats().items()
        }
        with torch.no_grad():
            encoded_obs = self.obs_encoder(example_obs)
            result_shape = self.model.output_shape(encoded_obs)
        return result_shape


# ====== TEST FUNCTIONS ======
def main():
    test_conditional_sample()
    test_compute_loss()
    test_predict_action()
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
    print("test_conditional_sample passed!")


def test_predict_action():
    """Test the predict_action function."""
    B, To, C, H, W = 4, 12, 3, 224, 224  # Image-based observations
    D = 16  # Action dimensions

    obs_dict = {
        "obs": torch.randn(B, To, C, H, W),  # Image-based observations
    }

    shape_meta = {
        "action": {"shape": (D,), "type": "low_dim"},  # Action metadata
        "obs": {"shape": (C, H, W), "type": "rgb"},  # Image metadata
    }

    policy = DiffusionUnetImagePolicy(
        shape_meta=shape_meta,
        noise_scheduler=DDPMScheduler(num_train_timesteps=1000),
        obs_encoder=ObsEncoder(shape_meta, vision_backbone=None),
        horizon=To,
        n_action_steps=5,
        n_obs_steps=2,
    )

    # Fit the normalizer with example data
    example_data = {
        "obs": torch.randn(B, To, C, H, W),
        "action": torch.randn(B, 5, D),  # Match action time steps
    }
    policy.normalizer.fit(example_data)

    output = policy.predict_action(obs_dict)
    assert (
        "action" in output and "action_pred" in output
    ), "Missing keys in predict_action output."
    print("test_predict_action passed!")


def test_compute_loss():
    """Test the compute_loss function."""
    B, T, C, H, W = 4, 32, 3, 224, 224  # Image-based observations
    D = 16  # Action dimensions

    batch = {
        "obs": torch.randn(B, T, C, H, W),  # Image-based observations
        "action": torch.randn(B, T, D),  # Action data
    }

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

    loss = policy.compute_loss(batch)
    assert isinstance(loss, torch.Tensor), "Loss is not a torch.Tensor."
    print("test_compute_loss passed!")


if __name__ == "__main__":
    main()
