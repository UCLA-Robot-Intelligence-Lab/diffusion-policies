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
    def __init__(
        self,
        shape_meta: Dict[str, Dict[str, Tuple[int, ...]]],
        noise_scheduler: DDPMScheduler,
        obs_encoder: ObsEncoder,
        horizon: int,
        num_action_steps: int,
        num_obs_steps: int,
        num_inference_steps: Optional[int] = None,
        obs_as_global_cond: bool = True,
        diffusion_step_embed_dim: int = 256,
        down_dims: Tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        **kwargs,  # keyword arguments passed to .step, used in conditional_sample
    ):
        super().__init__()

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1, "Action shape must be 1-dimensional."

        self.action_dim = action_shape[0]
        self.obs_feature_dim = obs_encoder.output_shape()[0]

        input_dim = self.action_dim + self.obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = self.action_dim
            global_cond_dim = self.obs_feature_dim * num_obs_steps

        self.model = ConditionalUnet1d(
            input_dim=input_dim,
            local_cond_dim=None,  # TODO: Test with this not none
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
            max_num_obs_steps=num_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.mask_generator.device = next(self.model.parameters()).device
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.num_action_steps = num_action_steps
        self.num_obs_steps = num_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        self.num_inference_steps = (
            num_inference_steps or noise_scheduler.config.num_train_timesteps
        )

    # ====== INFERENCE ======
    def conditional_sample(
        self,
        condition_data_BTD: torch.Tensor,
        condition_mask_BTD: torch.Tensor,
        local_cond_BTC: Optional[torch.Tensor] = None,
        global_cond_BC: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> torch.Tensor:
        # TODO: Update with dimension keys
        trajectory = torch.randn(
            condition_data_BTD.shape,
            dtype=condition_data_BTD.dtype,
            device=condition_data_BTD.device,
            generator=generator,
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            # 1. Apply conditioning, only model p(A_t|O_t)
            trajectory[condition_mask_BTD] = condition_data_BTD[condition_mask_BTD]

            # 2. Predict the model output
            # TODO: Update with dimension keys
            model_output = self.model(
                trajectory, t, local_cond=local_cond_BTC, global_cond_BC=global_cond_BC
            )

            # 3. Compute the previous image: x_t -> x_{t-1}
            trajectory = self.noise_scheduler.step(
                model_output, t, trajectory, generator=generator, **kwargs
            ).prev_sample

        trajectory[condition_mask_BTD] = condition_data_BTD[condition_mask_BTD]

        return trajectory

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        num_obs = self.normalizer.normalize(obs_dict)
        B, To = list(num_obs.values())[0].shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype

        flat_num_obs = dict_apply(num_obs, lambda x: x.reshape(-1, *x.shape[2:]))
        num_obs_features = self.obs_encoder(flat_num_obs).reshape(B, To, -1)

        if self.obs_as_global_cond:
            global_cond_BC = num_obs_features.reshape(B, -1)
            cond_data = torch.zeros((B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            cond_data = torch.zeros((B, T, Da + Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = num_obs_features
            cond_mask[:, :To, Da:] = True

        nsample = self.conditional_sample(
            cond_data, cond_mask, global_cond_BC=global_cond_BC
        )
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer.unnormalize({"action": naction_pred})["action"]

        start, end = To - 1, To - 1 + self.num_action_steps
        action = action_pred[:, start:end]

        return {"action": action, "action_pred": action_pred}

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        normalized_batch = self.normalizer.normalize(batch)
        num_obs = {
            key: normalized_batch[key] for key in self.obs_encoder.rgb_keys
        }  # Extract image data
        nactions = normalized_batch["action"]  # Extract normalized actions

        B, T = nactions.shape[:2]

        # Prepare `num_obs` as a dictionary for obs_encoder
        flat_num_obs = {
            key: value.reshape(-1, *value.shape[2:]) for key, value in num_obs.items()
        }
        num_obs_features = self.obs_encoder(flat_num_obs)
        num_obs_features = num_obs_features.reshape(B, T, -1)

        global_cond_BC = (
            num_obs_features.reshape(B, -1) if self.obs_as_global_cond else None
        )
        condition_mask_BTD = self.mask_generator.forward(nactions.shape).to(
            nactions.device
        )

        noise = torch.randn_like(nactions)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (B,),
            device=nactions.device,
        )
        noisy_trajectory = self.noise_scheduler.add_noise(nactions, noise, timesteps)

        noisy_trajectory[condition_mask_BTD] = nactions[condition_mask_BTD]
        pred = self.model(noisy_trajectory, timesteps, global_cond=global_cond_BC)

        target = (
            noise
            if self.noise_scheduler.config.prediction_type == "epsilon"
            else nactions
        )
        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss.masked_fill(~condition_mask_BTD, 0).mean()

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
    test_predict_action()
    test_compute_loss()
    print("All tests passed!")


def test_conditional_sample():
    B, T, C, H, W = 4, 32, 3, 224, 224
    D = 16  # Action dimensions

    condition_data_BTD = torch.randn(B, T, D)
    condition_mask_BTD = torch.zeros(B, T, D, dtype=torch.bool)
    condition_mask_BTD[:, :5, :] = True

    shape_meta = {
        "action": {"shape": (D,), "type": "low_dim"},  # Action metadata
        "obs": {"shape": (C, H, W), "type": "rgb"},  # Image metadata
    }

    policy = DiffusionUnetImagePolicy(
        shape_meta=shape_meta,
        noise_scheduler=DDPMScheduler(num_train_timesteps=1000),
        obs_encoder=ObsEncoder(shape_meta, vision_backbone=None),
        horizon=T,
        num_action_steps=5,
        num_obs_steps=2,
    )

    output = policy.conditional_sample(condition_data_BTD, condition_mask_BTD)
    assert (
        output.shape == condition_data_BTD.shape
    ), "Shape mismatch in conditional_sample output."

    print("test_conditional_sample passed!")


def test_predict_action():
    B, To, C, H, W = 4, 12, 3, 224, 224
    D = 16
    Ta = 4

    obs_dict = {
        "obs": torch.randn(B, To, C, H, W),  # Image-based observations
    }

    shape_meta = {
        "action": {"shape": (D,), "type": "low_dim"},
        "obs": {"shape": (C, H, W), "type": "rgb"},
    }

    policy = DiffusionUnetImagePolicy(
        shape_meta=shape_meta,
        noise_scheduler=DDPMScheduler(num_train_timesteps=1000),
        obs_encoder=ObsEncoder(shape_meta, vision_backbone=None),
        horizon=To + Ta,  # Horizon includes observation and action time steps
        num_action_steps=Ta,
        num_obs_steps=To,
    )

    example_data = {
        "obs": torch.randn(B, To, C, H, W),
        "action": torch.randn(B, Ta, D),  # Match action time steps
    }
    policy.normalizer.fit(example_data)

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
    B, T, C, H, W = 4, 32, 3, 224, 224
    D = 16

    batch = {
        "rgb": torch.randn(B, T, C, H, W),
        "action": torch.randn(B, T, D),
    }

    # Flattened shape_meta
    shape_meta = {
        "rgb": {"shape": (C, H, W), "type": "rgb"},  # Flattened key for image metadata
        "action": {"shape": (D,), "type": "low_dim"},
    }

    policy = DiffusionUnetImagePolicy(
        shape_meta=shape_meta,
        noise_scheduler=DDPMScheduler(num_train_timesteps=1000),
        obs_encoder=ObsEncoder(shape_meta, vision_backbone=None),
        horizon=T,
        num_action_steps=5,
        num_obs_steps=2,
    )

    example_data = {
        "rgb": torch.randn(B, T, C, H, W),
        "action": torch.randn(B, T, D),
    }
    policy.normalizer.fit(example_data)

    loss = policy.compute_loss(batch)
    print("Loss: ", loss.item())
    assert isinstance(loss, torch.Tensor), "Loss is not a torch.Tensor."
    print("test_compute_loss passed!")


def test_integration():
    # This is not really an integration test and left out for now
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
        num_action_steps=5,
        num_obs_steps=2,
    )

    example_data = {
        "rgb": torch.randn(B, T, C, H, W),
        "action": torch.randn(B, T, D),
    }
    policy.normalizer.fit(example_data)

    obs_dict = {"rgb": torch.randn(B, T, C, H, W)}
    predict_output = policy.predict_action(obs_dict)
    loss = policy.compute_loss(batch)


if __name__ == "__main__":
    main()
