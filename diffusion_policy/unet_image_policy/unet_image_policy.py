import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Optional
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from shared.models.common.normalizer import LinearNormalizer
from shared.models.unet.conditional_unet1d import ConditionalUnet1D
from shared.models.common.mask_generator import LowdimMaskGenerator
from shared.vision.common.multi_image_obs_encoder import MultiImageObsEncoder
from shared.utils.pytorch_util import dict_apply

"""
Selected Dimension Keys:

B: batch size
T: prediction horizon
    To: observation horizon
    Ta: action horizon
F: feature dimension
    Fo: observation feature dimension
    Fa: action feature dimension
G: global conditioning dimension
L: local conditioning dimension
"""


class DiffusionUnetImagePolicy(nn.Module):
    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        obs_encoder: MultiImageObsEncoder,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        obs_as_global_cond_BG=True,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        # parameters passed to step
        **kwargs,
    ):
        super().__init__()

        # parse shapes
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]

        # create diffusion model
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond_BG:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond_BG else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond_BG = obs_as_global_cond_BG
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    def reset(self):
        pass  # Only used for stateful policies?

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    # ========= INFERENCE  ============
    def conditional_sample(
        self,
        condition_data_BTF: torch.Tensor,
        condition_mask_BTF: torch.Tensor,
        local_cond_BTL: Optional[torch.Tensor] = None,
        global_cond_BG: Optional[torch.Tensor] = None,
        generator: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        args:
            condition_data_BTF : [ B, T, F ] Conditioning data that we want the final sampled
                                             trajectory to represent. These are *known* values
                                             for time steps and features in the trajectory.
            condition_mask_BTF : [ B, T, F ] Mask that we apply on our condition data. Where true,
                                             we should overwrite our predicted trajectory since we
                                             know these values.
            local_cond_BTL : [ B, T, L ] Temporal context: conditioning specific to each timestep
            global_cond_BG : [ B, G ] Global context: in this case, encoded observations O_t
            generator : Pseudo random number generator kwarg
            **kwargs : Parameters passed into noise_scheduler.step()
        returns:
            trajectory_BTF : [ B, T, F ] This is our predicted trajectory that we generate by running
                                         our conditioned diffusion sampling.
        """
        model = self.model
        noise_scheduler = self.noise_scheduler
        num_inference_steps = self.num_inference_steps

        trajectory_BTF = torch.randn(
            size=condition_data_BTF.shape,
            dtype=condition_data_BTF.dtype,
            device=condition_data_BTF.device,
            generator=generator,
        )

        noise_scheduler.set_timesteps(num_inference_steps)

        for t in scheduler.timesteps:
            # 1. Apply conditioning
            trajectory_BTF[condition_mask_BTF] = condition_data_BTF[condition_mask_BTF]

            # 2. Predict model output
            model_output = model(
                trajectory_BTF,
                t,
                local_cond_BTL=local_cond_BTL,
                global_cond_BG=global_cond_BG,
            )

            # 3. compute previous observation: x_t -> x_t-1
            trajectory_BTF = noise_scheduler.step(
                model_output, t, trajectory_BTF, generator=generator, **kwargs
            ).prev_sample

        # Just in case we have any drift, apply conditioning one last time.
        trajectory_BTF[condition_mask_BTF] = condition_data_BTF[condition_mask_BTF]

        return trajectory_BTF

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond_BTL = None
        global_cond_BG = None
        if self.obs_as_global_cond_BG:
            # condition through global feature
            this_nobs = dict_apply(
                nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
            )
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond_BG = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(
                nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
            )
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond_BTL=local_cond_BTL,
            global_cond_BG=global_cond_BG,
            **self.kwargs,
        )

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {"action": action, "action_pred": action_pred}
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond_BTL = None
        global_cond_BG = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond_BG:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(
                nobs, lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
            )
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond_BG = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask_BTF = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=trajectory.device,
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask_BTF

        # apply conditioning
        noisy_trajectory[condition_mask_BTF] = cond_data[condition_mask_BTF]

        # Predict the noise residual
        pred = self.model(
            noisy_trajectory,
            timesteps,
            local_cond_BTL=local_cond_BTL,
            global_cond_BG=global_cond_BG,
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        return loss
