import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Optional, Tuple
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from shared.models.common.normalizer import LinearNormalizer
from shared.models.unet.conditional_unet1d import ConditionalUnet1D
from shared.models.common.mask_generator import LowdimMaskGenerator
from shared.vision.common.multi_image_obs_encoder import ObsEncoder
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
        obs_encoder: ObsEncoder,
        horizon: int,
        num_action_steps: int,
        num_obs_steps: int,
        num_inference_steps: int = None,
        global_obs_cond: bool = True,
        embed_dim_D: int = 256,
        down_dims: Tuple[int] = (256, 512, 1024),
        kernel_size: int = 5,
        num_groups: int = 8,
        film_modulation_scale: bool = True,
        **kwargs,
    ):
        """
        args:
            shape_meta : Dictionary containing metadata for the input/output
                         shapes for our model, shapes specific to each task.
                         Contains 'obs' key with nested dictionary containing
                         'image' and 'agent_pos' keys with relevant shapes.
                         Contains 'action' key with action dimension shape.
            noise_scheduler : Scheduler that manages the diffusion noise
                              during training and inference.
            obs_encoder : Encodes observations (i.e., images) into 1d latent
                          space for our 1d unet.
            horizon : Length of *planning*/*prediction* horizon for the policy.
            num_action_steps : Action horizon; number of steps during which
                               actions are taken.
            num_obs_steps : Observation horizon; number of steps to consider
                            for observation history.
            num_inference_steps : Number of steps used for iterative denoising.
            global_obs_cond : Flag to indicate if observations should be used
                              as global conditioning.
            down_dims: Specifies the dimensions for down-sampling layers.
            embed_dim_D : Diffusion step embedding dimension.
            kernel_size : Kernel size for our network blocks.
            num_groups : Number of groups used in the block GroupNorms
            film_modulation_scale : Set to True if you want FiLM conditioning
                                    to also modulate the scale (gamma). By
                                    default we always modulate with a bias
                                    (beta). Read more in conv1d_components.py
        """
        super().__init__()

        action_dim_Fa = shape_meta["action"]["shape"][0]
        obs_feat_dim_Fo = obs_encoder.output_shape()[0]

        inp_dim_F = action_dim_Fa + obs_feat_dim_Fo
        cond_dim_G = None
        if global_obs_cond:
            inp_dim_F = action_dim_Fa  # F := Fa
            cond_dim_G = obs_feat_dim_Fo * num_obs_steps

        model = ConditionalUnet1D(
            inp_dim_F=inp_dim_F,
            cond_dim_L=None,
            cond_dim_G=cond_dim_G,
            embed_dim_D=embed_dim_D,
            down_dims=down_dims,
            kernel_size=kernel_size,
            num_groups=num_groups,
            film_modulation_scale=film_modulation_scale,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim_Fa=action_dim_Fa,
            obs_feat_dim_Fo=0 if global_obs_cond else obs_feat_dim_Fo,
            max_num_obs_steps=num_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feat_dim_Fo = obs_feat_dim_Fo
        self.action_dim_Fa = action_dim_Fa
        self.num_action_steps = num_action_steps
        self.num_obs_steps = num_obs_steps
        self.global_obs_cond = global_obs_cond
        self.kwargs = kwargs
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    def reset(self):
        # This method is required for our env runner,
        # but is only used for stateful policies.
        pass

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def observe(self, obs_dict):
        """
        Encodes observations using the obs_encoder
        
        Args:
            obs_dict: Dictionary of observations
                
        Returns:
            obs_feat: Observation features
            global_obs_feat: Global observation features 
        """
        B = next(iter(obs_dict.values())).shape[0]
        To = self.num_obs_steps
        
        # Prepare tensors for encoder
        flat_obs = dict_apply(
            obs_dict, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
        )
        
        # Encode observations
        obs_feat_flat = self.obs_encoder(flat_obs)
        
        # Reshape back to batch dimension
        obs_feat = obs_feat_flat.reshape(B, To, -1)
        
        # Global observation feature is flattened version
        global_obs_feat = obs_feat.reshape(B, -1)
        
        return obs_feat, global_obs_feat

    # ========= INFERENCE =========
    def conditional_sample(
        self,
        cond_data_BTF: torch.Tensor,
        cond_mask_BTF: torch.Tensor,
        cond_BTL: Optional[torch.Tensor] = None,
        cond_BG: Optional[torch.Tensor] = None,
        generator: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        args:
            cond_data_BTF : [ B, T, F ] Conditioning data that we want the
                                             final sampled trajectory to retain
                                             These are *known* values for the
                                             time steps and features.
            cond_mask_BTF : [ B, T, F ] Mask that we apply on our
                                             condition data. Where True, we
                                             should overwrite our predicted
                                             trajectory since they are known,
                                             and our prediction doesn't matter.
            cond_BTL : [ B, T, L ] Temporal context: (local) conditioning
                                         specific to each timestep.
            cond_BG : [ B, G ] Global context: (global) condition, which
                                      in this case are encoded observations O_t
            generator : Pseudo random number generator kwarg
            **kwargs : Parameters passed into noise_scheduler.step()
        returns:
            trajectory_BTF : [ B, T, F ] Predicted trajectory generated using
                                         the conditioned diffusion sampling
                                         process. This trajectory represents
                                         the denoised version of the input
                                         sample (noised trajectory) and keeps
                                         the known values (cond_data_BTF)
        """
        model = self.model
        noise_scheduler = self.noise_scheduler
        num_inference_steps = self.num_inference_steps

        # Sample a noisy trajectory.
        trajectory_BTF = torch.randn(
            size=cond_data_BTF.shape,
            dtype=cond_data_BTF.dtype,
            device=cond_data_BTF.device,
            generator=generator,
        )

        noise_scheduler.set_timesteps(num_inference_steps)

        # Denoise and predict previous observation
        for t in noise_scheduler.timesteps:
            # 1. Apply conditioning
            trajectory_BTF[cond_mask_BTF] = cond_data_BTF[cond_mask_BTF]

            # 2. Predict observation
            denoised_trajectory_BTF = model(
                sample_BTF=trajectory_BTF,
                timesteps_B=t,
                cond_BTL=cond_BTL,
                cond_BG=cond_BG,
            )

            # 3. Compute previous observation: x_t -> x_t-1
            trajectory_BTF = noise_scheduler.step(
                denoised_trajectory_BTF,
                t,
                trajectory_BTF,
                generator=generator,
                **kwargs,
            ).prev_sample

        # Just in case we have any drift, apply conditioning one last time.
        trajectory_BTF[cond_mask_BTF] = cond_data_BTF[cond_mask_BTF]

        return trajectory_BTF

    def predict_action(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        args:
            obs : Dictionary with two keys, 'image' and 'agent_pos'. 'image'
                  maps to a tensor of shape (B, F, C, H, W): where C is
                  channels here, which is the batched visual observations
                  (+ action features) in p(A_t|O_t). 'agent_pos' maps to a
                  tensor of shape (B, F, Fa): th batched representation of
                  the agent's pos/state.
        """
        normalizer = self.normalizer
        obs_encoder = self.obs_encoder
        global_obs_cond = self.global_obs_cond
        device = self.device
        dtype = self.dtype
        T = self.horizon
        Ta = self.num_action_steps
        To = self.num_obs_steps
        Fa = self.action_dim_Fa
        Fo = self.obs_feat_dim_Fo

        normalized_obs = normalizer.normalize(obs)
        value  = next(iter(obs.values()))
        B, To_cp = value.shape[:2]
        assert To_cp == To, f"To_cp: {To_cp}, To: {To}"

        cond_BTL = None
        cond_BG = None
        if global_obs_cond:
            # Prep tensors for encoder: [ B, To, C, H, W ] -> [ B*To, C, H, W ]
            flat_normalized_obs = dict_apply(
                normalized_obs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
            )
            # [ B*To, C, H, W ] -> [ B*To, Fo ] -> [ B, G ]; G := Fo*To
            normalized_obs_feats = obs_encoder(flat_normalized_obs)
            cond_BG = normalized_obs_feats.reshape(B, -1)

            # Observations are global, so we have the entire set of obs
            # Model treats entire action sequence as "unknown"
            cond_data_BTF = torch.zeros(size=(B, T, Fa), device=device, dtype=dtype)
            cond_mask_BTF = torch.zeros_like(cond_data_BTF, dtype=torch.bool)
        else:
            # "Condition through inpainting": segments of the action trajectory
            # known and fixed, and the model *inpaints* the missing parts
            flat_normalized_obs = dict_apply(
                normalized_obs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
            )
            # [ B*To, C, H, W ] -> [ B*To, Fo ] -> [ B, To, Fo ]
            normalized_obs_feats = obs_encoder(flat_normalized_obs)
            normalized_obs_feats = normalized_obs_feats.reshape(B, To, -1)

            cond_data_BTF = torch.zeros(
                size=(B, T, Fa + Fo), device=device, dtype=dtype
            )
            cond_mask_BTF = torch.zeros_like(cond_data_BTF, dtype=torch.bool)

            # Set known observation features in conditioning data; update mask
            cond_data_BTF[:, :To, Fa:] = normalized_obs_feats
            cond_mask_BTF[:, :To, Fa:] = True

        # F can be Fa or Fo + Fa; we are overloading F here.
        sample_BTF = self.conditional_sample(
            cond_data_BTF=cond_data_BTF,
            cond_mask_BTF=cond_mask_BTF,
            cond_BTL=cond_BTL,
            cond_BG=cond_BG,
            **self.kwargs,
        )

        action_pred_BTFa = sample_BTF[..., :Fa]
        action_pred_BTFa = normalizer["action"].unnormalize(action_pred_BTFa)

        obs_act_horizon = To + Ta - 1
        action_BTaFa = action_pred_BTFa[:, obs_act_horizon - Ta : obs_act_horizon]

        result = {"action": action_BTaFa, "action_pred": action_pred_BTFa}

        return result

    # ========= TRAINING =========
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        """
        args:
            batch: dictionary of tensors with structure of shape_meta:
                   'obs': {'image': (B, T, C, H, W),
                           'agent_pos': (B, T, F) }
                   'action': (B, T, F)
        """
        normalizer = self.normalizer
        obs_encoder = self.obs_encoder
        mask_generator = self.mask_generator
        noise_scheduler = self.noise_scheduler
        global_obs_cond = self.global_obs_cond
        model = self.model
        To = self.num_obs_steps
        T = batch["action"].shape[1]
        B = batch["action"].shape[0]

        normalized_obs = normalizer.normalize(batch["obs"])
        normalized_acts = normalizer["action"].normalize(batch["action"])

        cond_BTL = None
        cond_BG = None
        trajectory_BTF = normalized_acts  # We now consider it a trajectory
        cond_data_BTF = trajectory_BTF  # This is for else and .detach()
        if global_obs_cond:
            # Prep tensors for encoder: [ B, To, C, H, W ] -> [ B*To, C, H, W ]
            flat_normalized_obs = dict_apply(
                normalized_obs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
            )

            # [ B*To, C, H, W ] -> [ B*To, Fo ] -> [ B, G ]; G := Fo*To
            normalized_obs_feats = obs_encoder(flat_normalized_obs)
            cond_BG = normalized_obs_feats.reshape(B, -1)
        else:
            # Prep tensors for encoder: [ B, T, C, H, W ] -> [ B*T, C, H, W ]
            flat_normalized_obs = dict_apply(
                normalized_obs, lambda x: x.reshape(-1, *x.shape[2:])
            )
            # [ B*T, C, H, W ] -> [ B*T, Fo ] -> [ B, T, Fo ]
            normalized_obs_feats = obs_encoder(flat_normalized_obs)
            normalized_obs_feats = normalized_obs_feats.reshape(B, T, -1)

            cond_data_BTF = torch.cat([normalized_acts, normalized_obs_feats], dim=-1)
            trajectory_BTF = cond_data_BTF.detach()  # Detach to save loss

        cond_mask_BTF = mask_generator(trajectory_BTF.shape)
        loss_mask_BTF = ~cond_mask_BTF  # Find where loss *should* be computed.

        noise_BTF = torch.randn(trajectory_BTF.shape, device=trajectory_BTF.device)
        timesteps_B = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (B,),
            device=trajectory_BTF.device,
        ).long()

        noisy_trajectory_BTF = noise_scheduler.add_noise(
            trajectory_BTF, noise_BTF, timesteps_B
        )  # Forward diffusion.
        noisy_trajectory_BTF[cond_mask_BTF] = cond_data_BTF[cond_mask_BTF]

        denoised_trajectory_BTF = model(
            sample_BTF=noisy_trajectory_BTF,
            timesteps_B=timesteps_B,
            cond_BTL=cond_BTL,
            cond_BG=cond_BG,
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target_BTF = noise_BTF
        elif pred_type == "sample":
            target_BTF = trajectory_BTF
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = (
            F.mse_loss(denoised_trajectory_BTF, target_BTF, reduction="none")
            * loss_mask_BTF
        ).mean()

        return loss