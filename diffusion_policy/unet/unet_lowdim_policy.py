import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from typing import Dict, Optional, Tuple
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from shared.models.common.normalizer import LinearNormalizer
from shared.models.unet.conditional_unet1d import ConditionalUnet1D
from shared.models.common.mask_generator import LowdimMaskGenerator

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

class DiffusionUnetLowdimPolicy(nn.Module):
    def __init__(
        self,
        model: ConditionalUnet1D,
        noise_scheduler: DDPMScheduler,
        horizon: int,
        obs_dim: int,
        action_dim: int,
        num_action_steps: int,
        num_obs_steps: int,
        num_inference_steps: int = None,
        obs_as_local_cond: bool = False,
        obs_as_global_cond: bool = True,
        pred_action_steps_only: bool = False,
        **kwargs
    ):
        """
        args:
            model : ConditionalUnet1D model that predicts denoised trajectory
            noise_scheduler : Scheduler for diffusion process
            horizon : Length of planning/prediction horizon
            obs_dim : Dimension of observations
            action_dim : Dimension of actions
            num_action_steps : Number of action steps to predict
            num_obs_steps : Number of observation steps to condition on
            num_inference_steps : Number of denoising steps during inference
            obs_as_local_cond : Whether to use observations as local conditioning
            obs_as_global_cond : Whether to use observations as global conditioning
            pred_action_steps_only : Whether to only predict action steps
            **kwargs : Additional parameters passed to scheduler
        """
        super().__init__()
        
        assert not (obs_as_local_cond and obs_as_global_cond), "Cannot use both local and global conditioning"
        
        self.model = model
        self.noise_scheduler = noise_scheduler
        
        # LowdimMaskGenerator creates masks for the conditioning process
        self.mask_generator = LowdimMaskGenerator(
            action_dim_Fa=action_dim,
            obs_feat_dim_Fo=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_num_obs_steps=num_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_action_steps = num_action_steps
        self.num_obs_steps = num_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.kwargs = kwargs
        
        # Set default inference steps if not specified
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    def reset(self):
        # This method is required for env runners
        # but is only used for stateful policies
        pass

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    # ========= INFERENCE =========
    def conditional_sample(
        self,
        condition_data: torch.Tensor,
        condition_mask: torch.Tensor,
        local_cond: Optional[torch.Tensor] = None,
        global_cond: Optional[torch.Tensor] = None,
        generator=None,
        **kwargs
    ) -> torch.Tensor:
        """
        args:
            condition_data : Conditioning data that we want the
                             final sample to match at condition_mask locations
            condition_mask : Boolean mask indicating where conditions apply
            local_cond : Optional local conditioning tensor
            global_cond : Optional global conditioning tensor
            generator : Optional random generator for reproducibility
            **kwargs : Additional parameters passed to scheduler
        returns:
            trajectory : Sampled trajectory after diffusion process
        """
        model = self.model
        scheduler = self.noise_scheduler
        
        # Sample initial noise
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator
        )
        
        # Set timesteps for denoising
        scheduler.set_timesteps(self.num_inference_steps)
        
        # Iterative denoising process
        for t in scheduler.timesteps:
            # 1. Apply condition mask - keep known values fixed
            trajectory[condition_mask] = condition_data[condition_mask]
            
            # 2. Predict denoised state with model
            model_output = model(
                sample_BTF=trajectory,
                timesteps_B=t,
                cond_BTL=local_cond,
                cond_BG=global_cond
            )
            
            # 3. Compute previous step: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output,
                t,
                trajectory,
                generator=generator,
                **kwargs
            ).prev_sample
        
        # Apply condition one last time to ensure perfect conditioning
        trajectory[condition_mask] = condition_data[condition_mask]
        
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'obs' in obs_dict
        assert 'robot_eef_pose' in obs_dict['obs']

        # Extract and normalize the robot_eef_pose directly
        nobs = self.normalizer['robot_eef_pose'].normalize(obs_dict['obs']['robot_eef_pose'])
        B, _, Do = nobs.shape
        To = self.num_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            # condition through local feature
            local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
            local_cond[:,:To] = nobs[:,:To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            # condition through global feature
            global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.num_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            end = start + self.num_action_steps
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[...,Da:]
            obs_pred = self.normalizer['robot_eef_pose'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        return result

    # ========= TRAINING =========
    def set_normalizer(self, normalizer: LinearNormalizer):
        """Set the normalizer for data normalization/denormalization"""
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        """
        Compute diffusion loss for training
        """
        # normalize input
        assert 'obs' in batch
        assert 'action' in batch
        assert 'robot_eef_pose' in batch['obs']

        # Normalize obs and action separately
        nobs = self.normalizer['robot_eef_pose'].normalize(batch['obs']['robot_eef_pose'])
        naction = self.normalizer['action'].normalize(batch['action'])

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = naction
        if self.obs_as_local_cond:
            # zero out observations after n_obs_steps
            local_cond = nobs
            local_cond[:,self.num_obs_steps:,:] = 0
        elif self.obs_as_global_cond:
            # Use observations as global conditioning
            global_cond = nobs[:,:self.num_obs_steps,:].reshape(nobs.shape[0], -1)
            if self.pred_action_steps_only:
                To = self.num_obs_steps
                start = To
                end = start + self.num_action_steps
                trajectory = naction[:,start:end]
        else:
            # Inpainting approach - concatenate actions and observations
            trajectory = torch.cat([naction, nobs], dim=-1)

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            cond_BTL=local_cond, cond_BG=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = einops.reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss 