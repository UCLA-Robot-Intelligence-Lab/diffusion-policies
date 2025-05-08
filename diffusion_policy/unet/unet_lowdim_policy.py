import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from typing import Dict, Optional, Tuple
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from shared.models.common.normalizer import LinearNormalizer, NestedDictNormalizer
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
        n_action_steps: int,
        n_obs_steps: int,
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
            n_action_steps : Number of action steps to predict
            n_obs_steps : Number of observation steps to condition on
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
            max_num_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        
        self.normalizer = NestedDictNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
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

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        args:
            obs_dict : Dictionary with 'obs' field containing observations
                       Expected shape: { 'obs': { 'robot_eef_pose': [B, To, Fo] } }
        returns:
            result : Dictionary with action predictions
                     Contains 'action' and 'action_pred' keys
        """
        normalizer = self.normalizer
        device = self.device
        dtype = self.dtype
        
        assert 'obs' in obs_dict
        assert 'robot_eef_pose' in obs_dict['obs'], "Expected 'robot_eef_pose' in observations"
        
        # Normalize observations directly with the nested normalizer
        try:
            # Ensure robot_eef_pose has the correct shape [B, T, D]
            if obs_dict['obs']['robot_eef_pose'].ndim != 3:
                obs_shape = obs_dict['obs']['robot_eef_pose'].shape
                # Try to reshape to the expected format
                B = obs_shape[0]  # Batch dimension
                if obs_shape[-1] == 6:
                    # If the last dimension is already 6, just need to ensure 3D tensor
                    obs_dict['obs']['robot_eef_pose'] = obs_dict['obs']['robot_eef_pose'].reshape(B, -1, 6)
                else:
                    # Try to reshape and ensure 6 dimensions
                    total_elements = obs_dict['obs']['robot_eef_pose'].numel()
                    if total_elements % 6 == 0:
                        # Can reshape to have 6 feature dimensions
                        new_shape = (B, total_elements//(B*6), 6)
                        obs_dict['obs']['robot_eef_pose'] = obs_dict['obs']['robot_eef_pose'].reshape(new_shape)
                    else:
                        # Pad with zeros to reach 6 dimensions
                        current_feat_dim = obs_dict['obs']['robot_eef_pose'].shape[-1]
                        if current_feat_dim < 6:
                            pad_size = 6 - current_feat_dim
                            # Reshape to [B, T, D]
                            reshaped = obs_dict['obs']['robot_eef_pose'].reshape(B, -1, current_feat_dim)
                            # Pad to [B, T, 6]
                            padded = torch.cat([
                                reshaped, 
                                torch.zeros(*reshaped.shape[:-1], pad_size, device=device, dtype=dtype)
                            ], dim=-1)
                            obs_dict['obs']['robot_eef_pose'] = padded
            
            print(f"After reshaping: {obs_dict['obs']['robot_eef_pose'].shape}")
            nobs = normalizer.normalize(obs_dict)
        except Exception as e:
            print(f"Error in normalization: {e}")
            # Fallback: skip normalization
            nobs = {'obs': {'robot_eef_pose': obs_dict['obs']['robot_eef_pose']}}
        
        # Get observation shape
        B, _, Fo = nobs['obs']['robot_eef_pose'].shape
        To = self.n_obs_steps
        T = self.horizon
        Fa = self.action_dim
        
        # Prepare conditions for different conditioning types
        local_cond = None
        global_cond = None
        
        if self.obs_as_local_cond:
            # Local temporal conditioning
            local_cond = torch.zeros(size=(B, T, Fo), device=device, dtype=dtype)
            local_cond[:, :To] = nobs['obs']['robot_eef_pose'][:, :To]
            cond_data = torch.zeros(size=(B, T, Fa), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            # Global conditioning: reshape observations to a single vector
            global_cond = nobs['obs']['robot_eef_pose'][:, :To].reshape(B, -1)
            
            # Size depends on whether we predict full trajectory or just action steps
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Fa)
            else:
                shape = (B, T, Fa)
                
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # Inpainting approach
            shape = (B, T, Fa + Fo)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            
            # Set known observation features in condition data
            cond_data[:, :To, Fa:] = nobs['obs']['robot_eef_pose'][:, :To]
            cond_mask[:, :To, Fa:] = True
        
        # Run diffusion sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs
        )
        
        # Extract and unnormalize action predictions
        naction_pred = nsample[..., :Fa]
        action_pred = normalizer['action'].unnormalize(naction_pred)
        
        # Output the requested action steps
        if self.pred_action_steps_only:
            action = action_pred
        else:
            # By default, we take actions starting after observation horizon
            start = To
            end = start + self.n_action_steps
            action = action_pred[:, start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        
        # For inpainting approach, also return predicted observations
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[..., Fa:]
            obs_pred = normalizer['obs'].unnormalize({'robot_eef_pose': nobs_pred})
            action_obs_pred = obs_pred['robot_eef_pose'][:, start:end]
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
        
        args:
            batch: Dictionary with observations and actions
                  {'obs': {'robot_eef_pose': [B, T, Fo]}, 'action': [B, T, Fa]}
        returns:
            loss: Scalar loss tensor
        """
        # Normalize input data
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']
        
        # Debug: print shapes
        print(f"Observation shape: {obs['robot_eef_pose'].shape}")
        print(f"Action shape: {action.shape}")
        
        # Prepare conditions based on conditioning method
        local_cond = None
        global_cond = None
        trajectory = action  # Default target is actions only
        
        if self.obs_as_local_cond:
            # Use observations as local conditioning
            local_cond = obs['robot_eef_pose']
            # Zero out observations after first n_obs_steps
            local_cond[:, self.n_obs_steps:] = 0
        elif self.obs_as_global_cond:
            # Use observations as global conditioning
            global_cond = obs['robot_eef_pose'][:, :self.n_obs_steps].reshape(obs['robot_eef_pose'].shape[0], -1)
            print(f"Global cond shape: {global_cond.shape}")
            
            # Use only relevant action steps if specified
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To
                end = start + self.n_action_steps
                trajectory = action[:, start:end]
        else:
            # Inpainting approach - concatenate actions and observations
            trajectory = torch.cat([action, obs['robot_eef_pose']], dim=-1)
        
        # Generate conditioning mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)
        
        # Add noise to the trajectory (forward diffusion)
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        
        # Sample random timesteps for each batch element
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=trajectory.device
        ).long()
        
        # Forward diffusion: add noise to clean trajectory
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps
        )
        
        # Calculate loss mask (loss is computed where mask is False)
        loss_mask = ~condition_mask
        
        # Apply conditioning by overwriting noisy values where condition is True
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict denoised trajectory
        pred = self.model(
            noisy_trajectory,
            timesteps,
            cond_BTL=local_cond,
            cond_BG=global_cond
        )
        
        # Determine target based on scheduler configuration
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")
        
        # Compute MSE loss masked by loss_mask
        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = einops.reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        
        return loss 