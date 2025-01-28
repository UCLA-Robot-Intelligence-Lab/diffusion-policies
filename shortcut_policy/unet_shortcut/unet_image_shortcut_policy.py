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
from shortcut_policy.shortcut_model import ShortcutModel

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


class UnetImageShortcutPolicy(nn.Module):
    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        obs_encoder: ObsEncoder,
        horizon: int,
        num_action_steps: int,
        num_obs_steps: int,
        num_inference_steps: int = 128,
        global_obs_cond: bool = True,
        embed_dim_D: int = 256,
        down_dims: Tuple[int] = (256, 512, 1024),
        kernel_size: int = 5,
        num_groups: int = 8,
        film_modulation_scale: bool = True,
        **kwargs,
    ):
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
        self.num_inference_steps = num_inference_steps
        self.global_obs_cond = global_obs_cond

        # The ShortcutModel wrapper
        self.shortcut_model = ShortcutModel(
            self.forward_model, num_inference_steps=num_inference_steps, device=self.device
        )
        self.kwargs = kwargs

    def forward_model(self, z, t, distance=None, cond_BTL=None, cond_BG=None):
        t.to(self.device)
        out = self.model(
            sample_BTF=z,
            timesteps_B=t,
            cond_BTL=cond_BTL,
            cond_BG=cond_BG,
        )
        return out

    def reset(self):
        pass

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    # ========= INFERENCE =========
    def predict_action(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Default inference: for example, 2-step approach.
        You may keep or revert. We'll do 2-step here by default.
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
        B = obs["image"].shape[0]

        # Normalize obs
        normalized_obs = normalizer.normalize(obs)

        cond_BG = None
        if global_obs_cond:
            flat_normalized_obs = dict_apply(
                normalized_obs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
            )
            normalized_obs_feats = obs_encoder(flat_normalized_obs)
            cond_BG = normalized_obs_feats.reshape(B, -1)
        else:
            print("Error, local obs conditioning not implemented for now!")

        # z0 is random noise in [ B, T, Fa ]
        z0 = torch.randn(size=(B, T, Fa), dtype=dtype, device=device)

        # 2-step approach
        traj = self.shortcut_model.sample_2step_shortcut(z0=z0, cond_BG=cond_BG)
        action_pred_BTFa = traj[-1]
        action_pred_BTFa = normalizer["action"].unnormalize(action_pred_BTFa)

        # Slice out the final portion for the actual actions
        obs_act_horizon = To + Ta - 1
        action_BTaFa = action_pred_BTFa[:, obs_act_horizon - Ta : obs_act_horizon]

        result = {"action": action_BTaFa, "action_pred": action_pred_BTFa}
        return result

    # --- CHANGED HERE: new method to do custom # of Euler steps
    def predict_action_shortcut(self, obs: Dict[str, torch.Tensor], num_inference_steps: int = None):
        """
        Inference for a user-specified # of Euler steps.
        This calls sample_ode_shortcut(..., num_inference_steps=N).
        """
        num_inference_steps = num_inference_steps or self.num_inference_steps
        print("Predict action shortcut with : ", num_inference_steps)
        normalizer = self.normalizer
        obs_encoder = self.obs_encoder
        global_obs_cond = self.global_obs_cond
        device = self.device
        dtype = self.dtype
        T = self.horizon
        Ta = self.num_action_steps
        To = self.num_obs_steps
        B = obs["image"].shape[0]

        normalized_obs = normalizer.normalize(obs)
        cond_BG = None
        if global_obs_cond:
            flat_normalized_obs = dict_apply(
                normalized_obs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
            )
            normalized_obs_feats = obs_encoder(flat_normalized_obs)
            cond_BG = normalized_obs_feats.reshape(B, -1)

        z0 = torch.randn(size=(B, T, self.action_dim_Fa), dtype=dtype, device=device)

        # Now do num_inference_steps Euler steps
        traj = self.shortcut_model.sample_ode_shortcut(
            z0=z0, num_inference_steps=num_inference_steps, cond_BG=cond_BG
        )
        action_pred_BTFa = traj[-1]
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
        Shortcut training approach:
          1) Flow-matching pass (get_train_tuple)
          2) Shortcut pass (get_shortcut_train_tuple)
          => sum of MSE losses
        """
        normalizer = self.normalizer
        obs_encoder = self.obs_encoder
        global_obs_cond = self.global_obs_cond
        To = self.num_obs_steps
        T = batch["action"].shape[1]
        B = batch["action"].shape[0]

        # 1) Normalize data
        normalized_obs = normalizer.normalize(batch["obs"])
        normalized_acts = normalizer["action"].normalize(batch["action"])

        cond_BG = None
        if global_obs_cond:
            flat_normalized_obs = dict_apply(
                normalized_obs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
            )
            normalized_obs_feats = obs_encoder(flat_normalized_obs)
            cond_BG = normalized_obs_feats.reshape(B, -1)

        # 2) Our ground truth is z1
        z1 = normalized_acts
        z0 = torch.randn_like(z1)

        # --- (A) Flow-matching pass
        z_t, t, target, distance = self.shortcut_model.get_train_tuple(z0=z0, z1=z1)
        pred = self.shortcut_model.model(z_t, t, distance=distance, cond_BG=cond_BG)
        loss_flow = F.mse_loss(pred, target)

        # --- (B) Shortcut pass
        z_t2, t2, target2, distance2 = self.shortcut_model.get_shortcut_train_tuple(
            z0=torch.randn_like(z1), z1=z1, cond_BG=cond_BG
        )
        pred2 = self.shortcut_model.model(z_t2, t2, distance=distance2, cond_BG=cond_BG)
        loss_shortcut = F.mse_loss(pred2, target2)

        loss = loss_flow + loss_shortcut
        return loss
