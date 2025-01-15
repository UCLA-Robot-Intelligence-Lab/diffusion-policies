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
        num_steps: int = 100,
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
        self.shortcut_model = ShortcutModel(
            self.forward_model, num_steps=num_steps, device=self.device
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
        # This method is required for our env runner,
        # but is only used for stateful policies.
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
        B = obs["image"].shape[0]

        normalized_obs = normalizer.normalize(obs)

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
        else:
            # Needs to be changed...
            print("Error, this method branch is not implemented!")
            pass

        z0 = torch.randn(size=(B, T, Fa), dtype=dtype, device=device)

        traj = self.shortcut_model.sample_ode_shortcut(
            z0=z0, num_steps=self.shortcut_model.num_steps, cond_BG=cond_BG
        )
        print("shortcut model num steps: ", self.shortcut_model.num_steps)

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
        Shortcut training approach. We'll do two passes:
          1) A rectified-flow pass with get_train_tuple
          2) A shortcut pass with get_shortcut_train_tuple
        Both compute an MSE loss to the predicted direction.
        """
        normalizer = self.normalizer
        obs_encoder = self.obs_encoder
        global_obs_cond = self.global_obs_cond
        To = self.num_obs_steps
        T = batch["action"].shape[1]
        B = batch["action"].shape[0]

        # 1) Normalize
        normalized_obs = normalizer.normalize(batch["obs"])
        normalized_acts = normalizer["action"].normalize(batch["action"])

        cond_BG = None
        if global_obs_cond:
            flat_normalized_obs = dict_apply(
                normalized_obs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
            )
            normalized_obs_feats = obs_encoder(flat_normalized_obs)
            cond_BG = normalized_obs_feats.reshape(B, -1)

        # 2) We'll treat z1=the ground truth trajectory (actions).
        z1 = normalized_acts

        # 3) For z0, we can choose random or zero. Let's do random for training:
        z0 = torch.randn_like(z1)

        # --- (A) Regular rectified flow pass ---
        z_t, t, target, distance = self.shortcut_model.get_train_tuple(z0=z0, z1=z1)
        pred = self.shortcut_model.model(z_t, t, distance=distance, cond_BG=cond_BG)
        loss_flow = F.mse_loss(pred, target)

        # --- (B) Shortcut pass ---
        z_t2, t2, target2, distance2 = self.shortcut_model.get_shortcut_train_tuple(
            z0=torch.randn_like(z1), z1=z1, cond_BG=cond_BG
        )
        pred2 = self.shortcut_model.model(z_t2, t2, distance=distance2, cond_BG=cond_BG)
        loss_shortcut = F.mse_loss(pred2, target2)

        # 4) Combine or log both
        loss = loss_flow + loss_shortcut
        return loss
