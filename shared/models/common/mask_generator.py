from typing import Sequence, Optional
import torch
from torch import nn


# This code is taken from the official repository; all credit goes to the original authors
class LowdimMaskGenerator:
    def __init__(
        self,
        action_dim,
        obs_dim,
        max_num_obs_steps=2,
        fix_obs_steps=True,
        action_visible=False,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.max_num_obs_steps = max_num_obs_steps
        self.fix_obs_steps = fix_obs_steps
        self.action_visible = action_visible

    @torch.no_grad()
    def forward(self, shape, seed=None):
        device = self.device
        B, T, D = shape
        assert D == (self.action_dim + self.obs_dim)

        rng = torch.Generator(device=device)
        if seed is not None:
            rng.manual_seed(seed)

        dim_mask = torch.zeros(size=shape, dtype=torch.bool, device=device)
        is_action_dim = dim_mask.clone()
        is_action_dim[..., : self.action_dim] = True
        if D != self.action_dim + self.obs_dim:
            raise ValueError(
                f"[ERROR] Dimension mismatch: {D} != {self.action_dim + self.obs_dim}"
            )

        # Handle case where obs_dim is zero (e.g., global conditioning)
        if self.obs_dim > 0:
            is_obs_dim = ~is_action_dim
        else:
            is_obs_dim = torch.ones(size=shape, dtype=torch.bool, device=device)
            is_obs_dim[..., : self.action_dim] = True

        # Observation steps
        if self.fix_obs_steps:
            obs_steps = torch.full(
                (B,), fill_value=self.max_num_obs_steps, device=device
            )
        else:
            obs_steps = torch.randint(
                low=1,
                high=self.max_num_obs_steps + 1,
                size=(B,),
                generator=rng,
                device=device,
            )

        time_steps = torch.arange(0, T, device=device).unsqueeze(0).expand(B, T)
        obs_mask = time_steps < obs_steps.unsqueeze(1)

        obs_mask = obs_mask.unsqueeze(-1).expand(B, T, D)

        obs_mask = obs_mask & is_obs_dim
        if self.action_visible:
            action_steps = torch.clamp(obs_steps - 1, min=0)
            action_mask = time_steps < action_steps.unsqueeze(1)
            action_mask = action_mask.unsqueeze(-1).expand(B, T, D)
            action_mask = action_mask & is_action_dim
        else:
            action_mask = torch.zeros_like(obs_mask)

        mask = obs_mask | action_mask
        return mask


# ====== TEST FUNCTION ======
def test_lowdim_mask_generator():
    B, T, D = 4, 32, 16
    action_dim = 4
    obs_dim = D - action_dim
    max_num_obs_steps = 3
    fix_obs_steps = True  # Fix observation steps
    action_visible = True  # Action visibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mask_generator = LowdimMaskGenerator(
        action_dim=action_dim,
        obs_dim=obs_dim,
        max_num_obs_steps=max_num_obs_steps,
        fix_obs_steps=fix_obs_steps,
        action_visible=action_visible,
    )
    mask_generator.device = device

    mask = mask_generator.forward(shape=(B, T, D))

    print("Generated mask shape:", mask.shape)
    assert mask.shape == (B, T, D), "Mask shape does not match the expected dimensions."
    print("Mask sum per batch:", mask.sum(dim=(1, 2)))

    print("test_lowdim_mask_generator passed!")


if __name__ == "__main__":
    test_lowdim_mask_generator()
