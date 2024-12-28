import torch
from typing import Optional, Sequence


def create_slice_mask(
    shape: tuple, dim_slices: Sequence[slice], union=True, device=None
) -> torch.Tensor:
    """
    Creates a mask from slices, supporting union or intersection of slices.

    Args:
        shape (tuple): Shape of the mask.
        dim_slices (Sequence[slice]): Slices for each dimension.
        union (bool): Use union of slices if True, intersection if False.
        device (torch.device): Device for the mask tensor.

    Returns:
        torch.Tensor: The resulting mask.
    """
    mask = (
        torch.ones(size=shape, dtype=torch.bool, device=device)
        if not union
        else torch.zeros(size=shape, dtype=torch.bool, device=device)
    )
    for i, s in enumerate(dim_slices):
        slice_mask = torch.zeros(size=shape, dtype=torch.bool, device=device)
        slice_mask[(slice(None),) * i + (s,)] = True
        if union:
            mask |= slice_mask
        else:
            mask &= slice_mask
    return mask


class LowdimMaskGenerator:
    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        max_n_obs_steps: int = 2,
        fix_obs_steps: bool = True,
        action_visible: bool = False,
    ):
        """
        Generates masks for low-dimensional trajectory tasks.

        Args:
            action_dim (int): Dimension of actions.
            obs_dim (int): Dimension of observations.
            max_n_obs_steps (int): Maximum number of observation steps to mask.
            fix_obs_steps (bool): Fix the number of observation steps if True.
            action_visible (bool): Include action visibility in the mask.
        """
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.max_n_obs_steps = max_n_obs_steps
        self.fix_obs_steps = fix_obs_steps
        self.action_visible = action_visible

    def __call__(self, shape: tuple, seed: Optional[int] = None) -> torch.Tensor:
        B, T, D = shape
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert (
            D == self.action_dim + self.obs_dim
        ), f"Dimension mismatch: {D} != {self.action_dim + self.obs_dim}"

        rng = torch.Generator(device=device)
        if seed is not None:
            rng.manual_seed(seed)

        # Create dimension masks
        is_action_dim = torch.zeros(size=shape, dtype=torch.bool, device=device)
        is_action_dim[..., : self.action_dim] = True
        is_obs_dim = ~is_action_dim

        print(
            f"[DEBUG] is_action_dim shape: {is_action_dim.shape}, is_obs_dim shape: {is_obs_dim.shape}"
        )

        # Generate observation steps
        obs_steps = (
            torch.full((B,), self.max_n_obs_steps, device=device)
            if self.fix_obs_steps
            else torch.randint(
                1, self.max_n_obs_steps + 1, size=(B,), generator=rng, device=device
            )
        )
        print(f"[DEBUG] Observation steps (per batch): {obs_steps.tolist()}")

        # Generate observation mask
        steps = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        obs_mask = (steps < obs_steps.unsqueeze(1)).unsqueeze(2).expand(
            B, T, D
        ) & is_obs_dim
        print(
            f"[DEBUG] obs_mask shape: {obs_mask.shape}, obs_mask sum: {obs_mask.sum().item()}"
        )

        # Generate action mask
        action_mask = torch.zeros_like(obs_mask)
        if self.action_visible:
            action_steps = torch.clamp(
                obs_steps - 1, min=0
            )  # Actions visible for obs_steps - 1
            action_mask = (steps < action_steps.unsqueeze(1)).unsqueeze(2).expand(
                B, T, D
            ) & is_action_dim
            print(f"[DEBUG] Action steps (per batch): {action_steps.tolist()}")

        print(
            f"[DEBUG] action_mask shape: {action_mask.shape}, action_mask sum: {action_mask.sum().item()}"
        )

        # Combine masks
        mask = obs_mask | action_mask if self.action_visible else obs_mask
        print(f"[DEBUG] Final mask shape: {mask.shape}, mask sum: {mask.sum().item()}")
        return mask


# ====== TEST FUNCTION ======
def main():
    shape = (4, 5, 6)
    slices = [slice(1, 3), slice(0, 4), slice(2, 5)]
    union_mask = create_slice_mask(shape, slices, union=True)
    assert union_mask.sum() > 0, "Union mask failed."

    intersection_mask = create_slice_mask(shape, slices, union=False)
    assert intersection_mask.sum() > 0, "Intersection mask failed."

    print("Slice mask tests passed!")

    generator = LowdimMaskGenerator(2, 20, max_n_obs_steps=3, action_visible=True)
    shape = (4, 10, 22)
    mask = generator(shape)
    assert mask.shape == shape, "Mask shape mismatch."

    # Test without action visibility
    generator_no_action = LowdimMaskGenerator(
        2, 20, max_n_obs_steps=3, action_visible=False
    )
    mask_no_action = generator_no_action(shape)
    assert mask_no_action.sum() <= mask.sum(), "Action visibility not handled properly."

    print("LowdimMaskGenerator tests passed!")

    # More tests here as we add more types of masks
    print("All tests passed!")


if __name__ == "__main__":
    main()
