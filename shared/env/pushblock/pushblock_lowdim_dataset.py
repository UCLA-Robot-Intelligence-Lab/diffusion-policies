import torch
import numpy as np
import copy
import os
from typing import Dict, Union
from torch.utils.data import Dataset
from shared.utils.pytorch_util import dict_apply
from shared.models.common.normalizer import LinearNormalizer, NestedDictNormalizer
from shared.utils.replay_buffer import ReplayBuffer
from shared.utils.sampler import SequenceSampler, get_val_mask, downsample_mask
import torch.nn as nn

class PushBlockLowdimDataset(Dataset):
    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        num_obs_steps=None,
        num_latency_steps=0,
        delta_action=False,
    ):
        """
        Dataset for real robot pushblock lowdim data.
        
        Args:
            zarr_path: Path to zarr data
            horizon: Sequence length
            pad_before: Padding before sequence
            pad_after: Padding after sequence
            seed: Random seed
            val_ratio: Validation ratio
            max_train_episodes: Maximum training episodes
            num_obs_steps: Number of observation steps (when not None, only use first num_obs_steps)
            num_latency_steps: Number of latency steps
            delta_action: Whether to use delta actions
        """
        super().__init__()
        
        # Load low-dimensional data
        low_dim_keys = ["robot_eef_pose", "action"]
        
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=low_dim_keys
        )
        
        # Convert to delta actions if needed
        if delta_action:
            # Replace action as relative to previous frame
            actions = self.replay_buffer['action'][:]
            # Print action shape for debugging
            print(actions.shape)
            # Support actions of any dimension
            actions_diff = np.zeros_like(actions)
            episode_ends = self.replay_buffer.episode_ends[:]
            for i in range(len(episode_ends)):
                start = 0
                if i > 0:
                    start = episode_ends[i-1]
                end = episode_ends[i]
                # Delta action is the difference between previous desired position and the current
                # It should be scheduled at the previous timestep for the current timestep
                # to ensure consistency with positional mode
                actions_diff[start+1:end] = np.diff(actions[start:end], axis=0)
            self.replay_buffer['action'][:] = actions_diff
        
        # Setup train/validation split
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, max_n=max_train_episodes, seed=seed
        )
        
        # Setup key_first_k for num_obs_steps
        key_first_k = dict()
        if num_obs_steps is not None:
            # Only take first k obs from robot state
            key_first_k["robot_eef_pose"] = num_obs_steps

        # Create sampler
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon+num_latency_steps,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k
        )
        
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.num_obs_steps = num_obs_steps
        self.num_latency_steps = num_latency_steps

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon+self.num_latency_steps,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask
        )
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        # Create nested dict normalizer for handling obs dictionary
        normalizer = NestedDictNormalizer()
        
        # Create data structure with proper nesting for obs
        data = {
            "action": self.replay_buffer["action"],
            "obs": {
                "robot_eef_pose": self.replay_buffer["robot_eef_pose"]
            }
        }
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get sequence data from sampler
        data = self.sampler.sample_sequence(idx)
        
        # When self.num_obs_steps is None, this slice does nothing (takes all)
        T_slice = slice(self.num_obs_steps)
        
        # Process observations
        obs_dict = dict()
        
        # Process low-dim observations (robot state)
        obs_dict["robot_eef_pose"] = data["robot_eef_pose"][T_slice].astype(np.float32)
        del data["robot_eef_pose"]
        
        # Process actions
        action = data["action"].astype(np.float32)
        
        # Handle latency by dropping first num_latency_steps action
        if self.num_latency_steps > 0:
            action = action[self.num_latency_steps:]
            
        # Convert to torch tensors
        torch_data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(action)
        }
        
        return torch_data

def test():
    import os

    zarr_path = os.path.expanduser("data/pushblock_real/replay_buffer.zarr")
    dataset = PushBlockLowdimDataset(zarr_path, horizon=16, delta_action=True)
    
    # Test getting an item
    item = dataset[0]
    print(f"Item shapes: {item['obs']['robot_eef_pose'].shape}, {item['action'].shape}")
    
    # Test normalizer
    normalizer = dataset.get_normalizer()
    # Print normalizer structure
    print(f"Normalizer parameters: action and obs.robot_eef_pose")
    
    # Just print normalizer params
    print(f"Done creating normalizer")
    
    # Test action values with delta actions
    actions = dataset.replay_buffer["action"][:]
    print(f"Action mean: {np.mean(actions, axis=0)}, std: {np.std(actions, axis=0)}")

if __name__ == "__main__":
    test() 