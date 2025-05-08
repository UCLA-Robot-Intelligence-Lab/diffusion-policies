import torch
import numpy as np
import copy
import cv2
import os
import zarr
from pathlib import Path
from threadpoolctl import threadpool_limits

from typing import Dict
from torch.utils.data import Dataset
from shared.utils.pytorch_util import dict_apply
from shared.models.common.normalizer import LinearNormalizer
from shared.utils.replay_buffer import ReplayBuffer
from shared.utils.sampler import SequenceSampler, get_val_mask, downsample_mask
from shared.utils.normalize_util import get_image_range_normalizer

class PushBlockImageDataset(Dataset):
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
        Dataset for real robot pushblock image data.
        
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
        
        # First load low-dimensional data
        dataset_path = os.path.dirname(zarr_path)
        
        # Try to read the available keys in the zarr file
        try:
            import zarr
            zarr_store = zarr.open(zarr_path, 'r')
            available_keys = list(zarr_store.keys())
            print(f"Available keys in zarr file: {available_keys}")
            
            # Check what's in the zarr file and decide what to load
            low_dim_keys = []
            
            # Check for robot state
            if "robot_eef_pose" in available_keys:
                low_dim_keys.append("robot_eef_pose")
            else:
                print("Warning: 'robot_eef_pose' not found in zarr file")
                
            # Check for action
            if "action" in available_keys:
                low_dim_keys.append("action")
            else:
                print("Warning: 'action' not found in zarr file")
                
            # Load camera keys if available
            camera_keys = []
            if "camera_0" in available_keys:
                camera_keys.append("camera_0")
            if "camera_1" in available_keys:
                camera_keys.append("camera_1")
                
            # Load all found keys
            all_keys = low_dim_keys + camera_keys
            
            if len(all_keys) == 0:
                raise ValueError(f"No usable keys found in zarr file. Available: {available_keys}")
                
            print(f"Loading keys: {all_keys}")
            
        except Exception as e:
            print(f"Error inspecting zarr file: {e}")
            # Fallback to default keys
            low_dim_keys = ["robot_eef_pose", "action"]
            camera_keys = []
            all_keys = low_dim_keys
            
        # Load the replay buffer
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=all_keys
        )
        
        # Get shapes for debugging
        print("Data shapes:")
        for key in all_keys:
            try:
                print(f"  {key}: {self.replay_buffer[key].shape}")
            except Exception as e:
                print(f"  {key}: Error getting shape - {e}")
        
        # Convert to delta actions if needed
        if delta_action:
            try:
                # Replace action as relative to previous frame
                actions = self.replay_buffer['action'][:]
                
                # No need to assert specific dimensions, accept any action dimension
                action_shape = actions.shape[1]
                print(f"Action shape: {action_shape}")
                
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
                print("Successfully converted to delta actions")
            except Exception as e:
                print(f"Error converting to delta actions: {e}")
                raise
        
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
            # Only take first k obs from state
            for key in low_dim_keys:
                key_first_k[key] = num_obs_steps
                
            # Also limit camera frames if available
            for key in camera_keys:
                key_first_k[key] = num_obs_steps

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
        self.low_dim_keys = low_dim_keys
        self.camera_keys = camera_keys

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
        # Create normalizer for low-dimensional data
        normalizer = LinearNormalizer()
        
        # Fit normalizer on available low-dim data
        data = {}
        for key in self.low_dim_keys:
            data[key] = self.replay_buffer[key]
        
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        
        # Add image normalizers if needed
        for key in self.camera_keys:
            normalizer[key] = get_image_range_normalizer()
        
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        if "action" in self.low_dim_keys:
            return torch.from_numpy(self.replay_buffer["action"])
        else:
            raise ValueError("No action data available in dataset")

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get sequence data from sampler
        data = self.sampler.sample_sequence(idx)
        
        # When self.num_obs_steps is None, this slice does nothing (takes all)
        T_slice = slice(self.num_obs_steps)
        
        # Process observations
        obs_dict = dict()
        
        # Process camera images if available
        for key in self.camera_keys:
            # Convert from NHWC to NCHW format and normalize to [0,1]
            obs_dict[key] = np.moveaxis(data[key][T_slice], -1, 1).astype(np.float32) / 255.0
            # Save RAM
            del data[key]
            
        # Process low-dim observations (states, etc.)
        for key in self.low_dim_keys:
            if key != "action":  # Skip action, will be processed separately
                obs_dict[key] = data[key][T_slice].astype(np.float32)
                del data[key]
        
        # Process actions
        if "action" in self.low_dim_keys:
            action = data["action"].astype(np.float32)
            
            # Handle latency by dropping first num_latency_steps action
            if self.num_latency_steps > 0:
                action = action[self.num_latency_steps:]
                
            # Add to output dict
            torch_data = {
                "obs": dict_apply(obs_dict, torch.from_numpy),
                "action": torch.from_numpy(action)
            }
        else:
            # If no action data, just return observations
            torch_data = {
                "obs": dict_apply(obs_dict, torch.from_numpy)
            }
        
        return torch_data

def test():
    import os

    zarr_path = os.path.expanduser("data/pushblock_real/replay_buffer.zarr")
    dataset = PushBlockImageDataset(zarr_path, horizon=16, delta_action=True)
    
    # Print info about the dataset
    print(f"Dataset size: {len(dataset)}")
    print(f"Low-dim keys: {dataset.low_dim_keys}")
    print(f"Camera keys: {dataset.camera_keys}")
    
    # Test getting an item
    item = dataset[0]
    print("Item keys:", item.keys())
    print("Observation keys:", item['obs'].keys())
    
    # Print shapes of all items
    print("Item shapes:")
    for k, v in item.items():
        if k == 'obs':
            for obs_k, obs_v in v.items():
                print(f"  obs.{obs_k}: {obs_v.shape}")
        else:
            print(f"  {k}: {v.shape}")
    
    # Test normalizer
    normalizer = dataset.get_normalizer()
    print(f"Normalizer keys: {normalizer.keys()}")
    
    # Test action values with delta actions
    if "action" in dataset.low_dim_keys:
        actions = dataset.replay_buffer["action"][:]
        print(f"Action mean: {np.mean(actions, axis=0)}, std: {np.std(actions, axis=0)}")
        
    return dataset 