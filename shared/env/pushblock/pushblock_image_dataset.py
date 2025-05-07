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


def process_videos_to_replay_buffer(dataset_path, replay_buffer):
    """
    Process videos from dataset_path/videos into replay_buffer
    
    Args:
        dataset_path: Path to dataset directory
        replay_buffer: ReplayBuffer to add images to
    """
    print(f"Processing videos from {dataset_path}")
    dataset_path = Path(dataset_path)
    video_dir = dataset_path / "videos"
    
    if not video_dir.exists():
        raise ValueError(f"Video directory {video_dir} does not exist")
    
    # Get episode end indices
    episode_ends = replay_buffer.episode_ends[:]
    n_episodes = len(episode_ends)
    
    # Get dimensions of first frame to determine array shape
    sample_video_path = None
    for episode_idx in range(n_episodes):
        for camera_id in [0, 1]:
            test_path = video_dir / f"ep_{episode_idx}_camera_{camera_id}.mp4"
            if not test_path.exists():
                test_path = video_dir / f"ep_{episode_idx:05d}_camera_{camera_id}.mp4"
            
            if test_path.exists():
                sample_video_path = test_path
                break
        
        if sample_video_path is not None:
            break
    
    if sample_video_path is None:
        raise ValueError("Could not find any video files")
    
    cap = cv2.VideoCapture(str(sample_video_path))
    ret, frame = cap.read()
    if not ret:
        raise ValueError(f"Could not read first frame from {sample_video_path}")
    
    height, width, channels = frame.shape
    cap.release()
    
    # Create camera arrays if they don't exist
    n_frames = replay_buffer.n_steps
    
    for camera_id in [0, 1]:
        camera_key = f'camera_{camera_id}'
        if camera_key not in replay_buffer:
            # Create new array for camera images at original resolution
            replay_buffer.add_buffer(
                name=camera_key,
                shape=(n_frames, height, width, channels),
                dtype=np.uint8
            )
    
    # Get video paths
    cv2.setNumThreads(1)  # Prevent OpenCV from using too many threads
    
    # Loop through episodes
    with threadpool_limits(limits=1):
        start_idx = 0
        for episode_idx in range(n_episodes):
            end_idx = episode_ends[episode_idx]
            episode_len = end_idx - start_idx
            
            # Expected video file paths for this episode
            video_paths = {}
            for camera_id in [0, 1]:
                video_path = video_dir / f"ep_{episode_idx}_camera_{camera_id}.mp4"
                if not video_path.exists():
                    video_path = video_dir / f"ep_{episode_idx:05d}_camera_{camera_id}.mp4"
                
                if not video_path.exists():
                    raise ValueError(f"Video file {video_path} not found")
                
                video_paths[camera_id] = str(video_path)
            
            # Load videos
            for camera_id, video_path in video_paths.items():
                camera_key = f'camera_{camera_id}'
                
                # Open video
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise ValueError(f"Could not open video {video_path}")
                
                # Check video length
                video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if video_frame_count < episode_len:
                    raise ValueError(f"Video {video_path} has {video_frame_count} frames, but episode has {episode_len} steps")
                
                # Read frames
                for i in range(episode_len):
                    ret, frame = cap.read()
                    if not ret:
                        raise ValueError(f"Failed to read frame {i} from video {video_path}")
                    
                    # Store original frame without resizing
                    # Let the obs_encoder handle resizing and cropping
                    replay_buffer[camera_key][start_idx + i] = frame
                
                # Release video
                cap.release()
            
            # Update start index
            start_idx = end_idx
            print(f"Processed episode {episode_idx+1}/{n_episodes}")
    
    print(f"Finished processing videos. Processed {n_frames} frames.")
    return replay_buffer


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
        process_video=True,
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
            process_video: Whether to process videos
        """
        super().__init__()
        
        # First load low-dimensional data
        dataset_path = os.path.dirname(zarr_path)
        low_dim_keys = ["robot_eef_pose", "action"]
        
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=low_dim_keys
        )
        
        # Process videos if needed
        if process_video:
            self.replay_buffer = process_videos_to_replay_buffer(
                dataset_path=dataset_path, 
                replay_buffer=self.replay_buffer
            )
        
        # Convert to delta actions if needed
        if delta_action:
            # Replace action as relative to previous frame
            actions = self.replay_buffer['action'][:]
            # Support positions only at this time
            assert actions.shape[1] <= 3
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
            # Only take first k obs from images and state
            for key in ["camera_0", "camera_1", "robot_eef_pose"]:
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
        # Create normalizer for actions and robot_eef_pose
        normalizer = LinearNormalizer()
        
        # Fit normalizer on action and robot_eef_pose
        data = {
            "action": self.replay_buffer["action"],
            "robot_eef_pose": self.replay_buffer["robot_eef_pose"],
        }
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        
        # Image normalizers (range [0,1])
        normalizer["camera_0"] = get_image_range_normalizer()
        normalizer["camera_1"] = get_image_range_normalizer()
        
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
        
        # Process RGB observations (camera images)
        for key in ["camera_0", "camera_1"]:
            # Convert from NHWC to NCHW format and normalize to [0,1]
            obs_dict[key] = np.moveaxis(data[key][T_slice], -1, 1).astype(np.float32) / 255.0
            # Save RAM
            del data[key]
            
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

    zarr_path = os.path.expanduser("data/pushblock/pushblock_real_data.zarr")
    dataset = PushBlockImageDataset(zarr_path, horizon=16, delta_action=True)
    
    # Test getting an item
    item = dataset[0]
    print(f"Item shapes: {item['obs']['camera_0'].shape}, {item['action'].shape}")
    
    # Test normalizer
    normalizer = dataset.get_normalizer()
    print(f"Normalizer keys: {normalizer.keys()}")
    
    # Test action values with delta actions
    actions = dataset.replay_buffer["action"][:]
    print(f"Action mean: {np.mean(actions, axis=0)}, std: {np.std(actions, axis=0)}") 