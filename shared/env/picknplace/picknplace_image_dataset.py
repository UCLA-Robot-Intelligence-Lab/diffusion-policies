import torch
import numpy as np
import copy
import cv2
import os
import zarr
import json
import hashlib
import shutil

from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits
from filelock import FileLock
from typing import Dict
from torch.utils.data import Dataset
from shared.utils.pytorch_util import dict_apply
from shared.models.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from shared.utils.replay_buffer import ReplayBuffer
from shared.utils.sampler import SequenceSampler, get_val_mask, downsample_mask
from shared.utils.normalize_util import get_image_range_normalizer
from shared.real_world.real_data_conversion import real_data_to_replay_buffer


class PickNPlaceImageDataset(Dataset):
    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        num_obs_steps: int = None,
        num_latency_steps: int = 0,
        use_cache: bool = False,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: int = None,
        delta_action: bool = True,
    ):
        super().__init__()
        assert os.path.isdir(
            dataset_path
        ), f"Dataset path {dataset_path} does not exist"
        replay_buffer = None
        if use_cache:
            # fingerprint shape_meta
            shape_meta_json = json.dumps(
                OmegaConf.to_container(shape_meta), sort_keys=True
            )
            shape_meta_hash = hashlib.md5(shape_meta_json.encode("utf-8")).hexdigest()
            cache_zarr_path = os.path.join(dataset_path, shape_meta_hash + ".zarr.zip")
            cache_lock_path = cache_zarr_path + ".lock"
            print("Acquiring lock on cache.")
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print("Cache does not exist. Creating!")
                        replay_buffer = _get_replay_buffer(
                            dataset_path=dataset_path,
                            shape_meta=shape_meta,
                            store=zarr.MemoryStore(),
                        )
                        print("Saving cache to disk.")
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(store=zip_store)
                    except Exception as e:
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print("Loading cached ReplayBuffer from Disk.")
                    with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore()
                        )
                    print("Loaded!")
        else:
            replay_buffer = _get_replay_buffer(
                dataset_path=dataset_path,
                shape_meta=shape_meta,
                store=zarr.MemoryStore(),
            )

        if delta_action:
            actions = replay_buffer["action"][:]
            episode_ends = replay_buffer.episode_ends[:]
            for i in range(len(episode_ends)):
                start = 0
                if i > 0:
                    start = episode_ends[i - 1]
                end = episode_ends[i]
                actions[start + 1 : end, :6] = np.diff(actions[start:end, :6], axis=0)
            replay_buffer["action"][:] = actions

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            if type == "rgb":
                rgb_keys.append(key)
            elif type == "low_dim":
                lowdim_keys.append(key)

        key_first_k = dict()
        if num_obs_steps is not None:
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = num_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, max_n=max_train_episodes, seed=seed
        )
        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon + num_latency_steps,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k,
        )

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.num_obs_steps = num_obs_steps
        self.val_mask = val_mask
        self.horizon = horizon
        self.num_latency_steps = num_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon + self.num_latency_steps,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask,
        )
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # Use all 7 dimensions of the action data (including grasp)
        actions = self.replay_buffer["action"][:]
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(actions)

        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer[key]
            )

        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()

        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        T_slice = slice(self.num_obs_steps)
        obs_dict = dict()
        for key in self.rgb_keys:
            obs_dict[key] = (
                np.moveaxis(data[key][T_slice], -1, 1).astype(np.float32) / 255.0
            )
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        action = data["action"].astype(np.float32)
        if self.num_latency_steps > 0:
            action = action[self.num_latency_steps :]

        # Use all action dimensions including grasp
        torch_data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(action),
        }

        return torch_data


def zarr_resize_index_last_dim(zarr_arr, idxs):
    actions = zarr_arr[:]
    actions = actions[..., idxs]
    zarr_arr.resize(zarr_arr.shape[:-1] + (len(idxs),))
    zarr_arr[:] = actions
    return zarr_arr


def _get_replay_buffer(dataset_path, shape_meta, store):
    rgb_keys = list()
    lowdim_keys = list()
    out_resolutions = dict()
    lowdim_shapes = dict()
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        type = attr.get("type", "low_dim")
        shape = tuple(attr.get("shape"))
        if type == "rgb":
            rgb_keys.append(key)
            c, h, w = shape
            out_resolutions[key] = (w, h)
            # print(f"RGB key {key} has resolution {out_resolutions[key]}")
        elif type == "low_dim":
            lowdim_keys.append(key)
            lowdim_shapes[key] = tuple(shape)
            if "pose" in key:
                assert shape[0] == 6, f"Pose key {key} must have 6 elements"
        else:
            raise ValueError(f"Unknown shape type: {type}")

    action_shape = tuple(shape_meta["action"]["shape"])
    # Make sure we have 7 actions (including grasp)
    assert action_shape[0] == 7, f"Action shape must be 7, got {action_shape[0]}"

    cv2.setNumThreads(1)
    with threadpool_limits(1):
        replay_buffer = real_data_to_replay_buffer(
            dataset_path=dataset_path,
            out_store=store,
            out_resolutions=out_resolutions,
            lowdim_keys=lowdim_keys + ["action"],
            image_keys=rgb_keys,
        )

    print(f"Initial action shape in replay buffer: {replay_buffer['action'].shape}")
    zarr_arr = replay_buffer["action"]
    print(f"Raw action shape: {zarr_arr.shape}")

    return replay_buffer
