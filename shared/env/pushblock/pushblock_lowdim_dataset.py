import torch
import numpy as np
import copy
import cv2
import os
import json
import hashlib
import zarr
import shutil

from omegaconf import OmegaConf
from typing import Dict
from filelock import FileLock
from torch.utils.data import Dataset
from shared.utils.pytorch_util import dict_apply
from shared.models.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from shared.utils.replay_buffer import ReplayBuffer
from shared.utils.sampler import SequenceSampler, get_val_mask, downsample_mask
from shared.real_world.real_data_conversion import real_data_to_replay_buffer
from threadpoolctl import threadpool_limits


class PushBlockLowdimDataset(Dataset):
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
        delta_action: bool = False,
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
            # Should adapt so that we only take delta actions on position.
            actions = replay_buffer["action"][:]
            episode_ends = replay_buffer.episode_ends[:]
            for i in range(len(episode_ends)):
                start = 0
                if i > 0:
                    start = episode_ends[i - 1]
                end = episode_ends[i]
                actions[start + 1 : end] = np.diff(actions[start:end], axis=0)
            replay_buffer["action"][:] = actions

        lowdim_keys = list()
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            if type == "low_dim":
                lowdim_keys.append(key)
            else:
                raise ValueError(f"Unknown shape type: {type}")

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
            keys=lowdim_keys + ["action"],
        )

        self.train_mask = train_mask
        self.val_mask = val_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.num_obs_steps = num_obs_steps
        self.num_latency_steps = num_latency_steps
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.lowdim_keys = lowdim_keys
        self.shape_meta = shape_meta

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

    def get_normalizer(self, **kwargs):
        normalizer = LinearNormalizer()
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer["action"]
        )
        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer[key]
            )
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
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        action = data["action"].astype(np.float32)
        if self.num_latency_steps > 0:
            action = action[self.num_latency_steps :]

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
    # Parse shape meta
    lowdim_keys = list()
    lowdim_shapes = dict()
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        type = attr.get("type", "low_dim")
        shape = tuple(attr.get("shape"))
        if type == "low_dim":
            lowdim_keys.append(key)
            lowdim_shapes[key] = tuple(shape)
            if "pose" in key:
                assert shape[0] == 6, f"Pose key {key} must have 6 elements"
        else:
            raise ValueError(f"Unknown shape type: {type}")

    action_shape = tuple(shape_meta["action"]["shape"])
    # Should we be checking against 6 or 7 actions, maybe
    # Make 6 since we don't need grasp.
    cv2.setNumThreads(1)
    with threadpool_limits(1):
        replay_buffer = real_data_to_replay_buffer(
            dataset_path,
            out_store=store,
            lowdim_keys=lowdim_keys + ["action"],
            image_keys=[],
            lowdim_compressor=None,
        )

    if action_shape == (7,):
        zarr_arr = replay_buffer["action"]
        zarr_resize_index_last_dim(zarr_arr, idxs=[0, 1, 2, 3, 4, 5])

    return replay_buffer


def test():
    import hydra
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)

    this_file = os.path.abspath(__file__)
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(this_file)))
    )
    config_dir = os.path.join(repo_root, "diffusion_policy", "config")
    with hydra.initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = hydra.compose(config_name="train_pushblock_real_lowdim")
        dataset = hydra.utils.instantiate(cfg.tasks.dataset)
        print(f"Dataset length: {len(dataset)}")
        print(
            f"Dataset shape: "
            f"{dataset[0]['obs']['robot_eef_pose'].shape}, "
            f"{dataset[0]['action'].shape}"
        )

    normalizer = dataset.get_normalizer()
    all_actions = dataset.replay_buffer["action"][:]
    normalized_actions = normalizer["action"].normalize(all_actions)

    if isinstance(normalized_actions, torch.Tensor):
        normalized_actions = normalized_actions.detach().cpu().numpy()

    print(
        f"Normalized action mean: {np.mean(normalized_actions, axis=0)}, "
        f"std: {np.std(normalized_actions, axis=0)}"
    )


if __name__ == "__main__":
    test()
