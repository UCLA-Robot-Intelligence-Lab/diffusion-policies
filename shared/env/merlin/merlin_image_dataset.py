import copy
import hashlib
import json
import os
import shutil

import cv2
import numpy as np
import torch
import zarr

from filelock import FileLock
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits
from torch.utils.data import Dataset
from typing import Dict, List, Tuple

from shared.models.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from shared.utils.normalize_util import get_image_range_normalizer
from shared.utils.pytorch_util import dict_apply
from shared.utils.replay_buffer import ReplayBuffer
from shared.utils.sampler import SequenceSampler, get_val_mask, downsample_mask


# # NOTE(raayan)
# # This code was generated with the help of codex. That's why there is so much validation everywhere.


def _resolve_merlin_root(dataset_path: str) -> str:
    """
    Accept either:
      - <dataset_path>/camera, <dataset_path>/syncs, ...
      - <dataset_path>/data/camera, <dataset_path>/data/syncs, ...
    """
    direct_candidates = [
        os.path.join(dataset_path, "camera"),
        os.path.join(dataset_path, "processed_encoder_t"),
        os.path.join(dataset_path, "processed_action_t"),
        os.path.join(dataset_path, "syncs"),
    ]
    if all(os.path.isdir(p) for p in direct_candidates):
        return dataset_path

    nested_root = os.path.join(dataset_path, "data")
    nested_candidates = [
        os.path.join(nested_root, "camera"),
        os.path.join(nested_root, "processed_encoder_t"),
        os.path.join(nested_root, "processed_action_t"),
        os.path.join(nested_root, "syncs"),
    ]
    if all(os.path.isdir(p) for p in nested_candidates):
        return nested_root

    return dataset_path


class MerlinImageDataset(Dataset):
    """
    Based off the real-world PushBlockImageDataset class
    + MERLIN/train/dataset.py

    The onstructor builds from a Hydra config
    (shape_meta, horizon, num_obs_steps, etc)
    We expect a certain dataset disk layout. See _resolve_merlin_root.

    We try to convert the dataset as collected into a replay-buffer style
    dataset, which aligns with the rest of the repository.
    See shared/utils/replay_buffer.py

    NOTE(raayan): test different combos of seq_len
    We use a sampler to get sequences of (horizon+num_latency_steps) and
    generate valid indices. There is some padding if needed.
    See shared/utils/sampler.py
    """

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
        first_cut: int = 0,
        encoder_window: int = 0,
    ):
        super().__init__()

        assert os.path.isdir(dataset_path), f"Dataset path does not exist: {dataset_path}"

        replay_buffer = None
        if use_cache:
            # # cache_key fingerprints shape + some other preprocessing (first_cut, encoder window from MERLIN)
            # # fingerprint shape_meta exists since obs keys can change the already loaded data
            # # assume cache path is <dataset_path>/merlin_<md5>.zarr.zip
            cache_key = {
                "shape_meta": OmegaConf.to_container(shape_meta),
                "first_cut": int(first_cut),
                "encoder_window": int(encoder_window),
            }
            cache_hash = hashlib.md5(
                json.dumps(cache_key, sort_keys=True).encode("utf-8")
            ).hexdigest()
            cache_zarr_path = os.path.join(dataset_path, f"merlin_{cache_hash}.zarr.zip")
            cache_lock_path = cache_zarr_path + ".lock"

            print("[MerlinImageDataset] Acquiring cache lock")
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    try:
                        print("[MerlinImageDataset] Cache missing, building replay buffer")
                        replay_buffer = _build_replay_buffer(
                            dataset_path=dataset_path,
                            shape_meta=shape_meta,
                            first_cut=first_cut,
                            encoder_window=encoder_window,
                            store=zarr.MemoryStore(),
                        )
                        print("[MerlinImageDataset] Saving cache")
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(store=zip_store)
                    except Exception as exc:
                        if os.path.exists(cache_zarr_path):
                            if os.path.isdir(cache_zarr_path):
                                shutil.rmtree(cache_zarr_path)
                            else:
                                os.remove(cache_zarr_path)
                        raise exc
                else:
                    print("[MerlinImageDataset] Loading cached replay buffer")
                    with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store,
                            store=zarr.MemoryStore(),
                        )
        else:
            replay_buffer = _build_replay_buffer(
                dataset_path=dataset_path,
                shape_meta=shape_meta,
                first_cut=first_cut,
                encoder_window=encoder_window,
                store=zarr.MemoryStore(),
            )

        # # we find obs keys by type in shape_meta
        # # we expect 1 of each for use in _build_replay_buffer
        rgb_keys = []
        lowdim_keys = []
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            data_type = attr.get("type", "low_dim")
            if data_type == "rgb":
                rgb_keys.append(key)
            elif data_type == "low_dim":
                lowdim_keys.append(key)
            else:
                raise ValueError(f"Unsupported obs type '{data_type}' for key '{key}'")

        # # for obs keys only, the sampler can load the first num_obs_steps instead of the full sequence
        key_first_k = {}
        if num_obs_steps is not None:
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = num_obs_steps

        # # we split by *episode*, not step
        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed,
        )

        # # reminder: see shared/utils/sampler.py
        # # create a sampler over the remaining training episodes
        # # the padding ensures that we have valid samples near episode boundaries
        # # recall: seq_len = horizon + num_latency_steps
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
        self.first_cut = int(first_cut)
        self.encoder_window = int(encoder_window)

        print(
            "[MerlinImageDataset] Replay buffer: "
            f"episodes={self.replay_buffer.n_episodes}, steps={self.replay_buffer.n_steps}"
        )

    def get_validation_dataset(self):
        # # ReplayBuffer is shared and FileLock prevents concurrent writes
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
        # # ref: shared/models/common/normalizer.py
        # # from the original Diffusion Policy repository
        # # a LinearNormalizer is a dict of SingleFieldLinearNormalizers that store:
        # # scale, offset, and input stats (min/max/mean/std)
        # # when _fit is called, stats are computed and features are re-mapped ([-1, 1])
        # # this is so the model trains in normalized space and the optimization is smoother.
        # # during inference, we unnormalize predicted actions to "real" action unit scales
        normalizer = LinearNormalizer()

        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer["action"]
        )
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

        obs_t_slice = slice(self.num_obs_steps)
        obs_dict = {}

        for key in self.rgb_keys:
            obs_dict[key] = (
                np.moveaxis(data[key][obs_t_slice], -1, 1).astype(np.float32) / 255.0
            )
            del data[key]

        for key in self.lowdim_keys:
            obs_dict[key] = data[key][obs_t_slice].astype(np.float32)
            del data[key]

        action = data["action"].astype(np.float32)
        if self.num_latency_steps > 0:
            action = action[self.num_latency_steps :]

        torch_data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(action),
        }
        return torch_data


def _sorted_files(directory: str, suffix: str) -> List[str]:
    files = [f for f in os.listdir(directory) if f.endswith(suffix)]
    files.sort()
    return [os.path.join(directory, f) for f in files]


def _rolling_average(data: np.ndarray, window: int) -> np.ndarray:
    # taken from MERLIN/training/dataset.py
    if window <= 0:
        return data.astype(np.float32)

    padded = np.pad(data, ((window, window), (0, 0)), mode="edge")
    kernel_size = window * 2 + 1
    smoothed = np.zeros_like(data, dtype=np.float32)

    for dim in range(data.shape[1]):
        col = padded[:, dim].astype(np.float64)
        cumsum = np.cumsum(col, dtype=np.float64)
        cumsum[kernel_size:] = cumsum[kernel_size:] - cumsum[:-kernel_size]
        smoothed[:, dim] = (
            cumsum[kernel_size - 1 : kernel_size - 1 + len(data)] / kernel_size
        )

    return smoothed.astype(np.float32)


def _load_video_rgb(video_path: str, out_hw: Tuple[int, int]) -> np.ndarray:
    out_h, out_w = out_hw
    frames = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.shape[0] != out_h or frame.shape[1] != out_w:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        frames.append(frame)

    cap.release()

    if not frames:
        print("[MerlinImageDataSet] frames failed to load in _load_video_rgb(...)")
        return np.zeros((0, out_h, out_w, 3), dtype=np.uint8)

    return np.asarray(frames, dtype=np.uint8)


def _build_replay_buffer(
    dataset_path: str,
    shape_meta: dict,
    first_cut: int,
    encoder_window: int,
    store,
) -> ReplayBuffer:
    obs_shape_meta = shape_meta["obs"]

    rgb_keys = [k for k, v in obs_shape_meta.items() if v.get("type", "low_dim") == "rgb"]
    lowdim_keys = [
        k for k, v in obs_shape_meta.items() if v.get("type", "low_dim") == "low_dim"
    ]

    if len(rgb_keys) != 1:
        raise ValueError(
            f"MerlinImageDataset expects exactly 1 RGB obs key, got {len(rgb_keys)}"
        )
    if len(lowdim_keys) != 1:
        raise ValueError(
            "MerlinImageDataset expects exactly 1 low-dim obs key "
            f"(state vector), got {len(lowdim_keys)}"
        )

    rgb_key = rgb_keys[0]
    lowdim_key = lowdim_keys[0]

    rgb_shape = tuple(obs_shape_meta[rgb_key]["shape"])
    if len(rgb_shape) != 3 or rgb_shape[0] != 3:
        raise ValueError(
            f"RGB obs shape must be [3, H, W], got {rgb_shape} for key '{rgb_key}'"
        )
    out_h, out_w = rgb_shape[1], rgb_shape[2]

    lowdim_shape = tuple(obs_shape_meta[lowdim_key]["shape"])
    if lowdim_shape != (12,):
        raise ValueError(
            f"Low-dim obs shape must be [12] for MERLIN (6 arm + 6 encoder), got {lowdim_shape}"
        )

    action_shape = tuple(shape_meta["action"]["shape"])
    if action_shape != (12,):
        raise ValueError(
            f"Action shape must be [12] for MERLIN (6 arm + 6 encoder), got {action_shape}"
        )

    merlin_root = _resolve_merlin_root(dataset_path)

    video_dir = os.path.join(merlin_root, "camera", "mp4_files")
    encoder_dir = os.path.join(merlin_root, "processed_encoder_t")
    action_dir = os.path.join(merlin_root, "processed_action_t")
    sync_dir = os.path.join(merlin_root, "syncs")

    for required_dir in [video_dir, encoder_dir, action_dir, sync_dir]:
        if not os.path.isdir(required_dir):
            raise FileNotFoundError(f"Required MERLIN directory not found: {required_dir}")

    vid_files = _sorted_files(video_dir, ".mp4")
    enc_files = _sorted_files(encoder_dir, ".npy")
    act_files = _sorted_files(action_dir, ".npy")
    sync_files = _sorted_files(sync_dir, ".npz")

    n = len(vid_files)
    if not (len(enc_files) == n and len(act_files) == n and len(sync_files) == n):
        raise RuntimeError(
            "MERLIN file count mismatch after sorting: "
            f"video={len(vid_files)} encoder={len(enc_files)} "
            f"action={len(act_files)} sync={len(sync_files)}"
        )

    replay_buffer = ReplayBuffer.create_empty_zarr(storage=store)

    kept_episodes = 0
    dropped_episodes = 0
    total_steps = 0

    for i in range(n):
        raw_enc = np.load(enc_files[i]).astype(np.float32)
        raw_act = np.load(act_files[i]).astype(np.float32)
        sync_data = np.load(sync_files[i])

        frame_idx = sync_data["frame_idx"].astype(np.int64)
        encoder_idx = sync_data["encoder_idx"].astype(np.int64)
        pose_idx = sync_data["pose_idx"].astype(np.int64)

        if not (len(frame_idx) == len(encoder_idx) == len(pose_idx)):
            print(
                f"[MerlinImageDataset] Dropping episode {i}: sync array length mismatch"
            )
            dropped_episodes += 1
            continue

        if len(frame_idx) <= first_cut:
            dropped_episodes += 1
            continue

        frame_idx = frame_idx[first_cut:]
        encoder_idx = encoder_idx[first_cut:]
        pose_idx = pose_idx[first_cut:]

        smoothed_enc = _rolling_average(raw_enc, window=encoder_window)
        video_frames = _load_video_rgb(vid_files[i], out_hw=(out_h, out_w))

        if len(video_frames) == 0:
            print(f"[MerlinImageDataset] Dropping episode {i}: video has no frames")
            dropped_episodes += 1
            continue

        if len(raw_act) == 0 or len(smoothed_enc) == 0:
            print(
                f"[MerlinImageDataset] Dropping episode {i}: empty action or encoder array"
            )
            dropped_episodes += 1
            continue

        max_frame = len(video_frames) - 1
        max_pose = len(raw_act) - 1
        max_enc = len(smoothed_enc) - 1

        frame_idx = np.clip(frame_idx, 0, max_frame)
        pose_idx = np.clip(pose_idx, 0, max_pose)
        encoder_idx = np.clip(encoder_idx, 0, max_enc)

        episode_images = video_frames[frame_idx]
        episode_arm = raw_act[pose_idx].astype(np.float32)
        episode_hand = smoothed_enc[encoder_idx]
        episode_state = np.concatenate([episode_arm, episode_hand], axis=-1).astype(
            np.float32
        )

        episode = {
            rgb_key: episode_images.astype(np.uint8),
            lowdim_key: episode_state,
            "action": episode_state,
        }
        replay_buffer.add_episode(episode)

        kept_episodes += 1
        total_steps += len(episode_state)

    if replay_buffer.n_episodes == 0:
        raise RuntimeError("No usable episodes found in MERLIN dataset")

    print(
        "[MerlinImageDataset] Built replay buffer: "
        f"kept={kept_episodes}, dropped={dropped_episodes}, total_steps={total_steps}"
    )

    return replay_buffer
