import time
from collections import deque
from typing import Deque, Dict, Optional

import cv2
import dill
import hydra
import numpy as np
import torch


class MerlinPolicyInference:
    """
    Lightweight MERLIN policy inference wrapper.

    This class only handles model-side inference:
      image + robot_state -> action
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        use_ema: bool = True,
        num_inference_steps: Optional[int] = None,
        action_mode: str = "first",
    ):
        if action_mode not in {"first", "chunk", "all"}:
            raise ValueError(
                f"Unsupported action_mode '{action_mode}'. Use one of: first, chunk, all."
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.action_mode = action_mode
        self.checkpoint_path = checkpoint_path

        payload = torch.load(open(checkpoint_path, "rb"), pickle_module=dill)
        cfg = payload["cfg"]
        workspace_cls = hydra.utils.get_class(cfg._target_)
        workspace = workspace_cls(cfg)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        policy = workspace.model
        if use_ema and getattr(cfg.training, "use_ema", False):
            if workspace.ema_model is None:
                raise RuntimeError(
                    "Requested EMA policy, but checkpoint/workspace has no EMA model."
                )
            policy = workspace.ema_model

        policy.eval().to(self.device)
        if num_inference_steps is not None:
            policy.num_inference_steps = int(num_inference_steps)

        self.workspace = workspace
        self.cfg = cfg
        self.policy = policy

        self.shape_meta = self._resolve_shape_meta(cfg)
        self._validate_shape_meta(self.shape_meta)

        self.rgb_key = "camera_0"
        self.state_key = "robot_state"
        self.expected_chw = tuple(self.shape_meta["obs"][self.rgb_key]["shape"])
        self.state_dim = int(self.shape_meta["obs"][self.state_key]["shape"][0])
        self.action_dim = int(self.shape_meta["action"]["shape"][0])
        self.num_obs_steps = int(policy.num_obs_steps)

        self._img_history: Deque[np.ndarray] = deque(maxlen=self.num_obs_steps)
        self._state_history: Deque[np.ndarray] = deque(maxlen=self.num_obs_steps)

    def reset(self) -> None:
        self._img_history.clear()
        self._state_history.clear()
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def predict(self, image: np.ndarray, robot_state: np.ndarray) -> Dict[str, object]:
        """
        Run one inference step.

        Args:
            image: RGB image in HWC format.
            robot_state: 1D state array with shape (12,) for MERLIN.
        Returns:
            Dict with action outputs and metadata.
        """
        t0 = time.time()
        obs_torch = self._prepare_obs_torch(image=image, robot_state=robot_state)
        with torch.no_grad():
            result = self.policy.predict_action(obs_torch)
        latency = time.time() - t0

        action_chunk = result["action"][0].detach().to("cpu").numpy().astype(np.float32)
        action_pred = (
            result["action_pred"][0].detach().to("cpu").numpy().astype(np.float32)
        )

        output: Dict[str, object] = {
            "latency_sec": float(latency),
            "device": str(self.device),
            "num_inference_steps": int(self.policy.num_inference_steps),
        }

        if self.action_mode == "first":
            output["action"] = action_chunk[0].astype(np.float32)
        elif self.action_mode == "chunk":
            output["action"] = action_chunk
            output["action_chunk"] = action_chunk
        else:  # all
            output["action"] = action_chunk[0].astype(np.float32)
            output["action_chunk"] = action_chunk
            output["action_pred"] = action_pred

        return output

    @staticmethod
    def _resolve_shape_meta(cfg) -> dict:
        if hasattr(cfg, "shape_meta"):
            return cfg.shape_meta
        if hasattr(cfg, "tasks") and hasattr(cfg.tasks, "shape_meta"):
            return cfg.tasks.shape_meta
        if hasattr(cfg, "task") and hasattr(cfg.task, "shape_meta"):
            return cfg.task.shape_meta
        raise ValueError(
            "Could not find shape_meta in cfg. Checked cfg.shape_meta, cfg.tasks.shape_meta, cfg.task.shape_meta."
        )

    @staticmethod
    def _validate_shape_meta(shape_meta: dict) -> None:
        obs_meta = shape_meta.get("obs", {})
        if "camera_0" not in obs_meta:
            raise ValueError("shape_meta.obs must contain key 'camera_0'.")
        if "robot_state" not in obs_meta:
            raise ValueError("shape_meta.obs must contain key 'robot_state'.")

        cam_shape = tuple(obs_meta["camera_0"].get("shape", ()))
        if len(cam_shape) != 3 or cam_shape[0] != 3:
            raise ValueError(
                f"Expected camera_0 shape [3,H,W], got {obs_meta['camera_0'].get('shape')}."
            )

        state_shape = tuple(obs_meta["robot_state"].get("shape", ()))
        if state_shape != (12,):
            raise ValueError(
                f"Expected robot_state shape [12] for MERLIN, got {obs_meta['robot_state'].get('shape')}."
            )

        action_shape = tuple(shape_meta.get("action", {}).get("shape", ()))
        if action_shape != (6,):
            raise ValueError(
                f"Expected action shape [6] for MERLIN, got {shape_meta.get('action', {}).get('shape')}."
            )

    def _prepare_obs_torch(self, image: np.ndarray, robot_state: np.ndarray) -> Dict[str, torch.Tensor]:
        img = self._prepare_image(image)
        state = self._prepare_state(robot_state)

        self._img_history.append(img)
        self._state_history.append(state)

        while len(self._img_history) < self.num_obs_steps:
            self._img_history.appendleft(self._img_history[0].copy())
            self._state_history.appendleft(self._state_history[0].copy())

        img_stack = np.stack(list(self._img_history), axis=0)  # To, H, W, C
        img_stack = np.moveaxis(img_stack, -1, 1).astype(np.float32)  # To, C, H, W
        state_stack = np.stack(list(self._state_history), axis=0).astype(np.float32)  # To, D

        obs_np = {
            self.rgb_key: img_stack[None],     # B, To, C, H, W
            self.state_key: state_stack[None],  # B, To, D
        }

        obs_torch = {
            k: torch.from_numpy(v).to(self.device, non_blocking=True)
            for k, v in obs_np.items()
        }
        return obs_torch

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy array.")
        if image.ndim != 3:
            raise ValueError(
                f"Expected HWC image (3D array), got shape {image.shape}."
            )
        if image.shape[2] != 3:
            raise ValueError(
                f"Expected 3-channel RGB image in HWC format, got shape {image.shape}."
            )

        expected_h = int(self.expected_chw[1])
        expected_w = int(self.expected_chw[2])
        if image.shape[0] != expected_h or image.shape[1] != expected_w:
            image = cv2.resize(
                image, (expected_w, expected_h), interpolation=cv2.INTER_AREA
            )

        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)
            if image.min() < 0.0 or image.max() > 1.0:
                raise ValueError(
                    f"Float image must be in [0,1], got min={image.min()} max={image.max()}."
                )
        return image

    def _prepare_state(self, robot_state: np.ndarray) -> np.ndarray:
        if not isinstance(robot_state, np.ndarray):
            robot_state = np.asarray(robot_state, dtype=np.float32)
        robot_state = robot_state.astype(np.float32).reshape(-1)
        if robot_state.shape[0] != self.state_dim:
            raise ValueError(
                f"robot_state must have shape ({self.state_dim},), got {robot_state.shape}."
            )
        return robot_state
