import time
from collections import deque
from typing import Deque, Dict, Optional

import cv2
import dill
import hydra
import numpy as np
import torch

from shared.env.merlin.se3_utils import relative_to_absolute


# # NOTE(raayan)
# # This code was generated with the help of Codex and Opus. Blame the overwhelming validation on Codex.


class MerlinPolicyInference:
    """
    Lightweight MERLIN policy inference wrapper.
    
    Currently:
    image -> model -> relative actions -> + current state -> absolute actions
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
        self.expected_chw = tuple(self.shape_meta["obs"][self.rgb_key]["shape"])
        self.action_dim = int(self.shape_meta["action"]["shape"][0])
        self.num_obs_steps = int(policy.num_obs_steps)
        self.hand_pose_key = (
            "hand_pose_abs" if "hand_pose_abs" in self.shape_meta["obs"] else None
        )

        self._img_history: Deque[np.ndarray] = deque(maxlen=self.num_obs_steps)

    def reset(self) -> None:
        self._img_history.clear()
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def predict(self, image: np.ndarray, robot_state: np.ndarray) -> Dict[str, object]:
        """
        Run one inference step.

        Args:
            image: RGB image in HWC format. Used as conditioning.
            robot_state: 1D state array with shape (12,) for MERLIN.
                         Used to convert the predicted relative 
                         actions back to absolute.
        Returns:
            Dict with absolute action outputs and metadata.
            NOTE: the model is expected to predict *relative* outputs.
                  We convert them to absolute outputs using current robot state
                  which is passed into the prediction.
        """
        if not isinstance(robot_state, np.ndarray):
            robot_state = np.asarray(robot_state, dtype=np.float32)
        robot_state = robot_state.astype(np.float32).reshape(-1)
        if robot_state.shape[0] != self.action_dim:
            raise ValueError(
                f"robot_state must have shape ({self.action_dim},), got {robot_state.shape}."
            )

        t0 = time.time()
        obs_torch = self._prepare_obs_torch(image=image, robot_state=robot_state)

        # # NOTE(raayan)
        # # We are using *just* the image observation in order to do prediction.
        # # I decided to walk through what was happening just for my understanding.
        # # In model code, this works as follows:
        # # 1. self.policy_predict_action(obs_torch) will normalized the data, which is just the image.
        # #    normalized_obs = normalizer.normalize(obs) 
        # #    | unet_image_policy::259
        # # 2. since global_obs_cond=True, we flatten and encode the image into a global conditioning vector
        # #    cond_BG = normalized_obs_feats = obs_encoder(flat_normalized_obs = dict_apply(normalized_obs ... ))
        # #    | unet_image_policy::266-278
        # # 3. ObsEncoder.forward() will only collect visual features because it skips low_dim_keys
        # #    | multi_image_obs_encoder::181-233
        # # 4. Our mask becomes only false
        # #    cond_mask_BTF = torch.zeros_like(cond_data_BTF = torch.zeros(size=(B, T, Fa), ...), dtype=torch.bool)
        # #    | unet_image_policy::277-278
        # # 5. For our sampling, the sample we create looks like this:
        # #    sample_BTF = self.conditional_sample(
        # #        cond_data_BTF=cond_data_BTF,  # zeros [1, 16, 12]
        # #        cond_mask_BTF=cond_mask_BTF,  # all False [1, 16, 12]
        # #        cond_BTL=None,
        # #        cond_BG=cond_BG,              # image features [1, Fo]
        # #    )
        # #    | unet_image_policy::299-305
        # # 6. When we actually do our denoising, at each step, we assume we have completely unknown action trajectory
        # #    trajectory_BTF[cond_mask_BTF] = cond_data_BTF[cond_mask_BTF] # Since it all false, we don't overwrite at all
        # #    | unet_image_policy::214
        # # 7. The model forward pass gets our noised trajectory, and the only global condition (cond_BG) are the image features
        # #    denoised_trajectory_BTF = model(
        # #        sample_BTF=trajectory_BTF, # Current *noisy* trajectory at timestep t
        # #        timesteps_B=t, # Embedded in ConditionalUnet1d as timestep conditioning
        # #        cond_BTL=cond_BTL, # No temporal conditioning. This will be None.
        # #        cond_BG=cond_BG, # This is our image features
        # #    )
        # #    | unet_image_policy::217-222
        # # 8. Complete all scheduler steps and get a denoised trajectory of *relative* actions in *normalized* space
        # #    | unet_image_policy::225-231
        # # 9. Unnormalize from [-1, 1] to real relative action units
        # #    | unet_image_policy::307-313
        # # 0. Return result here.

        with torch.no_grad():
            # # NOTE(raayan)
            # # DiffusionUnetImagePolicy.predict_action(...) internally
            # # 1. normalizes obs with normalizer
            # # 2. samples action trajectory in normalized space
            # # 3. unnormalizes action back to real units before returning
            result = self.policy.predict_action(obs_torch)
        latency = time.time() - t0

        # # Model outputs are unnormalized *relative* actions.
        rel_action_chunk = result["action"][0].detach().cpu().numpy().astype(np.float32)
        rel_action_pred = result["action_pred"][0].detach().cpu().numpy().astype(np.float32)

        # # Convert relative -> absolute using current robot state.
        action_chunk = relative_to_absolute(rel_action_chunk, robot_state)
        action_pred = relative_to_absolute(rel_action_pred, robot_state)

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

        cam_shape = tuple(obs_meta["camera_0"].get("shape", ()))
        if len(cam_shape) != 3 or cam_shape[0] != 3:
            raise ValueError(
                f"Expected camera_0 shape [3,H,W], got {obs_meta['camera_0'].get('shape')}."
            )

        action_shape = tuple(shape_meta.get("action", {}).get("shape", ()))
        if action_shape != (12,):
            raise ValueError(
                f"Expected action shape [12] for MERLIN, got {shape_meta.get('action', {}).get('shape')}."
            )

        if "hand_pose_abs" in obs_meta:
            hand_meta = obs_meta["hand_pose_abs"]
            hand_shape = tuple(hand_meta.get("shape", ()))
            hand_type = hand_meta.get("type", "low_dim")
            if hand_type != "low_dim" or hand_shape != (6,):
                raise ValueError(
                    "Expected hand_pose_abs to be low_dim with shape [6], "
                    f"got type={hand_type}, shape={hand_meta.get('shape')}."
                )

    def _prepare_obs_torch(
        self, image: np.ndarray, robot_state: np.ndarray
    ) -> Dict[str, torch.Tensor]:
        img = self._prepare_image(image)
        self._img_history.append(img)

        while len(self._img_history) < self.num_obs_steps:
            self._img_history.appendleft(self._img_history[0].copy())

        img_stack = np.stack(list(self._img_history), axis=0)  # To, H, W, C
        img_stack = np.moveaxis(img_stack, -1, 1).astype(np.float32)  # To, C, H, W

        obs_np = {
            self.rgb_key: img_stack[None],  # B, To, C, H, W
        }
        if self.hand_pose_key is not None:
            hand_pose = robot_state[6:12].astype(np.float32)
            hand_pose_hist = np.repeat(
                hand_pose[None], repeats=self.num_obs_steps, axis=0
            )
            obs_np[self.hand_pose_key] = hand_pose_hist[None]  # B, To, 6

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
