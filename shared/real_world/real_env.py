import logging
import time
import math
import numpy as np
import shutil
import pathlib

from multiprocessing.managers import SharedMemoryManager
from shared.real_world.control.xarm_controller import (
    XArmConfig,
    XArmController,
    XArmInterpolationController,
)
from shared.real_world.record_utils.replay_buffer import ReplayBuffer
from shared.real_world.realsense.single_realsense import SingleRealsense
from shared.real_world.record_utils.cv2_util import get_image_transform
from shared.real_world.record_utils.video_recorder import VideoRecorder
from shared.real_world.realsense.multi_realsense import MultiRealsense
from shared.real_world.record_utils.timestamp_accumulator import (
    TimestampActionAccumulator,
    TimestampObsAccumulator,
)
from typing import Tuple, List, Optional, Dict, Union

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

"""
DEFAULT_OBS_KEY_MAP = {
    # Robot - using the correct key names that match XArmController.get_all_state()
    "TCPPose": "robot_eef_pose",
    "TCPSpeed": "robot_eef_pose_vel",
    "JointAngles": "robot_joint",
    "JointSpeeds": "robot_joint_vel",
    # Additional keys if they exist
    "Grasp": "robot_gripper",
    "robot_receive_timestamp": "robot_timestamp",
    # Camera stuff,
    "Camera_0": "camera_0",
    "Camera_1": "camera_1",
    "Camera_2": "camera_2",
    # Timestamps
    "step_idx": "step_idx",
    "timestamp": "timestamp",
}
"""
DEFAULT_OBS_KEY_MAP = {
    # Robot - using the correct key names that match XArmController.get_all_state()
    "robot_eef_pose": "robot_eef_pose",
    "robot_eef_pose_vel": "robot_eef_pose_vel",
    "robot_joint": "robot_joint",
    "robot_joint_vel": "robot_joint_vel",
    # Additional keys if they exist
    "robot_gripper": "robot_gripper",
    "robot_timestamp": "robot_timestamp",
    # Camera stuff,
    "camera_0": "camera_0",
    "camera_1": "camera_1",
    "camera_2": "camera_2",
    # Timestamps
    "step_idx": "step_idx",
    "timestamp": "timestamp",
}


class RealEnv:
    def __init__(
        self,
        output_dir: Union[pathlib.Path, str] = "./recordings/",
        xarm_config: Optional[XArmConfig] = None,
        frequency: int = 30,
        num_obs_steps: int = 2,
        obs_image_resolution: Tuple[int, int] = (640, 480),
        max_obs_buffer_size: int = 30,
        camera_serial_numbers: Optional[List[int]] = None,
        obs_key_map: Dict = DEFAULT_OBS_KEY_MAP,
        obs_float32: bool = False,
        video_capture_fps: int = 30,
        video_capture_resolution: Tuple[int, int] = (1280, 720),
        thread_per_video: int = 3,
        video_crf: int = 3,
        shm_manager: Optional[SharedMemoryManager] = None,
        use_interpolation: bool = False,
        max_pos_speed: float = 0.25,
        max_rot_speed: float = 0.6,
        xarm_controller: Optional[
            Union[XArmController, XArmInterpolationController]
        ] = None,
    ):
        logger.info("[RealEnv] Initializing environment.")

        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        video_dir = output_dir.joinpath("videos")
        video_dir.mkdir(parents=True, exist_ok=True)

        zarr_path = str(output_dir.joinpath("replay_buffer.zarr").absolute())
        replay_buffer = ReplayBuffer.create_from_path(zarr_path=zarr_path, mode="a")

        logger.info(f"[RealEnv] Output directory: {output_dir}")
        logger.info(f"[RealEnv] Video directory: {video_dir}")
        logger.info(f"[RealEnv] Replay buffer path: {zarr_path}")

        if xarm_config is None:
            xarm_config = XArmConfig()

        assert (
            frequency <= video_capture_fps
        ), "Cannot run frequency faster than video capture."

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()
            logger.info("[RealEnv] Started local SharedMemoryManager")
        self.shm_manager = shm_manager

        if camera_serial_numbers is None:
            camera_serial_numbers = SingleRealsense.get_connected_devices_serial()

        logger.info(f"[RealEnv] Camera serial numbers: {camera_serial_numbers}")

        color_tf = get_image_transform(
            input_res=video_capture_resolution,
            output_res=obs_image_resolution,
            bgr_to_rgb=True,
        )

        if obs_float32:

            def float_transform(img):
                return color_tf(img).astype(np.float32) / 255.0


        def transform(data):
            if "color" in data:
                img = color_tf(data["color"])
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
                data["color"] = img
            return data

        recording_transform = transform
        recording_fps = video_capture_fps
        recording_pix_fmt = "rgb24"

        video_recorder = VideoRecorder.create_h264(
            fps=recording_fps,
            codec="libx264",
            input_pix_fmt=recording_pix_fmt,
            crf=video_crf,
            thread_type="FRAME",
            thread_count=thread_per_video,
        )
        logger.info(f"[RealEnv] Recording FPS: {recording_fps}")

        realsense = MultiRealsense(
            serial_numbers=camera_serial_numbers,
            shm_manager=shm_manager,
            resolution=video_capture_resolution,
            record_fps=recording_fps,
            capture_fps=video_capture_fps,
            put_fps=video_capture_fps,
            put_downsample=True,
            enable_color=True,
            enable_depth=False,
            enable_infrared=False,
            get_max_k=max_obs_buffer_size,
            transform=transform,
            recording_transform=recording_transform,
            video_recorder=video_recorder,
            verbose=False,
        )

        if xarm_controller is not None:
            # Use the provided controller (either standard or interpolation)
            robot = xarm_controller
            logger.info("[RealEnv] Using provided XArm controller")
        elif use_interpolation:
            # Create an interpolation controller if requested
            robot = XArmInterpolationController(
                shm_manager=shm_manager,
                xarm_config=xarm_config,
                max_pos_speed=max_pos_speed,
                max_rot_speed=max_rot_speed,
            )
            logger.info(
                "[RealEnv] Created XArmInterpolationController with interpolation"
            )
        else:
            # Create a standard controller
            robot = XArmController(
                shm_manager=shm_manager,
                xarm_config=xarm_config,
            )
            logger.info("[RealEnv] Created standard XArmController")

        self.realsense = realsense
        self.robot = robot
        self.xarm_config = xarm_config
        self.video_capture_fps = video_capture_fps
        self.frequency = frequency
        self.num_obs_steps = num_obs_steps
        self.max_obs_buffer_size = max_obs_buffer_size
        self.obs_key_map = obs_key_map

        self.output_dir = output_dir
        self.video_dir = video_dir
        self.replay_buffer = replay_buffer

        self.last_realsense_data = None

        self.obs_accumulator = None
        self.action_accumulator = None
        self.stage_accumulator = None

        self.start_time = None

    # Start-stop API
    @property
    def is_ready(self):
        return self.realsense.is_ready and self.robot.is_ready

    def start(self, wait=True):
        if wait:
            self.realsense.start(wait=True)
            self.robot.start(wait=True)
        else:
            self.realsense.start(wait=False)
            self.robot.start(wait=False)

    def stop(self, wait=True):
        self.end_episode()
        if wait:
            self.realsense.stop(wait=True)
            self.robot.stop(wait=True)
        else:
            self.realsense.stop(wait=False)
            self.robot.stop(wait=False)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def get_obs(self) -> Dict:
        "observation dict"
        assert self.is_ready

        k = math.ceil(self.num_obs_steps * (self.video_capture_fps / self.frequency))
        self.last_realsense_data = self.realsense.get(k=k, out=self.last_realsense_data)

        # Running at 50 hz
        last_robot_data = self.robot.get_all_state()

        dt = 1 / self.frequency
        last_timestamp = np.max(
            [x["timestamp"][-1] for x in self.last_realsense_data.values()]
        )
        obs_align_timestamps = last_timestamp - (
            np.arange(self.num_obs_steps)[::-1] * dt
        )

        camera_obs = dict()
        for camera_idx, value in self.last_realsense_data.items():
            this_timestamps = value["timestamp"]
            this_idxs = list()
            for t in obs_align_timestamps:
                is_before_idxs = np.nonzero(this_timestamps < t)[0]
                this_idx = 0
                if len(is_before_idxs) > 0:
                    this_idx = is_before_idxs[-1]
                this_idxs.append(this_idx)

            camera_obs[f"camera_{camera_idx}"] = value["color"][this_idxs]

        robot_timestamps = last_robot_data["robot_receive_timestamp"]
        this_timestamps = robot_timestamps
        this_idxs = list()
        for t in obs_align_timestamps:
            is_before_idxs = np.nonzero(this_timestamps < t)[0]
            this_idx = 0
            if len(is_before_idxs) > 0:
                this_idx = is_before_idxs[-1]
            this_idxs.append(this_idx)

        robot_obs_raw = dict()
        for k, v in last_robot_data.items():
            if k in self.obs_key_map:
                robot_obs_raw[self.obs_key_map[k]] = v

        robot_obs = dict()
        for k, v in robot_obs_raw.items():
            robot_obs[k] = v[this_idxs]

        if self.obs_accumulator is not None:
            complete_obs = dict()

            for k, v in last_robot_data.items():
                if k in self.obs_key_map:
                    complete_obs[k] = v

            now = time.time()
            relative_timestamps = robot_timestamps
            absolute_timestamps = np.array(
                [now + (t - relative_timestamps[0]) for t in relative_timestamps]
            )

            self.obs_accumulator.put(complete_obs, absolute_timestamps)

        obs_data = dict(camera_obs)
        obs_data.update(robot_obs)
        obs_data["timestamp"] = obs_align_timestamps
        return obs_data

    def exec_actions(
        self,
        actions: np.ndarray,
        timestamps: np.ndarray,
        stages: Optional[np.ndarray] = None,
    ):
        assert self.is_ready
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)
        if stages is None:
            stages = np.zeros_like(timestamps, dtype=np.int64)
        elif not isinstance(stages, np.ndarray):
            stages = np.array(stages, dtype=np.int64)

        receive_time = time.time()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]
        new_stages = stages[is_new]

        for i in range(len(new_actions)):
            new_action = new_actions[i]
            pose = new_action[:6]
            grasp = new_action[-1]
            self.robot.step(pose, grasp)

        if self.action_accumulator is not None:
            self.action_accumulator.put(
                new_actions,
                new_timestamps,
            )

        if self.stage_accumulator is not None:
            self.stage_accumulator.put(
                new_stages,
                new_timestamps,
            )

    def exec_action_waypoints(
        self,
        actions: np.ndarray,
        timestamps: np.ndarray,
        stages: Optional[np.ndarray] = None,
    ):
        """
        Execute actions as waypoints with interpolation for smooth motion.
        This method should be used with XArmInterpolationController.

        Args:
            actions: Array of action vectors, each containing 7 values (6 for pose, 1 for grasp)
            timestamps: Array of target timestamps for each action
            stages: Optional array of stage values for each action
        """
        assert (
            self.is_ready
        ), "RealEnv must be initialized and ready before executing actions"

        # Convert inputs to numpy arrays if they aren't already
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)
        if stages is None:
            stages = np.zeros_like(timestamps, dtype=np.int64)
        elif not isinstance(stages, np.ndarray):
            stages = np.array(stages, dtype=np.int64)

        # Verify action dimensions
        assert (
            actions.ndim >= 1 and actions.shape[-1] == 7
        ), f"Actions must have shape (N, 7), got {actions.shape}"
        if actions.ndim == 1:
            # Single action, add batch dimension
            actions = np.expand_dims(actions, axis=0)
            timestamps = np.expand_dims(timestamps, axis=0)
            stages = np.expand_dims(stages, axis=0)

        # Ensure arrays have the same length
        n_actions = len(actions)
        assert (
            len(timestamps) == n_actions
        ), f"Timestamps length {len(timestamps)} must match actions length {n_actions}"
        assert (
            len(stages) == n_actions
        ), f"Stages length {len(stages)} must match actions length {n_actions}"

        # Filter actions that are in the future
        receive_time = time.time()
        is_new = timestamps > receive_time

        # For policy mode, we want to limit how many actions we send at once
        # This helps prevent overloading the controller
        if n_actions > 1:  # If this is the policy sending actions (typically batched)
            # Print info for debugging
            print("Scheduling waypoint...")
            print(f"Pose: {actions[0, :6]}")

            # If we have many actions, just use the first one
            # The policy will send more in the next iteration
            if np.sum(is_new) > 1:
                # Take only the first action that's in the future
                future_indices = np.where(is_new)[0]
                first_idx = future_indices[0]
                new_actions = actions[first_idx : first_idx + 1]
                new_timestamps = timestamps[first_idx : first_idx + 1]
                new_stages = stages[first_idx : first_idx + 1]
            elif np.sum(is_new) == 1:
                # Just one future action, use it
                new_actions = actions[is_new]
                new_timestamps = timestamps[is_new]
                new_stages = stages[is_new]
            else:
                # No future actions, use the last one with a small offset
                new_actions = actions[[-1]]
                new_timestamps = np.array([receive_time + 0.05])  # 50ms in the future
                new_stages = stages[[-1]]
        else:
            # Human control typically sends one action at a time
            # Check if we have any valid future actions
            if np.sum(is_new) == 0:
                # If no future actions, use the last action with a small time offset
                if n_actions > 0:
                    new_actions = actions  # Just use the provided action
                    new_timestamps = np.array(
                        [receive_time + 0.05]
                    )  # 50ms in the future
                    new_stages = stages
                else:
                    # No actions provided, return early
                    return
            else:
                # Use only future actions
                new_actions = actions[is_new]
                new_timestamps = timestamps[is_new]
                new_stages = stages[is_new]

        # Schedule waypoints for future actions
        for i in range(len(new_actions)):
            action = new_actions[i]
            pose = action[:6]
            grasp = action[-1]
            print(f"[realenv] Action: {action}")
            print(f"[realenv] Grasp: {grasp}")
            try:
                # Schedule the waypoint with the interpolation controller
                # Ensure the timestamp is at least a bit in the future for smoothing
                target_time = max(new_timestamps[i], time.time() + 0.02)

                self.robot.schedule_waypoint(
                    pose=pose, target_time=target_time, grasp=grasp
                )
            except Exception as e:
                logger.error(f"Error scheduling waypoint: {e}")
                # Fall back to regular step command on error
                self.robot.step(pose, grasp)

        # Record actions if recording
        if self.action_accumulator is not None:
            self.action_accumulator.put(
                new_actions,
                new_timestamps,
            )

        if self.stage_accumulator is not None:
            self.stage_accumulator.put(
                new_stages,
                new_timestamps,
            )

    def get_robot_state(self):
        return self.robot.get_state()

    def start_episode(self, start_time=None):
        if start_time is None:
            start_time = time.time()

        self.start_time = start_time

        assert self.is_ready

        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        this_video_dir.mkdir(parents=True, exist_ok=True)
        n_cameras = self.realsense.n_cameras
        video_paths = list()
        for i in range(n_cameras):
            video_paths.append(str(this_video_dir.joinpath(f"{i}.mp4").absolute()))

        self.realsense.restart_put(start_time=start_time)
        self.realsense.start_recording(video_path=video_paths, start_time=start_time)

        self.obs_accumulator = TimestampObsAccumulator(
            start_time=start_time,
            dt=1 / self.frequency,
        )
        self.action_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1 / self.frequency,
        )
        self.stage_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1 / self.frequency,
        )
        print(f"Episode {episode_id} started!")

    def end_episode(self):
        assert self.is_ready

        self.realsense.stop_recording()

        if self.obs_accumulator is not None:
            assert self.action_accumulator is not None
            assert self.stage_accumulator is not None

            obs_data = self.obs_accumulator.data
            obs_timestamps = self.obs_accumulator.timestamps

            actions = self.action_accumulator.actions
            action_timestamps = self.action_accumulator.timestamps
            stages = self.stage_accumulator.actions
            n_steps = min(len(obs_timestamps), len(action_timestamps))
            if n_steps > 0:
                episode = dict()
                episode["timestamp"] = obs_timestamps[:n_steps]
                episode["action"] = actions[:n_steps]
                episode["stage"] = stages[:n_steps]
                for key, value in obs_data.items():
                    episode[key] = value[:n_steps]
                self.replay_buffer.add_episode(episode, compressors="disk")
                episode_id = self.replay_buffer.n_episodes - 1
                print(f"Episode {episode_id} saved!")

            self.obs_accumulator = None
            self.action_accumulator = None
            self.stage_accumulator = None

    def drop_episode(self):
        self.end_episode()
        self.replay_buffer.drop_episode()
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        if this_video_dir.exists():
            shutil.rmtree(str(this_video_dir))
        print(f"Episode {episode_id} dropped!")
