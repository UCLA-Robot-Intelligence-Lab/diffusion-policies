import time
from multiprocessing.managers import SharedMemoryManager
import click
import numpy as np
import torch
import traceback
import dill
import hydra
import logging
import pathlib
from omegaconf import OmegaConf
import scipy.spatial.transform as st

from shared.real_world.control.spacemouse import Spacemouse
from shared.real_world.record_utils.keystroke_counter import (
    KeystrokeCounter,
    Key,
    KeyCode,
)
from shared.real_world.record_utils.precise_sleep import precise_wait
from shared.real_world.control.xarm_controller import (
    XArmConfig,
)
from shared.real_world.real_env import RealEnv
from shared.real_world.real_inference_util import (
    get_real_obs_resolution,
    get_real_obs_dict,
)
from shared.utils.pytorch_util import dict_apply

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver("eval", eval, replace=True)


def main(
    input="./checkpoints/epoch=0250-train_loss=0.005.ckpt",
    output="./output/",
    init_joints=True,
    frequency=30,
    command_latency=0.01,
    steps_per_inference=2,
    record_res=(1280, 720),
    spacemouse_deadzone=0.05,
    use_interpolation=True,  # Enable interpolation by default for smoother motion
    max_pos_speed=0.25,  # Position speed limit (m/s)
    max_rot_speed=0.6,  # Rotation speed limit (degrees/s)
):
    dt = 1.0 / frequency
    output_dir = pathlib.Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    ckpt_path = input
    print("CHECKPOINT: ", ckpt_path, "\n\n")
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    print("CONFIG: ", cfg)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # Setup for different policy types
    action_offset = 0
    delta_action = False
    if "diffusion" in cfg.name:
        # diffusion model
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device("cuda")
        policy.eval().to(device)

        # set inference params
        policy.num_inference_steps = 16  # DDIM inference iterations
        policy.num_action_steps = policy.horizon - policy.num_obs_steps + 1
    else:
        raise RuntimeError("Unsupported policy type: ", cfg.name)

    xarm_config = XArmConfig()

    # Handle both cfg.shape_meta and cfg.task.shape_meta cases
    shape_meta = None
    if hasattr(cfg, "shape_meta"):
        shape_meta = cfg.shape_meta
        print("Using shape_meta from cfg.shape_meta")
    elif hasattr(cfg, "tasks") and hasattr(cfg.tasks, "shape_meta"):
        shape_meta = cfg.tasks.shape_meta
        print("Using shape_meta from cfg.tasks.shape_meta")
    elif hasattr(cfg, "task") and hasattr(cfg.task, "shape_meta"):
        shape_meta = cfg.task.shape_meta
        print("Using shape_meta from cfg.task.shape_meta")
    else:
        print("Available keys in cfg:", cfg.keys())
        raise ValueError(
            "Could not find shape_meta in config. Please specify either cfg.shape_meta, cfg.tasks.shape_meta, or cfg.task.shape_meta"
        )

    get_real_obs_resolution(shape_meta)
    num_obs_steps = cfg.num_obs_steps
    logger.info(f"num_obs_steps: {num_obs_steps}")
    logger.info(f"steps_per_inference: {steps_per_inference}")
    logger.info(f"action_offset: {action_offset}")
    logger.info(f"use_interpolation: {use_interpolation}")

    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, Spacemouse(
            deadzone=spacemouse_deadzone, shm_manager=shm_manager
        ) as sm, RealEnv(
            output_dir=output_dir,
            xarm_config=xarm_config,
            frequency=frequency,
            num_obs_steps=num_obs_steps,
            obs_image_resolution=record_res,
            max_obs_buffer_size=30,
            obs_float32=True,
            video_capture_fps=30,
            video_capture_resolution=record_res,
            thread_per_video=3,
            video_crf=21,
            shm_manager=shm_manager,
            use_interpolation=use_interpolation,  # Use the new interpolation controller
            max_pos_speed=max_pos_speed,
            max_rot_speed=max_rot_speed,
        ) as env:
            logger.info("Configuring camera settings...")
            env.realsense.set_exposure(exposure=120, gain=0)
            env.realsense.set_white_balance(white_balance=3000)

            # Warm up policy inference
            logger.info("Warming up policy inference...")
            obs = env.get_obs()

            # Debug: print observation keys and shape_meta
            print("\nObservation keys:", list(obs.keys()))
            print("\nShape meta structure:", shape_meta)

            # Get robot state and add it to the observation
            robot_state = env.get_robot_state()
            print("\nRobot state keys:", list(robot_state.keys()))
            obs["robot_eef_pose"] = np.array(
                [robot_state["TCPPose"]] * obs["timestamp"].shape[0]
            )
            print("\nUpdated observation keys:", list(obs.keys()))
            print("\n")

            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_obs_dict(env_obs=obs, shape_meta=shape_meta)

                # The policy may expect different formats of the observation
                # Create both formats to handle different model types
                if "obs" not in obs_dict_np:
                    # Format 1: With 'obs' wrapper
                    obs_dict_with_wrapper = {"obs": obs_dict_np}

                    # Format 2: Direct features (no wrapper)
                    obs_dict_direct = obs_dict_np

                    # Convert both to torch tensors
                    obs_dict_with_wrapper_torch = dict_apply(
                        obs_dict_with_wrapper,
                        lambda x: torch.from_numpy(x).unsqueeze(0).to(device),
                    )
                    obs_dict_direct_torch = dict_apply(
                        obs_dict_direct,
                        lambda x: torch.from_numpy(x).unsqueeze(0).to(device),
                    )

                    # Try both formats
                    try:
                        # First try with 'obs' wrapper
                        result = policy.predict_action(obs_dict_with_wrapper_torch)
                        print("Model expects 'obs' wrapper")
                    except (KeyError, AttributeError) as e:
                        # If that fails, try direct features
                        print(f"First approach failed with: {e}")
                        print("Trying direct observation features instead")
                        result = policy.predict_action(obs_dict_direct_torch)
                        print("Model expects direct features (no 'obs' wrapper)")
                else:
                    # Normal case: obs_dict_np already has 'obs' key
                    obs_dict = dict_apply(
                        obs_dict_np,
                        lambda x: torch.from_numpy(x).unsqueeze(0).to(device),
                    )
                    result = policy.predict_action(obs_dict)

                action = result["action"][0].detach().to("cpu").numpy()
                assert action.shape[-1] == 6  # xy position
                del result

            time.sleep(1)
            logger.info("System initialized and ready!")

            while True:
                # ========= Human control loop ==========
                logger.info("Human in control!")
                state = env.get_robot_state()
                print("STATE: ", state.keys())
                target_pose = np.array(state["TCPPose"], dtype=np.float32)
                logger.info(f"Initial pose: {target_pose}")

                t_start = time.monotonic()
                iter_idx = 0
                stop = False
                is_recording = False

                try:
                    while not stop:
                        # Calculate timing
                        t_cycle_end = t_start + (iter_idx + 1) * dt
                        t_command_target = t_cycle_end + dt
                        t_sample = t_cycle_end - command_latency

                        # Get observations
                        obs = env.get_obs()

                        # Add robot state
                        robot_state = env.get_robot_state()
                        obs["robot_eef_pose"] = np.array(
                            [robot_state["TCPPose"]] * obs["timestamp"].shape[0]
                        )

                        # Handle key presses
                        press_events = key_counter.get_press_events()

                        for key_stroke in press_events:
                            if key_stroke == KeyCode(char="q"):
                                logger.info("Quit requested...")
                                env.end_episode()
                                exit(0)
                            elif key_stroke == KeyCode(char="c"):
                                # Exit human control loop, hand control to policy
                                stop = True
                                break

                        stage_val = key_counter[Key.space]

                        # Visualize

                        precise_wait(t_sample)

                        # Get spacemouse state
                        sm_state = sm.get_motion_state_transformed()

                        dpos = sm_state[:3]
                        drot = sm_state[3:]
                        grasp = sm.grasp

                        # Check if movement is significant
                        input_magnitude = np.linalg.norm(dpos) + np.linalg.norm(drot)
                        significant_movement = (
                            input_magnitude > spacemouse_deadzone * 8.0
                        )
                        if significant_movement:
                            dpos *= xarm_config.position_gain
                            drot *= xarm_config.orientation_gain

                            curr_rot = st.Rotation.from_euler(
                                "xyz", target_pose[3:], degrees=True
                            )
                            delta_rot = st.Rotation.from_euler(
                                "xyz", drot, degrees=True
                            )
                            final_rot = delta_rot * curr_rot

                            target_pose[:3] += dpos
                            target_pose[3:] = final_rot.as_euler("xyz", degrees=True)

                            action = np.concatenate([target_pose, [grasp]])

                            exec_timestamp = (
                                t_command_target - time.monotonic() + time.time()
                            )

                            # Use interpolation if enabled
                            if use_interpolation:
                                env.exec_action_waypoints(
                                    actions=[action],
                                    timestamps=[exec_timestamp],
                                    stages=[stage_val],
                                )
                            else:
                                env.exec_actions(
                                    actions=[action],
                                    timestamps=[exec_timestamp],
                                    stages=[stage_val],
                                )
                            logger.debug(
                                "Significant movement detected, executing action."
                            )
                        else:
                            action = np.concatenate([target_pose, [grasp]])
                            exec_timestamp = (
                                t_command_target - time.monotonic() + time.time()
                            )

                            # Use interpolation if enabled
                            if use_interpolation:
                                env.exec_action_waypoints(
                                    actions=[action],
                                    timestamps=[exec_timestamp],
                                    stages=[stage_val],
                                )
                            else:
                                env.exec_actions(
                                    actions=[action],
                                    timestamps=[exec_timestamp],
                                    stages=[stage_val],
                                )
                            logger.debug("No significant movement detected.")

                        precise_wait(t_cycle_end)
                        iter_idx += 1

                    # ========== Policy control loop ==============
                    # Start policy evaluation
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)
                    # Wait for 1/30 sec to get the closest frame
                    frame_latency = 1 / 30
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    logger.info("Policy evaluation started!")
                    iter_idx = 0
                    float("inf")
                    prev_target_pose = None
                    is_recording = True

                    while True:
                        # Calculate timing for policy control
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # Get observations
                        obs = env.get_obs()
                        obs_timestamps = obs["timestamp"]

                        # Add robot state
                        robot_state = env.get_robot_state()
                        obs["robot_eef_pose"] = np.array(
                            [robot_state["TCPPose"]] * obs["timestamp"].shape[0]
                        )

                        logger.debug(f"Obs latency {time.time() - obs_timestamps[-1]}")

                        # Check for key presses during policy control
                        press_events = key_counter.get_press_events()
                        for key_stroke in press_events:
                            if key_stroke == KeyCode(char="s"):
                                # Stop episode, hand control back to human
                                env.end_episode()
                                is_recording = False
                                logger.info("Policy evaluation stopped.")
                                break
                            elif key_stroke == Key.backspace:
                                if click.confirm(
                                    "Drop the most recently recorded episode?"
                                ):
                                    env.drop_episode()
                                    is_recording = False
                                    logger.info("Episode dropped.")

                        if not is_recording:
                            break

                        # Run policy inference
                        with torch.no_grad():
                            s = time.time()
                            obs_dict_np = get_real_obs_dict(
                                env_obs=obs, shape_meta=shape_meta
                            )

                            # Handle different model architectures
                            if "obs" not in obs_dict_np:
                                # Format 1: With 'obs' wrapper
                                obs_dict_with_wrapper = {"obs": obs_dict_np}

                                # Format 2: Direct features (no wrapper)
                                obs_dict_direct = obs_dict_np

                                # Convert both to torch tensors
                                obs_dict_with_wrapper_torch = dict_apply(
                                    obs_dict_with_wrapper,
                                    lambda x: torch.from_numpy(x)
                                    .unsqueeze(0)
                                    .to(device),
                                )
                                obs_dict_direct_torch = dict_apply(
                                    obs_dict_direct,
                                    lambda x: torch.from_numpy(x)
                                    .unsqueeze(0)
                                    .to(device),
                                )

                                # Try both formats, using what we learned in the initialization
                                try:
                                    # First try with 'obs' wrapper
                                    result = policy.predict_action(
                                        obs_dict_with_wrapper_torch
                                    )
                                except (KeyError, AttributeError):
                                    # If that fails, try direct features
                                    result = policy.predict_action(
                                        obs_dict_direct_torch
                                    )
                            else:
                                # Normal case: obs_dict_np already has 'obs' key
                                obs_dict = dict_apply(
                                    obs_dict_np,
                                    lambda x: torch.from_numpy(x)
                                    .unsqueeze(0)
                                    .to(device),
                                )
                                result = policy.predict_action(obs_dict)

                            action = result["action"][0].detach().to("cpu").numpy()
                            print("ACTION: ", action)

                            # The action array only has 6 columns (not 7), so we can't check index 6
                            # Setting grasp to a fixed value since we have no grasp data in the actions
                            grasp = 0.0  # Default to no grasp

                            logger.debug(f"Inference latency: {time.time() - s}")

                        # Convert policy action to robot actions
                        if delta_action:
                            assert len(action) == 1
                            if prev_target_pose is None:
                                prev_target_pose = obs["robot_eef_pose"][-1]
                            this_target_pose = prev_target_pose.copy()
                            this_target_pose[[0, 1]] += action[-1]
                            prev_target_pose = this_target_pose
                            this_target_poses = np.expand_dims(this_target_pose, axis=0)
                        else:
                            # Get current robot state to use as base
                            robot_state = env.get_robot_state()
                            current_pose = np.array(
                                robot_state["TCPPose"], dtype=np.float32
                            )

                            # Convert model's predictions to robot's coordinate space
                            # Model was trained with delta actions, so we apply them with scaling
                            this_target_poses = np.zeros(
                                (len(action), len(current_pose)), dtype=np.float32
                            )

                            # Scale up the deltas for more substantial movement
                            # Apply different scaling for each axis
                            # Assuming X axis is forward direction
                            position_scale_x = 2.0  # Forward/backward scaling (X-axis)
                            position_scale_y = 1.0  # Left/right scaling (Y-axis)
                            position_scale_z = 1.0  # Up/down scaling (Z-axis)
                            rotation_scale = 1.0  # Rotation scaling

                            for i in range(len(action)):
                                # Apply scaled delta to current position
                                this_target_poses[i] = current_pose.copy()

                                # Apply different scaling to each position axis
                                this_target_poses[i, 0] += (
                                    action[i, 0] * position_scale_x
                                )  # X-axis (forward/backward)
                                this_target_poses[i, 1] += (
                                    action[i, 1] * position_scale_y
                                )  # Y-axis (left/right)
                                this_target_poses[i, 2] += (
                                    action[i, 2] * position_scale_z
                                )  # Z-axis (up/down)

                                # Apply rotation scaling
                                this_target_poses[i, 3:] += (
                                    action[i, 3:6] * rotation_scale
                                )

                                # Print delta for debugging
                                if i == 0:  # Just print the first action for clarity
                                    print("Current pose:", current_pose)
                                    print("Delta action (raw):", action[i, :6])
                                    print(
                                        "Delta action (scaled X,Y,Z):",
                                        [
                                            action[i, 0] * position_scale_x,
                                            action[i, 1] * position_scale_y,
                                            action[i, 2] * position_scale_z,
                                        ],
                                    )
                                    print("Target pose:", this_target_poses[i])

                        # Handle timing for actions
                        action_timestamps = (
                            np.arange(len(action), dtype=np.float64) + action_offset
                        ) * dt + obs_timestamps[-1]
                        action_exec_latency = 1
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)
                        if np.sum(is_new) == 0:
                            # Exceeded time budget, still do something
                            this_target_poses = this_target_poses[[-1]]
                            # Schedule on next available step
                            next_step_idx = int(
                                np.ceil((curr_time - eval_t_start) / dt)
                            )
                            action_timestamp = eval_t_start + (next_step_idx) * dt
                            logger.debug(f"Over budget: {action_timestamp - curr_time}")
                            action_timestamps = np.array([action_timestamp])
                        else:
                            this_target_poses = this_target_poses[is_new]
                            action_timestamps = action_timestamps[is_new]

                        # Execute actions
                        full_actions = []
                        for i in range(len(this_target_poses)):
                            # Add grasp parameter
                            full_action = np.concatenate(
                                [this_target_poses[i], [grasp]]
                            )
                            full_actions.append(full_action)

                        # Use interpolation if enabled
                        if use_interpolation:
                            env.exec_actions(
                                actions=full_actions,
                                timestamps=action_timestamps,
                                stages=[stage_val] * len(action_timestamps),
                            )
                        else:
                            for i in range(len(full_actions)):
                                env.exec_actions(
                                    actions=[full_actions[i]],
                                    timestamps=[action_timestamps[i]],
                                    stages=[stage_val],
                                )
                        logger.info(
                            f"Submitted {len(this_target_poses)} steps of actions."
                        )

                        # Visualize

                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += steps_per_inference

                except KeyboardInterrupt:
                    logger.info("Interrupted!")
                    env.end_episode()
                except Exception:
                    logger.error("Exception occurred during control loop:")
                    traceback.print_exc()
                finally:
                    if is_recording:
                        env.end_episode()
                    logger.info("Control loop ended. Returning to human control.")


if __name__ == "__main__":
    main()
