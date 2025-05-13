import time
import enum
import multiprocessing as mp
import scipy.spatial.transform as st
import numpy as np
import traceback
import logging

from typing import List
from shared.real_world.xarm.wrapper import XArmAPI
from multiprocessing.managers import SharedMemoryManager
from shared.real_world.shared_memory.shared_memory_queue import SharedMemoryQueue, Empty
from shared.real_world.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from dataclasses import dataclass, field
from shared.real_world.control.spacemouse import Spacemouse
from shared.real_world.control.pose_trajectory_interpolator import PoseTrajectoryInterpolator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class XArmConfig:
    robot_ip: str = "192.168.1.223"
    frequency: int = 30
    position_gain: float = 2.0
    orientation_gain: float = 2.0
    home_pos: List[int] = field(default_factory=lambda: [0, 0, 0, 70, 0, 70, 0])
    home_speed: float = 50.0
    tcp_maxacc: int = 5000
    verbose: bool = False  # switch off


class XArm:
    """
    At this point, this is legacy code. If you want to record
    data, please use the multiprocessing-emabled XArmController.
    """

    def __init__(self, xarm_config: XArmConfig):
        self.config = xarm_config
        self.init = False

        self.current_position = None
        self.current_orientation = None
        self.previous_grasp = 0.0

        if self.config.verbose:
            logger.setLevel(logging.DEBUG)

    @property
    def is_ready(self):
        return self.init

    def initialize(self):
        self.arm = XArmAPI(self.config.robot_ip)
        arm = self.arm
        arm.connect()
        arm.clean_error()
        arm.clean_warn()

        code = arm.motion_enable(enable=True)
        if code != 0:
            logger.error(f"Error in motion_enable: {code}")
            raise RuntimeError(f"Error in motion_enable: {code}")

        arm.set_tcp_maxacc(self.config.tcp_maxacc)

        code = arm.set_mode(1)
        if code != 0:
            logger.error(f"Error in set_mode: {code}")
            raise RuntimeError(f"Error in set_mode: {code}")

        code = arm.set_state(0)
        if code != 0:
            logger.error(f"Error in set_state: {code}")
            raise RuntimeError(f"Error in set_state: {code}")

        code, state = arm.get_state()
        if code != 0:
            logger.error(f"Error getting robot state: {code}")
            raise RuntimeError(f"Error getting robot state: {code}")
        if state != 0:
            logger.error(f"Robot is not ready to move. Current state: {state}")
            raise RuntimeError(f"Robot is not ready to move. Current state: {state}")
        else:
            logger.info(f"Robot is ready to move. Current state: {state}")

        err_code, warn_code = arm.get_err_warn_code()
        if err_code != 0 or warn_code != 0:
            logger.error(
                f"Error code: {err_code}, Warning code: {warn_code}. Cleaning error and warning."
            )
            arm.clean_error()
            arm.clean_warn()
            arm.motion_enable(enable=True)
            arm.set_state(0)

        code = arm.set_gripper_mode(0)
        if code != 0:
            logger.error(f"Error in set_gripper_mode: {code}")
            raise RuntimeError(f"Error in set_gripper_mode: {code}")

        code = arm.set_gripper_enable(True)
        if code != 0:
            logger.error(f"Error in set_gripper_enable: {code}")
            raise RuntimeError(f"Error in set_gripper_enable: {code}")

        code = arm.set_gripper_speed(1000)
        if code != 0:
            logger.error(f"Error in set_gripper_speed: {code}")
            raise RuntimeError(f"Error in set_gripper_speed: {code}")

        self.init = True
        time.sleep(3)
        self.home()
        time.sleep(3)
        logger.info("Successfully initialized xArm.")

    def shutdown(self):
        if not self.init:
            logger.error("shutdown() called on an uninitialized xArm.")
            return
        self.home()
        self.arm.disconnect()
        logger.info("xArm shutdown complete.")

    def home(self):
        logger.info("Homing robot.")
        if not self.init:
            logger.error("xArm not initialized.")
            raise RuntimeError("xArm not initialized.")

        arm = self.arm
        arm.set_mode(0)
        arm.set_state(0)
        code = arm.set_gripper_position(850, wait=False)
        if code != 0:
            logger.error(f"Error in set_gripper_position (open, homing): {code}")
            raise RuntimeError(f"Error in set_gripper_position (open, homing): {code}")

        code = arm.set_servo_angle(
            angle=self.config.home_pos, speed=self.config.home_speed, wait=True
        )
        if code != 0:
            logger.error(f"Error in set_servo_angle (homing): {code}")
            raise RuntimeError(f"Error in set_servo_angle (homing): {code}")
        arm.set_mode(1)
        arm.set_state(0)

        code, pose = arm.get_position()
        if code != 0:
            logger.error(f"Failed to query initial pose: {code}")
            raise RuntimeError(f"Failed to query initial pose: {code}")
        else:
            self.current_position = np.array(pose[:3])
            self.current_orientation = np.array(pose[3:])
            logger.debug(
                f"Initial pose set: pos={self.current_position}, ori={self.current_orientation}"
            )

    def step(self, dpos, drot, grasp):
        if not self.init:
            logger.error("xArm not initialized. Use it in a 'with' block")
            raise RuntimeError("xArm not initialized. Use it in a 'with' block")

        dpos *= self.config.position_gain
        drot *= self.config.orientation_gain

        curr_rot = st.Rotation.from_euler("xyz", self.current_orientation, degrees=True)
        delta_rot = st.Rotation.from_euler("xyz", drot, degrees=True)
        final_rot = delta_rot * curr_rot

        self.current_orientation = final_rot.as_euler("xyz", degrees=True)
        self.current_position += dpos

        logger.debug(f"Current position: {self.current_position}")
        logger.debug(f"Current orientation: {self.current_orientation}")

        code = self.arm.set_servo_cartesian(
            np.concatenate((self.current_position, self.current_orientation)),
            is_radian=False,
        )
        if code != 0:
            logger.error(f"Error in set_servo_cartesian in step(): {code}")
            raise RuntimeError(f"Error in set_servo_cartesian in step(): {code}")

        if grasp != self.previous_grasp:
            if grasp == 1.0:
                code = self.arm.set_gripper_position(0, wait=False)
                if code != 0:
                    logger.error(f"Error in set_gripper_position (close): {code}")
                    raise RuntimeError(f"Error in set_gripper_position (close): {code}")
            else:
                code = self.arm.set_gripper_position(850, wait=False)
                if code != 0:
                    logger.error(f"Error in set_gripper_position (open): {code}")
                    raise RuntimeError(f"Error in set_gripper_position (open): {code}")
            self.previous_grasp = grasp

    def get_state(self):
        state = {}

        code, actual_pose = self.arm.get_position(is_radian=False)
        if code != 0:
            logger.error(f"Error getting TCP pose: code {code}")
            raise RuntimeError(f"Error getting TCP pose: code {code}")
        state["ActualTCPPose"] = actual_pose

        actual_tcp_speed = self.arm.realtime_tcp_speed()
        state["ActualTCPSpeed"] = actual_tcp_speed

        code, actual_angles = self.arm.get_servo_angle(is_radian=False)
        if code != 0:
            logger.error(f"Error getting joint angles: code {code}")
            raise RuntimeError(f"Error getting joint angles: code {code}")
        state["ActualQ"] = actual_angles

        actual_joint_speeds = self.arm.realtime_joint_speeds()
        state["ActualQd"] = actual_joint_speeds

        return state

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()


class Command(enum.Enum):
    STOP = 0
    STEP = 1
    HOME = 2
    SCHEDULE_WAYPOINT = 3
    # May add more commands here. e.g. SCHEDULE_WAYPOINT


class XArmController(mp.Process):
    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        xarm_config: XArmConfig,
    ):
        super().__init__(name="XArmController")

        self.robot_ip = xarm_config.robot_ip
        self.frequency = xarm_config.frequency
        self.position_gain = xarm_config.position_gain
        self.orientation_gain = xarm_config.orientation_gain
        self.home_pos = xarm_config.home_pos
        self.home_speed = xarm_config.home_speed
        self.tcp_maxacc = xarm_config.tcp_maxacc
        self.verbose = xarm_config.verbose

        if self.verbose:
            logger.setLevel(logging.DEBUG)

        # Events for synchronization
        self.ready_event = mp.Event()
        self.stop_event = mp.Event()

        # Build Input Queue
        queue_example = {
            "cmd": Command.STEP.value,
            "target_pose": np.zeros(6, dtype=np.float64),
            "grasp": 0.0,
            "duration": 0.0,
            "target_time": 0.0,
        }
        self.input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager, examples=queue_example, buffer_size=256
        )

        # Build Ring Buffer.
        try:
            arm_temp = XArmAPI(self.robot_ip)
            arm_temp.connect()
            arm_temp.clean_error()
            arm_temp.clean_warn()
            arm_temp.set_tcp_maxacc(xarm_config.tcp_maxacc)
            code = arm_temp.motion_enable(True)
            if code != 0:
                raise RuntimeError(f"motion_enable error: {code}")
            code = arm_temp.set_mode(1)
            if code != 0:
                raise RuntimeError(f"set_mode error: {code}")
            code = arm_temp.set_state(0)
            if code != 0:
                raise RuntimeError(f"set_state error: {code}")

            state_example = {}

            # Get TCPPose: use get_position.
            code, pos = arm_temp.get_position(is_radian=False)
            if code == 0:
                state_example["TCPPose"] = np.array(pos[:6], dtype=np.float64)
            else:
                state_example["TCPPose"] = np.zeros(6, dtype=np.float64)

            # Get TCPSpeed: use realtime_tcp_speed.
            try:
                if callable(arm_temp.realtime_tcp_speed):
                    tcp_speed = arm_temp.realtime_tcp_speed()
                else:
                    tcp_speed = arm_temp.realtime_tcp_speed
                state_example["TCPSpeed"] = np.array(tcp_speed, dtype=np.float64)
            except Exception:
                state_example["TCPSpeed"] = np.zeros(6, dtype=np.float64)

            # Get JointAngles: use get_servo_angle()
            code, angles = arm_temp.get_servo_angle(is_radian=False)
            if code == 0:
                state_example["JointAngles"] = np.array(angles, dtype=np.float64)
            else:
                state_example["JointAngles"] = np.zeros(7, dtype=np.float64)

            # Get JointSpeeds: handle callable or value directly.
            try:
                if callable(arm_temp.realtime_joint_speeds):
                    joint_speeds = arm_temp.realtime_joint_speeds()
                else:
                    joint_speeds = arm_temp.realtime_joint_speeds
                state_example["JointSpeeds"] = np.array(joint_speeds, dtype=np.float64)
            except Exception:
                state_example["JointSpeeds"] = np.zeros(7, dtype=np.float64)

            # Robot timestamp (absolute for now).
            state_example["robot_receive_timestamp"] = time.time()

            # Initialize our grasp state.
            self.previous_grasp = 0.0
            state_example["Grasp"] = self.previous_grasp

            self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
                shm_manager=shm_manager,
                examples=state_example,
                get_max_k=128,
                get_time_budget=0.2,
                put_desired_frequency=self.frequency,
            )

            # Disconnect the temporary connection; the main loop will reconnect.
            arm_temp.disconnect()

        except Exception as e:
            logger.error(f"Error during initial state fetch: {e}")
            raise e

        # Store the last target pose; initialize it from the example.
        self.last_target_pose = state_example["TCPPose"]

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def start(self, wait=True):
        super().start()
        if wait:
            self.ready_event.wait(3)
            assert self.is_alive(), "XArmController did not start correctly."
        logger.debug(f"[XArmController] Process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {"cmd": Command.STOP.value}
        self.input_queue.put(message)
        self.stop_event.set()
        if wait:
            self.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def get_state(self, k=None):
        if k is None:
            logger.debug("[XArmController] In get_state(), k is None")
            return self.ring_buffer.get()
        else:
            return self.ring_buffer.get_last_k(k)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    def step(self, pose, grasp):
        assert self.is_alive()
        pose = np.array(pose)
        assert pose.shape == (6,)

        cmd = {
            "cmd": Command.STEP.value,
            "target_pose": pose,
            "grasp": grasp,
            "duration": 0.02,
            "target_time": time.time() + 0.02,
        }

        self.input_queue.put(cmd)

    def run(self):
        try:
            logger.info(f"[XArmController] Connecting to xArm at {self.robot_ip}")
            arm = XArmAPI(self.robot_ip)
            arm.connect()
            arm.clean_error()
            arm.clean_warn()
            arm.set_tcp_maxacc(self.tcp_maxacc)

            code = arm.motion_enable(True)
            if code != 0:
                raise RuntimeError(f"[XArmController] motion_enable error: {code}")
            code = arm.set_mode(1)
            if code != 0:
                raise RuntimeError(f"[XArmController] set_mode error: {code}")
            code = arm.set_state(0)
            if code != 0:
                raise RuntimeError(f"[XArmController] set_state error: {code}")

            code, pos = arm.get_position(is_radian=False)
            if code == 0:
                self.last_target_pose = np.array(pos[:6], dtype=np.float64)
            else:
                logger.error(
                    "[XArmController] Failed to get initial position; defaulting to zeros."
                )
                self.last_target_pose = np.zeros(6, dtype=np.float64)

            start_time = time.time()
            self.ready_event.set()

            dt = 1.0 / self.frequency
            iter_idx = 0

            while not self.stop_event.is_set():
                grasp = self.previous_grasp
                t_start = time.time()

                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands["cmd"])
                except Empty:
                    n_cmd = 0

                for i in range(n_cmd):
                    command = {key: commands[key][i] for key in commands}
                    cmd = command["cmd"]
                    if cmd == Command.STOP.value:
                        logger.debug("[XArmController] Received STOP command.")
                        self.stop_event.set()
                        break
                    elif cmd == Command.STEP.value:
                        target_pose = np.array(command["target_pose"], dtype=np.float64)
                        grasp = command["grasp"]
                        self.last_target_pose = target_pose
                        logger.debug(f"[XArmController] New target pose: {target_pose}")
                    elif cmd == Command.HOME.value:
                        # Currently, there are some issues here. It is best to move closer
                        # to home before homing, otherwise it is *very* dangerous.
                        logger.info("[XArmController] Received HOME command.")
                        arm.set_mode(0)
                        arm.set_state(0)
                        code = arm.set_gripper_position(850, wait=False)
                        if code != 0:
                            logger.error(
                                f"Error in set_gripper_position (HOME open): {code}"
                            )
                        code = arm.set_servo_angle(
                            angle=self.home_pos, speed=self.home_speed, wait=True
                        )
                        arm.set_mode(1)
                        arm.set_state(0)
                        code, pos = arm.get_position(is_radian=False)
                        if code == 0:
                            self.last_target_pose = np.array(pos[:6], dtype=np.float64)
                    else:
                        logger.error(f"[XArmController] Unknown command: {cmd}")

                # If the last command wasn't STOP or HOME, we do a servo step
                code = arm.set_servo_cartesian(
                    list(self.last_target_pose), is_radian=False
                )

                # Update gripper.
                if grasp != self.previous_grasp:
                    if grasp == 1.0:
                        code = arm.set_gripper_position(0, wait=False)
                        if code != 0:
                            logger.error(
                                f"Error in set_gripper_position (close): {code}"
                            )
                            raise RuntimeError(
                                f"Error in set_gripper_position (close): {code}"
                            )
                    else:
                        code = arm.set_gripper_position(850, wait=False)
                        if code != 0:
                            logger.error(
                                f"Error in set_gripper_position (open): {code}"
                            )
                            raise RuntimeError(
                                f"Error in set_gripper_position (open): {code}"
                            )
                    self.previous_grasp = grasp

                if code != 0:
                    logger.error(f"[XArmController] set_servo_cartesian error: {code}")

                # Update robot state
                state = {}
                code, pos = arm.get_position(is_radian=False)
                if code == 0:
                    state["TCPPose"] = np.array(pos[:6], dtype=np.float64)
                else:
                    state["TCPPose"] = np.zeros(6, dtype=np.float64)

                try:
                    if callable(arm.realtime_tcp_speed):
                        tcp_speed = arm.realtime_tcp_speed()
                    else:
                        tcp_speed = arm.realtime_tcp_speed
                    state["TCPSpeed"] = np.array(tcp_speed, dtype=np.float64)
                except Exception as e:
                    logger.error(f"Error in realtime_tcp_speed: {e}")
                    state["TCPSpeed"] = np.zeros(6, dtype=np.float64)

                code, angles = arm.get_servo_angle(is_radian=False)
                if code == 0:
                    state["JointAngles"] = np.array(angles, dtype=np.float64)
                else:
                    state["JointAngles"] = np.zeros(7, dtype=np.float64)

                try:
                    if callable(arm.realtime_joint_speeds):
                        joint_speeds = arm.realtime_joint_speeds()
                    else:
                        joint_speeds = arm.realtime_joint_speeds
                    state["JointSpeeds"] = np.array(joint_speeds, dtype=np.float64)
                except Exception as e:
                    logger.error(f"Error in realtime_joint_speeds: {e}")
                    state["JointSpeeds"] = np.zeros(7, dtype=np.float64)

                state["Grasp"] = self.previous_grasp
                state["robot_receive_timestamp"] = time.time() - start_time

                # Update ring buffer (data)
                self.ring_buffer.put(state)

                elapsed = time.time() - t_start
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                iter_idx += 1
                logger.debug(
                    f"[XArmController] Iteration {iter_idx} at {1.0/(time.time()-t_start):.2f} Hz"
                )
        except Exception as e:
            logger.error(f"[XArmController] Exception in control loop: {e}")
        finally:
            try:
                arm.set_mode(0)
                arm.set_state(0)
                arm.disconnect()
                logger.info(
                    f"[XArmController] Disconnected from xArm at {self.robot_ip}"
                )
            except Exception as e:
                logger.error(f"[XArmController] Cleanup error: {e}")
            self.ready_event.set()


def main():
    with SharedMemoryManager() as shm_manager, Spacemouse(deadzone=0.4) as sm:
        xarm_config = XArmConfig()
        xarm_ctrl = XArmController(
            shm_manager=shm_manager,
            xarm_config=xarm_config,
        )
        xarm_ctrl.start(wait=True)
        print("XArmController started and ready.")

        # Keep our local "target_pose" so orientation accumulates properly
        current_target_pose = xarm_ctrl.last_target_pose.copy()

        last_timestamp = None
        try:
            while True:
                loop_start = time.monotonic()

                sm_state = sm.get_motion_state_transformed()
                dpos = sm_state[:3]
                drot = sm_state[3:]
                grasp = sm.grasp

                # Right button -> HOME
                # DO NOT MOVE THE SPACEMOUSE WHILE CLICKING THIS.
                # The HOME command is buggy and best avoided...
                """
                if sm.is_button_pressed(1):
                    command = {
                        "cmd": Command.HOME.value,
                        "target_pose": np.zeros(6, dtype=np.float64),
                        "grasp": 0.0,
                        "duration": 0.0,
                        "target_time": time.time(),
                    }
                    xarm_ctrl.input_queue.put(command)

                    time.sleep(1.0)
                    updated_state = xarm_ctrl.get_state()
                    new_pose = updated_state["TCPPose"]
                    current_target_pose[:] = new_pose
                    continue
                """

                dpos *= xarm_ctrl.position_gain
                drot *= xarm_ctrl.orientation_gain

                curr_orientation = current_target_pose[3:]
                curr_rot = st.Rotation.from_euler("xyz", curr_orientation, degrees=True)
                delta_rot = st.Rotation.from_euler("xyz", drot, degrees=True)
                final_rot = delta_rot * curr_rot

                current_target_pose[:3] += dpos
                current_target_pose[3:] = final_rot.as_euler("xyz", degrees=True)

                # This is a workaround that does not use home_pos
                # This will have to be fixed later.
                # This is also like insanely dangerous...
                """
                if sm.is_button_pressed(1):
                    current_target_pose = [
                        475.791901,
                        -1.143693,
                        244.719421,
                        179.132906,
                        -0.010084,
                        0.77567,
                    ]
                """

                command = {
                    "cmd": Command.STEP.value,
                    "target_pose": current_target_pose,
                    "grasp": grasp,
                    "duration": 0.02,
                    "target_time": time.time() + 0.02,
                }
                xarm_ctrl.input_queue.put(command)

                # Check the ring buffer to see if the child updated
                state = xarm_ctrl.get_state(k=1)
                logger.debug(f"Most recent state: {state}")
                ts = state.get("robot_receive_timestamp")[0]
                if ts != last_timestamp:
                    logger.debug(f"Ring buffer updated, time: {ts:.3f}")
                    last_timestamp = ts

                elapsed = time.monotonic() - loop_start
                time.sleep(max(0, 0.02 - elapsed))
        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            xarm_ctrl.stop(wait=True)
            print("XArmController stopped.")


class XArmInterpolationController(mp.Process):
    """
    Controller for xArm that provides smooth motion through trajectory interpolation.
    Similar to RTDEInterpolationController but adapted for xArm API.
    """
    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        xarm_config,  # Use XArmConfig from xarm_controller.py
        max_pos_speed=0.05,  # m/s - use a lower default speed for safety
        max_rot_speed=0.2,   # degrees/s - use a lower default speed for safety
        launch_timeout=3,
        soft_real_time=False,
        verbose=False,
    ):
        # Tested different max_pos_speed, max_rot_speed, still issue
        super().__init__(name="XArmInterpolationController")
        
        # Store configuration
        self.robot_ip = xarm_config.robot_ip
        self.frequency = xarm_config.frequency
        self.position_gain = xarm_config.position_gain
        self.orientation_gain = xarm_config.orientation_gain
        self.home_pos = xarm_config.home_pos
        self.home_speed = xarm_config.home_speed
        self.tcp_maxacc = xarm_config.tcp_maxacc
        
        # Additional parameters
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.launch_timeout = launch_timeout
        self.soft_real_time = soft_real_time
        self.verbose = verbose

        if self.verbose:
            logging.getLogger(__name__).setLevel(logging.DEBUG)
            
        # Set up logger
        self.logger = logging.getLogger(__name__)
        
        # Events for synchronization
        self.ready_event = mp.Event()
        self.stop_event = mp.Event()
        
        # Build Input Queue
        queue_example = {
            "cmd": Command.STEP.value,
            "target_pose": np.zeros(6, dtype=np.float64),
            "grasp": 0.0,
            "duration": 0.0,
            "target_time": 0.0,
        }
        self.input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager, examples=queue_example, buffer_size=256
        )

        # Build Ring Buffer (similar to XArmController)
        try:
            arm_temp = XArmAPI(self.robot_ip)
            arm_temp.connect()
            arm_temp.clean_error()
            arm_temp.clean_warn()
            arm_temp.set_tcp_maxacc(xarm_config.tcp_maxacc)
            code = arm_temp.motion_enable(True)
            if code != 0:
                raise RuntimeError(f"motion_enable error: {code}")
            code = arm_temp.set_mode(1)
            if code != 0:
                raise RuntimeError(f"set_mode error: {code}")
            code = arm_temp.set_state(0)
            if code != 0:
                raise RuntimeError(f"set_state error: {code}")

            state_example = {}

            # Get TCPPose: use get_position.
            code, pos = arm_temp.get_position(is_radian=False)
            if code == 0:
                state_example["TCPPose"] = np.array(pos[:6], dtype=np.float64)
            else:
                state_example["TCPPose"] = np.zeros(6, dtype=np.float64)

            # Get TCPSpeed: use realtime_tcp_speed.
            try:
                if callable(arm_temp.realtime_tcp_speed):
                    tcp_speed = arm_temp.realtime_tcp_speed()
                else:
                    tcp_speed = arm_temp.realtime_tcp_speed
                state_example["TCPSpeed"] = np.array(tcp_speed, dtype=np.float64)
            except Exception:
                state_example["TCPSpeed"] = np.zeros(6, dtype=np.float64)

            # Get JointAngles: use get_servo_angle()
            code, angles = arm_temp.get_servo_angle(is_radian=False)
            if code == 0:
                state_example["JointAngles"] = np.array(angles, dtype=np.float64)
            else:
                state_example["JointAngles"] = np.zeros(7, dtype=np.float64)

            # Get JointSpeeds
            try:
                if callable(arm_temp.realtime_joint_speeds):
                    joint_speeds = arm_temp.realtime_joint_speeds()
                else:
                    joint_speeds = arm_temp.realtime_joint_speeds
                state_example["JointSpeeds"] = np.array(joint_speeds, dtype=np.float64)
            except Exception:
                state_example["JointSpeeds"] = np.zeros(7, dtype=np.float64)

            # Robot timestamp (absolute for now).
            state_example["robot_receive_timestamp"] = time.time()

            # Initialize grasp state.
            self.previous_grasp = 0.0
            state_example["Grasp"] = self.previous_grasp

            self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
                shm_manager=shm_manager,
                examples=state_example,
                get_max_k=128,
                get_time_budget=0.2,
                put_desired_frequency=self.frequency,
            )

            # Disconnect the temporary connection; the main loop will reconnect.
            arm_temp.disconnect()

        except Exception as e:
            self.logger.error(f"Error during initial state fetch: {e}")
            raise e

        # Store the last target pose; initialize it from the example.
        self.last_target_pose = state_example["TCPPose"]

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def start(self, wait=True):
        super().start()
        if wait:
            self.ready_event.wait(self.launch_timeout)
            assert self.is_alive(), "XArmInterpolationController did not start correctly."
        self.logger.debug(f"[XArmInterpolationController] Process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {"cmd": Command.STOP.value}
        self.input_queue.put(message)
        self.stop_event.set()
        if wait:
            self.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def get_state(self, k=None):
        if k is None:
            self.logger.debug("[XArmInterpolationController] In get_state(), k is None")
            return self.ring_buffer.get()
        else:
            return self.ring_buffer.get_last_k(k)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    def step(self, pose, grasp):
        """
        Execute a single step, moving directly to the specified pose.
        This is for backward compatibility with the original XArmController.
        """
        assert self.is_alive()
        pose = np.array(pose)
        assert pose.shape == (6,)

        cmd = {
            "cmd": Command.STEP.value,
            "target_pose": pose,
            "grasp": grasp,
            "duration": 0.02,
            "target_time": time.time() + 0.02,
        }

        self.input_queue.put(cmd)
        
    def schedule_waypoint(self, pose, target_time, grasp=None):
        """
        Schedule a waypoint for the robot to reach at the specified time.
        Uses interpolation for smooth motion.
        """
        assert self.is_alive()
        pose = np.array(pose)
        assert pose.shape == (6,)
        
        # Ensure target_time is in the future
        now = time.time()
        if target_time <= now:
            target_time = now + 0.05  # Small offset to ensure it's in the future
        
        cmd = {
            "cmd": Command.SCHEDULE_WAYPOINT.value,
            "target_pose": pose,
            "grasp": self.previous_grasp if grasp is None else grasp,
            "target_time": target_time,
        }
        
        self.input_queue.put(cmd)

    def run(self):
        """
        Main control loop for the XArmInterpolationController.
        Handles waypoint scheduling and trajectory interpolation.
        """
        if self.soft_real_time:
            # enable soft real-time if requested
            try:
                os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(20))
            except Exception as e:
                self.logger.warning(f"Failed to set real-time scheduler: {e}")
        
        try:
            self.logger.info(f"[XArmInterpolationController] Connecting to xArm at {self.robot_ip}")
            arm = XArmAPI(self.robot_ip)
            arm.connect()
            arm.clean_error()
            arm.clean_warn()
            arm.set_tcp_maxacc(self.tcp_maxacc)

            # Enable the robot
            code = arm.motion_enable(True)
            if code != 0:
                raise RuntimeError(f"[XArmInterpolationController] motion_enable error: {code}")
            
            # Set the robot mode (1 for servo mode)
            code = arm.set_mode(1)
            if code != 0:
                raise RuntimeError(f"[XArmInterpolationController] set_mode error: {code}")
            
            # Set the robot state (0 for ready)
            code = arm.set_state(0)
            if code != 0:
                raise RuntimeError(f"[XArmInterpolationController] set_state error: {code}")

            # Get the current position
            code, pos = arm.get_position(is_radian=False)
            if code == 0:
                self.last_target_pose = np.array(pos[:6], dtype=np.float64)
                self.logger.info(f"[XArmInterpolationController] Initial position: {self.last_target_pose}")
            else:
                self.logger.error(
                    "[XArmInterpolationController] Failed to get initial position; defaulting to zeros."
                )
                self.last_target_pose = np.zeros(6, dtype=np.float64)
        
            # Create a specialized policy controller
            policy_controller = PolicyExecutionController(arm, logger=self.logger)
        
            # Current control mode (DIRECT for human, POLICY for policy)
            control_mode = ControlMode.DIRECT
        
            # Initialize pose interpolator (for direct control)
            curr_t = time.monotonic()
            pose_interp = PoseTrajectoryInterpolator(
                times=np.array([curr_t]),
                poses=np.array([self.last_target_pose])
            )
        
            start_time = time.time()
            self.ready_event.set()

            dt = 1.0 / self.frequency
            iter_idx = 0
            grasp = self.previous_grasp
            last_reset_time = 0
        
            # Reset robot at the start to ensure clean state
            arm.clean_error()
            arm.clean_warn()
            arm.set_mode(1)
            arm.set_state(0)

            while not self.stop_event.is_set():
                t_start = time.time()
            
                # Get current time for interpolation
                t_now = time.monotonic()

                try:
                    # Process commands
                    try:
                        commands = self.input_queue.get_all()
                        n_cmd = len(commands["cmd"])
                    except Empty:
                        n_cmd = 0

                    # Process any incoming commands
                    for i in range(n_cmd):
                        command = {key: commands[key][i] for key in commands}
                        cmd = command["cmd"]
                    
                        if cmd == Command.STOP.value:
                            self.logger.debug("[XArmInterpolationController] Received STOP command.")
                            self.stop_event.set()
                            break
                        
                        elif cmd == Command.STEP.value:
                            # Direct step command - switch to DIRECT mode
                            control_mode = ControlMode.DIRECT
                        
                            target_pose = np.array(command["target_pose"], dtype=np.float64)
                            grasp = command["grasp"]
                            self.last_target_pose = target_pose
                        
                            # Create a new interpolator for direct control
                            pose_interp = PoseTrajectoryInterpolator(
                                times=np.array([t_now]),
                                poses=np.array([target_pose])
                            )
                        
                            # Reset policy controller
                            policy_controller.last_executed_pose = None
                            policy_controller.waypoints = []
                        
                            self.logger.debug(f"[XArmInterpolationController] STEP command, mode: DIRECT")
                        
                        elif cmd == Command.SCHEDULE_WAYPOINT.value:
                            # Schedule waypoint - switch to POLICY mode for smoother control
                            control_mode = ControlMode.POLICY
                        
                            target_pose = np.array(command["target_pose"], dtype=np.float64)
                            target_time = float(command["target_time"])
                            grasp = command.get("grasp", self.previous_grasp)
                        
                            # Add to policy controller
                            policy_controller.add_waypoint(target_pose, target_time)
                            self.last_target_pose = target_pose
                        
                            self.logger.debug(f"[XArmInterpolationController] Scheduling waypoint, mode: POLICY")
                        
                        elif cmd == Command.HOME.value:
                            # Home command - always use DIRECT mode after homing
                            control_mode = ControlMode.DIRECT
                        
                            self.logger.info("[XArmInterpolationController] Received HOME command.")
                            arm.set_mode(0)
                            arm.set_state(0)
                            code = arm.set_gripper_position(850, wait=False)
                            if code != 0:
                                self.logger.error(f"Error in set_gripper_position (HOME open): {code}")
                            
                            code = arm.set_servo_angle(
                                angle=self.home_pos, speed=self.home_speed, wait=True
                            )
                            arm.set_mode(1)
                            arm.set_state(0)
                            code, pos = arm.get_position(is_radian=False)
                            if code == 0:
                                self.last_target_pose = np.array(pos[:6], dtype=np.float64)
                            
                                # Reset the interpolator after homing
                                pose_interp = PoseTrajectoryInterpolator(
                                    times=np.array([t_now]),
                                    poses=np.array([self.last_target_pose])
                                )
                            
                                # Reset policy controller
                                policy_controller.last_executed_pose = None
                                policy_controller.waypoints = []
                        else:
                            self.logger.error(f"[XArmInterpolationController] Unknown command: {cmd}")

                    # Check for errors and reset if needed (every 2 seconds)
                    current_time = time.time()
                    if current_time - last_reset_time > 2.0:
                        code, state = arm.get_state()
                        if state != 0:  # Not in ready state
                            self.logger.warning(f"[XArmInterpolationController] Robot not ready, state: {state}")
                            arm.clean_error()
                            arm.clean_warn()
                            arm.set_mode(1)
                            arm.set_state(0)
                            last_reset_time = current_time

                    # Execute motion based on current control mode
                    if control_mode == ControlMode.DIRECT:
                        # Direct control mode - use interpolator
                        try:
                            pose_command = pose_interp(t_now)
                        
                            # Send the command to the robot
                            code = arm.set_servo_cartesian(
                                list(pose_command), is_radian=False
                            )
                        
                            if code != 0:
                                self.logger.error(f"[XArmInterpolationController] set_servo_cartesian error: {code}")
                                # Try to reset robot on error
                                arm.clean_error()
                                arm.clean_warn()
                                arm.set_mode(1)
                                arm.set_state(0)
                        except Exception as e:
                            self.logger.error(f"[XArmInterpolationController] Direct mode error: {e}")
                            self.logger.error(traceback.format_exc())
                        
                    elif control_mode == ControlMode.POLICY:
                        # Policy control mode - use specialized controller
                        policy_controller.process_waypoints()

                    # Update gripper state if needed
                    if grasp != self.previous_grasp:
                        if grasp == 1.0:
                            code = arm.set_gripper_position(0, wait=False)
                            if code != 0:
                                self.logger.error(f"Error in set_gripper_position (close): {code}")
                        else:
                            code = arm.set_gripper_position(850, wait=False)
                            if code != 0:
                                self.logger.error(f"Error in set_gripper_position (open): {code}")
                        self.previous_grasp = grasp

                    # Update robot state for the ring buffer
                    state = {}
                    code, pos = arm.get_position(is_radian=False)
                    if code == 0:
                        state["TCPPose"] = np.array(pos[:6], dtype=np.float64)
                    else:
                        state["TCPPose"] = np.zeros(6, dtype=np.float64)

                    try:
                        if callable(arm.realtime_tcp_speed):
                            tcp_speed = arm.realtime_tcp_speed()
                        else:
                            tcp_speed = arm.realtime_tcp_speed
                        state["TCPSpeed"] = np.array(tcp_speed, dtype=np.float64)
                    except Exception as e:
                        self.logger.error(f"Error in realtime_tcp_speed: {e}")
                        state["TCPSpeed"] = np.zeros(6, dtype=np.float64)

                    code, angles = arm.get_servo_angle(is_radian=False)
                    if code == 0:
                        state["JointAngles"] = np.array(angles, dtype=np.float64)
                    else:
                        state["JointAngles"] = np.zeros(7, dtype=np.float64)

                    try:
                        if callable(arm.realtime_joint_speeds):
                            joint_speeds = arm.realtime_joint_speeds()
                        else:
                            joint_speeds = arm.realtime_joint_speeds
                        state["JointSpeeds"] = np.array(joint_speeds, dtype=np.float64)
                    except Exception as e:
                        self.logger.error(f"Error in realtime_joint_speeds: {e}")
                        state["JointSpeeds"] = np.zeros(7, dtype=np.float64)

                    state["Grasp"] = self.previous_grasp
                    state["robot_receive_timestamp"] = time.time() - start_time

                    # Update ring buffer (data)
                    self.ring_buffer.put(state)

                except Exception as e:
                    self.logger.error(f"[XArmInterpolationController] Main loop exception: {e}")
                    self.logger.error(traceback.format_exc())
                
                    # Try to reset the robot state on error
                    try:
                        arm.clean_error()
                        arm.clean_warn()
                        arm.set_mode(1)
                        arm.set_state(0)
                    except:
                        pass

                # Sleep to maintain the desired frequency
                elapsed = time.time() - t_start
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    if elapsed > dt * 1.5:  # Only warn for significant overruns
                        self.logger.warning(f"[XArmInterpolationController] Loop took too long: {elapsed:.4f}s > {dt:.4f}s")
                
                iter_idx += 1
                if iter_idx % 100 == 0:
                    self.logger.debug(
                        f"[XArmInterpolationController] Iteration {iter_idx} at {1.0/(time.time()-t_start):.2f} Hz"
                    )
        except Exception as e:
            self.logger.error(f"[XArmInterpolationController] Fatal exception in control loop: {e}")
            self.logger.error(traceback.format_exc())
        finally:
            try:
                arm.set_mode(0)
                arm.set_state(0)
                arm.disconnect()
                self.logger.info(
                    f"[XArmInterpolationController] Disconnected from xArm at {self.robot_ip}"
                )
            except Exception as e:
                self.logger.error(f"[XArmInterpolationController] Cleanup error: {e}")
            self.ready_event.set()

class ControlMode(enum.Enum):
    """
    Control modes for the XArmInterpolationController
    """
    DIRECT = 0    # Direct control mode (used for human control)
    POLICY = 1    # Policy control mode (uses smoother motion and additional checks)


class PolicyExecutionController:
    """
    Specialized controller for policy execution with XArm
    Handles error states and provides smoother control with scaling
    """
    def __init__(self, xarm_api, logger=None):
        self.arm = xarm_api
        self.logger = logger or logging.getLogger(__name__)
        self.last_error_reset = 0
        self.error_reset_cooldown = 0.5  # seconds
        self.waypoints = []
        self.last_executed_pose = None
        self.motion_filter_alpha = 0.3  # Lower = smoother but laggier motion
        
        # Scaling factors to reduce movement amplitude
        self.position_scale = 1.0  # Scale position movements by 50%
        self.rotation_scale = 1.0  # Scale rotation movements by 30%
        
        # Center point for scaling (typically current robot position)
        self.center_position = None
        
    def reset_error_state(self):
        """
        Reset the robot error state if needed
        """
        current_time = time.time()
        if current_time - self.last_error_reset < self.error_reset_cooldown:
            return False
            
        self.logger.info("Resetting XArm error state")
        self.arm.clean_error()
        self.arm.clean_warn()
        
        # Reset servo mode
        self.arm.set_mode(1)
        self.arm.set_state(0)
        
        self.last_error_reset = current_time
        return True
        
    def update_center_position(self, position):
        """
        Update the center position around which scaling happens
        """
        if self.center_position is None:
            self.center_position = position
        else:
            # Slowly move the center point to adapt to policy drift
            alpha = 0.05  # Very slow adaptation
            self.center_position = alpha * position + (1 - alpha) * self.center_position
        
    def scale_movements(self, target_pose):
        """
        Scale movements to reduce amplitude
        """
        # Get current position if we don't have one
        if self.center_position is None:
            code, pos = self.arm.get_position(is_radian=False)
            if code == 0:
                self.center_position = np.array(pos[:3], dtype=np.float64)
            else:
                # If we can't get position, use the target as center
                self.center_position = target_pose[:3]
        
        # Create scaled pose
        scaled_pose = target_pose.copy()
        
        # Scale position (XYZ) relative to center position
        delta_pos = target_pose[:3] - self.center_position
        scaled_pose[:3] = self.center_position + delta_pos * self.position_scale
        
        # Get current orientation if we don't have executed pose
        if self.last_executed_pose is None:
            code, pos = self.arm.get_position(is_radian=False)
            if code == 0:
                self.last_executed_pose = np.array(pos[:6], dtype=np.float64)
            else:
                self.last_executed_pose = target_pose
        
        # Scale rotation relative to last executed pose
        delta_rot = target_pose[3:] - self.last_executed_pose[3:]
        # Handle angle wrapping (e.g., -179 vs +179 degrees)
        for i in range(3):
            if delta_rot[i] > 180:
                delta_rot[i] -= 360
            elif delta_rot[i] < -180:
                delta_rot[i] += 360
                
        scaled_pose[3:] = self.last_executed_pose[3:] + delta_rot * self.rotation_scale
        
        # Log the scaling effect
        self.logger.debug(f"Original pose: {target_pose[:3]}, Scaled: {scaled_pose[:3]}")
        
        return scaled_pose
        
    def filter_waypoint(self, target_pose):
        """
        Apply scaling and smoothing to target poses
        """
        # First scale the movements
        scaled_pose = self.scale_movements(target_pose)
        
        # Then apply smoothing filter
        if self.last_executed_pose is None:
            self.last_executed_pose = scaled_pose
            return scaled_pose
            
        # Simple EMA filter
        filtered_pose = scaled_pose.copy()
        alpha = self.motion_filter_alpha
        
        # Filter position (XYZ)
        filtered_pose[:3] = alpha * scaled_pose[:3] + (1-alpha) * self.last_executed_pose[:3]
        
        # Filter orientation 
        filtered_pose[3:] = alpha * scaled_pose[3:] + (1-alpha) * self.last_executed_pose[3:]
        
        # Update last executed pose
        self.last_executed_pose = filtered_pose
        
        # Update center position (slow drift)
        self.update_center_position(filtered_pose[:3])
        
        return filtered_pose
        
    def add_waypoint(self, pose, timestamp):
        """
        Add a waypoint to the queue
        """
        # Don't queue too many future waypoints
        if len(self.waypoints) >= 3:
            # Just keep the most recent one
            self.waypoints = [self.waypoints[-1]]
            
        # Add the new waypoint
        self.waypoints.append((pose, timestamp))
        
    def process_waypoints(self):
        """
        Process pending waypoints
        """
        if not self.waypoints:
            return
            
        current_time = time.time()
        
        # Remove waypoints that are in the past
        self.waypoints = [(pose, ts) for pose, ts in self.waypoints if ts > current_time]
        
        if not self.waypoints:
            return
            
        # Choose the oldest waypoint to execute
        pose, timestamp = self.waypoints[0]
        
        # Apply scaling and filtering
        processed_pose = self.filter_waypoint(pose)
        
        # Execute the waypoint
        try:
            code = self.arm.set_servo_cartesian(list(processed_pose), is_radian=False)
            if code != 0:
                self.logger.error(f"Policy controller: set_servo_cartesian error: {code}")
                # Reset on error
                self.reset_error_state()
                
                # If there's an error, clear all waypoints to prevent continued errors
                self.waypoints = []
        except Exception as e:
            self.logger.error(f"Policy controller execution error: {e}")
            self.reset_error_state()
            self.waypoints = []
            
        # Remove the executed waypoint only if successful
        if len(self.waypoints) > 0:
            self.waypoints.pop(0)


if __name__ == "__main__":
    main()