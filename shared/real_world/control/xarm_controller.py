import time
import enum
import multiprocessing as mp
import scipy.spatial.transform as st
import numpy as np
import logging

from typing import List
from xarm.wrapper import XArmAPI
from multiprocessing.managers import SharedMemoryManager
from shared.real_world.shared_memory.shared_memory_queue import SharedMemoryQueue, Empty
from shared.real_world.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from shared.real_world.control.spacemouse import Spacemouse
from dataclasses import dataclass, field


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
    home_on_shutdown: bool = True
    home_speed: float = 50.0
    tcp_maxacc: int = 5000
    verbose: bool = False  # Switch off


class XArm:
    """
    This XArm object is a wrapper around the xArm API.
    It is not thread-safe and should not be used in a
    multi-threaded context. If you want to record data,
    please use the XArmController class instead. You can
    use this class if you just want to control the xArm
    directly.
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
        if self.config.home_on_shutdown:
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
    
    def set_gripper_position(self, position):
        code = self.arm.set_gripper_position(position, wait=False)
        if code != 0:
            logger.error(f"Error in set_gripper_position: {code}")
            raise RuntimeError(f"Error in set_gripper_position: {code}")
        
    def move_joints(self, joints):
        code = self.arm.set_servo_angle(
            angle=joints, speed=self.config.home_speed, wait=True
        )
        if code != 0:
            logger.error(f"Error in set_servo_angle (moving joints): {code}")
            raise RuntimeError(f"Error in set_servo_angle (moving joints): {code}")

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()


class Command(enum.Enum):
    STOP = 0
    STEP = 1
    HOME = 2
    WAYPOINT = 3


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
        self.arm = None

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
            print("Setting arm")
            self.arm = arm
            print("Arm set")

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
                        
                        # Check for special clean_errors flag
                        if command.get("clean_errors", False):
                            logger.info("[XArmController] Cleaning errors and resetting state...")
                            arm.clean_error()
                            arm.clean_warn()
                            arm.motion_enable(True)
                            arm.set_mode(0)  # Set to position control mode first
                            arm.set_state(0)  # Set to ready state
                            time.sleep(0.1)
                            arm.set_mode(1)  # Set back to servo mode
                            arm.set_state(0)  # Set to ready state
                            time.sleep(0.1)
                    elif cmd == Command.HOME.value:
                        # Currently, there are some issues here. It is best to move closer
                        # to home before homing, otherwise it is *very* dangerous.
                        logger.info("[XArmController] Received HOME command.")
                        # First clean errors and ensure correct mode
                        arm.clean_error()
                        arm.clean_warn()
                        arm.motion_enable(True)
                        
                        # Proper mode switching for homing
                        arm.set_mode(0)  # Position control mode
                        arm.set_state(0)  # Ready state
                        time.sleep(0.2)  # Give time for mode change to take effect
                        
                        code = arm.set_gripper_position(850, wait=False)
                        if code != 0:
                            logger.error(
                                f"Error in set_gripper_position (HOME open): {code}"
                            )
                        code = arm.set_servo_angle(
                            angle=self.home_pos, speed=self.home_speed, wait=True
                        )
                        
                        # Reset back to servo mode after homing
                        time.sleep(0.2)
                        arm.set_mode(1)  # Servo mode
                        arm.set_state(0)  # Ready state
                        
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


if __name__ == "__main__":
    main()