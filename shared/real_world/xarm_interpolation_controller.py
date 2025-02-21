import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import scipy.interpolate as si
import scipy.spatial.transform as st
import numpy as np

from xarm.wrapper import XArmAPI
from shared.utils.shared_memory.shared_memory_queue import SharedMemoryQueue, Empty
from shared.utils.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer

# We will keep PoseTrajectoryInterpolator in a separate file (shown below)


class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2


def wrap_angles_deg(angles_deg):
    """Wrap angles (deg) into [-180, 180)."""
    return ((angles_deg + 180) % 360) - 180


class xArmInterpolationController(mp.Process):
    """
    High-frequency controller for xArm7, streaming smooth Cartesian trajectories
    in real-time servo mode, while using shared memory for I/O.
    """

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        robot_ip="192.168.1.223",
        frequency=20,
        lookahead_time=0.1,
        gain=100,
        max_pos_speed=0.25,
        max_rot_speed=0.16,
        launch_timeout=3,
        tcp_offset_pose=None,
        payload_mass=None,
        payload_cog=None,
        joints_init=None,
        joints_init_speed=1.05,
        soft_real_time=False,
        verbose=False,
        receive_keys=None,
        get_max_k=128,
    ):
        super().__init__(name="xArmPositionalController")
        self.robot_ip = robot_ip
        self.frequency = frequency
        self.lookahead_time = lookahead_time
        self.gain = gain
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.launch_timeout = launch_timeout
        self.tcp_offset_pose = tcp_offset_pose
        self.payload_mass = payload_mass
        self.payload_cog = payload_cog
        self.joints_init = joints_init
        self.joints_init_speed = joints_init_speed
        self.soft_real_time = soft_real_time
        self.verbose = verbose

        # SharedMemoryQueue for commands
        example_cmd = {
            "cmd": Command.SERVOL.value,
            "target_pose": np.zeros(6, dtype=np.float64),
            "duration": 0.0,
            "target_time": 0.0,
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager, examples=example_cmd, buffer_size=256
        )

        # Build the ring buffer for states
        if receive_keys is None:
            receive_keys = ["ActualTCPPose", "ActualQ", "ActualQd"]
        example_state = {}
        try:
            # Probe the robot once to build an example
            arm = XArmAPI(robot_ip)
            arm.connect()
            ret, pose = arm.get_position(
                is_radian=False
            )  # pose in [mm, mm, mm, deg, deg, deg]
            if ret != 0:
                raise Exception(f"Error calling get_position (ret={ret}).")
            example_state["ActualTCPPose"] = np.array(pose[:6])

            ret, joint_states = arm.get_joint_states(is_radian=False)
            if ret != 0:
                raise Exception(f"Error calling get_joint_states (ret={ret}).")
            example_state["ActualQ"] = np.array(joint_states[0])  # deg
            example_state["ActualQd"] = np.array(joint_states[1])  # deg/s

            example_state["robot_receive_timestamp"] = time.time()
            arm.disconnect()
        except Exception as e:
            raise Exception(f"Failed to build ring buffer example state: {e}")

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example_state,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency,
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys

    # ========== Launch/Stop Methods ==========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(
                f"[xArmPositionalController] Controller process spawned at {self.pid}"
            )

    def stop(self, wait=True):
        message = {"cmd": Command.STOP.value}
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========== Context Manager ==========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========== Command Methods ==========
    def servoL(self, pose, duration=0.1):
        """
        Send a servoL command (move to pose in 'duration' seconds).
        `pose` is [x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg].
        """
        assert self.is_alive()
        assert len(pose) == 6
        assert duration >= (1 / self.frequency)

        message = {
            "cmd": Command.SERVOL.value,
            "target_pose": np.array(pose, dtype=np.float64),
            "duration": duration,
        }
        self.input_queue.put(message)

    def schedule_waypoint(self, pose, target_time):
        """
        Schedule a waypoint at absolute wall-clock `target_time` (seconds).
        `pose` is [x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg].
        """
        assert self.is_alive()
        assert len(pose) == 6
        assert target_time > time.time()

        message = {
            "cmd": Command.SCHEDULE_WAYPOINT.value,
            "target_pose": np.array(pose, dtype=np.float64),
            "target_time": float(target_time),
        }
        self.input_queue.put(message)

    # ========== State Retrieval ==========
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    # ========== The Real-Time Loop ==========
    def run(self):
        # Optionally set soft real-time scheduling
        if self.soft_real_time:
            try:
                os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(20))
            except PermissionError:
                if self.verbose:
                    print(
                        "[xArmPositionalController] Warning: no permission for SCHED_RR"
                    )

        # Connect to the xArm
        try:
            if self.verbose:
                print(f"[xArmPositionalController] Connect to robot: {self.robot_ip}")
            arm = XArmAPI(self.robot_ip)
            arm.connect()
            arm.clean_error()
            arm.clean_warn()
            ret = arm.motion_enable(True)
            if ret != 0:
                raise RuntimeError(f"Failed to enable motion: error {ret}")
            # set_mode(1) => servo motion mode
            ret = arm.set_mode(1)
            if ret != 0:
                raise RuntimeError(f"Failed to set mode=1 (servo) ret={ret}")
            ret = arm.set_state(0)
            if ret != 0:
                raise RuntimeError(f"Failed to set state=0 ret={ret}")

            if self.verbose:
                print(
                    f"[xArmPositionalController] Connected to xArm at: {self.robot_ip}"
                )
        except Exception as e:
            print(f"[xArmPositionalController] Connection error: {e}")
            self.ready_event.set()
            return

        dt = 1.0 / self.frequency

        # Initialize your trajectory with the current pose in axis-angle form
        ret, pose_full = arm.get_position(is_radian=False)
        if ret != 0 or pose_full is None:
            # fallback
            pose_full = [300, 0, 300, 0, 0, 0]  # mm, deg

        # Convert xArm's Euler-deg to axis-angle for internal storage
        # pose_full[:3] in mm, pose_full[3:6] in deg
        curr_xyz = np.array(pose_full[:3], dtype=np.float64)
        r_euler_deg = pose_full[3:6]
        r = st.Rotation.from_euler(
            "xyz", r_euler_deg, degrees=True
        )  # convert roll–pitch–yaw (deg) to Rotation
        curr_axis_angle = r.as_rotvec()  # [rx, ry, rz] axis-angle (rad)

        curr_pose = np.concatenate([curr_xyz, curr_axis_angle])
        curr_t = time.monotonic()

        from pose_trajectory_interpolator import (
            PoseTrajectoryInterpolator,
        )  # or wherever you put it

        pose_interp = PoseTrajectoryInterpolator(times=[curr_t], poses=[curr_pose])

        iter_idx = 0
        keep_running = True

        while keep_running:
            t_start = time.time()
            t_now = time.monotonic()

            # Get interpolated target pose in axis-angle
            pose_command = pose_interp(
                t_now
            )  # shape (6,) = [x_mm, y_mm, z_mm, ax, ay, az]
            xyz_cmd = pose_command[:3]
            axis_angle_cmd = pose_command[3:6]

            # Convert axis-angle back to Euler (roll, pitch, yaw in radians)
            r_euler = st.Rotation.from_rotvec(axis_angle_cmd).as_euler(
                "xyz", degrees=False
            )
            # So final command to xArm is [x_mm, y_mm, z_mm, roll_rad, pitch_rad, yaw_rad]
            pose_to_send = np.concatenate([xyz_cmd, r_euler])

            # Send the servo command
            # is_radian=True => first 3 in mm, last 3 in rad
            ret = arm.set_servo_cartesian(
                pose_to_send, is_radian=True, is_tool_coord=False, relative=False
            )
            if ret != 0 and self.verbose:
                print(f"[xArmPositionalController] set_servo_cartesian error: {ret}")

            # Read back robot state
            ret2, pose_full = arm.get_position(is_radian=False)
            if ret2 == 0 and pose_full is not None:
                # convert Euler-deg => axis-angle
                curr_xyz = np.array(pose_full[:3], dtype=np.float64)
                r_euler_deg = pose_full[3:6]
                r_now = st.Rotation.from_euler("xyz", r_euler_deg, degrees=True)
                curr_axis_angle = r_now.as_rotvec()
                curr_pose = np.concatenate([curr_xyz, curr_axis_angle])
            else:
                # fallback if error
                curr_pose = pose_command  # just assume we are at the last command

            # Store in ring buffer. We'll store exactly what we have in the same format.
            # The ring buffer example had "ActualTCPPose" in 6D, but we used the original pose format (mm, deg).
            # For consistency let's store the raw xArm read (mm, deg) + a timestamp:
            state_dict = {
                "ActualTCPPose": np.array(pose_full[:6], dtype=np.float64),
                "robot_receive_timestamp": time.time(),
            }
            # optionally also store joint states if you like:
            # ret3, joint_states = arm.get_joint_states(is_radian=False)
            # if ret3 == 0:
            #     state_dict["ActualQ"] = np.array(joint_states[0])  # deg
            #     state_dict["ActualQd"] = np.array(joint_states[1]) # deg/s

            self.ring_buffer.put(state_dict)

            # Drain the input_queue and process commands
            try:
                commands = self.input_queue.get_all()
                n_cmd = len(commands["cmd"])
            except Empty:
                n_cmd = 0

            for i in range(n_cmd):
                cmd_data = {k: commands[k][i] for k in commands}
                cmd_type = cmd_data["cmd"]

                if cmd_type == Command.STOP.value:
                    keep_running = False
                    break

                elif cmd_type == Command.SERVOL.value:
                    # Convert from [mm, mm, mm, deg, deg, deg] => axis-angle
                    target_pose = cmd_data["target_pose"]
                    duration = float(cmd_data["duration"])
                    # Convert orientation to axis-angle
                    r_target = st.Rotation.from_euler(
                        "xyz", target_pose[3:6], degrees=True
                    )
                    axis_angle_target = r_target.as_rotvec()
                    target_pose_axis = np.concatenate(
                        [target_pose[:3], axis_angle_target]
                    )

                    # Insert into the trajectory with an end time = t_now + duration
                    now_plus_dt = t_now + dt  # slightly in the future
                    final_time = now_plus_dt + duration
                    pose_interp = pose_interp.drive_to_waypoint(
                        pose=target_pose_axis,
                        time=final_time,
                        curr_time=now_plus_dt,
                        max_pos_speed=self.max_pos_speed,
                        max_rot_speed=self.max_rot_speed,
                    )
                    if self.verbose:
                        print(
                            f"[xArmPositionalController] servoL => target={target_pose}, duration={duration}s"
                        )

                elif cmd_type == Command.SCHEDULE_WAYPOINT.value:
                    target_pose = cmd_data["target_pose"]
                    t_target = float(cmd_data["target_time"])

                    # Convert orientation to axis-angle
                    r_target = st.Rotation.from_euler(
                        "xyz", target_pose[3:6], degrees=True
                    )
                    axis_angle_target = r_target.as_rotvec()
                    target_pose_axis = np.concatenate(
                        [target_pose[:3], axis_angle_target]
                    )

                    # Convert the user-specified real clock => monotonic clock
                    monotonic_target_time = time.monotonic() + (t_target - time.time())
                    now_plus_dt = t_now + dt
                    pose_interp = pose_interp.schedule_waypoint(
                        pose=target_pose_axis,
                        time=monotonic_target_time,
                        max_pos_speed=self.max_pos_speed,
                        max_rot_speed=self.max_rot_speed,
                        curr_time=now_plus_dt,
                        last_waypoint_time=pose_interp.times[-1],
                    )
                    if self.verbose:
                        print(
                            f"[xArmPositionalController] schedule_waypoint => target={target_pose}, at t={t_target}"
                        )

                else:
                    # Unknown command => stop
                    keep_running = False
                    break

            # Regulate loop frequency
            elapsed = time.time() - t_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            if iter_idx == 0:
                self.ready_event.set()  # signal "ready"
            iter_idx += 1

            if self.verbose:
                actual_freq = 1.0 / (time.time() - t_start)
                print(f"[xArmPositionalController] Actual freq: {actual_freq:.2f} Hz")

        # Cleanup
        arm.set_state(0)
        arm.disconnect()
        self.ready_event.set()
        if self.verbose:
            print(
                f"[xArmPositionalController] Disconnected from xArm at: {self.robot_ip}"
            )


def main(robot_ip="192.168.1.223", frequency=20, duration=3):
    """
    Minimal test of the xArmInterpolationController:
    1) Start the controller,
    2) Move to some "home pose" for 3 seconds,
    3) Shut down.
    """
    from multiprocessing.managers import SharedMemoryManager

    shm_manager = SharedMemoryManager()
    shm_manager.start()

    # Example "home pose" (in mm + deg) - pick something that is actually reachable
    home_pose = [
        0,
        0,
        0,
        0,
        0,
        0,
    ]  # x=300mm, y=0mm, z=300mm, roll=0, pitch=0, yaw=0 deg

    controller = xArmInterpolationController(
        shm_manager=shm_manager, robot_ip=robot_ip, frequency=frequency, verbose=True
    )

    controller.start(wait=True)
    print("Controller process started. Waiting for initialization...")
    time.sleep(1.5)

    print(f"Sending servoL to home_pose={home_pose} over {duration}s")
    controller.servoL(home_pose, duration=duration)

    time.sleep(duration + 1)  # wait for motion

    controller.stop(wait=True)
    print("Controller process stopped.")
    shm_manager.shutdown()


if __name__ == "__main__":
    main()
