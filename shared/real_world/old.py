# xarm_interpolation_controller.py
import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import numpy as np
from xarm.wrapper import XArmAPI
from shared.utils.shared_memory.shared_memory_queue import SharedMemoryQueue, Empty
from shared.utils.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from shared.utils.pose_trajectory_interpolator import PoseTrajectoryInterpolator


class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2


class xArmInterpolationController(mp.Process):
    """
    This process sends continuous Cartesian commands to the xArm using
    set_servo_cartesian exclusively. An optional Cartesian initialization pose
    (initial_tcp_pose) can be provided.

    Note: For Cartesian commands, the pose should be a 6- or 7-element array:
          [x, y, z, rx, ry, rz] where rx,ry,rz are in degrees for API calls.
          Internally, the pose trajectory interpolator works in radians for orientation.
          For xArm7 you may supply 7-element arrays (the extra element will be passed through).
    """

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        robot_ip,
        frequency=125,
        lookahead_time=0.1,
        gain=300,
        max_pos_speed=0.25,  # m/s
        max_rot_speed=0.16,  # rad/s
        launch_timeout=3,
        tcp_offset_pose=None,
        initial_tcp_pose=None,
        soft_real_time=False,
        verbose=False,
        receive_keys=None,
        get_max_k=128,
    ):
        # Validate parameters
        assert 0 < frequency <= 500
        assert 0.03 <= lookahead_time <= 0.2
        assert 100 <= gain <= 2000
        assert max_pos_speed > 0
        assert max_rot_speed > 0
        if tcp_offset_pose is not None:
            tcp_offset_pose = np.array(tcp_offset_pose)
            assert tcp_offset_pose.shape[0] in [
                6,
                7,
            ], "tcp_offset_pose must be a 6- or 7-element pose"
        if initial_tcp_pose is not None:
            initial_tcp_pose = np.array(initial_tcp_pose)
            assert initial_tcp_pose.shape[0] in [
                6,
                7,
            ], "initial_tcp_pose must be a 6- or 7-element pose"

        super().__init__(name="xArmInterpolationController")
        self.robot_ip = robot_ip
        self.frequency = frequency
        self.lookahead_time = lookahead_time
        self.gain = gain
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.launch_timeout = launch_timeout
        self.tcp_offset_pose = tcp_offset_pose
        self.initial_tcp_pose = initial_tcp_pose
        self.soft_real_time = soft_real_time
        self.verbose = verbose

        # Build input command queue.
        example = {
            "cmd": Command.SERVOL.value,
            "target_pose": np.zeros((6,), dtype=np.float64),
            "duration": 0.0,
            "target_time": 0.0,
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager, examples=example, buffer_size=256
        )

        # Build ring buffer for state logging.
        if receive_keys is None:
            # For xArm we at least log the actual TCP pose.
            receive_keys = ["ActualTCPPose"]
        dummy_state = {}
        for key in receive_keys:
            dummy_state[key] = np.zeros((6,), dtype=np.float64)
        dummy_state["robot_receive_timestamp"] = time.time()
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=dummy_state,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency,
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys

    # ========= Process launch and stop methods ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[xArmInterpolationController] Process spawned at {self.pid}")

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

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= Command API ===========
    def servoL(self, pose, duration=0.1):
        """
        Send a continuous Cartesian pose command.
        pose: 6- or 7-element array [x, y, z, rx, ry, rz] (rx,ry,rz in degrees)
        duration: time in seconds to drive to the pose
        """
        assert self.is_alive(), "Controller process not alive"
        assert duration >= (1 / self.frequency)
        pose = np.array(pose, dtype=np.float64)
        assert pose.shape[0] in [6, 7], "Pose must be a 6- or 7-dimensional array"
        # Convert orientation (elements 3:6) from degrees to radians for internal processing
        pose[:6][3:6] = np.radians(pose[:6][3:6])
        message = {
            "cmd": Command.SERVOL.value,
            "target_pose": pose,
            "duration": duration,
        }
        self.input_queue.put(message)

    def schedule_waypoint(self, pose, target_time):
        assert target_time > time.time(), "Target time must be in the future"
        pose = np.array(pose, dtype=np.float64)
        assert pose.shape[0] in [6, 7], "Pose must be a 6- or 7-dimensional array"
        # Convert orientation from degrees to radians for internal processing
        pose[:6][3:6] = np.radians(pose[:6][3:6])
        message = {
            "cmd": Command.SCHEDULE_WAYPOINT.value,
            "target_pose": pose,
            "target_time": target_time,
        }
        self.input_queue.put(message)

    # ========= State APIs ===========
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    # ========= Main control loop ===========
    def run(self):
        if self.soft_real_time:
            os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(20))
        robot_ip = self.robot_ip

        # Connect and initialize the xArm.
        try:
            arm = XArmAPI(robot_ip)
            arm.connect()
            arm.clean_error()
            arm.clean_warn()
            ret = arm.motion_enable(True)
            if ret != 0:
                raise RuntimeError(f"Failed to enable motion: error {ret}")
            ret = arm.set_mode(1)  # Servo motion mode.
            if ret != 0:
                raise RuntimeError(f"Failed to set mode: error {ret}")
            ret = arm.set_state(0)  # Ready state.
            if ret != 0:
                raise RuntimeError(f"Failed to set state: error {ret}")
            if self.verbose:
                print(f"[xArmInterpolationController] Connected to xArm at: {robot_ip}")
        except Exception as e:
            print(f"[xArmInterpolationController] Connection error: {e}")
            self.ready_event.set()
            return

        # Warn if tcp_offset_pose was provided.
        if self.tcp_offset_pose is not None:
            if self.verbose:
                print(
                    "[xArmInterpolationController] Warning: tcp_offset_pose provided but not applied (not supported by xArm API)."
                )

        # Use Cartesian initialization if an initial TCP pose is provided.
        # (Assumes initial_tcp_pose is given in degrees for orientation.)
        if self.initial_tcp_pose is not None:
            ret = arm.set_servo_cartesian(self.initial_tcp_pose, is_radian=False)
            if ret != 0:
                raise AssertionError("Failed to initialize TCP pose")
            if self.verbose:
                print(
                    f"[xArmInterpolationController] TCP pose initialized to: {self.initial_tcp_pose}"
                )

        # Main control loop.
        dt = 1.0 / self.frequency
        ret, pose_full = arm.get_position(is_radian=False)
        if ret != 0:
            # Fallback to zeros if unable to read
            pose_full = [0, 0, 0, 0, 0, 0]
        # Convert the orientation part from degrees to radians for internal interpolation.
        curr_pose = np.array(pose_full[:6], dtype=np.float64)
        curr_pose[3:6] = np.radians(curr_pose[3:6])
        curr_t = time.monotonic()
        last_waypoint_time = curr_t
        pose_interp = PoseTrajectoryInterpolator(times=[curr_t], poses=[curr_pose])
        iter_idx = 0
        keep_running = True

        try:
            while keep_running:
                t_start = time.time()
                t_now = time.monotonic()
                # Interpolate to get the target pose (in radians for orientation)
                pose_command = pose_interp(t_now)

                # --- Debug and conversion block ---
                if self.verbose:
                    print("Raw interpolated pose:", pose_command)
                    # Orientation (radians)
                    print("Orientation (radians):", pose_command[3:6])
                    print("Orientation (degrees):", np.degrees(pose_command[3:6]))
                # Convert only the rotational part (elements 3 to 5) from radians to degrees for the API.
                pose_to_send = pose_command.copy()
                pose_to_send[3:6] = np.degrees(pose_command[3:6])
                if self.verbose:
                    print("Pose to send (converted):", pose_to_send)
                # --- End conversion block ---

                # Send the Cartesian command.
                ret = arm.set_servo_cartesian(pose_to_send, is_radian=False)
                if ret != 0 and self.verbose:
                    print(
                        f"[xArmInterpolationController] set_servo_cartesian error: {ret}"
                    )
                # Update and log state.
                ret, pose_full = arm.get_position(is_radian=False)
                if ret == 0:
                    curr_pose = np.array(pose_full[:6], dtype=np.float64)
                    # Convert orientation part from degrees to radians for internal use.
                    curr_pose[3:6] = np.radians(curr_pose[3:6])
                else:
                    curr_pose = np.zeros((6,))
                state = {
                    "ActualTCPPose": curr_pose,
                    "robot_receive_timestamp": time.time(),
                }
                self.ring_buffer.put(state)

                # Process incoming commands.
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands["cmd"])
                except Empty:
                    n_cmd = 0

                for i in range(n_cmd):
                    command = {key: commands[key][i] for key in commands}
                    cmd = command["cmd"]
                    if cmd == Command.STOP.value:
                        keep_running = False
                        break
                    elif cmd == Command.SERVOL.value:
                        target_pose = command["target_pose"]
                        duration = float(command["duration"])
                        # Convert the target pose orientation (elements 3:6) from degrees to radians.
                        target_pose = np.array(target_pose, dtype=np.float64)
                        target_pose[3:6] = np.radians(target_pose[3:6])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            print(
                                f"[xArmInterpolationController] New target: {target_pose} over {duration}s"
                            )
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command["target_pose"]
                        target_time = float(command["target_time"])
                        target_pose = np.array(target_pose, dtype=np.float64)
                        target_pose[3:6] = np.radians(target_pose[3:6])
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time,
                        )
                        last_waypoint_time = target_time
                    else:
                        keep_running = False
                        break

                elapsed = time.time() - t_start
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose:
                    actual_freq = 1.0 / (time.time() - t_start)
                    print(
                        f"[xArmInterpolationController] Actual frequency: {actual_freq}"
                    )
        finally:
            arm.set_state(0)
            arm.disconnect()
            self.ready_event.set()
            if self.verbose:
                print(
                    f"[xArmInterpolationController] Disconnected from xArm at: {robot_ip}"
                )
