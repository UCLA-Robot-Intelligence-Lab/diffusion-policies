import os
import time
import enum
import multiprocessing as mp
import scipy.interpolate as si
import scipy.spatial.transform as st
import numpy as np

from multiprocessing.managers import SharedMemoryManager
from shared.utils.shared_memory.shared_memory_queue import SharedMemoryQueue, Empty
from shared.utils.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from shared.utils.pose_trajectory_interpolator import PoseTrajectoryInterpolator

# BEFORE: The original code imported RTDE interfaces and the xArmAPI directly
# from rtde_control import RTDEControlInterface
# from rtde_receive import RTDEReceiveInterface
# from xarm.wrapper.xarm_api import XArmAPI
#
# NOW: We use the xArmEnv wrapper from the ril_env package.
from ril_env.xarm_env import XArmEnv, XArmConfig


class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2


class xArmInterpolationController(mp.Process):
    """
    This controller runs in its own process to achieve predictable, real-time command sending.
    Originally written for UR robots (using RTDE), it is now adapted to use the xArmEnv from ril_env.
    """

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        robot_ip,
        frequency=125,  # use 125 Hz as default (for UR, but here sets control loop rate)
        lookahead_time=0.1,
        gain=300,
        max_pos_speed=0.25,  # m/s
        max_rot_speed=0.16,  # rad/s
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
        """
        frequency: originally for UR robots (e.g. CB2=125, UR3e=500) but here it defines the control loop rate.
        lookahead_time: time window for smoothing the trajectory.
        gain: proportional gain.
        max_pos_speed: maximum positional speed (m/s).
        max_rot_speed: maximum rotational speed (rad/s).
        tcp_offset_pose: tool center point offset (6D pose).
        payload_mass, payload_cog: payload parameters.
        joints_init: initial joint configuration.
        soft_real_time: enable real-time priority.
        """
        # Verify parameters
        assert 0 < frequency <= 500
        assert 0.03 <= lookahead_time <= 0.2
        assert 100 <= gain <= 2000
        assert 0 < max_pos_speed
        assert 0 < max_rot_speed
        if tcp_offset_pose is not None:
            tcp_offset_pose = np.array(tcp_offset_pose)
            assert tcp_offset_pose.shape == (6,)
        if payload_mass is not None:
            assert 0 <= payload_mass <= 5
        if payload_cog is not None:
            payload_cog = np.array(payload_cog)
            assert payload_cog.shape == (3,)
            assert payload_mass is not None
        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (6,)

        super().__init__(name="RTDEPositionalController")
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

        # Build input queue (unchanged)
        example = {
            "cmd": Command.SERVOL.value,
            "target_pose": np.zeros((6,), dtype=np.float64),
            "duration": 0.0,
            "target_time": 0.0,
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager, examples=example, buffer_size=256
        )

        # Build ring buffer.
        # BEFORE: The UR version would use RTDEReceiveInterface to populate state keys.
        # NOW: We simply initialize a ring buffer with a basic state (only ActualTCPPose).
        if receive_keys is None:
            receive_keys = [
                "ActualTCPPose",
                "ActualTCPSpeed",
                "ActualQ",
                "ActualQd",
                "TargetTCPPose",
                "TargetTCPSpeed",
                "TargetQ",
                "TargetQd",
            ]
        code, pos = (0, [0, 0, 0, 0, 0, 0])
        example = {
            "ActualTCPPose": np.array(pos),
            "robot_receive_timestamp": time.time(),
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency,
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys

    # ========= Launch Methods ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(
                f"[RTDEPositionalController] Controller process spawned at {self.pid}"
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

    # ========= Context Manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= Command Methods ===========
    def servoL(self, pose, duration=0.1):
        """
        Send a new immediate pose command.
        duration: desired time to reach pose.
        """
        assert self.is_alive()
        assert duration >= (1 / self.frequency)
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            "cmd": Command.SERVOL.value,
            "target_pose": pose,
            "duration": duration,
        }
        self.input_queue.put(message)

    def schedule_waypoint(self, pose, target_time):
        assert target_time > time.time()
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            "cmd": Command.SCHEDULE_WAYPOINT.value,
            "target_pose": pose,
            "target_time": target_time,
        }
        self.input_queue.put(message)

    # ========= Receive APIs ===========
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    # ========= Main Loop ===========
    def run(self):
        # Enable soft real-time scheduling if requested.
        if self.soft_real_time:
            os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(20))

        # BEFORE: For a UR robot, the code would create RTDEControlInterface and RTDEReceiveInterface.
        # NOW: We use the xArmEnv wrapper from ril_env.
        from ril_env.xarm_env import XArmEnv, XArmConfig

        robot_ip = self.robot_ip
        # Create a configuration for the xArm (using default parameters from XArmConfig)
        xarm_config = XArmConfig(ip=robot_ip, verbose=self.verbose)
        xarm_env = XArmEnv(xarm_config)
        arm = xarm_env.arm  # Get the underlying xArm API instance

        if self.verbose:
            print(f"[RTDEPositionalController] Connected to xArm at: {robot_ip}")

        try:
            # Set parameters (TCP offset, payload, etc.)
            if self.tcp_offset_pose is not None:
                # BEFORE: rtde_c.setTcp(self.tcp_offset_pose)
                code = arm.set_tcp_offset(self.tcp_offset_pose, is_radian=True)
                assert code == 0, "Failed to set TCP offset"
            if self.payload_mass is not None:
                # xArm SDK may not support payload setting in the same way.
                if self.verbose:
                    print("Payload setting is not implemented in this xArm adaptation.")

            # Initialize joints if provided.
            if self.joints_init is not None:
                # BEFORE: assert rtde_c.moveJ(self.joints_init, self.joints_init_speed, 1.4)
                # NOW: Using xArmEnv, we assume joints_init is provided in degrees.
                code = arm.set_servo_angle(
                    angle=self.joints_init, is_radian=False, wait=True
                )
                assert code == 0, "Failed to initialize joints"

            # Main control loop initialization.
            dt = 1.0 / self.frequency
            # BEFORE: curr_pose = rtde_r.getActualTCPPose()
            code, curr_pose = arm.get_position(is_radian=False)
            assert code == 0, "Failed to get current position"
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(times=[curr_t], poses=[curr_pose])

            iter_idx = 0
            keep_running = True
            while keep_running:
                # Simulate start of control period.
                t_start = time.perf_counter()

                # Compute the interpolated pose command.
                t_now = time.monotonic()
                pose_command = pose_interp(t_now)
                # BEFORE: For UR, the pose_command (in SI units) was sent directly.
                # NOW: xArmEnv expects position in mm and orientation in degrees.
                xarm_pose = np.copy(pose_command)
                xarm_pose[:3] = xarm_pose[:3] * 1000  # Convert meters to mm
                xarm_pose[3:] = np.degrees(xarm_pose[3:])  # Convert radians to degrees

                # Set example speed and acceleration parameters.
                speed = 100  # mm/s
                mvacc = 500  # mm/s^2
                mvtime = 0  # Reserved

                code = arm.set_servo_cartesian(
                    xarm_pose,
                    speed=speed,
                    mvacc=mvacc,
                    mvtime=mvtime,
                    is_radian=False,  # xArmEnv uses degrees/mm
                    is_tool_coord=False,
                )
                assert code == 0, "xArm command failed!"

                # Update robot state.
                # BEFORE: state was read using RTDEReceiveInterface.
                # NOW: we call arm.get_position() from xArmEnv.
                code, pos = arm.get_position(is_radian=False)
                state = {
                    "ActualTCPPose": np.array(pos),
                    "robot_receive_timestamp": time.time(),
                }
                self.ring_buffer.put(state)

                # Fetch and process commands from the input queue.
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands["cmd"])
                except Empty:
                    n_cmd = 0

                for i in range(n_cmd):
                    command = {}
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command["cmd"]

                    if cmd == Command.STOP.value:
                        keep_running = False
                        break
                    elif cmd == Command.SERVOL.value:
                        # BEFORE: using rtde_c.servoL with interpolation drive_to_waypoint.
                        target_pose = command["target_pose"]
                        duration = float(command["duration"])
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
                                "[RTDEPositionalController] New pose target:{} duration:{}s".format(
                                    target_pose, duration
                                )
                            )
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command["target_pose"]
                        target_time = float(command["target_time"])
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

                # Regulate frequency: sleep if needed.
                elapsed = time.perf_counter() - t_start
                if dt > elapsed:
                    time.sleep(dt - elapsed)

                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose:
                    actual_freq = 1 / (time.perf_counter() - t_start)
                    print(f"[RTDEPositionalController] Actual frequency {actual_freq}")

        finally:
            # Cleanup.
            # BEFORE: rtde_c.servoStop(), rtde_c.stopScript(), rtde_c.disconnect(), rtde_r.disconnect()
            # NOW: We reset the arm via xArmEnv and disconnect the arm.
            xarm_env._arm_reset()
            arm.disconnect()
            self.ready_event.set()
            if self.verbose:
                print(f"[RTDEPositionalController] Disconnected from robot: {robot_ip}")
