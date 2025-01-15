"""
Usage:
python demo_xarm.py --robot_ip <ip_of_xarm> --max_speed 100

Robot movement:
Move your SpaceMouse to move the robot EEF (3 spatial DoF only).
Press SpaceMouse left button once to reset to initial pose.
"""

import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import sys
import numpy as np
import scipy.spatial.transform as st

from shared.utils.real_world.precise_util import precise_wait
from shared.utils.real_world.spacemouse import Spacemouse
from shared.utils.real_world.keystroke_counter import KeystrokeCounter, Key

# xArm API
sys.path.append("/home/u-ril/URIL/xArm-Python-SDK")
from xarm.wrapper import XArmAPI


@click.command()
@click.option(
    "--robot_ip",
    "-ri",
    default="192.168.1.223",
    help="xArm's IP address e.g. 192.168.1.223",
)
@click.option(
    "--init_joints",
    "-j",
    is_flag=True,
    default=True,
    help="Whether to initialize robot joint configuration in the beginning.",
)
@click.option(
    "--frequency", "-f", default=10, type=float, help="Control frequency in Hz."
)
@click.option(
    "--command_latency",
    "-cl",
    default=0.01,
    type=float,
    help="Latency between receiving SpaceMouse command to executing on Robot in Sec.",
)
@click.option(
    "--max_speed",
    "-ms",
    default=100,
    type=float,
    help="Max speed of the robot in mm/s.",
)
def main(robot_ip, init_joints, frequency, command_latency, max_speed):
    max_speed = max_speed * frequency  # Max displacement per control cycle
    dt = 1 / frequency

    # Initialize XArm
    arm = XArmAPI(robot_ip, is_radian=True)
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(state=0)
    time.sleep(1)
    if init_joints:
        arm.reset(wait=True)

    # Set arm mode for servo control
    arm.set_mode(1)  # Servo motion mode
    arm.set_state(0)
    time.sleep(1)

    # Get initial pose for reference
    _, initial_pose = arm.get_position(is_radian=False)
    target_pose = np.array(initial_pose)  # [x, y, z, rx, ry, rz]

    cv2.setNumThreads(1)

    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, Spacemouse(
            shm_manager=shm_manager
        ) as sm:
            print("Ready!")
            t_start = time.monotonic()
            iter_idx = 0
            stop = False

            while not stop:
                # Calculate timing for control loop
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency

                # Wait until sample time
                precise_wait(t_sample)

                # Get input from SpaceMouse
                sm_state = sm.get_motion_state_transformed()
                dpos = sm_state[:3] * (max_speed / frequency)  # Displacement change
                drot_xyz = sm_state[3:] * (max_speed / frequency)

                # Limit orientation changes if desired
                drot_xyz[:] = 0  # Zero out rotation for 3 DoF translation only

                # Reset to home on button press
                if sm.is_button_pressed(0):
                    arm.set_mode(0)
                    arm.set_state(0)
                    time.sleep(0.1)
                    arm.reset(wait=True)
                    # Re-enter servo mode after reset
                    arm.set_mode(1)
                    arm.set_state(0)
                    # Update target_pose after reset
                    # _, initial_pose = arm.get_position(is_radian=False)
                    initial_pose = [0, 0, 0, 70, 0, 70, 0]
                    target_pose = np.array(initial_pose)

                # Update target pose based on SpaceMouse input
                target_pose[:3] += dpos
                # For 3 DoF movement, drot_xyz is zero; if orientation control is added, update target_pose[3:] accordingly.

                # Use set_servo_cartesian to command the robot to the new target pose
                ret = arm.set_servo_cartesian(target_pose.tolist(), is_radian=False)
                if ret != 0:
                    print(f"Error in set_servo_cartesian: {ret}")

                # Wait until end of cycle
                precise_wait(t_cycle_end)
                iter_idx += 1


if __name__ == "__main__":
    main()
