"""
python demo_xarm.py (optional params:) -o <demo_save_dir> -ri <robot_ip> ... etc

This file is largely taken from the original Diffusion Policy repository, modified for
the xArm7. It can probably be used with any xArm robot.

Robot movement:
Move your SpaceMouse to move the robot EEF.
Please make sure your SpaceMouse has the gripper button
facing the left side (i.e., the wire goes out back)

Recording control:
MISSING!
"""

import time
import click
import cv2
import sys
import numpy as np
import scipy.spatial.transform as st

from multiprocessing.managers import SharedMemoryManager

# RealEnv missing!
from shared.real_world.precise_util import precise_wait
from shared.real_world.spacemouse import Spacemouse
from shared.real_world.keystroke_counter import KeystrokeCounter, Key, KeyCode

# This assumes you have ril_env installed in the same level.
# i.e., running ls in the directory above should list something like
# (robodiff) u-ril@uril-workstation-1:~/URIL/raayan$ ls
# diffusion-policies  ril-env
# Make sure to run pip install -e . in ril-env
from ril_env.xarm_env import XArmEnv, XArmConfig


@click.command()
@click.option(
    "--output", "-o", default="/data", help="Directory to save demonstration dataset."
)
@click.option(
    "--robot_ip",
    "-ri",
    default="192.168.1.223",
    help="xArm's IP address e.g. 192.168.1.223",
)
@click.option(
    "--vis_camera_idx",
    default="0",
    type=int,
    help="Which RealSense camera to visualize.",
)
@click.option(
    "--init_joints",
    "-j",
    is_flag=True,
    default=True,
    help="Whether to initialize robot joint configuration in the beginning.",
)
@click.option(
    "--frequency", "-f", default=20.0, type=float, help="Control frequency in Hz."
)
@click.option(
    "--command_latency",
    "-cl",
    default=0.01,
    type=float,
    help="Latency between receiving SpaceMouse command to executing on Robot in Sec.",
)
def main(output, robot_ip, vis_camera_idx, init_joints, frequency, command_latency):
    dt = 1 / frequency

    # Create config and change parameters here.
    # See ril_env for more details
    xarm_config = XArmConfig(
        position_gain=2.0,
        orientation_gain=2.0,
        ip=robot_ip,
    )
    # Initialize XArm environment API
    xarm_env = XArmEnv(xarm_config)

    if init_joints:
        # If you want to change the home position, change it in xarm_config initialization.
        xarm_env._arm_reset()

    loop_period = xarm_env.control_loop_period

    cv2.setNumThreads(1)

    with Spacemouse(deadzone=0.4) as sm, KeystrokeCounter() as key_counter:
        try:
            while True:
                loop_start = time.monotonic()

                if command_latency > 0:
                    time.sleep(command_latency)

                sm_state = sm.get_motion_state_transformed()
                dpos = sm_state[:3] * xarm_config.position_gain
                drot = sm_state[3:] * xarm_config.orientation_gain
                grasp = sm.grasp

                if sm.is_button_pressed(1):
                    xarm_env._arm_reset()
                    continue

                xarm_env.step(dpos, drot, grasp)
                elapsed = time.monotonic() - loop_start
                sleep_time = loop_period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("\nStopped.")

    xarm_env._arm_reset()


if __name__ == "__main__":
    main()
