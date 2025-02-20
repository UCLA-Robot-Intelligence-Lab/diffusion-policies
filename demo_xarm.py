"""
This file is largely taken from the original Diffusion Policy repository, modified for
the xArm7. It can probably be used with any xArm robot.


"""


import time
import click
import cv2
import sys
import numpy as np
import scipy.spatial.transform as st

from multiprocessing.managers import SharedMemoryManager

from shared.utils.real_world.precise_util import precise_wait
from shared.utils.real_world.spacemouse import Spacemouse
from shared.utils.real_world.keystroke_counter import KeystrokeCounter, Key, KeyCode

from ril_env.xarm_env import XArmEnv, XArmConfig

@click.command()
@click.option(
    "--robot_ip",
    "-ri",
    default="192.168.1.223",
    help="xArm's IP address e.g. 192.168.1.223",
)
@click.option(
    "--frequency", "-f", default=50, type=float, help="Control frequency in Hz."
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
def main(robot_ip, frequency, command_latency, max_speed):
    max_speed = max_speed * frequency
    dt = 1 / frequency

    xarm_config = XArmConfig(position_gain=2.0, orientation_gain=2.0)
    xarm_config.ip = robot_ip

    xarm_env = XArmEnv(xarm_config)

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
