import time
import click
import cv2
import sys
import numpy as np

from shared.utils.real_world.precise_util import precise_wait
from shared.utils.real_world.spacemouse import Spacemouse
from shared.utils.real_world.keystroke_counter import KeystrokeCounter, Key

from ril_env.xarm_env import XArmEnv, XArmConfig

@click.command()
@click.option(
    "--robot_ip",
    "-ri",
    default="192.168.1.223",
    help="xArm's IP address e.g. 192.168.1.223",
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
def main(robot_ip, frequency, command_latency, max_speed):
    max_speed = max_speed * frequency
    dt = 1 / frequency

    xarm_config = XArmConfig()
    xarm_config.ip = robot_ip  # override IP if necessary
    # Other configuration parameters can be set here if needed.

    xarm_env = XArmEnv(xarm_config)

    # Optionally, do an initial reset if needed
    xarm_env._arm_reset()

    cv2.setNumThreads(1)

    with Spacemouse() as sm, KeystrokeCounter() as key_counter:
        print("Ready!")
        t_start = time.monotonic()
        iter_idx = 0

        while True:
            # Calculate timing for control loop
            t_cycle_end = t_start + (iter_idx + 1) * dt
            t_sample = t_cycle_end - command_latency

            precise_wait(t_sample)

            # Get input from SpaceMouse
            sm_state = sm.get_motion_state_transformed()
            # Compute displacement and rotation changes based on SpaceMouse input.
            dpos = sm_state[:3] * (max_speed / frequency)
            drot = sm_state[3:] * (max_speed / frequency)

            # For 3 DoF movement, ignore rotation if desired
            drot[:] = 0

            # Check for reset command (e.g., SpaceMouse button press)
            if sm.is_button_pressed(0):
                # Use the environment's reset method to go back to home
                xarm_env._arm_reset()
                # Optionally, reinitialize your target state if needed.
                continue

            # Here, we assume grasp value of 0.0 (open) for simplicity.
            grasp = 0.0

            xarm_env.step(dpos, drot, grasp)

            precise_wait(t_cycle_end)
            iter_idx += 1

if __name__ == "__main__":
    main()
