import time
import click
import cv2
import sys
import numpy as np

from shared.utils.real_world.keystroke_counter import KeystrokeCounter, Key

from ril_env.xarm_env import XArmEnv, XArmConfig
from ril_env.controller import SpaceMouse, SpaceMouseConfig

@click.command()
@click.option(
    "--command_latency",
    "-cl",
    default=0.01,
    type=float,
    help="(Optional) artificial latency before each robot command.",
)
@click.option(
    "--max_speed",
    "-ms",
    default=100,
    type=float,
    help="Max speed of the robot in mm/s (if you want to clamp dpos).",
)
def main(robot_ip, command_latency, max_speed):
    xarm_config = XArmConfig()
    xarm_config.ip = robot_ip

    xarm_env = XArmEnv(xarm_config)
    xarm_env._arm_reset()

    loop_period = xarm_env.control_loop_period

    spacemouse_cfg = SpaceMouseConfig(
        pos_sensitivity=2.0,
        rot_sensitivity=2.0,
        verbose=False,
        vendor_id=9583,
        product_id=50741,
    )

    cv2.setNumThreads(1)

    last_print_time = time.monotonic()
    loops_since_print = 0

    with SpaceMouse(spacemouse_cfg) as sm, KeystrokeCounter() as key_counter:
        print("Ready!")

        while True:
            loop_start = time.monotonic()

            if command_latency > 0:
                time.sleep(command_latency)

            state = sm.get_controller_state()
            dpos = state["dpos"] * xarm_config.position_gain
            drot = state["raw_drotation"] * xarm_config.orientation_gain

            if state["reset"] == 1:
                xarm_env._arm_reset()
                continue

            grasp = state["grasp"]
            xarm_env.step(dpos, drot, grasp)

            elapsed = time.monotonic() - loop_start
            sleep_time = loop_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            loops_since_print += 1
            now = time.monotonic()
            if now - last_print_time >= 1.0:
                freq_measured = loops_since_print / (now - last_print_time)
                print(f"[demo_xarm] Current loop frequency: {freq_measured:.2f} Hz")
                loops_since_print = 0
                last_print_time = now

if __name__ == "__main__":
    main()
