"""
Demo script to test the RTDEInterpolationController (adapted for xArm) on the robot.

Usage example:
    python demo_rtde_controller.py --robot_ip 192.168.1.223 --frequency 20 --duration 2.0

This script:
  - Creates a shared memory manager.
  - Instantiates and starts the RTDEInterpolationController process.
  - Sends a test servoL command (i.e., a pose command) to the controller.
  - Waits for the command to execute and then stops the controller.

Note: This demo tests the RTDEInterpolationController (the “RTDEController stuff”) on the real robot.
Make sure you have set up all safety parameters and that your test pose is safe.
"""

import time
import click
import numpy as np
from multiprocessing.managers import SharedMemoryManager

# Import the adapted RTDEInterpolationController.
# (This file is the modified version of the controller that now uses the xArm API.)
from shared.utils.real_world.rtde_interpolation_controller import (
    xArmInterpolationController,
)


@click.command()
@click.option(
    "--robot_ip", "-ri", default="192.168.1.223", help="IP address of the xArm robot."
)
@click.option(
    "--frequency", "-f", default=20.0, type=float, help="Control loop frequency in Hz."
)
@click.option(
    "--duration",
    "-d",
    default=2.0,
    type=float,
    help="Duration for the servoL command (in sec).",
)
def main(robot_ip, frequency, duration):
    dt = 1.0 / frequency

    # Create a shared memory manager.
    shm_manager = SharedMemoryManager()
    shm_manager.start()

    # Create an instance of the RTDEInterpolationController.
    # BEFORE: In the UR version this would create RTDEControlInterface and RTDEReceiveInterface.
    # NOW: We pass robot_ip and parameters to the controller that now uses the xArmAPI.
    controller = xArmInterpolationController(
        shm_manager=shm_manager,
        robot_ip=robot_ip,
        frequency=frequency,
        lookahead_time=0.1,
        gain=300,
        max_pos_speed=0.25,  # m/s
        max_rot_speed=0.16,  # rad/s
        launch_timeout=5,
        tcp_offset_pose=[0, 0, 0, 0, 0, 0],  # Example offset; adjust if needed.
        joints_init=[0, 0, 0, 0, 0, 0],  # Example initial joint pose.
        verbose=True,
    )

    # Start the controller process.
    controller.start(wait=True)
    print("Controller process started and ready.")

    # Send a servoL command with a test pose.
    # The pose is given in SI units: first three values in meters, last three in radians.
    test_pose = [0.4, 0.0, 0.3, 0.0, 0.0, 0.0]
    # BEFORE: The UR version would send this to rtde_c.servoL.
    # NOW: Our controller will use its run() loop (which calls arm.set_servo_cartesian) to move the arm.
    print(f"Sending servoL command with pose: {test_pose} for duration {duration} sec.")
    controller.servoL(test_pose, duration=duration)

    # Let the controller run for the duration plus a short margin.
    time.sleep(duration + 1)

    # Optionally, retrieve and print the latest state from the ring buffer.
    state = controller.get_state()
    print("Latest state from controller:", state)

    # Stop the controller.
    controller.stop(wait=True)
    print("Controller process stopped.")

    # Shutdown the shared memory manager.
    shm_manager.shutdown()


if __name__ == "__main__":
    main()
