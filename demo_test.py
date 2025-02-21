import time
import numpy as np
import cv2
from multiprocessing.managers import SharedMemoryManager

from shared.real_world.spacemouse import Spacemouse
from shared.real_world.xarm_interpolation_controller import (
    xArmInterpolationController,
)  # your modified file


def main():
    # 1. Set up the interpolation controller
    robot_ip = "192.168.1.223"
    frequency = 20
    shm_manager = SharedMemoryManager()
    shm_manager.start()

    controller = xArmInterpolationController(
        shm_manager=shm_manager,
        robot_ip=robot_ip,
        frequency=frequency,
        verbose=True,
    )

    # 2. Start the controller process
    controller.start(wait=True)
    print("Controller process started and ready.")
    time.sleep(2)

    # 3. Initialize a “current target pose”.
    #    For example, read from the robot or just use a known default/home.
    state = controller.get_state()
    current_pose = state["ActualTCPPose"]
    current_pose = np.array(current_pose, dtype=float)  # shape (6,)

    position_gain = 15.0
    orientation_gain = 15.0

    # 4. Open the SpaceMouse
    with Spacemouse(deadzone=0.2) as sm:
        try:
            print("Move the SpaceMouse to move the robot EEF.")
            while True:
                loop_start = time.monotonic()

                # 4a. Read SpaceMouse
                sm_state = sm.get_motion_state_transformed()  # shape (6,)
                # Typically: sm_state[:3] = dpos, sm_state[3:] = drot
                dpos = sm_state[:3] * position_gain
                drot = sm_state[3:] * orientation_gain

                # 4b. Convert drot from axis-angle or some representation into rpy deltas
                #     but commonly in SpaceMouse, drot is a small incremental euler-ish.
                #     You might just add it in place if it’s small enough.
                current_pose[:3] += dpos  # x,y,z
                current_pose[
                    3:
                ] += drot  # roll, pitch, yaw in degrees if that’s your representation

                # 4c. Send a short servoL command to the interpolation controller
                #     small duration to keep streaming
                controller.servoL(current_pose, duration=0.1)

                # 4d. Regulate frequency
                elapsed = time.monotonic() - loop_start
                sleep_time = (1.0 / frequency) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("\nSpaceMouse control stopped.")

    # 5. Stop the interpolation controller
    controller.stop(wait=True)
    print("Controller process stopped.")
    shm_manager.shutdown()


if __name__ == "__main__":
    main()
