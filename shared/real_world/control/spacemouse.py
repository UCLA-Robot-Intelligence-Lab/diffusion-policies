import numpy as np
import time

from spnav import (
    spnav_open,
    spnav_poll_event,
    spnav_close,
    SpnavMotionEvent,
    SpnavButtonEvent,
)
from threading import Thread, Event
from collections import defaultdict


class Spacemouse(Thread):
    def __init__(
        self,
        max_value=300,
        deadzone=(0, 0, 0, 0, 0, 0),
        dtype=np.float32,
        shm_manager=None,
    ):
        """
        Continuously listen to 3Dconnexion SpaceMouse events and update the latest state.

        max_value: {300, 500} — 300 for wired version and 500 for wireless.
        deadzone: a number or a tuple of numbers for each axis; values lower than this will be set to 0.
        """
        if np.issubdtype(type(deadzone), np.number):
            deadzone = np.full(6, fill_value=deadzone, dtype=dtype)
        else:
            deadzone = np.array(deadzone, dtype=dtype)
        assert (deadzone >= 0).all()

        super().__init__()
        self.stop_event = Event()
        self.max_value = max_value
        self.dtype = dtype
        self.deadzone = deadzone

        self.motion_event = SpnavMotionEvent([0, 0, 0], [0, 0, 0], 0)
        # Dictionary to keep track of button states (True if pressed)
        self.button_state = defaultdict(lambda: False)
        # Grasp state: 0.0 for open, 1.0 for closed
        self.grasp = 0.0

        # Translation transform:
        #   Mapping native translation: [translation_x, translation_y, translation_z]
        #   where native: right: x, up: y, front: z,
        #   to desired robot frame: [forward, right, up]
        self.tx_translation = np.array(
            [
                [0, 0, 1],  # native +z becomes robot +x (forward)
                [-1, 0, 0],  # native +x becomes robot +y (right)
                [0, 1, 0],  # native +y becomes robot +z (up)
            ],
            dtype=dtype,
        )

        # Rotation transform:
        self.tx_rotation = np.array(
            [
                [0, 0, -1],  # native +z rotation aligns with robot -x (roll)
                [1, 0, 0],  # native +x rotation aligns with robot +y (pitch)
                [0, 1, 0],  # native +y rotation aligns with robot +z (yaw)
            ],
            dtype=dtype,
        )

    def get_motion_state(self):
        me = self.motion_event
        state = (
            np.array(me.translation + me.rotation, dtype=self.dtype) / self.max_value
        )
        # Zero out values within the deadzone.
        is_dead = (-self.deadzone < state) & (state < self.deadzone)
        state[is_dead] = 0
        return state

    def get_motion_state_transformed(self):
        """
        Return the motion state in a right-handed coordinate system using separate
        transformations for translation and rotation.

        Translation (first three elements) is mapped by tx_translation.
        Rotation (last three elements) is mapped by tx_rotation.
        """
        state = self.get_motion_state()
        tf_state = np.zeros_like(state)
        tf_state[:3] = self.tx_translation @ state[:3]
        tf_state[3:] = self.tx_rotation @ state[3:]
        return tf_state

    def is_button_pressed(self, button_id):
        """
        Returns whether the given button is currently pressed.
        """
        return self.button_state[button_id]

    def stop(self):
        self.stop_event.set()
        self.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def run(self):
        spnav_open()
        try:
            while not self.stop_event.is_set():
                event = spnav_poll_event()
                if isinstance(event, SpnavMotionEvent):
                    self.motion_event = event
                elif isinstance(event, SpnavButtonEvent):
                    if event.bnum == 0 and event.press:
                        self.grasp = 1.0 if self.grasp == 0.0 else 0.0
                    self.button_state[event.bnum] = event.press
                else:
                    time.sleep(1 / 200)
        finally:
            spnav_close()


def test():
    with Spacemouse(deadzone=0.3) as sm:
        for i in range(2000):
            print("Motion state:", sm.get_motion_state_transformed())
            print("Grasp state:", sm.grasp)
            time.sleep(1 / 100)


if __name__ == "__main__":
    test()
