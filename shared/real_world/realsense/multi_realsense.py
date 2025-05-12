import time
import pathlib
import numpy as np
import logging

from multiprocessing.managers import SharedMemoryManager
from typing import List, Optional, Union, Dict, Callable

from shared.real_world.realsense.single_realsense import SingleRealsense
from shared.real_world.record_utils.video_recorder import VideoRecorder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def repeat_to_list(x, n: int, cls):
    if x is None:
        x = [None] * n
    if isinstance(x, cls):
        x = [x] * n
    assert len(x) == n, f"repeat_to_list got len(x)={len(x)}, expected {n}"
    return x


class MultiRealsense:
    def __init__(
        self,
        serial_numbers: Optional[List[str]] = None,
        shm_manager: Optional[SharedMemoryManager] = None,
        resolution=(1280, 720),
        capture_fps=30,
        put_fps=None,
        put_downsample=True,
        record_fps=None,
        enable_color=True,
        enable_depth=False,
        enable_infrared=False,
        get_max_k=30,
        advanced_mode_config: Optional[Union[dict, List[dict]]] = None,
        transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]] = None,
        recording_transform: Optional[
            Union[Callable[[Dict], Dict], List[Callable]]
        ] = None,
        video_recorder: Optional[Union[VideoRecorder, List[VideoRecorder]]] = None,
        verbose=False,
    ):
        logger.info("Initializing MultiRealsense...")

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()

        if serial_numbers is None:
            serial_numbers = SingleRealsense.get_connected_devices_serial()
        n_cameras = len(serial_numbers)
        logger.info(f"MultiRealsense found {n_cameras} cameras: {serial_numbers}")

        advanced_mode_config = repeat_to_list(advanced_mode_config, n_cameras, dict)
        transform = repeat_to_list(transform, n_cameras, Callable)
        recording_transform = repeat_to_list(recording_transform, n_cameras, Callable)
        video_recorder = repeat_to_list(video_recorder, n_cameras, VideoRecorder)

        self.cameras = {}
        for i, serial in enumerate(serial_numbers):
            logger.info(f"Creating SingleRealsense {i} with serial={serial}")
            self.cameras[serial] = SingleRealsense(
                shm_manager=shm_manager,
                serial_number=serial,
                resolution=resolution,
                capture_fps=capture_fps,
                put_fps=put_fps,
                put_downsample=put_downsample,
                record_fps=record_fps,
                enable_color=enable_color,
                enable_depth=enable_depth,
                enable_infrared=enable_infrared,
                get_max_k=get_max_k,
                advanced_mode_config=advanced_mode_config[i],
                transform=transform[i],
                recording_transform=recording_transform[i],
                video_recorder=video_recorder[i],
                verbose=verbose,
            )
        self.shm_manager = shm_manager
        logger.info("MultiRealsense init done.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @property
    def n_cameras(self):
        return len(self.cameras)

    @property
    def is_ready(self):
        # Must have all cameras ready
        for camera in self.cameras.values():
            if not camera.is_ready:
                return False
        return True

    def start(self, wait=True, put_start_time=None):
        logger.info(
            f"MultiRealsense.start(wait={wait}), put_start_time={put_start_time}"
        )
        if put_start_time is None:
            put_start_time = time.time()
        for camera in self.cameras.values():
            camera.start(wait=False, put_start_time=put_start_time)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        logger.info(f"MultiRealsense.stop(wait={wait})")
        for camera in self.cameras.values():
            camera.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        logger.info(
            "MultiRealsense.start_wait() => waiting for each camera.start_wait()"
        )
        for camera in self.cameras.values():
            camera.start_wait()

    def stop_wait(self):
        logger.info("MultiRealsense.stop_wait() => joining each camera.")
        for camera in self.cameras.values():
            camera.join()

    def get(self, k=None, out=None) -> Dict[int, Dict[str, np.ndarray]]:
        if out is None:
            out = {}
        i = 0
        for serial, camera in self.cameras.items():
            this_out = out.get(i, None)
            this_out = camera.get(k=k, out=this_out)
            out[i] = this_out
            i += 1
        return out


    def set_color_option(self, option, value):
        logger.info(f"MultiRealsense.set_color_option({option}, {value})")
        for camera in self.cameras.values():
            camera.set_color_option(option, value)

    def set_exposure(self, exposure=None, gain=None):
        logger.info(f"MultiRealsense.set_exposure(exposure={exposure}, gain={gain})")
        for camera in self.cameras.values():
            camera.set_exposure(exposure=exposure, gain=gain)

    def set_white_balance(self, white_balance=None):
        logger.info(f"MultiRealsense.set_white_balance({white_balance})")
        for camera in self.cameras.values():
            camera.set_white_balance(white_balance)

    def get_intrinsics(self):
        return np.array([cam.get_intrinsics() for cam in self.cameras.values()])

    def get_depth_scale(self):
        return np.array([cam.get_depth_scale() for cam in self.cameras.values()])

    def start_recording(self, video_path: Union[str, List[str]], start_time: float):
        logger.info(
            f"MultiRealsense.start_recording({video_path}, start_time={start_time})"
        )
        # If user passed a single string, interpret it as a directory
        if isinstance(video_path, str):
            video_dir = pathlib.Path(video_path)
            assert video_dir.parent.is_dir(), "Directory's parent not found"
            video_dir.mkdir(parents=True, exist_ok=True)
            # We build a separate .mp4 for each camera
            new_paths = []
            i = 0
            for cam_serial in self.cameras:
                new_paths.append(str(video_dir.joinpath(f"{i}.mp4").absolute()))
                i += 1
            video_path = new_paths

        # Now we have a list of paths, one per camera
        assert len(video_path) == len(
            self.cameras
        ), f"Number of video paths {len(video_path)} != number of cameras {len(self.cameras)}"

        i = 0
        for cam_serial, camera in self.cameras.items():
            camera.start_recording(video_path[i], start_time)
            i += 1

    def stop_recording(self):
        logger.info("MultiRealsense.stop_recording()")
        for camera in self.cameras.values():
            camera.stop_recording()

    def restart_put(self, start_time):
        logger.info(f"MultiRealsense.restart_put(start_time={start_time})")
        for camera in self.cameras.values():
            camera.restart_put(start_time)