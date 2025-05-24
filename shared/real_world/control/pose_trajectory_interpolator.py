import numbers
import numpy as np
import scipy.interpolate as si
import scipy.spatial.transform as st
import logging
from typing import Union

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def rotation_distance(a: st.Rotation, b: st.Rotation) -> float:
    return (b * a.inv()).magnitude()


def pose_distance(start_pose, end_pose):
    start_pose = np.array(start_pose)
    end_pose = np.array(end_pose)
    start_pos = start_pose[:3]
    end_pos = end_pose[:3]

    # For xArm, rotations are in euler angles (degrees)
    start_rot = st.Rotation.from_euler("xyz", start_pose[3:], degrees=True)
    end_rot = st.Rotation.from_euler("xyz", end_pose[3:], degrees=True)

    pos_dist = np.linalg.norm(end_pos - start_pos)
    rot_dist = rotation_distance(start_rot, end_rot)
    return pos_dist, rot_dist


class PoseTrajectoryInterpolator:
    def __init__(self, times: np.ndarray, poses: np.ndarray):
        assert len(times) >= 1
        assert len(poses) == len(times)
        if not isinstance(times, np.ndarray):
            times = np.array(times)
        if not isinstance(poses, np.ndarray):
            poses = np.array(poses)

        # Ensure we have unique and increasing times
        if len(times) > 1:
            # Check if times are strictly increasing
            if not np.all(np.diff(times) > 0):
                # If not, ensure they are by adding small deltas
                logger.warning(
                    "Times are not strictly increasing. Adding small deltas."
                )
                for i in range(1, len(times)):
                    if times[i] <= times[i - 1]:
                        times[i] = times[i - 1] + 0.001

        if len(times) == 1:
            # Special treatment for single step interpolation
            # Add a second point just ahead in time with the same pose
            # to make a valid interpolator
            self.single_step = True
            self._times = np.array([times[0], times[0] + 0.1])
            self._poses = np.tile(poses, (2, 1))

            # Create interpolators with our manufactured points
            pos = self._poses[:, :3]
            self.pos_interp = si.interp1d(
                self._times, pos, axis=0, bounds_error=False, fill_value="extrapolate"
            )

            # For rotation, we need to create a Slerp
            rot_euler = self._poses[:, 3:]
            rot = st.Rotation.from_euler("xyz", rot_euler, degrees=True)
            self.rot_interp = st.Slerp(self._times, rot)
        else:
            self.single_step = False
            self._times = times
            self._poses = poses

            pos = poses[:, :3]
            rot_euler = poses[:, 3:]
            try:
                rot = st.Rotation.from_euler("xyz", rot_euler, degrees=True)

                # Create interpolators
                self.pos_interp = si.interp1d(
                    times, pos, axis=0, bounds_error=False, fill_value="extrapolate"
                )
                self.rot_interp = st.Slerp(times, rot)
            except Exception as e:
                logger.error(f"Error creating interpolator: {e}")
                logger.error(f"Times: {times}")
                logger.error(f"Poses: {poses}")
                # Fall back to single step interpolation
                self.single_step = True
                self._times = np.array([times[0], times[0] + 0.1])
                self._poses = np.tile(poses[0:1], (2, 1))

                pos = self._poses[:, :3]
                rot_euler = self._poses[:, 3:]
                rot = st.Rotation.from_euler("xyz", rot_euler, degrees=True)

                self.pos_interp = si.interp1d(
                    self._times,
                    pos,
                    axis=0,
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                self.rot_interp = st.Slerp(self._times, rot)

    @property
    def times(self) -> np.ndarray:
        return self._times

    @property
    def poses(self) -> np.ndarray:
        return self._poses

    def trim(self, start_t: float, end_t: float) -> "PoseTrajectoryInterpolator":
        """
        Trim the trajectory to be between start_t and end_t
        """
        assert start_t <= end_t
        times = self.times
        should_keep = (start_t <= times) & (times <= end_t)

        # If no points are in range, create a new trajectory with just these two points
        if not np.any(should_keep):
            start_pose = self(start_t)
            end_pose = self(end_t)
            return PoseTrajectoryInterpolator(
                times=np.array([start_t, end_t]), poses=np.stack([start_pose, end_pose])
            )

        # Otherwise keep points in range and add boundary points
        keep_times = times[should_keep]
        keep_poses = self._poses[should_keep]

        # Add boundary points if they're not already included
        times_to_use = list(keep_times)
        poses_to_use = list(keep_poses)

        if start_t < keep_times[0]:
            times_to_use.insert(0, start_t)
            poses_to_use.insert(0, self(start_t))

        if end_t > keep_times[-1]:
            times_to_use.append(end_t)
            poses_to_use.append(self(end_t))

        return PoseTrajectoryInterpolator(
            times=np.array(times_to_use), poses=np.array(poses_to_use)
        )

    def drive_to_waypoint(
        self, pose, time, curr_time, max_pos_speed=np.inf, max_rot_speed=np.inf
    ) -> "PoseTrajectoryInterpolator":
        """
        Create a new trajectory that goes from the current pose to the target pose
        with speed limits.
        """
        assert max_pos_speed > 0
        assert max_rot_speed > 0
        time = max(time, curr_time)

        # Get current pose from interpolator
        curr_pose = self(curr_time)

        # Calculate position and rotation distances
        pos_dist, rot_dist = pose_distance(curr_pose, pose)

        # Calculate minimum duration based on speed limits
        pos_min_duration = pos_dist / max_pos_speed
        rot_min_duration = rot_dist / max_rot_speed
        duration = time - curr_time
        duration = max(duration, max(pos_min_duration, rot_min_duration))

        # Ensure we have some duration (avoid division by zero)
        duration = max(duration, 0.001)

        # Calculate the target time
        target_time = curr_time + duration

        # Create a new interpolator with just two points
        return PoseTrajectoryInterpolator(
            times=np.array([curr_time, target_time]), poses=np.stack([curr_pose, pose])
        )

    def schedule_waypoint(
        self, pose, time, max_pos_speed=np.inf, max_rot_speed=np.inf, curr_time=None
    ) -> "PoseTrajectoryInterpolator":
        """
        Simplified waypoint scheduling that avoids complex trimming operations
        """
        assert max_pos_speed > 0
        assert max_rot_speed > 0

        # If no current time specified, use the earliest time in the trajectory
        if curr_time is None:
            curr_time = self.times[0]

        # If target time is in the past, no change to trajectory
        if time <= curr_time:
            return self

        # Get current pose from interpolator
        curr_pose = self(curr_time)

        # Calculate position and rotation distances
        pos_dist, rot_dist = pose_distance(curr_pose, pose)

        # Calculate minimum duration based on speed limits
        pos_min_duration = pos_dist / max_pos_speed
        rot_min_duration = rot_dist / max_rot_speed
        duration = time - curr_time
        duration = max(duration, max(pos_min_duration, rot_min_duration))

        # Calculate the target time
        target_time = curr_time + duration

        # Get the poses at the current time and the waypoint
        # Create a simple two-point trajectory
        new_times = np.array([curr_time, target_time])
        new_poses = np.stack([curr_pose, pose])

        return PoseTrajectoryInterpolator(times=new_times, poses=new_poses)

    def __call__(self, t: Union[numbers.Number, np.ndarray]) -> np.ndarray:
        """
        Get interpolated pose(s) at time(s) t
        """
        is_single = False
        if isinstance(t, numbers.Number):
            is_single = True
            t = np.array([t])

        # Ensure we have at least one time point
        if len(t) == 0:
            if is_single:
                return self._poses[0]  # Return the first pose
            else:
                return np.zeros((0, 6))  # Return empty array with correct shape

        pose = np.zeros((len(t), 6))

        # Handle the case of a single step interpolation
        if self.single_step:
            pose[:] = self._poses[0]
            return pose[0] if is_single else pose

        # Normal interpolation with bounds checking
        t_clipped = np.clip(t, self.times[0], self.times[-1])

        try:
            # Get position interpolation
            pose[:, :3] = self.pos_interp(t_clipped)

            # Get rotation interpolation
            pose[:, 3:] = self.rot_interp(t_clipped).as_euler("xyz", degrees=True)
        except Exception as e:
            logger.error(f"Error in interpolation: {e}")
            # Fall back to nearest pose on error
            for i, ti in enumerate(t_clipped):
                # Find closest time
                idx = np.argmin(np.abs(self.times - ti))
                pose[i] = self._poses[idx]

        if is_single:
            pose = pose[0]
        return pose
