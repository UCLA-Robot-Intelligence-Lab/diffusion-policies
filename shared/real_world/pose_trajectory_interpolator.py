import numpy as np
import scipy.interpolate as si
import scipy.spatial.transform as st
import numbers


def rotation_distance(a: st.Rotation, b: st.Rotation) -> float:
    """Compute the magnitude of relative rotation between two orientations."""
    return (b * a.inv()).magnitude()


def pose_distance(start_pose, end_pose):
    """
    start_pose, end_pose are each [x, y, z, ax, ay, az] in mm + axis-angle(radians).
    """
    start_pose = np.array(start_pose, dtype=np.float64)
    end_pose = np.array(end_pose, dtype=np.float64)

    start_pos = start_pose[:3]
    end_pos = end_pose[:3]
    pos_dist = np.linalg.norm(end_pos - start_pos)

    # Convert axis-angle -> Rotation
    start_rot = st.Rotation.from_rotvec(start_pose[3:6])
    end_rot = st.Rotation.from_rotvec(end_pose[3:6])
    rot_dist = rotation_distance(start_rot, end_rot)
    return pos_dist, rot_dist


class PoseTrajectoryInterpolator:
    """
    Spline/linear interpolation for positions in R^3 plus SLERP for rotations in SO(3),
    where rotations are stored as axis-angle (rotvec).
    """

    def __init__(self, times, poses):
        """
        times: ascending list/array of monotonic times
        poses: same shape, each = [x, y, z, ax, ay, az]
        """
        times = np.array(times, dtype=np.float64)
        poses = np.array(poses, dtype=np.float64)
        assert times.shape[0] == poses.shape[0] >= 1

        self._times = times
        self._poses = poses
        self._single_step = len(times) == 1

        if not self._single_step:
            # We do linear interpolation for the xyz, and slerp for rotation
            self._pos_interp = si.interp1d(
                times, poses[:, :3], axis=0, assume_sorted=True
            )

            # Convert each axis-angle to a Rotation
            rots = st.Rotation.from_rotvec(poses[:, 3:6])
            self._rot_slerp = st.Slerp(times, rots)

    @property
    def times(self):
        return self._times

    @property
    def poses(self):
        """
        Return the key poses used by the interpolator, each [x, y, z, ax, ay, az].
        """
        return self._poses

    def __call__(self, t):
        """
        Evaluate the interpolator at time t (or array of times). Return [x, y, z, ax, ay, az].
        """
        single = False
        if isinstance(t, numbers.Number):
            t = np.array([t], dtype=np.float64)
            single = True

        # clip times to [t0, tN]
        t0, tN = self._times[0], self._times[-1]
        t_clamped = np.clip(t, t0, tN)

        out = np.zeros((len(t_clamped), 6), dtype=np.float64)
        if self._single_step:
            # We only have one pose, so everything is constant
            out[:] = self._poses[0]
        else:
            out[:, :3] = self._pos_interp(t_clamped)
            rots = self._rot_slerp(t_clamped)
            out[:, 3:6] = rots.as_rotvec()

        if single:
            return out[0]
        else:
            return out

    def trim(self, start_t, end_t):
        """
        Return a new PoseTrajectoryInterpolator restricted to [start_t, end_t].
        """
        # Evaluate at boundary times + any original sample within (start_t, end_t).
        times_in = self._times
        mask = (start_t < times_in) & (times_in < end_t)
        new_times = np.concatenate([[start_t], times_in[mask], [end_t]])
        new_times = np.unique(new_times)
        new_poses = self.__call__(new_times)
        return PoseTrajectoryInterpolator(new_times, new_poses)

    def drive_to_waypoint(
        self, pose, time, curr_time, max_pos_speed=np.inf, max_rot_speed=np.inf
    ):
        """
        Insert a new waypoint at `time`, ensuring we do not exceed speed constraints from
        the current pose at `curr_time` to the new `pose`.
        """
        if time < curr_time:
            time = curr_time
        # Evaluate current pose
        curr_pose = self(curr_time)
        pos_dist, rot_dist = pose_distance(curr_pose, pose)
        dt_desired = time - curr_time
        # Minimum time needed given the speed limits
        pos_dt_min = pos_dist / max_pos_speed
        rot_dt_min = rot_dist / max_rot_speed
        needed = max(pos_dt_min, rot_dt_min)
        if needed > dt_desired:
            # push time out
            time = curr_time + needed

        # Trim away everything after curr_time (we re-build from that point)
        truncated = self.trim(self._times[0], curr_time)
        # Add new waypoint
        new_times = np.concatenate([truncated.times, [time]])
        new_poses = np.vstack([truncated.poses, pose])
        return PoseTrajectoryInterpolator(new_times, new_poses)

    def schedule_waypoint(
        self,
        pose,
        time,
        max_pos_speed=np.inf,
        max_rot_speed=np.inf,
        curr_time=None,
        last_waypoint_time=None,
    ):
        """
        Insert a future waypoint at `time` (monotonic time). If `time <= curr_time`, no effect.
        If needed, we extend that time to satisfy speed limits from the last known pose.
        """
        if time <= curr_time:
            # no effect
            return self

        # figure out from where we are going to move
        if last_waypoint_time is None:
            last_waypoint_time = self._times[-1]
        if curr_time is None:
            curr_time = time
        start_time = max(curr_time, last_waypoint_time)
        if start_time > time:
            # nothing
            return self

        # Evaluate pose at start_time
        base_pose = self(start_time)
        pos_dist, rot_dist = pose_distance(base_pose, pose)
        dt_desired = time - start_time
        pos_dt_min = pos_dist / max_pos_speed
        rot_dt_min = rot_dist / max_rot_speed
        needed = max(pos_dt_min, rot_dt_min)
        if needed > dt_desired:
            time = start_time + needed

        # Trim away everything after start_time
        truncated = self.trim(self._times[0], start_time)
        new_times = np.concatenate([truncated.times, [time]])
        new_poses = np.vstack([truncated.poses, pose])
        return PoseTrajectoryInterpolator(new_times, new_poses)
