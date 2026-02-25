import numpy as np
from scipy.spatial.transform import Rotation as R


def vec6dof_to_homogeneous_matrix(vec6: np.ndarray) -> np.ndarray:
    """Convert a 6-DOF vector [x, y, z, rx, ry, rz] to a 4x4 homogeneous matrix."""
    T = np.eye(4)
    T[:3, 3] = vec6[:3]
    T[:3, :3] = R.from_rotvec(vec6[3:]).as_matrix()
    return T


def invert_transformation(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 homogeneous transformation matrix."""
    R_mat = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_mat.T
    T_inv[:3, 3] = -R_mat.T @ t
    return T_inv


def relative_transformation(T0: np.ndarray, Tt: np.ndarray) -> np.ndarray:
    """Compute relative transformation: T_rel = T0^{-1} @ Tt."""
    return invert_transformation(T0) @ Tt


def homogeneous_matrix_to_6dof(T: np.ndarray) -> np.ndarray:
    """Convert a 4x4 homogeneous matrix to a 6-DOF vector [x, y, z, rx, ry, rz]."""
    translation = T[:3, 3]
    rotvec = R.from_matrix(T[:3, :3]).as_rotvec()
    return np.concatenate((translation, rotvec))


def absolute_to_relative(abs_actions: np.ndarray) -> np.ndarray:
    """
    Convert a sequence of absolute 12D actions to relative.

    Args:
        abs_actions: [T, 12] absolute actions where [:, :6] is arm SE(3) pose
                     and [:, 6:] is hand encoder readings.

    Returns:
        [T, 12] relative actions. Arm uses SE(3) relative transform from
        the first timestep; hand uses simple subtraction.
    """
    T_len = abs_actions.shape[0]
    arm_abs = abs_actions[:, :6]
    hand_abs = abs_actions[:, 6:]

    # Arm: SE(3) relative from first timestep
    T0 = vec6dof_to_homogeneous_matrix(arm_abs[0])
    arm_rel = np.zeros_like(arm_abs)
    for i in range(T_len):
        Tt = vec6dof_to_homogeneous_matrix(arm_abs[i])
        T_rel = relative_transformation(T0, Tt)
        arm_rel[i] = homogeneous_matrix_to_6dof(T_rel)

    # Hand: simple subtraction from first timestep
    hand_rel = hand_abs - hand_abs[0:1]

    return np.concatenate([arm_rel, hand_rel], axis=-1).astype(np.float32)


def relative_to_absolute(
    rel_actions: np.ndarray, current_state: np.ndarray
) -> np.ndarray:
    """
    Convert predicted relative 12D actions back to absolute, given the current state.

    Args:
        rel_actions: [T, 12] relative actions (arm SE(3) relative + hand subtraction).
        current_state: [12,] current absolute state used as reference.

    Returns:
        [T, 12] absolute actions.
    """
    T_len = rel_actions.shape[0]
    arm_rel = rel_actions[:, :6]
    hand_rel = rel_actions[:, 6:]

    current_arm = current_state[:6]
    current_hand = current_state[6:]

    # Arm: T_target = T_current @ T_relative
    T_current = vec6dof_to_homogeneous_matrix(current_arm)
    arm_abs = np.zeros_like(arm_rel)
    for i in range(T_len):
        T_rel = vec6dof_to_homogeneous_matrix(arm_rel[i])
        T_target = T_current @ T_rel
        arm_abs[i] = homogeneous_matrix_to_6dof(T_target)

    # Hand: simple addition
    hand_abs = hand_rel + current_hand

    return np.concatenate([arm_abs, hand_abs], axis=-1).astype(np.float32)
