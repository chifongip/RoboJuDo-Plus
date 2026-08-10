import numpy as np


def get_fresh_zmq_velocity(ctrl_entry) -> np.ndarray | None:
    """Return planar ``[linear.x, linear.y, angular.z]`` from a fresh controller entry."""

    if not ctrl_entry.get("fresh", False):
        return None
    linear = np.asarray(ctrl_entry["linear_velocity"], dtype=np.float32)
    angular = np.asarray(ctrl_entry["angular_velocity"], dtype=np.float32)
    if linear.shape != (3,) or angular.shape != (3,):
        raise ValueError("VelocityZmqCtrl vectors must each have shape (3,)")
    if not np.isfinite(linear).all() or not np.isfinite(angular).all():
        raise ValueError("VelocityZmqCtrl vectors must be finite")
    return np.asarray([linear[0], linear[1], angular[2]], dtype=np.float32)


def clip_velocity(command, command_maps) -> np.ndarray:
    command = np.asarray(command, dtype=np.float32)
    bounds = np.asarray([[min(item[0], item[2]), max(item[0], item[2])] for item in command_maps], dtype=np.float32)
    return np.clip(command, bounds[:, 0], bounds[:, 1])
