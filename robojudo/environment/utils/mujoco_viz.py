import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as sRot

from robojudo.utils.rotation import TransformAlignment


class MujocoVisualizer:
    def __init__(self, viewer, alignment: TransformAlignment | None = None):
        self.viewer = viewer
        self.alignment = alignment

    # TODO: reset: clear all markers

    def update_teleop_frames(self, torso, target_poses):
        """Draw RGB XYZ axes in MuJoCo world coordinates, without alignment remapping."""
        self._draw_teleop_frame(np.eye(4), "world (fixed)", 6012, (1, 1, 1, 0.8), 0.2)
        self._draw_teleop_frame(torso, "XR torso reference", 6000, (1, 1, 1, 0.8), 0.15)
        for side, marker_id, color in (
            ("left", 6004, (1, 0.8, 0, 0.8)),
            ("right", 6008, (0, 0.9, 1, 0.8)),
        ):
            local = target_poses.get(side)
            world = None if local is None else torso @ np.asarray(local, dtype=float)
            self._draw_teleop_frame(world, f"{side} hand target", marker_id, color, 0.09)

    def _draw_teleop_frame(self, transform, label, marker_id, color, length):
        visible = transform is not None
        transform = np.eye(4) if transform is None else transform
        self.viewer.add_marker(
            id=marker_id,
            pos=transform[:3, 3],
            size=np.full(3, 0.015),
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            rgba=color if visible else (0, 0, 0, 0),
            label=label if visible else "",
        )
        axis_rotations = (
            np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
            np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
            np.eye(3),
        )
        for index, rotation in enumerate(axis_rotations):
            rgba = [0.0, 0.0, 0.0, 0.0]
            rgba[index] = 1.0
            rgba[3] = 0.9 if visible else 0.0
            self.viewer.add_marker(
                id=marker_id + index + 1,
                pos=transform[:3, 3],
                mat=transform[:3, :3] @ rotation,
                size=np.array([0.004, 0.004, length]),
                type=mujoco.mjtGeom.mjGEOM_ARROW,
                rgba=rgba,
                label="",
            )

    def update_rg_view(self, body_pos, body_rot, humanoid_id):
        if humanoid_id == 0:
            rgba = (1, 0, 0, 1)
        elif humanoid_id == 1:
            rgba = (0, 1, 0, 1)
        else:
            return

        for j in range(body_pos.shape[0]):
            # need to modify mujoco_viewer to support this
            self.viewer.add_marker(
                pos=body_pos[j],
                size=0.05,
                rgba=rgba,
                type=mujoco.mjtGeom.mjGEOM_SPHERE,  # pyright: ignore[reportAttributeAccessIssue]
                label="",
                id=humanoid_id * 1000 + j,
            )

    def draw_arrow(
        self,
        origin,
        root_quat,
        vec_local,
        color,
        scale=1.0,
        horizontal_only=False,
        id=0,
        aligned_frame=False,
    ):
        if aligned_frame and self.alignment is not None:
            root_quat, origin = self.alignment.unalign_transform(root_quat, origin)

        vec_local = np.array(vec_local, dtype=float)
        r = sRot.from_quat(root_quat)

        if horizontal_only:
            yaw = r.as_euler("xyz", degrees=False)[2]
            yaw_rot = sRot.from_euler("z", yaw).as_matrix()
            vec_world = yaw_rot @ vec_local
        else:
            vec_world = r.as_matrix() @ vec_local

        length = np.linalg.norm(vec_world)
        if length > 1e-6:
            dir_world = vec_world / length
            scaled_length = length * scale

            rot, _ = sRot.align_vectors([dir_world], [[0, 0, 1]])
            mat = rot.as_matrix()

            center_pos = origin  # dir_world * scaled_length * 0.5
        else:
            # zero length
            scaled_length = 0
            mat = np.eye(3)
            center_pos = origin

        self.viewer.add_marker(
            pos=center_pos,
            mat=mat,
            size=np.array([0.02, 0.02, scaled_length]),
            rgba=np.array(color),
            type=mujoco.mjtGeom.mjGEOM_ARROW,  # pyright: ignore[reportAttributeAccessIssue]
            id=3000 + id,
        )
