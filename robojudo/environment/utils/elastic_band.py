import logging

import mujoco
import numpy as np

from robojudo.environment.env_cfgs import ElasticBandCfg

logger = logging.getLogger(__name__)


class ElasticBand:
    """Apply a tension-only spring-damper force to a MuJoCo body."""

    BAND_MARKER_ID = 9000
    ANCHOR_MARKER_ID = 9001

    def __init__(self, cfg: ElasticBandCfg, model, data):
        self.cfg = cfg
        self.model = model
        self.data = data
        self.anchor_point = np.asarray(cfg.anchor_point, dtype=np.float64)
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.body_name)
        if self.body_id < 0:
            raise ValueError(f"ElasticBand body '{cfg.body_name}' was not found in the MuJoCo model")

        self.rest_length = float(cfg.rest_length)
        self.active = bool(cfg.active)
        logger.info(
            "ElasticBand attached to %s at %s (active=%s)",
            cfg.body_name,
            self.anchor_point.tolist(),
            self.active,
        )

    @staticmethod
    def compute_force(
        position: np.ndarray,
        velocity: np.ndarray,
        anchor_point: np.ndarray,
        rest_length: float,
        stiffness: float,
        damping: float,
    ) -> np.ndarray:
        position = np.asarray(position, dtype=np.float64)
        velocity = np.asarray(velocity, dtype=np.float64)
        anchor_point = np.asarray(anchor_point, dtype=np.float64)
        if not all(np.isfinite(value).all() for value in (position, velocity, anchor_point)):
            return np.zeros(3, dtype=np.float64)

        displacement = anchor_point - position
        distance = float(np.linalg.norm(displacement))
        if not np.isfinite(distance) or distance <= np.finfo(np.float64).eps:
            return np.zeros(3, dtype=np.float64)

        extension = distance - rest_length
        if extension <= 0.0:
            return np.zeros(3, dtype=np.float64)

        direction = displacement / distance
        radial_velocity = float(np.dot(velocity, direction))
        magnitude = max(0.0, stiffness * extension - damping * radial_velocity)
        force = magnitude * direction
        return force if np.isfinite(force).all() else np.zeros(3, dtype=np.float64)

    def _body_linear_velocity(self) -> np.ndarray:
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.body_id,
            velocity,
            0,
        )
        return velocity[3:]

    def apply(self) -> np.ndarray:
        force = np.zeros(3, dtype=np.float64)
        if self.active:
            force = self.compute_force(
                position=self.data.xpos[self.body_id],
                velocity=self._body_linear_velocity(),
                anchor_point=self.anchor_point,
                rest_length=self.rest_length,
                stiffness=self.cfg.stiffness,
                damping=self.cfg.damping,
            )
        self.data.xfrc_applied[self.body_id, :3] = force
        return force

    def update_visualization(self, viewer):
        if not self.cfg.visualize:
            return

        body_position = np.asarray(self.data.xpos[self.body_id], dtype=np.float64)
        displacement = self.anchor_point - body_position
        distance = float(np.linalg.norm(displacement))
        valid = np.isfinite(body_position).all() and np.isfinite(distance)

        midpoint = (body_position + self.anchor_point) * 0.5 if valid else self.anchor_point
        rotation = np.eye(3, dtype=np.float64)
        if valid and distance > np.finfo(np.float64).eps:
            quaternion = np.zeros(4, dtype=np.float64)
            mujoco.mju_quatZ2Vec(quaternion, displacement)
            mujoco.mju_quat2Mat(rotation.ravel(), quaternion)

        rgba = np.asarray(self.cfg.visual_rgba, dtype=np.float64).copy()
        if not self.active or not valid:
            rgba[3] = 0.0
        half_length = distance * 0.5 if valid else 0.0
        viewer.add_marker(
            pos=midpoint,
            mat=rotation,
            size=np.array([self.cfg.visual_radius, self.cfg.visual_radius, half_length]),
            rgba=rgba,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            label="",
            id=self.BAND_MARKER_ID,
        )
        viewer.add_marker(
            pos=self.anchor_point,
            size=np.full(3, self.cfg.anchor_radius),
            rgba=rgba,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            label="",
            id=self.ANCHOR_MARKER_ID,
        )

    def toggle(self) -> bool:
        self.active = not self.active
        if not self.active:
            self.data.xfrc_applied[self.body_id, :3] = 0.0
        logger.info("ElasticBand %s", "enabled" if self.active else "disabled")
        return self.active

    def lower(self) -> float:
        self.rest_length += self.cfg.length_step
        logger.info("ElasticBand rest length: %.3f m", self.rest_length)
        return self.rest_length

    def lift(self) -> float:
        self.rest_length = max(0.0, self.rest_length - self.cfg.length_step)
        logger.info("ElasticBand rest length: %.3f m", self.rest_length)
        return self.rest_length

    def reset(self):
        self.rest_length = float(self.cfg.rest_length)
        self.active = bool(self.cfg.active)
        self.data.xfrc_applied[self.body_id, :3] = 0.0
