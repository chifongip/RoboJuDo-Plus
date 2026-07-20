import logging

import numpy as np

from robojudo.controller.ctrl_cfgs import UpperBodyZmqCtrlCfg

logger = logging.getLogger(__name__)


class UpperBodyZmqPipelineMixin:
    """Named, filtered upper-body targets for pipelines with partial-body policies."""

    def __init__(self, cfg):
        self._upper_body_cfg = next(
            (ctrl_cfg for ctrl_cfg in cfg.ctrl if isinstance(ctrl_cfg, UpperBodyZmqCtrlCfg)),
            None,
        )
        self._upper_body_enabled = False
        self._upper_body_stream_was_fresh = False
        self._upper_body_indices = np.asarray([], dtype=np.int32)
        self._upper_body_default = np.asarray([], dtype=np.float32)
        self._upper_body_filtered = np.asarray([], dtype=np.float32)
        super().__init__(cfg=cfg)
        self._configure_upper_body_override()

    def reset(self):
        self._upper_body_enabled = False
        self._upper_body_stream_was_fresh = False
        super().reset()
        if self._upper_body_indices.size:
            self._upper_body_filtered = self.env.dof_pos[self._upper_body_indices].astype(np.float32)

    def _upper_body_action_joint_names(self) -> list[str]:
        return self.cfg.policy.action_dof.joint_names

    def _upper_body_control_available(self) -> bool:
        return True

    def _upper_body_enable_available(self) -> bool:
        if not self._upper_body_control_available():
            logger.warning("Ignored upper-body enable while upper-body control is unavailable")
            return False
        return True

    def _configure_upper_body_override(self):
        if self._upper_body_cfg is None:
            return
        missing = [name for name in self._upper_body_cfg.joint_names if name not in self.env.joint_names]
        if missing:
            raise ValueError(f"Upper-body ZMQ joints missing from environment: {missing}")
        action_joints = set(self._upper_body_action_joint_names())
        overlap = sorted(action_joints.intersection(self._upper_body_cfg.joint_names))
        if overlap:
            raise ValueError(f"Upper-body ZMQ joints overlap policy actions: {overlap}")

        self._upper_body_indices = np.asarray(
            [self.env.joint_names.index(name) for name in self._upper_body_cfg.joint_names],
            dtype=np.int32,
        )
        self._upper_body_default = self.env.default_pos[self._upper_body_indices].astype(np.float32)
        self._upper_body_filtered = self.env.dof_pos[self._upper_body_indices].astype(np.float32)

    def _set_upper_body_enabled(self, enabled: bool):
        enabled = bool(enabled and getattr(self, "_upper_body_cfg", None) is not None)
        if getattr(self, "_upper_body_enabled", False) == enabled:
            return
        self._upper_body_enabled = enabled
        self._upper_body_stream_was_fresh = False
        logger.warning("Upper-body ZMQ control %s", "enabled" if enabled else "disabled")

    def _toggle_upper_body(self):
        if self._upper_body_cfg is None:
            logger.warning("Ignored upper-body toggle: ZMQ controller is not configured")
        elif self._upper_body_enabled:
            self._set_upper_body_enabled(False)
        elif not self._upper_body_enable_available():
            return
        else:
            self._set_upper_body_enabled(True)

    def _apply_pd_target_override(self, pd_target: np.ndarray, ctrl_data) -> np.ndarray:
        return self._apply_upper_body_override(pd_target, ctrl_data)

    def _apply_upper_body_override(self, pd_target: np.ndarray, ctrl_data) -> np.ndarray:
        if self._upper_body_cfg is None:
            return pd_target

        desired = self._upper_body_default.copy()
        stream_data = ctrl_data.get("UpperBodyZmqCtrl", {})
        stream_is_fresh = bool(stream_data.get("fresh", False))
        if self._upper_body_enabled and stream_is_fresh:
            positions = stream_data.get("joint_positions", {})
            for local_index, name in enumerate(self._upper_body_cfg.joint_names):
                if name in positions:
                    desired[local_index] = positions[name]
            limits = self.env.position_limits[self._upper_body_indices]
            desired = np.clip(desired, limits[:, 0], limits[:, 1])
            if not self._upper_body_stream_was_fresh:
                logger.info("Upper-body ZMQ stream active")
        elif self._upper_body_enabled and self._upper_body_stream_was_fresh:
            logger.warning("Upper-body ZMQ stream timed out; returning to defaults")

        self._upper_body_stream_was_fresh = self._upper_body_enabled and stream_is_fresh
        alpha = self._upper_body_cfg.ema_alpha
        self._upper_body_filtered = alpha * self._upper_body_filtered + (1.0 - alpha) * desired
        snap = np.abs(self._upper_body_filtered - desired) < 0.001
        self._upper_body_filtered[snap] = desired[snap]

        target = np.asarray(pd_target, dtype=np.float32).copy()
        target[self._upper_body_indices] = self._upper_body_filtered
        return target
