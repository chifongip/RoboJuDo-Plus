import numpy as np

from robojudo.pipeline.four_mode_pipeline import ControlMode


class Gr00tLocomanipulationPipelineMixin:
    """Gate and rate-limit atomic GR00T arm and locomotion commands."""

    def _set_gr00t_takeover_state(self, enabled: bool):
        ctrl_manager = getattr(self, "ctrl_manager", None)
        controllers = getattr(ctrl_manager, "controllers", {})
        controller = controllers.get("Gr00tZmqCtrl")
        if controller is not None:
            return controller.inst.set_takeover_enabled(enabled)
        return False

    def _set_upper_body_enabled(self, enabled: bool):
        super()._set_upper_body_enabled(enabled)
        takeover_enabled = bool(
            self.mode == ControlMode.RL_DEFAULT
            and self._upper_body_enabled
            and self._upper_body_control_available()
        )
        if not takeover_enabled:
            self._set_gr00t_takeover_state(False)

    def _prepare_gr00t_stream(self, ctrl_data):
        stream = ctrl_data.get("Gr00tZmqCtrl", {})
        takeover_enabled = bool(
            self.mode == ControlMode.RL_DEFAULT
            and self._upper_body_enabled
            and self._upper_body_control_available()
        )
        stream["takeover_enabled"] = takeover_enabled
        session_changed = self._set_gr00t_takeover_state(takeover_enabled)
        if takeover_enabled and session_changed:
            # ctrl_data was read before this enable edge and belongs to the old session.
            stream["fresh"] = False
        # Reuse the existing arm override without changing its controller protocol.
        ctrl_data["UpperBodyZmqCtrl"] = stream
        return stream

    def _step_rl_policy(self, env_data, ctrl_data, dry_run: bool):
        self._prepare_gr00t_stream(ctrl_data)
        return super()._step_rl_policy(env_data, ctrl_data, dry_run)

    def _apply_pd_target_override(self, pd_target: np.ndarray, ctrl_data) -> np.ndarray:
        previous = self._upper_body_filtered.copy()
        target = super()._apply_pd_target_override(pd_target, ctrl_data)
        if not self._upper_body_indices.size or previous.shape != self._upper_body_filtered.shape:
            return target

        max_delta = float(self._upper_body_cfg.max_joint_velocity_rad_s) * self.dt
        limited = np.clip(
            target[self._upper_body_indices],
            previous - max_delta,
            previous + max_delta,
        ).astype(np.float32)
        self._upper_body_filtered = limited
        target[self._upper_body_indices] = limited
        return target
