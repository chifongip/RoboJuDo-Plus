import numpy as np

from robojudo.pipeline import pipeline_registry
from robojudo.pipeline.four_mode_pipeline import ControlMode
from robojudo.pipeline.x2_locomanipulation_pipeline import X2LocomanipulationPipeline


@pipeline_registry.register
class X2Gr00tLocomanipulationPipeline(X2LocomanipulationPipeline):
    """Couple atomic GR00T arm and locomotion commands to one takeover gate."""

    def _prepare_gr00t_stream(self, ctrl_data):
        stream = ctrl_data.get("Gr00tZmqCtrl", {})
        stream["takeover_enabled"] = bool(
            self.mode == ControlMode.RL_DEFAULT
            and self._upper_body_enabled
            and self._upper_body_control_available()
        )
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
