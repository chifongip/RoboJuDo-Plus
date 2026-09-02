import logging
import time

import numpy as np

logger = logging.getLogger(__name__)


class UpperBodyHandZmqPipelineMixin:
    """Add physical OmniHand takeover and recording to an upper-body pipeline."""

    def _set_upper_body_enabled(self, enabled: bool):
        was_enabled = getattr(self, "_upper_body_enabled", False)
        super()._set_upper_body_enabled(enabled)
        is_enabled = getattr(self, "_upper_body_enabled", False)
        if was_enabled == is_enabled:
            return
        controller = self.ctrl_manager.controllers.get("UpperBodyHandZmqCtrl")
        if controller is not None:
            controller.inst.set_takeover_enabled(is_enabled)

    def _apply_pd_target_override(self, pd_target: np.ndarray, ctrl_data) -> np.ndarray:
        stream_data = ctrl_data.get("UpperBodyHandZmqCtrl")
        if stream_data is None:
            return super()._apply_pd_target_override(pd_target, ctrl_data)
        upper_body_ctrl_data = dict(ctrl_data)
        upper_body_ctrl_data["UpperBodyZmqCtrl"] = stream_data
        return super()._apply_pd_target_override(pd_target, upper_body_ctrl_data)

    def _post_mode_step(self, env_data, ctrl_data, extras, pd_target, rl_active: bool):
        self._upper_body_hand_ctrl_data = ctrl_data
        try:
            return super()._post_mode_step(env_data, ctrl_data, extras, pd_target, rl_active)
        finally:
            self._upper_body_hand_ctrl_data = None

    def _record_upper_body_sample(self, env_data, extras, pd_target, *, rl_active: bool):
        recorder_client = getattr(self, "_recorder_client", None)
        if recorder_client is None or not self._recording_active:
            return

        can_record = rl_active and self._upper_body_enabled and self._upper_body_control_available()
        if not can_record:
            self._finish_recording_episode()
            return
        if self._recording_paused or not self._upper_body_stream_was_fresh:
            return

        locomotion_command = extras.get("locomotion_command")
        if locomotion_command is None or len(locomotion_command) < 4:
            logger.warning("Skipped recording frame without a velocity/height command")
            return

        ctrl_data = getattr(self, "_upper_body_hand_ctrl_data", None) or {}
        hand_data = ctrl_data.get("UpperBodyHandZmqCtrl", {}).get("omnihand")
        if not hand_data or not hand_data.get("fresh", False):
            return
        hand_names = list(hand_data.get("joint_names", []))
        hand_positions = np.asarray(hand_data.get("joint_positions"), dtype=np.float32)
        hand_commands = np.asarray(hand_data.get("joint_position_commands"), dtype=np.float32)
        expected_shape = (len(hand_names),)
        if not hand_names or hand_positions.shape != expected_shape or hand_commands.shape != expected_shape:
            logger.warning("Skipped recording frame with an invalid OmniHand snapshot")
            return

        arm_positions = np.asarray(env_data.dof_pos, dtype=np.float32)[self._upper_body_indices]
        arm_commands = np.asarray(pd_target, dtype=np.float32)[self._upper_body_indices]
        recorder_client.submit(
            joint_names=[*self._upper_body_cfg.joint_names, *hand_names],
            joint_positions=np.concatenate((arm_positions, hand_positions)),
            joint_position_commands=np.concatenate((arm_commands, hand_commands)),
            velocity_height_command=np.asarray(locomotion_command, dtype=np.float32)[:4],
            timestamp_ns=time.monotonic_ns(),
        )


__all__ = ["UpperBodyHandZmqPipelineMixin"]
