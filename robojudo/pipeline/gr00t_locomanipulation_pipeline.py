import logging
import time

import numpy as np

from robojudo.pipeline.four_mode_pipeline import ControlMode

logger = logging.getLogger(__name__)


class Gr00tLocomanipulationPipelineMixin:
    """Gate and rate-limit atomic GR00T arm and locomotion commands."""

    def _gr00t_controller(self):
        ctrl_manager = getattr(self, "ctrl_manager", None)
        controllers = getattr(ctrl_manager, "controllers", {})
        controller = controllers.get("Gr00tZmqCtrl")
        return None if controller is None else controller.inst

    def _set_gr00t_takeover_state(self, enabled: bool, *, return_hand_to_default: bool = False):
        controller = self._gr00t_controller()
        if controller is not None:
            if return_hand_to_default:
                return controller.set_takeover_enabled(enabled, return_hand_to_default=True)
            return controller.set_takeover_enabled(enabled)
        return False

    def _upper_body_enable_available(self) -> bool:
        if not super()._upper_body_enable_available():
            return False
        cfg = getattr(self, "_upper_body_cfg", None)
        if cfg is None or not getattr(cfg, "offline_dagger_enabled", False):
            return True
        controller = self._gr00t_controller()
        # Offline DAgger may start policy takeover only after dex-teleop has
        # echoed fresh measured feedback from this exact observation stream.
        if controller is None or not controller.expert_stream_ready():
            logger.warning(
                "Ignored GR00T enable: offline DAgger VR stream is not ready"
            )
            return False
        return True

    def _set_upper_body_enabled(self, enabled: bool):
        was_enabled = getattr(self, "_upper_body_enabled", False)
        super()._set_upper_body_enabled(enabled)
        is_enabled = getattr(self, "_upper_body_enabled", False)
        takeover_enabled = bool(
            self.mode == ControlMode.RL_DEFAULT and is_enabled and self._upper_body_control_available()
        )
        if not takeover_enabled:
            self._set_gr00t_takeover_state(
                False,
                return_hand_to_default=bool(was_enabled and not is_enabled),
            )

    def _prepare_gr00t_stream(self, ctrl_data):
        stream = ctrl_data.get("Gr00tZmqCtrl", {})
        cfg = getattr(self, "_upper_body_cfg", None)
        if (
            getattr(cfg, "offline_dagger_enabled", False)
            and self._upper_body_enabled
            and not stream.get("expert_stream_ready", False)
        ):
            # Offline DAgger requires VR readiness to remain live throughout a
            # rollout; losing it disables upper-body takeover immediately.
            self._set_upper_body_enabled(False)
        takeover_enabled = bool(
            self.mode == ControlMode.RL_DEFAULT and self._upper_body_enabled and self._upper_body_control_available()
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

    def _post_mode_step(self, env_data, ctrl_data, extras, pd_target, rl_active: bool):
        # Offline DAgger recording needs the exact policy/expert arbitration
        # result that produced this control step; keep it only for this call.
        self._gr00t_record_stream = ctrl_data.get("Gr00tZmqCtrl", {})
        try:
            return super()._post_mode_step(env_data, ctrl_data, extras, pd_target, rl_active)
        finally:
            self._gr00t_record_stream = None

    def _record_upper_body_sample(self, env_data, extras, pd_target, *, rl_active: bool):
        cfg = getattr(self, "_upper_body_cfg", None)
        if cfg is None or not getattr(cfg, "offline_dagger_enabled", False):
            return super()._record_upper_body_sample(
                env_data, extras, pd_target, rl_active=rl_active
            )
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
            logger.warning("Skipped offline DAgger recording frame without locomotion command")
            return
        stream = getattr(self, "_gr00t_record_stream", None) or {}
        hand_data = stream.get("casia_hand")
        if not hand_data or not hand_data.get("fresh", False):
            return
        hand_names = list(hand_data.get("joint_names", ()))
        hand_positions = np.asarray(hand_data.get("joint_positions"), dtype=np.float32)
        hand_commands = np.asarray(hand_data.get("joint_position_commands"), dtype=np.float32)
        if (
            not hand_names
            or hand_positions.shape != (len(hand_names),)
            or hand_commands.shape != (len(hand_names),)
        ):
            logger.warning("Skipped offline DAgger recording frame with invalid CASIA snapshot")
            return

        arm_positions = np.asarray(env_data.dof_pos, dtype=np.float32)[self._upper_body_indices]
        arm_commands = np.asarray(pd_target, dtype=np.float32)[self._upper_body_indices]
        # Offline DAgger stores the full rollout and an explicit per-frame label;
        # dataset finalization can then select expert-applied contiguous chunks.
        dagger = {
            "expert_intervention": bool(stream.get("expert_intervention", False)),
            "expert_applied": bool(stream.get("expert_applied", False)),
            "intervention_session": int(stream.get("intervention_session", 0)),
            "action_source": str(stream.get("action_source", "policy")),
            # Offline DAgger provenance remains in the finalized full rollout;
            # -1 is used only after finalization for frames without a candidate.
            "expert_frame_id": stream.get("expert_frame_id"),
        }
        recorder_client.submit(
            joint_names=[*cfg.joint_names, *hand_names],
            joint_positions=np.concatenate((arm_positions, hand_positions)),
            joint_position_commands=np.concatenate((arm_commands, hand_commands)),
            velocity_height_command=np.asarray(locomotion_command, dtype=np.float32)[:4],
            timestamp_ns=time.monotonic_ns(),
            dagger=dagger,
        )
