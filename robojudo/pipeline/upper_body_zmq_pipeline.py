import logging
import time

import numpy as np

from robojudo.controller.ctrl_cfgs import UpperBodyZmqCtrlCfg
from robojudo.recording import RecorderClient

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
        self._upper_body_generation = None
        self._upper_body_pending = None
        self._upper_body_command_submitted = False
        self._upper_body_indices = np.asarray([], dtype=np.int32)
        self._upper_body_default = np.asarray([], dtype=np.float32)
        self._upper_body_filtered = np.asarray([], dtype=np.float32)
        self._recorder_client = None
        self._recording_active = False
        self._recording_paused = False
        self._recording_review_pending = False
        super().__init__(cfg=cfg)
        self._configure_upper_body_override()
        if cfg.record.enabled:
            if self._upper_body_cfg is None:
                raise ValueError("recording requires an UpperBodyZmqCtrl configuration")
            self._recorder_client = RecorderClient(cfg.record, robot_type=cfg.robot)

    def reset(self):
        recorder_client = getattr(self, "_recorder_client", None)
        if recorder_client is not None and getattr(self, "_recording_active", False):
            recorder_client.finish_episode(save=True)
        self._recording_active = False
        self._recording_paused = False
        self._upper_body_enabled = False
        self._upper_body_stream_was_fresh = False
        self._upper_body_pending = None
        self._upper_body_command_submitted = False
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
        if self._upper_body_cfg.upper_body_default_pose is None:
            self._upper_body_default = self.env.default_pos[self._upper_body_indices].astype(np.float32)
        else:
            self._upper_body_default = np.asarray(
                self._upper_body_cfg.upper_body_default_pose,
                dtype=np.float32,
            )
        self._upper_body_filtered = self.env.dof_pos[self._upper_body_indices].astype(np.float32)

    def _set_upper_body_enabled(self, enabled: bool):
        enabled = bool(enabled and getattr(self, "_upper_body_cfg", None) is not None)
        if getattr(self, "_upper_body_enabled", False) == enabled:
            return
        self._upper_body_enabled = enabled
        self._upper_body_stream_was_fresh = False
        self._upper_body_pending = None
        manager = getattr(self, "ctrl_manager", None)
        controller_type = self._upper_body_controller_type()
        entry = getattr(manager, "controllers", {}).get(controller_type)
        if entry is not None:
            receiver = entry["inst"] if isinstance(entry, dict) else entry.inst
            if controller_type == "UpperBodyZmqCtrl" or getattr(self._upper_body_cfg, "feedback_endpoint", None):
                receiver.reset()
                self._upper_body_generation = receiver.reset_generation
            else:
                receiver.set_takeover_enabled(enabled)
        logger.warning("Upper-body ZMQ control %s", "enabled" if enabled else "disabled")

    def _upper_body_controller_type(self):
        return getattr(self._upper_body_cfg, "ctrl_type", getattr(self, "hand_controller_type", "UpperBodyZmqCtrl"))

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
        receiver = self._feedback_receiver()
        if receiver is not None:
            return self._apply_feedback_proposal(pd_target, ctrl_data, receiver)

        desired = self._upper_body_default.copy()
        stream_data = ctrl_data.get(self._upper_body_controller_type(), {})
        stream_is_fresh = bool(stream_data.get("fresh", False))
        generation = getattr(self, "_upper_body_generation", None)
        if generation is not None:
            stream_generation = stream_data.get("reset_generation")
            stream_is_fresh = stream_is_fresh and isinstance(stream_generation, int) and stream_generation >= generation
            if isinstance(stream_generation, int) and stream_generation >= generation:
                self._upper_body_generation = stream_generation
        positions = stream_data.get("joint_positions", {})
        stream_is_fresh = (
            stream_is_fresh and isinstance(positions, dict) and set(positions) == set(self._upper_body_cfg.joint_names)
        )
        received_at = stream_data.get("joint_received_at")
        if received_at is not None:
            now = time.monotonic()
            # Inference occurs after CtrlManager samples the receiver. Recheck
            # leases at application time, rather than trusting an earlier bool.
            stream_is_fresh = stream_is_fresh and all(
                name in received_at and 0 <= now - received_at[name] <= self._upper_body_cfg.timeout_s
                for name in self._upper_body_cfg.joint_names
            )
        if self._upper_body_enabled and not stream_is_fresh:
            if self._upper_body_stream_was_fresh:
                logger.warning("Upper-body ZMQ stream timed out or became incomplete; holding joint targets")
            self._upper_body_stream_was_fresh = False
            target = np.asarray(pd_target, dtype=np.float32).copy()
            target[self._upper_body_indices] = self._upper_body_filtered
            return target
        if self._upper_body_enabled and stream_is_fresh:
            for local_index, name in enumerate(self._upper_body_cfg.joint_names):
                if name in positions:
                    # Narrow only after bounding to the destination dtype; large
                    # finite JSON values must not overflow before joint clipping.
                    numeric_limit = np.finfo(desired.dtype).max
                    desired[local_index] = np.clip(positions[name], -numeric_limit, numeric_limit)
            if not self._upper_body_stream_was_fresh:
                logger.info("Upper-body ZMQ stream active")

        limits = self.env.position_limits[self._upper_body_indices]
        desired = np.clip(desired, limits[:, 0], limits[:, 1])
        self._upper_body_stream_was_fresh = self._upper_body_enabled and stream_is_fresh
        alpha = self._upper_body_cfg.ema_alpha
        previous = self._upper_body_filtered
        filtered = alpha * previous + (1.0 - alpha) * desired
        snap = np.abs(filtered - desired) < 0.001
        filtered[snap] = desired[snap]
        # Limit after smoothing and snapping so every emitted step respects the cap.
        max_delta = self._upper_body_cfg.max_joint_velocity_rad_s * self.dt
        self._upper_body_filtered = np.clip(filtered, previous - max_delta, previous + max_delta).astype(np.float32)

        target = np.asarray(pd_target, dtype=np.float32).copy()
        target[self._upper_body_indices] = self._upper_body_filtered
        return target

    def _feedback_receiver(self):
        cfg = getattr(self, "_upper_body_cfg", None)
        if cfg is None or getattr(cfg, "feedback_endpoint", None) is None:
            return None
        return self.ctrl_manager.controllers[cfg.ctrl_type]["inst"]

    def _feedback_intent(self):
        mode = getattr(getattr(self, "mode", None), "value", None)
        if (
            mode != "RL_DEFAULT"
            or not self._upper_body_control_available()
            or getattr(self.env, "_last_safety_state", None) == "HOLD"
        ):
            return "HOLD"
        return "TRACK" if self._upper_body_enabled else "HOME"

    def _reset_upper_body_feedback(self):
        receiver = self._feedback_receiver()
        if receiver is not None:
            receiver.reset()
            self._upper_body_pending = None
            self._upper_body_stream_was_fresh = False

    def _refresh_upper_body_feedback(self, ctrl_data):
        receiver = self._feedback_receiver()
        if receiver is not None:
            # Proposals may arrive while policy inference is running. Sample
            # them at application time rather than adding another control cycle.
            ctrl_data[self._upper_body_cfg.ctrl_type] = receiver.get_data()

    def _apply_feedback_proposal(self, pd_target, ctrl_data, receiver):
        # An outage never sends the arms home. Only HOME intent authorizes the
        # external solver's checked return, including while teleop is disabled.
        self._upper_body_pending = None
        self._upper_body_stream_was_fresh = False
        hands = receiver.feedback.schema_version == 2
        if hands:
            receiver.synchronize_intent(self._feedback_intent())
        if not self._upper_body_control_available():
            return pd_target
        stream = ctrl_data.get(self._upper_body_cfg.ctrl_type, {})
        proposal = stream.get("proposal")
        busy = hands and receiver.pending is not None
        if (
            not busy
            and stream.get("fresh")
            and proposal is not None
            and proposal.get("intent") == self._feedback_intent()
        ):
            positions = stream.get("joint_positions", {})
            try:
                receiver.feedback.validate(proposal, positions)
                if hands and not receiver.hand_snapshot()["fresh"]:
                    raise ValueError("finger measurements are stale")
                candidate = np.asarray([positions[name] for name in self._upper_body_cfg.joint_names], dtype=np.float32)
                # The bridge checks float32 targets. Do not reshape a checked path.
                checked = dict(positions)
                checked.update(zip(self._upper_body_cfg.joint_names, candidate.tolist(), strict=True))
                receiver.feedback.validate(proposal, checked)
            except (ValueError, KeyError, OverflowError, RuntimeError) as exc:
                receiver.feedback.command_counts["rejected"] += 1
                if hands and isinstance(exc, RuntimeError):
                    receiver.invalidate(str(exc))
                else:
                    receiver._log_invalid_message(exc, time.monotonic())
                    if hands:
                        receiver._clear_positions()
                        receiver.synchronize_generation()
            else:
                self._upper_body_previous = self._upper_body_filtered.copy()
                self._upper_body_filtered = candidate
                self._upper_body_pending = proposal
                self._upper_body_stream_was_fresh = self._upper_body_enabled and proposal.get("status") == "tracking"
        target = np.asarray(pd_target, dtype=np.float32).copy()
        target[self._upper_body_indices] = self._upper_body_filtered
        return target

    def _send_upper_body_target(self, pd_target, hand_pose=None):
        """Admit hands before issuing the corresponding arm target.

        Physical writes remain asynchronous; the bridge checks independent
        application paths. Failure revokes further writes, never rolls hardware back.
        """
        receiver = self._feedback_receiver()
        hands = receiver is not None and receiver.feedback.schema_version == 2
        proposal = getattr(self, "_upper_body_pending", None)
        if hands and proposal is not None:
            try:
                if receiver.synchronize_intent(self._feedback_intent()):
                    raise ValueError("control intent changed before submission")
                receiver.feedback.validate(proposal, proposal["positions"])
                if not receiver.hand_snapshot()["fresh"]:
                    raise ValueError("finger measurements are stale before submission")
                receiver.submit_hands(proposal)
            except (ValueError, KeyError, RuntimeError) as exc:
                receiver.invalidate(str(exc), recoverable=isinstance(exc, ValueError | KeyError))
                self._upper_body_pending = None
                self._upper_body_stream_was_fresh = False
                self._upper_body_filtered = self._upper_body_previous.copy()
                pd_target[self._upper_body_indices] = self._upper_body_filtered
        try:
            self.env.step(pd_target, hand_pose)
        except Exception:
            if hands:
                receiver.invalidate("arm submission failed")
                self._upper_body_pending = None
            raise
        self._upper_body_command_submitted = True

    def _update_upper_body_feedback(self, env_data, ctrl_data, pd_target):
        cfg = getattr(self, "_upper_body_cfg", None)
        if getattr(cfg, "visualize_targets", False):
            draw_targets = getattr(self.env, "draw_teleop_targets", None)
            if draw_targets is not None:
                stream = ctrl_data.get(cfg.ctrl_type, {})
                draw_targets(stream.get("target_poses", {}))
        self._publish_upper_body_feedback(env_data, pd_target)

    def _publish_upper_body_feedback(self, env_data, pd_target):
        assembly_started = time.perf_counter()
        receiver = self._feedback_receiver()
        if receiver is None or not self._upper_body_indices.size:
            return
        hands = receiver.feedback.schema_version == 2
        try:
            if hands and receiver.synchronize_intent(self._feedback_intent()):
                # Safety or ownership can change during environment submission.
                self._upper_body_pending = None
            if self._upper_body_command_submitted:
                self._upper_body_filtered = np.asarray(pd_target, dtype=np.float32)[self._upper_body_indices].copy()
                if self._upper_body_pending is not None:
                    if hands:
                        # Admission happened before env.step; do not re-apply or
                        # expire an already issued command after a blocking step.
                        if receiver.pending is None:
                            raise ValueError("combined command was not admitted before arm submission")
                    else:
                        receiver.acknowledge(self._upper_body_pending)
            hand_state = None
            timing = dict(getattr(self, "_upper_body_timing_ms", {}))
            if hands:
                started = time.perf_counter()
                hand_state = receiver.hand_snapshot()
                timing["hand_snapshot"] = 1000 * (time.perf_counter() - started)
                tracking = receiver.pending is not None and receiver.pending[0]["status"] == "tracking"
                self._upper_body_stream_was_fresh = (
                    receiver.finish_pending(hand_state) and tracking and self._upper_body_enabled
                )
            self._upper_body_pending = None
            started = time.perf_counter()
            get_feedback = getattr(self.env, "get_joint_feedback", None)
            if get_feedback is not None:
                body = get_feedback()
            else:
                get_data = getattr(self.env, "get_data", None)
                if self._upper_body_command_submitted and get_data is not None:
                    env_data = get_data()
                stamps = env_data.get("joint_sample_times")
                body = {
                    "positions": dict(zip(self.env.joint_names, np.asarray(env_data.dof_pos).tolist(), strict=True)),
                    "sample_times": {} if stamps is None else dict(zip(self.env.joint_names, stamps, strict=True)),
                }
            timing["body_snapshot"] = 1000 * (time.perf_counter() - started)
            names = self._upper_body_cfg.joint_names
            source_groups = {"body": list(body["positions"])}
            payload = {
                **body,
                "targets": dict(zip(names, self._upper_body_filtered.tolist(), strict=True)),
                "defaults": dict(zip(names, self._upper_body_default.tolist(), strict=True)),
                "limits": {
                    name: self.env.position_limits[index].tolist()
                    for name, index in zip(names, self._upper_body_indices, strict=True)
                },
            }
            diagnostics = {}
            if hands:
                hand_payload = receiver.hand_feedback_payload(hand_state)
                for side, prefix in (("left_hand", "L_"), ("right_hand", "R_")):
                    source_groups[side] = [name for name in hand_payload["positions"] if name.startswith(prefix)]
                for key, values in hand_payload.items():
                    payload[key].update(values)
                diagnostics = {
                    key: hand_state[key]
                    for key in (
                        "snapshot_sequence",
                        "snapshot_connection_generation",
                        "handoff_delay_ms",
                        "snapshot_cache_age_ms",
                    )
                    if key in hand_state
                }
            timing["feedback_assembly"] = 1000 * (time.perf_counter() - assembly_started)
            receiver.feedback.publish(
                **payload,
                intent=self._feedback_intent(),
                mode=self.mode.value,
                control_dt=self.dt,
                timing_ms=timing,
                source_groups=source_groups,
                diagnostics=diagnostics,
                measurements_available=hand_state["fresh"] if hands else True,
            )
            if hands:
                receiver.synchronize_generation()
        except (RuntimeError, ValueError, KeyError) as exc:
            if not hands:
                raise
            receiver.invalidate(str(exc), recoverable=isinstance(exc, ValueError | KeyError))
            self._upper_body_pending = None
            self._upper_body_stream_was_fresh = False

    # Recording methods
    def _record_upper_body_sample(self, env_data, extras, pd_target, *, rl_active: bool):
        recorder_client = getattr(self, "_recorder_client", None)
        if recorder_client is None:
            return

        if not self._recording_active:
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
        names = list(self._upper_body_cfg.joint_names)
        positions = np.asarray(env_data.dof_pos, dtype=np.float32)[self._upper_body_indices]
        commands = np.asarray(pd_target, dtype=np.float32)[self._upper_body_indices]
        hand_data = self._upper_body_record_hand_data()
        if hand_data is not None:
            if not hand_data.get("fresh", False):
                return
            hand_names = list(hand_data.get("joint_names", []))
            hand_positions = np.asarray(hand_data.get("joint_positions"), dtype=np.float32)
            hand_commands = np.asarray(hand_data.get("joint_position_commands"), dtype=np.float32)
            if (
                not hand_names
                or hand_positions.shape != (len(hand_names),)
                or hand_commands.shape != (len(hand_names),)
            ):
                logger.warning("Skipped recording frame with an invalid hand snapshot")
                return
            names.extend(hand_names)
            positions = np.concatenate((positions, hand_positions))
            commands = np.concatenate((commands, hand_commands))
        recorder_client.submit(
            joint_names=names,
            joint_positions=positions,
            joint_position_commands=commands,
            velocity_height_command=np.asarray(locomotion_command, dtype=np.float32)[:4],
            timestamp_ns=time.monotonic_ns(),
        )

    def _upper_body_record_hand_data(self):
        receiver = self._feedback_receiver()
        if receiver is not None and receiver.feedback.schema_version == 2:
            try:
                return receiver.record_hand_data()
            except RuntimeError as exc:
                receiver.invalidate(str(exc))
                self._upper_body_stream_was_fresh = False
                return {"fresh": False}
        return None

    def _toggle_recording_episode(self):
        if getattr(self, "_recorder_client", None) is None:
            logger.warning("Ignored record command: recorder is not enabled")
            return
        if self._recording_active:
            self._finish_recording_episode()
            return
        if getattr(self, "_recording_review_pending", False):
            logger.warning("Recording review is pending; confirm save or discard first")
            return
        if not self._upper_body_enabled:
            logger.warning("Ignored record start: upper-body control is not enabled")
            return
        if not self._upper_body_enable_available():
            return
        self._recording_active = True
        self._recording_paused = False
        logger.warning("Recording started; waiting for a fresh upper-body sample")

    def _toggle_recording_pause(self):
        if not getattr(self, "_recording_active", False):
            logger.warning("Ignored record pause/resume: no recording is active")
            return
        self._recording_paused = not self._recording_paused
        logger.warning("Recording %s", "paused" if self._recording_paused else "resumed")

    def _finish_recording_episode(self):
        recorder_client = getattr(self, "_recorder_client", None)
        if recorder_client is not None:
            review_episode = getattr(recorder_client, "review_episode", None)
            if review_episode is not None:
                review_episode()
            else:
                recorder_client.finish_episode(save=True)
                logger.warning("Recording stopped and episode saved (legacy recorder client)")
                self._recording_review_pending = False
                self._recording_active = False
                self._recording_paused = False
                return
        self._recording_active = False
        self._recording_paused = False
        self._recording_review_pending = True
        logger.warning("Recording stopped; confirm save or discard the pending episode")

    def _confirm_recording_save(self):
        if not self._recording_review_pending:
            logger.warning("Ignored recording save: no episode is pending review")
            return
        recorder_client = getattr(self, "_recorder_client", None)
        if recorder_client is not None:
            recorder_client.confirm_save()
        self._recording_review_pending = False
        logger.warning("Recording episode confirmed and saved")

    def _discard_recording(self):
        if not self._recording_review_pending:
            logger.warning("Ignored recording discard: no episode is pending review")
            return
        recorder_client = getattr(self, "_recorder_client", None)
        if recorder_client is not None:
            recorder_client.discard_review()
        self._recording_review_pending = False
        logger.warning("Recording episode discarded")

    def close_recording(self):
        recorder_client = getattr(self, "_recorder_client", None)
        if recorder_client is not None:
            self._recording_active = False
            self._recording_paused = False
            self._recording_review_pending = False
            recorder_client.close()
            self._recorder_client = None
