"""Shared synchronized upper-body and physical-hand ZMQ controller."""

import logging
import time
from numbers import Integral, Real

import numpy as np
import zmq

from robojudo.controller.ctrl_cfgs import UpperBodyZmqCtrlCfg
from robojudo.controller.upper_body_zmq_ctrl import UpperBodyZmqCtrl

logger = logging.getLogger(__name__)


class UpperBodyHandZmqCtrlBase(UpperBodyZmqCtrl):
    """Share arm/hand transport and lifecycle across streaming and feedback modes.

    Concrete hand controllers provide only their wire ``hand_type``, physical
    joint schema, output key, and runtime factory. Feedback mode stages checked
    proposals until environment submission; streaming mode accepts synchronized frames.
    """

    hand_type: str
    hand_data_key: str
    hand_display_name: str
    left_hand_joint_names: tuple[str, ...]
    right_hand_joint_names: tuple[str, ...]
    cfg_ctrl: UpperBodyZmqCtrlCfg
    _conflate_legacy_frames = True

    def __init__(self, cfg_ctrl: UpperBodyZmqCtrlCfg, env=None, device="cpu"):
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, device=device)
        self._arm_joint_names = tuple(cfg_ctrl.joint_names)
        self._latest_arm_joint_positions: dict[str, float] = {}
        self._latest_hand_joint_commands: np.ndarray | None = None
        self._latest_sync_frame_id: int | None = None
        self._latest_source_timestamp_ns: int | None = None
        self._last_received_at: float | None = None
        self._last_invalid_log_at = float("-inf")
        self._hand_runtime = None
        self._hand_stream_generation = None
        self.pending = None
        if self.feedback is not None:
            self._joint_names.update((*self.left_hand_joint_names, *self.right_hand_joint_names))
        self._hand_generation = None if self.feedback is None else self.feedback.generation
        try:
            simulated_feedback = self.feedback is not None and env is not None and env.cfg_env.is_sim
            if not simulated_feedback:
                self._hand_runtime = self._create_hand_runtime(cfg_ctrl)
        except Exception:
            self.close()
            raise

    def _create_hand_runtime(self, cfg_ctrl: UpperBodyZmqCtrlCfg):
        raise NotImplementedError

    def reset(self):
        super().reset()
        self._latest_arm_joint_positions.clear()
        self._latest_hand_joint_commands = None
        self._latest_sync_frame_id = None
        self._latest_source_timestamp_ns = None
        self.pending = None
        if self._hand_runtime is not None:
            self._hand_runtime.reset()
        self._hand_generation = None if self.feedback is None else self.feedback.generation

    def close(self):
        super().close()
        if self._hand_runtime is not None:
            self._hand_runtime.close()
            self._hand_runtime = None

    def set_takeover_enabled(self, enabled: bool):
        """Forward the pipeline hardware takeover gate to the hand runtime."""

        if self._hand_runtime is not None:
            self._hand_runtime.set_takeover_enabled(enabled)

    @staticmethod
    def _validate_joint_payload(payload, expected_names: tuple[str, ...], label: str) -> np.ndarray:
        """Validate one named joint array and return its finite values."""

        if not isinstance(payload, dict):
            raise ValueError(f"{label} must be an object")
        if payload.get("valid") is not True:
            raise ValueError(f"{label} is not valid")
        if payload.get("joint_names") != list(expected_names):
            raise ValueError(f"{label} joint names or order do not match the configured schema")
        joint_positions = payload.get("qpos")
        if not isinstance(joint_positions, list) or len(joint_positions) != len(expected_names):
            raise ValueError(f"{label} qpos must contain exactly {len(expected_names)} values")
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in joint_positions):
            raise ValueError(f"{label} qpos values must be numeric")
        try:
            values = np.asarray(joint_positions, dtype=np.float64)
        except OverflowError as exc:
            raise ValueError(f"{label} qpos values must fit float64") from exc
        if not np.isfinite(values).all():
            raise ValueError(f"{label} qpos values must be finite")
        return values

    def _validate_synchronized_frame(self, message) -> tuple[dict[str, float], np.ndarray, np.ndarray, int, int]:
        """Validate a complete frame without accepting partial arm/hand data."""

        if not isinstance(message, dict):
            raise ValueError("synchronized frame must be an object")
        if message.get("schema_version") != 1 or message.get("type") != "synchronized_teleop_frame":
            raise ValueError("expected synchronized_teleop_frame schema version 1")
        if message.get("mode") != "sim2real":
            raise ValueError(f"real {self.hand_display_name} control requires mode='sim2real'")
        if message.get("hand_type") != self.hand_type:
            raise ValueError(f"synchronized frame hand_type must be {self.hand_type!r}")

        frame_id = message.get("frame_id")
        if isinstance(frame_id, bool) or not isinstance(frame_id, Integral) or frame_id < 0:
            raise ValueError("frame_id must be a non-negative integer")
        timestamp_ns = message.get("timestamp_ns")
        if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, Integral) or timestamp_ns <= 0:
            raise ValueError("timestamp_ns must be a positive integer")

        arm_joint_positions = self._validate_joint_payload(message.get("arm"), self._arm_joint_names, "arm")
        left_hand_joint_commands = self._validate_joint_payload(
            message.get("left_hand"), self.left_hand_joint_names, "left_hand"
        )
        right_hand_joint_commands = self._validate_joint_payload(
            message.get("right_hand"), self.right_hand_joint_names, "right_hand"
        )
        arm_positions_by_name = dict(zip(self._arm_joint_names, arm_joint_positions.tolist(), strict=True))
        return (
            arm_positions_by_name,
            left_hand_joint_commands,
            right_hand_joint_commands,
            int(frame_id),
            int(timestamp_ns),
        )

    def _log_invalid_message(self, exc: Exception, now: float):
        if now - self._last_invalid_log_at >= 1.0:
            logger.warning("Rejected %s upper-body ZMQ frame: %s", self.hand_display_name, exc)
            self._last_invalid_log_at = now

    def _receive_latest(self, now: float):
        """Accept at most the current latest frame and enqueue both hands together."""

        try:
            message = self._socket.recv_json(flags=zmq.NOBLOCK)
        except zmq.Again:
            return
        except (TypeError, ValueError, zmq.ZMQError) as exc:
            self._log_invalid_message(exc, now)
            return

        try:
            (
                arm_positions_by_name,
                left_hand_joint_commands,
                right_hand_joint_commands,
                synchronized_frame_id,
                source_timestamp_ns,
            ) = self._validate_synchronized_frame(message)
            stream_is_fresh = (
                self._last_received_at is not None and now - self._last_received_at <= self.cfg_ctrl.timeout_s
            )
            # Enforce monotonic frame IDs only while the current publisher is
            # fresh; after a timeout, a restarted publisher may begin at zero.
            if (
                stream_is_fresh
                and self._latest_sync_frame_id is not None
                and synchronized_frame_id <= self._latest_sync_frame_id
            ):
                raise ValueError(f"frame_id {synchronized_frame_id} is not newer than {self._latest_sync_frame_id}")
            snapshot = self._hand_runtime.get_data()
            if self._synchronize_hand_stream(snapshot) or not snapshot.get("joint_state_fresh", False):
                return
            combined_hand_joint_commands = self._hand_runtime.set_joint_commands(
                left_hand_joint_commands,
                right_hand_joint_commands,
                source_timestamp_ns,
                synchronized_frame_id,
            )
        except ValueError as exc:
            self._log_invalid_message(exc, now)
            return

        # Publish arm and hand commands only after the entire frame has passed
        # validation and the dual-hand command has been accepted by the runtime.
        self._latest_arm_joint_positions = arm_positions_by_name
        self._latest_hand_joint_commands = combined_hand_joint_commands
        self._latest_sync_frame_id = synchronized_frame_id
        self._latest_source_timestamp_ns = source_timestamp_ns
        self._last_received_at = now

    def get_data(self):
        """Return arm commands plus measured hand positions and freshness."""

        if self._hand_runtime is not None:
            self._synchronize_hand_stream(self._hand_runtime.get_data())
        if getattr(self, "feedback", None) is not None:
            return super().get_data()
        now = time.monotonic()
        if getattr(self, "_discard_backlog", False):
            # Reset/toggle must discard queued frames in both protocol modes.
            self._receive_available(now)
        if not getattr(self, "_discard_backlog", False):
            self._receive_latest(now)
        hand_data = self._hand_runtime.get_data()
        self._synchronize_hand_stream(hand_data)
        has_received = self._last_received_at is not None
        age_s = None if self._last_received_at is None else now - self._last_received_at
        stream_is_fresh = age_s is not None and 0 <= age_s <= self.cfg_ctrl.timeout_s
        if self._latest_hand_joint_commands is not None:
            # Recorder action belongs to the same source frame as the arm
            # command; measured state remains independently sampled hardware data.
            hand_data["joint_position_commands"] = self._latest_hand_joint_commands.copy()
        hand_data["source_frame_id"] = self._latest_sync_frame_id
        hand_data["source_timestamp_ns"] = self._latest_source_timestamp_ns
        hand_data["fresh"] = bool(
            stream_is_fresh
            and self._latest_hand_joint_commands is not None
            and hand_data.get("enabled", False)
            and hand_data.get("joint_state_fresh", False)
        )
        return {
            "joint_positions": self._latest_arm_joint_positions.copy(),
            "has_received": has_received,
            "fresh": bool(stream_is_fresh and hand_data.get("joint_state_fresh", False)),
            "age_s": age_s,
            "frame_id": self._latest_sync_frame_id,
            "source_timestamp_ns": self._latest_source_timestamp_ns,
            self.hand_data_key: hand_data,
        }

    def _synchronize_hand_stream(self, snapshot):
        generation = snapshot.get("stream_generation")
        if generation is None:
            return False
        previous = getattr(self, "_hand_stream_generation", None)
        self._hand_stream_generation = generation
        if previous is not None and previous != generation:
            # Clear queued transport frames and checked proposals on both sides
            # of a hold. Preserve operator intent through this hardware outage.
            if self.feedback is not None:
                self.invalidate("hand feedback stream changed", recoverable=True)
            else:
                super().reset()
                self.pending = None
                self._latest_sync_frame_id = None
                self._latest_hand_joint_commands = None
                self._latest_source_timestamp_ns = None
                self._last_received_at = None
            log = logger.info if snapshot.get("joint_state_fresh", False) else logger.warning
            log(
                "%s hardware feedback %s; pending proposals cleared",
                self.hand_display_name,
                snapshot.get("connection_status", "changed"),
            )
            return True
        return False

    def process_triggers(self, ctrl_data):
        return ctrl_data, []

    def _decode_message(self, message):
        if self.pending is not None:
            raise ValueError("combined proposal still awaiting application")
        return super()._decode_message(message)

    def hand_snapshot(self):
        if self._hand_runtime is None:
            return self.env.hand_snapshot()
        snapshot = self._hand_runtime.get_data()
        self._synchronize_hand_stream(snapshot)
        # Robot feedback requires fresh measurements even before any command is admitted.
        snapshot["fresh"] = snapshot["joint_state_fresh"]
        return snapshot

    def record_hand_data(self):
        snapshot = self.hand_snapshot()
        names = [*self.left_hand_joint_names, *self.right_hand_joint_names]
        return {
            "fresh": snapshot["fresh"],
            "joint_names": names,
            "joint_positions": [snapshot["positions"][name] for name in names],
            "joint_position_commands": [snapshot["targets"][name] for name in names],
        }

    def invalidate(self, reason, *, recoverable=False):
        runtime = self._hand_runtime
        requested = getattr(runtime, "takeover_requested", False)
        self.reset()
        if recoverable and requested:
            runtime.set_takeover_enabled(True)
        # The bridge preserves clutch engagement across transient command outages.
        self.feedback.reset_reason = "tracking command timeout" if recoverable else reason
        if reason != "hand feedback stream changed":
            self._log_invalid_message(ValueError(reason), time.monotonic())

    def synchronize_generation(self):
        if self._hand_generation != self.feedback.generation:
            self.pending = None
            if self._hand_runtime is not None:
                invalidate = getattr(self._hand_runtime, "invalidate_commands", None)
                if invalidate is not None:
                    invalidate()
                else:
                    self._hand_runtime.reset()
            self._hand_generation = self.feedback.generation

    def synchronize_intent(self, intent):
        changed = self.feedback.intent != intent
        if changed:
            self.invalidate("control intent changed")
            self.feedback.intent = intent
        self.synchronize_generation()
        return changed

    def submit_hands(self, proposal):
        self.synchronize_generation()
        deadline = self.feedback._states[proposal["state_id"]][0]
        identity = (proposal["generation"], proposal["frame_id"])
        if self._hand_runtime is None:
            self.env.submit_hand_targets(proposal["positions"])
        else:
            self._hand_runtime.set_takeover_enabled(True)
            self._hand_runtime.submit(proposal["positions"], identity, deadline)
        self.pending = (proposal, identity, deadline)
        self._clear_positions()

    def finish_pending(self, state=None):
        self.synchronize_generation()
        if self.pending is None:
            return False
        pending = self.pending
        proposal, identity, deadline = pending
        generation = self.feedback.generation
        if state is None:
            state = self.hand_snapshot()
        if self.feedback.generation != generation or self.pending is not pending:
            return False
        # A matching hardware identity is only published after both writes
        # completed before their deadline. Observing it later is not a failure.
        applied = self._hand_runtime is not None and state["applied"] == identity
        if not state["fresh"] or (not applied and time.monotonic() > deadline):
            self.invalidate("finger application expired or measurements stale", recoverable=True)
            return False
        if self._hand_runtime is not None and state["applied"] != identity:
            return False
        self.feedback.acknowledge()
        if self._hand_runtime is not None:
            self._hand_runtime.acknowledge()
        self.pending = None
        return True


__all__ = ["UpperBodyHandZmqCtrlBase"]
