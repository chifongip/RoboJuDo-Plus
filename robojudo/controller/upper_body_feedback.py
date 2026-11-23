"""Robot-authoritative XR targets and completed-command feedback.

All deadlines belong to this process. A publisher echoes feedback identifiers;
its clock is never compared with the robot's monotonic clock.
"""

import logging
import math
import time
import uuid
from collections import OrderedDict

import numpy as np
import zmq


def _counter(value):
    if type(value) is not int or value < 0:
        raise ValueError("sequence fields must be non-negative integers")
    return value


class StaleVisualization(ValueError):
    """Display data references measurements that are no longer available."""


class UpperBodyFeedback:
    def __init__(self, cfg, context):
        self.cfg = cfg
        self.session_id = uuid.uuid4().hex
        self.generation = 0
        self.command_id = 0
        self.applied_bridge_id = None
        self.applied_frame_id = None
        self._completed_positions = None
        self._first_feedback_built = None
        self._first_ack_state_id = None
        self._completion_observation_source = None
        self._command_timestamps = {}
        self.command_counts = dict.fromkeys(("received", "superseded", "rejected", "completed"), 0)
        self.state_id = 0
        self.intent = "HOLD"
        self.reset_reason = "startup"
        self.bridge_id = None
        self.last_desired_at = None
        self._frame_id = -1
        self._states = OrderedDict()
        self._last_publication_timing = None
        from robojudo.controller.xr_target_executor import XRTargetExecutor

        self.latest_measurements = None
        self.executor = XRTargetExecutor(cfg)
        self.socket = context.socket(zmq.PUB)
        try:
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.setsockopt(zmq.SNDHWM, cfg.feedback_send_hwm)
            self.socket.bind(cfg.feedback_endpoint)
        except Exception:
            self.socket.close(linger=0)
            if self.executor is not None:
                self.executor.close()
            raise
        logging.getLogger(__name__).info("Upper-body feedback configured SNDHWM=%d", cfg.feedback_send_hwm)

    schema_version = 3

    @property
    def has_hands(self):
        return bool(self.executor.hand_names)

    @property
    def command_names(self):
        return [*self.cfg.joint_names, *self.executor.hand_names]

    def reset(self, reason="controller session reset"):
        self.reset_reason = reason
        self.generation += 1
        self._frame_id = -1
        self._states.clear()
        if self.executor is not None:
            self.executor.reset()
        self.latest_measurements = None
        self.last_desired_at = None
        self.applied_bridge_id = None
        self.applied_frame_id = None
        self._completed_positions = None
        self._first_feedback_built = None
        self._first_ack_state_id = None
        self._completion_observation_source = None
        self._command_timestamps = {}

    def expire_tracking(self, now):
        if (
            self.intent == "TRACK"
            and self.last_desired_at is not None
            and now - self.last_desired_at > self.cfg.timeout_s
        ):
            self.reset("tracking command timeout")
            return True
        return False

    def close(self):
        self.socket.close(linger=0)
        if self.executor is not None:
            self.executor.close()

    def publish(
        self,
        *,
        positions,
        sample_times,
        targets,
        defaults,
        limits,
        intent,
        mode,
        control_dt,
        timing_ms=None,
        source_groups=None,
        diagnostics=None,
        measurements_available=True,
        xr_posture=None,
        xr_velocity=None,
    ):
        started = time.perf_counter()
        now = time.monotonic()
        if intent != self.intent:
            self.reset("control intent changed")
            self.intent = intent
        current_targets = targets
        if self._completed_positions is not None:
            targets = self._completed_positions
        ages = {name: now - stamp for name, stamp in sample_times.items()}
        valid = (
            bool(measurements_available)
            and set(ages) == set(positions)
            and all(math.isfinite(age) and 0 <= age <= self.cfg.feedback_timeout_s for age in ages.values())
        )
        valid = valid and all(
            math.isfinite(value) for value in (*positions.values(), *targets.values(), *current_targets.values())
        )
        # An invalid startup snapshot must not become a permanent baseline.
        if valid and self._completed_positions is None:
            self._completed_positions = dict(targets)
        self.latest_measurements = dict(positions) if valid else None
        self.state_id += 1
        oldest = max(ages, key=ages.get, default=None)
        feedback_diagnostics = {
            **({} if diagnostics is None else diagnostics),
            "oldest_joint": oldest,
            "source_age_ms": {
                source: max((1000 * ages[name] for name in names if name in ages), default=None)
                for source, names in (source_groups or {}).items()
            },
            "previous_publication": self._last_publication_timing,
            "command_counts": self.command_counts.copy(),
        }
        feedback_diagnostics["source_age_ms"] = {
            source: value if value is not None and math.isfinite(value) else None
            for source, value in feedback_diagnostics["source_age_ms"].items()
        }
        if self._completion_observation_source is not None:
            feedback_diagnostics["completion_observation_source"] = self._completion_observation_source
        if self.applied_frame_id is not None:
            feedback_diagnostics["first_ack_state_id"] = (
                self.state_id if self._first_ack_state_id is None else self._first_ack_state_id
            )
        if self._command_timestamps:
            # All stamps use the robot host monotonic clock, including its workers.
            feedback_diagnostics["command_timestamps"] = {
                **self._command_timestamps,
                "feedback_built": now,
                "first_feedback_built": now if self._first_feedback_built is None else self._first_feedback_built,
            }
        frame = {
            "schema_version": self.schema_version,
            "type": "upper_body_state",
            "session_id": self.session_id,
            "generation": self.generation,
            "reset_reason": self.reset_reason,
            "state_id": self.state_id,
            "command_id": self.command_id,
            "applied_bridge_id": self.applied_bridge_id,
            "applied_frame_id": self.applied_frame_id,
            "intent": intent,
            "mode": mode,
            "valid": valid,
            "positions": {name: value if math.isfinite(value) else None for name, value in positions.items()},
            "joint_age_s": {name: age if math.isfinite(age) else None for name, age in ages.items()},
            "last_commanded_positions": {
                name: value if math.isfinite(value) else None for name, value in targets.items()
            },
            "default_positions": defaults,
            "position_limits": limits,
            "control_dt": control_dt,
            "arm_velocity_limit": self.cfg.max_joint_velocity_rad_s,
            "return_velocity_limit": self.cfg.return_joint_velocity_rad_s,
            "feedback_timeout_s": self.cfg.feedback_timeout_s,
            "visualize_targets": self.cfg.visualize_targets,
            "pipeline_timing_ms": {} if timing_ms is None else dict(timing_ms),
            "feedback_diagnostics": feedback_diagnostics,
        }
        frame["executor"] = dict(self.executor.contract)
        if xr_posture is not None:
            frame["xr_posture"] = xr_posture
        if xr_velocity is not None:
            frame["xr_velocity"] = xr_velocity
        deadline = min(sample_times.values(), default=float("-inf")) + self.cfg.feedback_timeout_s
        self._states[self.state_id] = (deadline, frame)
        while len(self._states) > 16:
            self._states.popitem(last=False)
        built = time.perf_counter()
        sent = True
        try:
            self.socket.send_json(frame, flags=zmq.NOBLOCK, allow_nan=False)
        except zmq.Again:
            sent = False
        if sent and self.applied_frame_id is not None and self._first_ack_state_id is None:
            self._first_ack_state_id = self.state_id
        if sent and self._command_timestamps and self._first_feedback_built is None:
            self._first_feedback_built = now
        self._last_publication_timing = {
            "state_id": self.state_id,
            "generation": self.generation,
            "build_ms": 1000 * (built - started),
            "encode_send_ms": 1000 * (time.perf_counter() - built),
            "sent": sent,
        }

    def validate_posture_reference(self, message, epoch, now):
        if (
            self.intent != "TRACK"
            or self.bridge_id is None
            or message["bridge_id"] != self.bridge_id
            or message["session_id"] != self.session_id
            or _counter(message["generation"]) != self.generation
            or _counter(message["epoch"]) != epoch
        ):
            raise ValueError("obsolete XR posture authority")
        state = self._states.get(_counter(message["state_id"]))
        if state is None:
            raise ValueError("unknown XR posture feedback state")
        deadline, feedback = state
        posture = feedback.get("xr_posture")
        if (
            now > deadline
            or self.latest_measurements is None
            or not feedback["valid"]
            or feedback["intent"] != "TRACK"
            or not posture
            or not posture["available"]
            or posture["epoch"] != epoch
        ):
            raise ValueError("XR posture feedback is expired or unavailable")
        return deadline

    def validate(self, message, positions, *, receiving=False):
        if (
            type(message.get("schema_version")) is not int
            or message["schema_version"] != self.schema_version
            or message.get("type") != "upper_body_desired"
        ):
            kind = "upper_body_desired"
            raise ValueError(f"expected {kind} schema version {self.schema_version}")
        if message.get("session_id") != self.session_id or _counter(message.get("generation")) != self.generation:
            raise ValueError("obsolete upper-body session or generation")
        frame_id = _counter(message.get("frame_id"))
        state_id = _counter(message.get("state_id"))
        _counter(message.get("command_id"))
        bridge_id = message.get("bridge_id")
        if not isinstance(bridge_id, str) or not 1 <= len(bridge_id) <= 64:
            raise ValueError("bridge_id must be a non-empty bounded string")
        if self.bridge_id is not None and bridge_id != self.bridge_id:
            # Rotate the robot-owned generation before allowing a restarted bridge.
            self.bridge_id = bridge_id
            self.reset("bridge restarted")
            raise ValueError("bridge restarted; waiting for a new feedback generation")
        state = self._states.get(state_id)
        if state is None:
            raise ValueError("unknown feedback state")
        deadline, feedback = state
        if not feedback["valid"]:
            raise ValueError("referenced robot measurements were invalid at publication")
        now = time.monotonic()
        if now > deadline:
            stage = "receipt" if receiving else "application"
            age_at_publish = max((age for age in feedback["joint_age_s"].values() if age is not None), default=0.0)
            raise ValueError(
                f"referenced robot measurements expired at {stage} "
                f"(overdue={max(0.0, now - deadline) * 1000:.1f} ms, "
                f"age_at_publish={age_at_publish * 1000:.1f} ms, "
                f"budget={self.cfg.feedback_timeout_s * 1000:.1f} ms)"
            )
        if message.get("intent") != self.intent or self.intent not in {"TRACK", "HOME"}:
            raise ValueError("arm desired is not authorized by the current intent")
        if set(positions) != set(self.command_names):
            raise ValueError("feedback desired targets must contain all configured arm joints")
        if receiving and frame_id <= self._frame_id:
            raise ValueError("duplicate or out-of-order arm desired")
        status = message.get("status")
        if status not in {"tracking", "homing"} or (self.intent == "HOME" and status != "homing"):
            raise ValueError("desired status is incompatible with its control intent")
        self.executor.validate(self, message, positions)
        if receiving:
            self.bridge_id = bridge_id
            self._frame_id = frame_id
            self.last_desired_at = time.monotonic()

    def acknowledge(self, desired=None):
        source = None if desired is None else desired.get("_completion_observation_source")
        self._completion_observation_source = (
            source if source in ("idle", "periodic", "before_policy", "after_policy") else None
        )
        self._first_feedback_built = None
        self._first_ack_state_id = None
        self.command_id += 1
        self.command_counts["completed"] += 1
        if desired is not None:
            self.applied_bridge_id = desired["bridge_id"]
            self.applied_frame_id = desired["frame_id"]
            self._completed_positions = dict(desired["positions"])
            self._command_timestamps = dict(desired.get("_command_timestamps", {}))
        else:
            self.applied_bridge_id = self.applied_frame_id = None
            self._completed_positions = None
            self._command_timestamps = {}
        # Fresh waiting desired targets and overlays retain their measurement leases.

    def visualization(self, message):
        """Validate display-only poses without authorizing or refreshing commands."""
        if (
            type(message.get("schema_version")) is not int
            or message["schema_version"] != self.schema_version
            or message.get("session_id") != self.session_id
        ):
            raise ValueError("invalid visualization session")
        if _counter(message.get("generation")) != self.generation:
            raise StaleVisualization("obsolete visualization generation")
        state = self._states.get(_counter(message.get("state_id")))
        if state is None or not state[1]["valid"] or time.monotonic() > state[0]:
            raise StaleVisualization("expired or superseded visualization measurements")
        poses = message.get("target_poses", {})
        if not isinstance(poses, dict) or set(poses) - {"left", "right"}:
            raise ValueError("invalid target pose names")
        result = {}
        for side, value in poses.items():
            transform = np.asarray(value, dtype=float)
            if (
                transform.shape != (4, 4)
                or not np.isfinite(transform).all()
                or np.any(np.abs(transform[:3, 3]) > 10.0)
                or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6)
                or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(transform[:3, :3]), 1.0, atol=1e-5)
            ):
                raise ValueError("target pose must be a finite rigid transform")
            result[side] = transform.tolist()
        return result, state[0]
