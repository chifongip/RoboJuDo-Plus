import logging
import time
from numbers import Integral, Real

import numpy as np
import zmq

from robojudo.controller import Controller, ctrl_registry
from robojudo.controller.casia_hand_runtime import (
    CASIA_LEFT_JOINT_NAMES,
    CASIA_RIGHT_JOINT_NAMES,
    CasiaHandRuntime,
)
from robojudo.controller.ctrl_cfgs import UpperBodyCasiaHandZmqCtrlCfg

logger = logging.getLogger(__name__)


@ctrl_registry.register
class UpperBodyCasiaHandZmqCtrl(Controller):
    """Receive one atomic arm-and-CASIA-hands frame and drive both physical hands."""

    cfg_ctrl: UpperBodyCasiaHandZmqCtrlCfg

    def __init__(self, cfg_ctrl: UpperBodyCasiaHandZmqCtrlCfg, env=None, device="cpu"):
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, device=device)
        self._arm_joint_names = tuple(cfg_ctrl.joint_names)
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        # The teleoperation stream is a control signal. Never replay old frames
        # after a temporary scheduling or hardware delay.
        self._socket.setsockopt(zmq.RCVHWM, 1)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.connect(cfg_ctrl.endpoint)
        self._latest_arm_joint_positions: dict[str, float] = {}
        self._latest_hand_joint_commands: np.ndarray | None = None
        self._latest_sync_frame_id: int | None = None
        self._latest_source_timestamp_ns: int | None = None
        self._last_received_at: float | None = None
        self._last_invalid_log_at = float("-inf")
        self._casia_hand = None
        logger.info("UpperBodyCasiaHandZmqCtrl subscribed to %s", cfg_ctrl.endpoint)
        try:
            self._casia_hand = CasiaHandRuntime(cfg_ctrl.casia_hand)
        except Exception:
            self._socket.close(linger=0)
            raise

    def reset(self):
        """Clear accepted stream state and discard frames already in the socket."""

        self._latest_arm_joint_positions.clear()
        self._latest_hand_joint_commands = None
        self._latest_sync_frame_id = None
        self._latest_source_timestamp_ns = None
        self._last_received_at = None
        self._casia_hand.reset()
        while True:
            try:
                self._socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break

    def close(self):
        casia_hand = getattr(self, "_casia_hand", None)
        if casia_hand is not None:
            casia_hand.close()
            self._casia_hand = None
        self._socket.close(linger=0)

    def set_takeover_enabled(self, enabled: bool):
        """Forward the pipeline hardware takeover gate to the CASIA runtime."""

        self._casia_hand.set_takeover_enabled(enabled)

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
        values = np.asarray(joint_positions, dtype=np.float64)
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
            raise ValueError("real CASIA Hand control requires mode='sim2real'")
        if message.get("hand_type") != "casia":
            raise ValueError("synchronized frame hand_type must be 'casia'")

        frame_id = message.get("frame_id")
        if isinstance(frame_id, bool) or not isinstance(frame_id, Integral) or frame_id < 0:
            raise ValueError("frame_id must be a non-negative integer")
        timestamp_ns = message.get("timestamp_ns")
        if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, Integral) or timestamp_ns <= 0:
            raise ValueError("timestamp_ns must be a positive integer")

        arm_joint_positions = self._validate_joint_payload(message.get("arm"), self._arm_joint_names, "arm")
        left_hand_joint_commands = self._validate_joint_payload(
            message.get("left_hand"), CASIA_LEFT_JOINT_NAMES, "left_hand"
        )
        right_hand_joint_commands = self._validate_joint_payload(
            message.get("right_hand"), CASIA_RIGHT_JOINT_NAMES, "right_hand"
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
            logger.warning("Rejected synchronized CASIA upper-body ZMQ frame: %s", exc)
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
            if (
                stream_is_fresh
                and self._latest_sync_frame_id is not None
                and synchronized_frame_id <= self._latest_sync_frame_id
            ):
                raise ValueError(
                    f"frame_id {synchronized_frame_id} is not newer than {self._latest_sync_frame_id}"
                )
            combined_hand_joint_commands = self._casia_hand.set_joint_commands(
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
        """Return arm commands plus measured CASIA positions and command freshness."""

        now = time.monotonic()
        self._receive_latest(now)
        has_received = self._last_received_at is not None
        age_s = None if self._last_received_at is None else now - self._last_received_at
        stream_is_fresh = age_s is not None and age_s <= self.cfg_ctrl.timeout_s
        casia_hand_data = self._casia_hand.get_data()
        if self._latest_hand_joint_commands is not None:
            # Recorder action belongs to the same synchronized source frame as
            # the arm command. Hardware feedback remains independently sampled.
            casia_hand_data["joint_position_commands"] = self._latest_hand_joint_commands.copy()
        casia_hand_data["source_frame_id"] = self._latest_sync_frame_id
        casia_hand_data["source_timestamp_ns"] = self._latest_source_timestamp_ns
        casia_hand_data["fresh"] = bool(
            stream_is_fresh
            and self._latest_hand_joint_commands is not None
            and casia_hand_data.get("enabled", False)
            and casia_hand_data.get("joint_state_fresh", False)
        )
        return {
            "joint_positions": self._latest_arm_joint_positions.copy(),
            "has_received": has_received,
            "fresh": stream_is_fresh,
            "age_s": age_s,
            "frame_id": self._latest_sync_frame_id,
            "source_timestamp_ns": self._latest_source_timestamp_ns,
            "casia_hand": casia_hand_data,
        }

    def process_triggers(self, ctrl_data):
        return ctrl_data, []


__all__ = ["UpperBodyCasiaHandZmqCtrl"]
