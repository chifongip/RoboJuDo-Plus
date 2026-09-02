import logging
import time
from numbers import Integral, Real

import numpy as np
import zmq

from robojudo.controller import Controller, ctrl_registry
from robojudo.controller.ctrl_cfgs import UpperBodyHandZmqCtrlCfg
from robojudo.controller.omnihand_runtime import (
    OMNIHAND_LEFT_JOINT_NAMES,
    OMNIHAND_RIGHT_JOINT_NAMES,
    OmniHandRuntime,
)

logger = logging.getLogger(__name__)


@ctrl_registry.register
class UpperBodyHandZmqCtrl(Controller):
    """Receive one atomic arm-and-hands frame and drive the two physical hands.

    The controller owns the wire-protocol boundary; SDK I/O remains asynchronous
    inside :class:`OmniHandRuntime` so polling ZMQ never blocks the robot loop.
    """

    cfg_ctrl: UpperBodyHandZmqCtrlCfg

    def __init__(self, cfg_ctrl: UpperBodyHandZmqCtrlCfg, env=None, device="cpu"):
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, device=device)
        self._arm_joint_names = tuple(cfg_ctrl.joint_names)
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        # Keep the control path latest-only. CONFLATE avoids applying a burst of
        # stale teleoperation frames after a temporary scheduling delay.
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
        self._omnihand: OmniHandRuntime | None = None
        logger.info("UpperBodyHandZmqCtrl subscribed to %s", cfg_ctrl.endpoint)
        try:
            self._omnihand = OmniHandRuntime(cfg_ctrl.omnihand)
        except Exception:
            self._socket.close(linger=0)
            raise

    def reset(self):
        """Clear the accepted stream state and discard frames already in the socket."""

        self._latest_arm_joint_positions.clear()
        self._latest_hand_joint_commands = None
        self._latest_sync_frame_id = None
        self._latest_source_timestamp_ns = None
        self._last_received_at = None
        self._omnihand.reset()
        while True:
            try:
                self._socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break

    def close(self):
        omnihand = getattr(self, "_omnihand", None)
        if omnihand is not None:
            omnihand.close()
            self._omnihand = None
        self._socket.close(linger=0)

    def set_takeover_enabled(self, enabled: bool):
        """Forward the pipeline's hardware takeover gate to the runtime."""

        self._omnihand.set_takeover_enabled(enabled)

    @staticmethod
    def _decode_joint_payload(payload, expected_names: tuple[str, ...], label: str) -> np.ndarray:
        """Validate one named joint array, including its exact ordering and validity flag."""

        if not isinstance(payload, dict):
            raise ValueError(f"{label} must be an object")
        if payload.get("valid") is not True:
            raise ValueError(f"{label} is not valid")
        if payload.get("joint_names") != list(expected_names):
            raise ValueError(f"{label} joint names or order do not match the configured schema")
        qpos = payload.get("qpos")
        if not isinstance(qpos, list) or len(qpos) != len(expected_names):
            raise ValueError(f"{label} qpos must contain exactly {len(expected_names)} values")
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in qpos):
            raise ValueError(f"{label} qpos values must be numeric")
        values = np.asarray(qpos, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{label} qpos values must be finite")
        return values

    def _decode_synchronized_frame(self, message) -> tuple[dict[str, float], np.ndarray, np.ndarray, int, int]:
        """Decode a complete dex-teleop frame without accepting partial arm/hand data."""

        if not isinstance(message, dict):
            raise ValueError("synchronized frame must be an object")
        if message.get("schema_version") != 1 or message.get("type") != "synchronized_teleop_frame":
            raise ValueError("expected synchronized_teleop_frame schema version 1")
        if message.get("mode") != "sim2real":
            raise ValueError("real OmniHand control requires mode='sim2real'")
        if message.get("hand_type") != "omnihand":
            raise ValueError("synchronized frame hand_type must be 'omnihand'")

        frame_id = message.get("frame_id")
        if isinstance(frame_id, bool) or not isinstance(frame_id, Integral) or frame_id < 0:
            raise ValueError("frame_id must be a non-negative integer")
        timestamp_ns = message.get("timestamp_ns")
        if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, Integral) or timestamp_ns <= 0:
            raise ValueError("timestamp_ns must be a positive integer")

        arm_joint_positions = self._decode_joint_payload(message.get("arm"), self._arm_joint_names, "arm")
        left_hand_joint_commands = self._decode_joint_payload(
            message.get("left_hand"), OMNIHAND_LEFT_JOINT_NAMES, "left_hand"
        )
        right_hand_joint_commands = self._decode_joint_payload(
            message.get("right_hand"), OMNIHAND_RIGHT_JOINT_NAMES, "right_hand"
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
            logger.warning("Rejected synchronized upper-body ZMQ frame: %s", exc)
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
            ) = self._decode_synchronized_frame(message)
            stream_is_fresh = (
                self._last_received_at is not None and now - self._last_received_at <= self.cfg_ctrl.timeout_s
            )
            # Frame IDs are monotonic while the stream is fresh. After a timeout,
            # accepting a restarted publisher's counter prevents permanent lockout.
            if (
                stream_is_fresh
                and self._latest_sync_frame_id is not None
                and synchronized_frame_id <= self._latest_sync_frame_id
            ):
                raise ValueError(
                    f"frame_id {synchronized_frame_id} is not newer than {self._latest_sync_frame_id}"
                )
            combined_hand_joint_commands = self._omnihand.set_joint_commands(
                left_hand_joint_commands,
                right_hand_joint_commands,
                source_timestamp_ns,
                synchronized_frame_id,
            )
        except ValueError as exc:
            self._log_invalid_message(exc, now)
            return

        self._latest_arm_joint_positions = arm_positions_by_name
        self._latest_hand_joint_commands = combined_hand_joint_commands
        self._latest_sync_frame_id = synchronized_frame_id
        self._latest_source_timestamp_ns = source_timestamp_ns
        self._last_received_at = now

    def get_data(self):
        """Return arm commands plus measured hand positions and command freshness state."""

        now = time.monotonic()
        self._receive_latest(now)
        has_received = self._last_received_at is not None
        age_s = None if self._last_received_at is None else now - self._last_received_at
        stream_is_fresh = age_s is not None and age_s <= self.cfg_ctrl.timeout_s
        omnihand_data = self._omnihand.get_data()
        if self._latest_hand_joint_commands is not None:
            omnihand_data["joint_position_commands"] = self._latest_hand_joint_commands.copy()
        omnihand_data["source_frame_id"] = self._latest_sync_frame_id
        omnihand_data["source_timestamp_ns"] = self._latest_source_timestamp_ns
        omnihand_data["fresh"] = bool(
            stream_is_fresh
            and self._latest_hand_joint_commands is not None
            and omnihand_data.get("enabled", False)
            and omnihand_data.get("joint_state_fresh", False)
        )
        return {
            "joint_positions": self._latest_arm_joint_positions.copy(),
            "has_received": has_received,
            "fresh": stream_is_fresh,
            "age_s": age_s,
            "frame_id": self._latest_sync_frame_id,
            "source_timestamp_ns": self._latest_source_timestamp_ns,
            "omnihand": omnihand_data,
        }

    def process_triggers(self, ctrl_data):
        return ctrl_data, []
