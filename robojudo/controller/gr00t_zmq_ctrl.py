import logging
import math
import time
from numbers import Integral, Real

import numpy as np
import zmq

from robojudo.controller import Controller, ctrl_registry
from robojudo.controller.ctrl_cfgs import Gr00tZmqCtrlCfg

logger = logging.getLogger(__name__)


@ctrl_registry.register
class Gr00tZmqCtrl(Controller):
    """Receive one atomic GR00T arm and locomotion command per message."""

    cfg_ctrl: Gr00tZmqCtrlCfg

    def __init__(self, cfg_ctrl: Gr00tZmqCtrlCfg, env=None, device="cpu"):
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, device=device)
        self._joint_names = tuple(cfg_ctrl.joint_names)
        self._joint_name_set = set(self._joint_names)
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVHWM, 100)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.connect(cfg_ctrl.endpoint)
        self._latest_positions: dict[str, float] = {}
        self._latest_locomotion_command: np.ndarray | None = None
        self._latest_sequence: int | None = None
        self._last_received_at: float | None = None
        self._last_invalid_log_at = float("-inf")
        logger.info("Gr00tZmqCtrl subscribed to %s", cfg_ctrl.endpoint)

    def reset(self):
        self._latest_positions.clear()
        self._latest_locomotion_command = None
        self._latest_sequence = None
        self._last_received_at = None
        for _ in range(100):
            try:
                self._socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break

    def close(self):
        self._socket.close(linger=0)

    def _decode_positions(self, value) -> dict[str, float]:
        if not isinstance(value, dict) or not value:
            raise ValueError("positions must be a non-empty object")
        unknown = sorted(set(value) - self._joint_name_set)
        if unknown:
            raise ValueError(f"unknown GR00T joints: {unknown}")
        if self.cfg_ctrl.require_complete_positions:
            missing = sorted(self._joint_name_set - set(value))
            if missing:
                raise ValueError(f"GR00T message is missing joints: {missing}")

        positions = {}
        for name, position in value.items():
            if isinstance(position, bool) or not isinstance(position, Real):
                raise ValueError(f"position for {name} must be numeric")
            position = float(position)
            if not math.isfinite(position):
                raise ValueError(f"position for {name} must be finite")
            positions[name] = position
        return positions

    @staticmethod
    def _decode_locomotion_command(value) -> np.ndarray:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ValueError("locomotion_command must be a four-element array [vx, vy, yaw_rate, height]")
        if any(isinstance(item, bool) or not isinstance(item, Real) for item in value):
            raise ValueError("locomotion_command values must be numeric")
        command = np.asarray(value, dtype=np.float32)
        if not np.isfinite(command).all():
            raise ValueError("locomotion_command values must be finite")
        return command

    def _decode_message(self, message) -> tuple[dict[str, float], np.ndarray, int | None]:
        if not isinstance(message, dict):
            raise ValueError("GR00T message must be an object")
        if "positions" not in message or "locomotion_command" not in message:
            raise ValueError("GR00T message must contain positions and locomotion_command")
        positions = self._decode_positions(message["positions"])
        locomotion_command = self._decode_locomotion_command(message["locomotion_command"])

        sequence = message.get("sequence")
        if sequence is not None:
            if isinstance(sequence, bool) or not isinstance(sequence, Integral) or sequence < 0:
                raise ValueError("sequence must be a non-negative integer")
            sequence = int(sequence)
        return positions, locomotion_command, sequence

    def _log_invalid_message(self, exc: Exception, now: float):
        if now - self._last_invalid_log_at >= 1.0:
            logger.warning("Rejected GR00T ZMQ message: %s", exc)
            self._last_invalid_log_at = now

    def _receive_available(self, now: float):
        for _ in range(100):
            try:
                message = self._socket.recv_json(flags=zmq.NOBLOCK)
            except zmq.Again:
                return
            except (TypeError, ValueError, zmq.ZMQError) as exc:
                self._log_invalid_message(exc, now)
                continue

            try:
                positions, locomotion_command, sequence = self._decode_message(message)
            except ValueError as exc:
                self._log_invalid_message(exc, now)
                continue
            stream_is_fresh = (
                self._last_received_at is not None and now - self._last_received_at <= self.cfg_ctrl.timeout_s
            )
            if (
                stream_is_fresh
                and sequence is not None
                and self._latest_sequence is not None
                and sequence <= self._latest_sequence
            ):
                self._log_invalid_message(
                    ValueError(f"sequence {sequence} is not newer than {self._latest_sequence}"),
                    now,
                )
                continue

            self._latest_positions = positions
            self._latest_locomotion_command = locomotion_command
            self._latest_sequence = sequence
            self._last_received_at = now

    def get_data(self):
        now = time.monotonic()
        self._receive_available(now)
        has_received = self._last_received_at is not None
        age_s = None if self._last_received_at is None else now - self._last_received_at
        fresh = age_s is not None and age_s <= self.cfg_ctrl.timeout_s
        return {
            "joint_positions": self._latest_positions.copy(),
            "locomotion_command": (
                None if self._latest_locomotion_command is None else self._latest_locomotion_command.copy()
            ),
            "sequence": self._latest_sequence,
            "has_received": has_received,
            "fresh": fresh,
            "age_s": age_s,
        }

    def process_triggers(self, ctrl_data):
        return ctrl_data, []
