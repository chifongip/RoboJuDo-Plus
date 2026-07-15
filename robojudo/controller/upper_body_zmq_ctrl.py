import logging
import math
import time
from numbers import Real

import zmq

from robojudo.controller import Controller, ctrl_registry
from robojudo.controller.ctrl_cfgs import UpperBodyZmqCtrlCfg

logger = logging.getLogger(__name__)


@ctrl_registry.register
class UpperBodyZmqCtrl(Controller):
    """Receive named upper-body joint positions without blocking the control loop."""

    cfg_ctrl: UpperBodyZmqCtrlCfg

    def __init__(self, cfg_ctrl: UpperBodyZmqCtrlCfg, env=None, device="cpu"):
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, device=device)
        self._joint_names = set(cfg_ctrl.joint_names)
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVHWM, 100)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.connect(cfg_ctrl.endpoint)
        self._latest_positions: dict[str, float] = {}
        self._last_received_at: float | None = None
        self._last_invalid_log_at = float("-inf")
        logger.info("UpperBodyZmqCtrl subscribed to %s", cfg_ctrl.endpoint)

    def reset(self):
        self._latest_positions.clear()
        self._last_received_at = None
        for _ in range(100):
            try:
                self._socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break

    def close(self):
        self._socket.close(linger=0)

    def _decode_message(self, message) -> dict[str, float]:
        if not isinstance(message, dict) or "positions" not in message:
            raise ValueError("message must contain a 'positions' object")
        positions = message["positions"]
        if not isinstance(positions, dict) or not positions:
            raise ValueError("positions must be a non-empty object")

        unknown = sorted(set(positions) - self._joint_names)
        if unknown:
            raise ValueError(f"unknown upper-body joints: {unknown}")

        decoded = {}
        for name, value in positions.items():
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"position for {name} must be numeric")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"position for {name} must be finite")
            decoded[name] = value
        return decoded

    def _log_invalid_message(self, exc: Exception, now: float):
        if now - self._last_invalid_log_at >= 1.0:
            logger.warning("Rejected upper-body ZMQ message: %s", exc)
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
                positions = self._decode_message(message)
            except ValueError as exc:
                self._log_invalid_message(exc, now)
                continue
            self._latest_positions.update(positions)
            self._last_received_at = now

    def get_data(self):
        now = time.monotonic()
        self._receive_available(now)
        has_received = self._last_received_at is not None
        age_s = None if self._last_received_at is None else now - self._last_received_at
        fresh = age_s is not None and age_s <= self.cfg_ctrl.timeout_s
        return {
            "joint_positions": self._latest_positions.copy(),
            "has_received": has_received,
            "fresh": fresh,
            "age_s": age_s,
        }

    def process_triggers(self, ctrl_data):
        return ctrl_data, []
