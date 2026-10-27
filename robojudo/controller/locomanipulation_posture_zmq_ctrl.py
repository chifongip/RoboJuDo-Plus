import logging
import math
import time
from numbers import Real

import zmq

from robojudo.controller import Controller, ctrl_registry
from robojudo.controller.ctrl_cfgs import LocomanipulationPostureZmqCtrlCfg

logger = logging.getLogger(__name__)


@ctrl_registry.register
class LocomanipulationPostureZmqCtrl(Controller):
    """Receive atomic absolute Locomanipulation posture setpoints without blocking."""

    cfg_ctrl: LocomanipulationPostureZmqCtrlCfg

    def __init__(self, cfg_ctrl: LocomanipulationPostureZmqCtrlCfg, env=None, device="cpu"):
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, device=device)
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVHWM, 100)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.connect(cfg_ctrl.endpoint)
        self._height: float | None = None
        self._waist_yaw: float | None = None
        self._last_received_at: float | None = None
        self._last_invalid_log_at = float("-inf")
        logger.info("LocomanipulationPostureZmqCtrl subscribed to %s", cfg_ctrl.endpoint)

    def reset(self):
        self._height = None
        self._waist_yaw = None
        self._last_received_at = None
        for _ in range(100):
            try:
                self._socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break

    def close(self):
        self._socket.close(linger=0)

    @staticmethod
    def _decode_message(message) -> tuple[float, float]:
        if not isinstance(message, dict) or set(message) != {"height", "waist_yaw"}:
            raise ValueError("message must contain exactly 'height' and 'waist_yaw'")

        values = []
        for name in ("height", "waist_yaw"):
            value = message[name]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be numeric")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            values.append(value)
        return values[0], values[1]

    def _log_invalid_message(self, exc: Exception, now: float):
        if now - self._last_invalid_log_at >= 1.0:
            logger.warning("Rejected Locomanipulation posture ZMQ message: %s", exc)
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
                height, waist_yaw = self._decode_message(message)
            except ValueError as exc:
                self._log_invalid_message(exc, now)
                continue
            self._height = height
            self._waist_yaw = waist_yaw
            self._last_received_at = now

    def get_data(self):
        now = time.monotonic()
        self._receive_available(now)
        has_received = self._last_received_at is not None
        age_s = None if self._last_received_at is None else now - self._last_received_at
        fresh = age_s is not None and age_s <= self.cfg_ctrl.timeout_s
        return {
            "height": self._height,
            "waist_yaw": self._waist_yaw,
            "has_received": has_received,
            "fresh": fresh,
            "age_s": age_s,
        }

    def process_triggers(self, ctrl_data):
        return ctrl_data, []
