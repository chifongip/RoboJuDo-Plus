import logging
import math
import time
from numbers import Real

import numpy as np
import zmq

from robojudo.controller import Controller, ctrl_registry
from robojudo.controller.ctrl_cfgs import VelocityZmqCtrlCfg

logger = logging.getLogger(__name__)


@ctrl_registry.register
class VelocityZmqCtrl(Controller):
    """Receive ROS Twist-shaped velocity commands without blocking the control loop."""

    cfg_ctrl: VelocityZmqCtrlCfg

    def __init__(self, cfg_ctrl: VelocityZmqCtrlCfg, env=None, device="cpu"):
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, device=device)
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVHWM, 100)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.connect(cfg_ctrl.endpoint)
        self._linear_velocity = np.zeros(3, dtype=np.float32)
        self._angular_velocity = np.zeros(3, dtype=np.float32)
        self._last_received_at: float | None = None
        self._last_invalid_log_at = float("-inf")
        logger.info("VelocityZmqCtrl subscribed to %s", cfg_ctrl.endpoint)

    def reset(self):
        self._linear_velocity.fill(0.0)
        self._angular_velocity.fill(0.0)
        self._last_received_at = None
        for _ in range(100):
            try:
                self._socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break

    def close(self):
        self._socket.close(linger=0)

    @staticmethod
    def _decode_vector(message, name: str) -> np.ndarray:
        vector = message.get(name)
        if not isinstance(vector, dict):
            raise ValueError(f"message must contain a '{name}' object")
        if set(vector) != {"x", "y", "z"}:
            raise ValueError(f"{name} must contain exactly x, y, and z")

        values = []
        for axis in ("x", "y", "z"):
            value = vector[axis]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name}.{axis} must be numeric")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{name}.{axis} must be finite")
            values.append(value)
        return np.asarray(values, dtype=np.float32)

    @classmethod
    def _decode_message(cls, message) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(message, dict):
            raise ValueError("message must be an object")
        return cls._decode_vector(message, "linear"), cls._decode_vector(message, "angular")

    def _log_invalid_message(self, exc: Exception, now: float):
        if now - self._last_invalid_log_at >= 1.0:
            logger.warning("Rejected velocity ZMQ message: %s", exc)
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
                linear_velocity, angular_velocity = self._decode_message(message)
            except ValueError as exc:
                self._log_invalid_message(exc, now)
                continue
            self._linear_velocity = linear_velocity
            self._angular_velocity = angular_velocity
            self._last_received_at = now

    def get_data(self):
        now = time.monotonic()
        self._receive_available(now)
        has_received = self._last_received_at is not None
        age_s = None if self._last_received_at is None else now - self._last_received_at
        fresh = age_s is not None and age_s <= self.cfg_ctrl.timeout_s
        return {
            "linear_velocity": self._linear_velocity.copy(),
            "angular_velocity": self._angular_velocity.copy(),
            "has_received": has_received,
            "fresh": fresh,
            "age_s": age_s,
        }

    def process_triggers(self, ctrl_data):
        return ctrl_data, []
