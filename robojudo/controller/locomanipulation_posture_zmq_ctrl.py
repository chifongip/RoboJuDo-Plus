import json
import logging
import math
import time
from numbers import Real

import zmq

from robojudo.controller import Controller, ctrl_registry
from robojudo.controller.ctrl_cfgs import LocomanipulationPostureZmqCtrlCfg

logger = logging.getLogger(__name__)
XR_POSTURE_MAX_BYTES = 4096
XR_POSTURE_FIELDS = {
    "schema_version",
    "bridge_id",
    "session_id",
    "generation",
    "state_id",
    "epoch",
    "sequence",
    "height",
    "waist_yaw",
}


def _unique_fields(pairs):
    fields = {}
    for key, value in pairs:
        if key in fields:
            raise ValueError("duplicate XR posture field")
        fields[key] = value
    return fields


@ctrl_registry.register
class LocomanipulationPostureZmqCtrl(Controller):
    """Receive atomic absolute Locomanipulation posture setpoints without blocking."""

    cfg_ctrl: LocomanipulationPostureZmqCtrlCfg

    def __init__(self, cfg_ctrl: LocomanipulationPostureZmqCtrlCfg, env=None, device="cpu"):
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, device=device)
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVHWM, 8 if cfg_ctrl.xr_authorized else 100)
        if cfg_ctrl.xr_authorized:
            self._socket.setsockopt(zmq.MAXMSGSIZE, XR_POSTURE_MAX_BYTES)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.connect(cfg_ctrl.endpoint)
        self._height: float | None = None
        self._waist_yaw: float | None = None
        self._last_received_at: float | None = None
        self._last_invalid_log_at = float("-inf")
        self._xr_epoch = 0
        self._xr_manual = False
        self._xr_key = None
        self._xr_sequence = -1
        self._xr_deadline = float("-inf")
        self._xr_multipart = False
        logger.info("LocomanipulationPostureZmqCtrl subscribed to %s", cfg_ctrl.endpoint)

    def reset(self):
        self._height = None
        self._waist_yaw = None
        self._last_received_at = None
        self._xr_epoch += 1
        self._xr_key = None
        self._xr_sequence = -1
        self._xr_deadline = float("-inf")
        self._xr_manual = False
        self._xr_multipart = False
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

    def _receive_available(self, now: float, feedback=None, limits=None):
        xr_authorized = getattr(self.cfg_ctrl, "xr_authorized", False)
        for _ in range(32 if xr_authorized else 100):
            try:
                if xr_authorized:
                    payload = self._socket.recv(flags=zmq.NOBLOCK)
                    more = bool(self._socket.getsockopt(zmq.RCVMORE))
                    multipart, self._xr_multipart = self._xr_multipart or more, more
                    if multipart or feedback is None:
                        continue
                    message = json.loads(payload, object_pairs_hook=_unique_fields)
                    self._accept_xr(message, feedback, limits, now)
                    continue
                message = self._socket.recv_json(flags=zmq.NOBLOCK)
                height, waist_yaw = self._decode_message(message)
            except zmq.Again:
                return
            except (TypeError, ValueError, OverflowError, RecursionError, zmq.ZMQError) as exc:
                self._log_invalid_message(exc, now)
                continue
            self._height = height
            self._waist_yaw = waist_yaw
            self._last_received_at = now

    def _accept_xr(self, message, feedback, limits, now):
        if (
            not isinstance(message, dict)
            or set(message) != XR_POSTURE_FIELDS
            or type(message["schema_version"]) is not int
            or message["schema_version"] != 1
        ):
            raise ValueError("invalid XR posture schema")
        sequence = message["sequence"]
        if type(sequence) is not int or sequence <= self._xr_sequence:
            raise ValueError("duplicate or out-of-order XR posture")
        height, yaw = self._decode_message({name: message[name] for name in ("height", "waist_yaw")})
        if not (limits[0][0] <= height <= limits[0][1] and limits[1][0] <= yaw <= limits[1][1]):
            raise ValueError("XR posture command outside policy limits")
        deadline = feedback.validate_posture_reference(message, self._xr_epoch, now)
        self._height, self._waist_yaw = height, yaw
        self._last_received_at = now
        self._xr_deadline = deadline
        self._xr_sequence = sequence

    def xr_feedback(self, *, limits, defaults, targets, available):
        return {
            "schema_version": 1,
            "available": available and not self._xr_manual,
            "epoch": self._xr_epoch,
            "limits": limits,
            "defaults": defaults,
            "targets": targets,
        }

    def update_xr(self, feedback, *, eligible, manual, limits):
        key = feedback.session_id, feedback.generation, feedback.bridge_id
        if key != self._xr_key or manual != self._xr_manual:
            self._last_received_at = None
            self._xr_deadline = float("-inf")
            self._xr_sequence = -1
            if manual != self._xr_manual:
                self._xr_epoch += 1
            self._xr_key = key
            self._xr_manual = manual
        now = time.monotonic()
        if eligible and not manual and feedback.intent == "TRACK":
            self._receive_available(now, feedback, limits)
        else:
            self._xr_deadline = float("-inf")
            self._receive_available(now)
        return self._data(now)

    def get_data(self):
        now = time.monotonic()
        if not getattr(self.cfg_ctrl, "xr_authorized", False):
            self._receive_available(now)
        return self._data(now)

    def _data(self, now):
        has_received = self._last_received_at is not None
        age_s = None if self._last_received_at is None else now - self._last_received_at
        fresh = age_s is not None and age_s <= self.cfg_ctrl.timeout_s
        if getattr(self.cfg_ctrl, "xr_authorized", False):
            fresh = fresh and now <= self._xr_deadline
        return {
            "height": self._height,
            "waist_yaw": self._waist_yaw,
            "has_received": has_received,
            "fresh": fresh,
            "age_s": age_s,
        }

    def process_triggers(self, ctrl_data):
        return ctrl_data, []
