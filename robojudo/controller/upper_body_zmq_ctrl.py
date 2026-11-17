import json
import logging
import math
import time
from numbers import Real

import zmq

from robojudo.controller import Controller, ctrl_registry
from robojudo.controller.ctrl_cfgs import UpperBodyZmqCtrlCfg

logger = logging.getLogger(__name__)
_MAX_MESSAGE_BYTES = 16_384
_MAX_MESSAGES_PER_POLL = 100


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON key: {name}")
        result[name] = value
    return result


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


@ctrl_registry.register
class UpperBodyZmqCtrl(Controller):
    """Receive single-frame JSON arm targets with bounded work and per-joint leases.

    Freshness is measured locally at receipt. Legacy publishers supply no source
    timestamp or sequence, so transit delay/replay cannot be inferred from JSON.
    """

    cfg_ctrl: UpperBodyZmqCtrlCfg

    def __init__(self, cfg_ctrl: UpperBodyZmqCtrlCfg, env=None, device="cpu"):
        cfg_ctrl = type(cfg_ctrl).model_validate(cfg_ctrl.model_dump())
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, device=device)
        self._joint_names = set(cfg_ctrl.joint_names)
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.SUB)
        try:
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.setsockopt(zmq.RCVHWM, _MAX_MESSAGES_PER_POLL)
            self._socket.setsockopt(zmq.MAXMSGSIZE, _MAX_MESSAGE_BYTES)
            self._socket.setsockopt(zmq.SUBSCRIBE, b"")
            self._socket.connect(cfg_ctrl.endpoint)
        except Exception:
            self._socket.close(linger=0)
            raise
        self._latest_positions: dict[str, float] = {}
        self._joint_received_at: dict[str, float] = {}
        self._last_received_at: float | None = None
        self._last_poll_at: float | None = None
        self._last_invalid_log_at = float("-inf")
        self._discard_backlog = False
        self._discard_multipart = False
        self._closed = False
        self._transport_error: str | None = None
        self._reset_generation = 0
        logger.info("UpperBodyZmqCtrl subscribed to %s", cfg_ctrl.endpoint)

    @property
    def reset_generation(self):
        return self._reset_generation

    def _clear_positions(self):
        self._latest_positions.clear()
        self._joint_received_at.clear()
        self._last_received_at = None

    def reset(self):
        self._clear_positions()
        self._reset_generation += 1
        # Drain at the next poll, within its normal budget. Commands queued before
        # a toggle/reset must not be promoted into a new control session.
        self._discard_backlog = True
        self._last_poll_at = None

    def close(self):
        if not self._closed:
            self.reset()
            self._closed = True
            self._socket.close(linger=0)

    def _decode_message(self, message) -> dict[str, float]:
        if not isinstance(message, dict) or "positions" not in message:
            raise ValueError("message must contain a 'positions' object")
        positions = message["positions"]
        if not isinstance(positions, dict) or not positions:
            raise ValueError("positions must be a non-empty object")
        if any(not isinstance(name, str) for name in positions):
            raise ValueError("joint names must be strings")
        unknown = sorted(set(positions) - self._joint_names)
        if unknown:
            raise ValueError(f"unknown upper-body joints: {unknown}")

        decoded = {}
        for name, value in positions.items():
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"position for {name} must be numeric")
            try:
                value = float(value)
            except OverflowError:
                raise ValueError(f"position for {name} must be finite") from None
            if not math.isfinite(value):
                raise ValueError(f"position for {name} must be finite")
            decoded[name] = value
        return decoded

    def _log_invalid_message(self, exc: Exception, now: float):
        if now - self._last_invalid_log_at >= 1.0:
            logger.warning("Rejected upper-body ZMQ message: %s", exc)
            self._last_invalid_log_at = now

    def _receive_available(self, now: float):
        for _ in range(_MAX_MESSAGES_PER_POLL):
            try:
                # Count frames against the budget, including rejected multipart
                # messages, so a large multipart envelope cannot monopolize a tick.
                payload = self._socket.recv(flags=zmq.NOBLOCK)
                more = bool(self._socket.getsockopt(zmq.RCVMORE))
            except zmq.Again:
                self._discard_backlog = False
                return
            except zmq.ZMQError as exc:
                self._clear_positions()
                self._transport_error = str(exc)
                self._log_invalid_message(exc, now)
                return
            multipart = self._discard_multipart or more
            self._discard_multipart = more
            if self._discard_backlog:
                continue
            if multipart:
                self._log_invalid_message(ValueError("expected a single-frame JSON message"), now)
                continue
            try:
                if len(payload) > _MAX_MESSAGE_BYTES:
                    raise ValueError("JSON frame exceeds size limit")
                message = json.loads(
                    payload.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant
                )
                positions = self._decode_message(message)
            except (TypeError, ValueError, RecursionError) as exc:
                self._log_invalid_message(exc, now)
                continue
            self._latest_positions.update(positions)
            self._joint_received_at.update(dict.fromkeys(positions, now))
            self._last_received_at = now
            self._transport_error = None
        # Reaching the budget cannot prove that the queue was drained. Discard
        # until an empty queue is observed, then require a subsequent new packet.
        self._clear_positions()
        self._discard_backlog = True

    def get_data(self):
        now = time.monotonic()
        if self._last_poll_at is not None and now - self._last_poll_at > self.cfg_ctrl.timeout_s:
            self.reset()
        self._last_poll_at = now
        if not self._closed:
            self._receive_available(now)
        positions = {
            name: value
            for name, value in self._latest_positions.items()
            if 0 <= now - self._joint_received_at[name] <= self.cfg_ctrl.timeout_s
        }
        age_s = None if self._last_received_at is None else now - self._last_received_at
        return {
            "joint_positions": positions,
            "joint_received_at": {name: self._joint_received_at[name] for name in positions},
            "has_received": self._last_received_at is not None,
            "fresh": bool(positions) and not self._closed,
            "age_s": age_s,
            "reset_generation": self._reset_generation,
            "transport_error": self._transport_error,
        }

    def process_triggers(self, ctrl_data):
        return ctrl_data, []
