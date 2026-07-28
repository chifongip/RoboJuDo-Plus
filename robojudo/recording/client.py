import logging
import time

import msgpack
import numpy as np
import zmq

from .record_cfgs import RecordCfg

logger = logging.getLogger(__name__)


class RecorderClient:
    """Non-blocking control-sample publisher for the standalone recorder."""

    def __init__(self, cfg: RecordCfg, robot_type: str):
        self.cfg = cfg
        self.robot_type = robot_type
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.PUSH)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SNDHWM, cfg.send_hwm)
        self._socket.setsockopt(zmq.IMMEDIATE, 1)
        self._socket.bind(cfg.endpoint)
        self._episode_id = 0
        self._active_episode_id: int | None = None
        self.dropped_samples = 0
        logger.info("Recorder client publishing to %s", cfg.endpoint)

    @property
    def active(self) -> bool:
        return self._active_episode_id is not None

    def _send(self, payload: dict, *, lifecycle: bool = False) -> bool:
        flags = 0 if lifecycle and self.cfg.lifecycle_timeout_ms else zmq.NOBLOCK
        if lifecycle and self.cfg.lifecycle_timeout_ms:
            self._socket.setsockopt(zmq.SNDTIMEO, self.cfg.lifecycle_timeout_ms)
        try:
            self._socket.send(msgpack.packb(payload, use_bin_type=True), flags=flags)
            return True
        except zmq.Again:
            return False
        finally:
            if lifecycle and self.cfg.lifecycle_timeout_ms:
                self._socket.setsockopt(zmq.SNDTIMEO, -1)

    def _ensure_episode(self) -> int:
        if self._active_episode_id is None:
            self._episode_id += 1
            self._active_episode_id = self._episode_id
            self._send(
                {
                    "kind": "episode_start",
                    "episode_id": self._active_episode_id,
                    "task": self.cfg.task,
                    "robot_type": self.robot_type,
                    "timestamp_ns": time.monotonic_ns(),
                },
                lifecycle=True,
            )
        return self._active_episode_id

    def submit(
        self,
        *,
        joint_names: list[str],
        joint_positions: np.ndarray,
        joint_position_commands: np.ndarray,
        velocity_height_command: np.ndarray,
        timestamp_ns: int | None = None,
    ) -> bool:
        episode_id = self._ensure_episode()
        payload = {
            "kind": "sample",
            "episode_id": episode_id,
            "task": self.cfg.task,
            "robot_type": self.robot_type,
            "timestamp_ns": timestamp_ns if timestamp_ns is not None else time.monotonic_ns(),
            "joint_names": joint_names,
            "joint_positions": np.asarray(joint_positions, dtype=np.float32).tolist(),
            "joint_position_commands": np.asarray(joint_position_commands, dtype=np.float32).tolist(),
            "velocity_height_command": np.asarray(velocity_height_command, dtype=np.float32).tolist(),
        }
        sent = self._send(payload)
        if not sent:
            self.dropped_samples += 1
            if self.dropped_samples == 1 or self.dropped_samples % 100 == 0:
                logger.warning("Recorder unavailable or saturated; dropped %d samples", self.dropped_samples)
        return sent

    def finish_episode(self, *, save: bool = True):
        if self._active_episode_id is None:
            return
        episode_id = self._active_episode_id
        self._active_episode_id = None
        sent = self._send(
            {
                "kind": "episode_end",
                "episode_id": episode_id,
                "save": save,
                "timestamp_ns": time.monotonic_ns(),
            },
            lifecycle=True,
        )
        if not sent:
            logger.warning("Recorder did not acknowledge delivery of episode end %d", episode_id)

    def close(self):
        self.finish_episode(save=True)
        self._send({"kind": "client_close", "timestamp_ns": time.monotonic_ns()}, lifecycle=True)
        self._socket.close(linger=0)
