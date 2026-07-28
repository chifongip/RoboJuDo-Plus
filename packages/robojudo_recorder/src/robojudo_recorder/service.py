import logging
import time
from collections import deque

import msgpack
import zmq

from .cameras import CameraFrame, CameraSource, create_camera
from .config import RecorderConfig
from .dataset import LeRobotV3Writer
from .protocol import ControlSample
from .profiles import NamedJointProfile

logger = logging.getLogger(__name__)


class RecorderService:
    """Pair camera frames with the latest preceding RoboJuDo control sample."""

    def __init__(self, cfg: RecorderConfig, camera: CameraSource | None = None):
        self.cfg = cfg
        self.camera = camera if camera is not None else create_camera(cfg.camera)
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.PULL)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVHWM, 512)
        self._socket.connect(cfg.control_endpoint)
        self._samples: deque[ControlSample] = deque(maxlen=512)
        self._active_episode_id: int | None = None
        self._active_task: str | None = None
        self._writer: LeRobotV3Writer | None = None
        self._profile: NamedJointProfile | None = None
        self._last_camera_sequence = -1
        self._last_written_sample_timestamp = -1
        self._running = False
        self.dropped_stale_frames = 0

    def _receive_messages(self):
        while True:
            try:
                raw = self._socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                return
            received_at = time.monotonic_ns()
            try:
                message = msgpack.unpackb(raw, raw=False)
                self._handle_message(message, received_at)
            except (KeyError, TypeError, ValueError, msgpack.UnpackException) as exc:
                logger.warning("Rejected recorder control message: %s", exc)

    def _handle_message(self, message: dict, received_at: int):
        kind = message.get("kind")
        if kind == "sample":
            sample = ControlSample.from_message(message, received_at)
            if self._active_episode_id != sample.episode_id:
                self._finish_episode(save=True)
                self._active_episode_id = sample.episode_id
                self._active_task = sample.task
            self._samples.append(sample)
        elif kind == "episode_start":
            episode_id = int(message["episode_id"])
            if self._active_episode_id not in (None, episode_id):
                self._finish_episode(save=True)
            self._active_episode_id = episode_id
            self._active_task = str(message["task"])
        elif kind == "episode_end":
            if int(message["episode_id"]) == self._active_episode_id:
                self._finish_episode(save=bool(message.get("save", True)))
        elif kind == "client_close":
            self._finish_episode(save=True)
        else:
            raise ValueError(f"unknown message kind {kind!r}")

    def _ensure_writer(self, sample: ControlSample):
        if self._profile is None:
            self._profile = NamedJointProfile.from_sample(sample)
        else:
            self._profile.validate(sample)
        if self._writer is None:
            self._writer = LeRobotV3Writer(
                root=self.cfg.dataset.root,
                repo_id=self.cfg.dataset.repo_id,
                robot_type=sample.robot_type,
                fps=self.cfg.dataset.fps,
                state_names=self._profile.state_names,
                action_names=self._profile.action_names,
                camera_name=self.cfg.camera.name,
                camera_shape=self.camera.shape,
                codec=self.cfg.dataset.codec,
                resume=self.cfg.dataset.resume,
            )
        if not self._writer.episode_open:
            self._writer.start_episode(self._active_task or sample.task)

    def _matching_sample(self, frame: CameraFrame) -> ControlSample | None:
        if self._active_episode_id is None:
            return None
        candidates = [
            sample
            for sample in self._samples
            if sample.episode_id == self._active_episode_id
            and sample.timestamp_ns(self.cfg.sync.clock) <= frame.timestamp_ns
        ]
        if not candidates:
            return None
        sample = candidates[-1]
        age_ns = frame.timestamp_ns - sample.timestamp_ns(self.cfg.sync.clock)
        if age_ns > self.cfg.sync.max_control_age_ms * 1_000_000:
            self.dropped_stale_frames += 1
            return None
        return sample

    def _record_frame(self, frame: CameraFrame):
        if frame.sequence == self._last_camera_sequence:
            return
        self._last_camera_sequence = frame.sequence
        sample = self._matching_sample(frame)
        if sample is None:
            return
        sample_timestamp = sample.timestamp_ns(self.cfg.sync.clock)
        if sample_timestamp == self._last_written_sample_timestamp:
            return
        self._ensure_writer(sample)
        self._writer.add_frame(sample.joint_positions, sample.action, frame.image)
        self._last_written_sample_timestamp = sample_timestamp

    def _finish_episode(self, *, save: bool):
        if self._writer is not None and self._writer.episode_open:
            if save and self._writer.has_pending_frames:
                self._writer.save_episode()
            else:
                self._writer.discard_episode()
        self._active_episode_id = None
        self._active_task = None
        self._last_written_sample_timestamp = -1

    def step(self):
        self._receive_messages()
        frame = self.camera.read(self.cfg.sync.poll_timeout_ms)
        if frame is not None:
            self._record_frame(frame)

    def run(self):
        self.camera.connect()
        self._running = True
        logger.info("Recorder connected to control endpoint %s", self.cfg.control_endpoint)
        try:
            while self._running:
                self.step()
        finally:
            self.close()

    def stop(self):
        self._running = False

    def close(self):
        self._running = False
        self._finish_episode(save=True)
        if self._writer is not None:
            self._writer.finalize()
        self.camera.close()
        self._socket.close(linger=0)
