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
    """Pair synchronized camera groups with the latest preceding control sample."""

    def __init__(
        self,
        cfg: RecorderConfig,
        camera: CameraSource | None = None,
        cameras: list[CameraSource] | tuple[CameraSource, ...] | None = None,
    ):
        self.cfg = cfg
        if camera is not None and cameras is not None:
            raise ValueError("provide either camera or cameras, not both")
        if camera is not None:
            cameras = (camera,)
        self.cameras = tuple(cameras) if cameras is not None else tuple(create_camera(item) for item in cfg.cameras)
        if len(self.cameras) != len(cfg.cameras):
            raise ValueError(f"expected {len(cfg.cameras)} camera sources, got {len(self.cameras)}")
        self.camera = self.cameras[0]
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
        self._last_camera_sequences = {camera.name: -1 for camera in cfg.cameras}
        self._last_written_sample_timestamp = -1
        self._running = False
        self.dropped_stale_frames = 0
        self._episode_frame_count = 0
        self._episode_stale_frames = 0
        self._episode_unmatched_frames = 0
        self._camera_stream_ready = False
        self._camera_missing_since_ns: int | None = None
        self._last_camera_wait_log_ns = 0

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
                self._episode_frame_count = 0
                self._episode_stale_frames = 0
                self._episode_unmatched_frames = 0
                logger.info("Episode %d started from first control sample: %s", sample.episode_id, sample.task)
            self._samples.append(sample)
        elif kind == "episode_start":
            episode_id = int(message["episode_id"])
            if self._active_episode_id not in (None, episode_id):
                self._finish_episode(save=True)
            self._active_episode_id = episode_id
            self._active_task = str(message["task"])
            self._episode_frame_count = 0
            self._episode_stale_frames = 0
            self._episode_unmatched_frames = 0
            logger.info("Episode %d armed: %s", episode_id, self._active_task)
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
                camera_shapes={cfg.name: camera.shape for cfg, camera in zip(self.cfg.cameras, self.cameras)},
                codec=self.cfg.dataset.codec,
                resume=self.cfg.dataset.resume,
            )
            camera_schema = ", ".join(
                f"{cfg.name}={camera.shape}" for cfg, camera in zip(self.cfg.cameras, self.cameras)
            )
            logger.info("Dataset initialized at %s with cameras: %s", self.cfg.dataset.root.resolve(), camera_schema)
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
            self._episode_unmatched_frames += 1
            if self._episode_unmatched_frames == 1 or self._episode_unmatched_frames % 100 == 0:
                logger.warning(
                    "Episode %s could not match %d camera frames to a preceding control sample; "
                    "check sync.clock and producer clock domains",
                    self._active_episode_id,
                    self._episode_unmatched_frames,
                )
            return None
        sample = candidates[-1]
        age_ns = frame.timestamp_ns - sample.timestamp_ns(self.cfg.sync.clock)
        if age_ns > self.cfg.sync.max_control_age_ms * 1_000_000:
            self.dropped_stale_frames += 1
            self._episode_stale_frames += 1
            if self._episode_stale_frames == 1 or self._episode_stale_frames % 100 == 0:
                logger.warning(
                    "Episode %s dropped %d stale camera frames (latest age %.1f ms, limit %.1f ms)",
                    self._active_episode_id,
                    self._episode_stale_frames,
                    age_ns / 1_000_000,
                    self.cfg.sync.max_control_age_ms,
                )
            return None
        return sample

    def _record_frames(self, frames: dict[str, CameraFrame]):
        if any(frame.sequence == self._last_camera_sequences[name] for name, frame in frames.items()):
            return
        for name, frame in frames.items():
            self._last_camera_sequences[name] = frame.sequence
        primary_frame = frames[self.cfg.cameras[0].name]
        sample = self._matching_sample(primary_frame)
        if sample is None:
            return
        sample_timestamp = sample.timestamp_ns(self.cfg.sync.clock)
        if sample_timestamp == self._last_written_sample_timestamp:
            return
        self._ensure_writer(sample)
        self._writer.add_frame(
            sample.joint_positions,
            sample.action,
            {name: frame.image for name, frame in frames.items()},
        )
        self._episode_frame_count += 1
        if self._episode_frame_count == 1:
            logger.info(
                "Episode %s recording first synchronized frame from cameras: %s",
                self._active_episode_id,
                ", ".join(frames),
            )
        progress_interval = max(self.cfg.dataset.fps * 5, 1)
        if self._episode_frame_count % progress_interval == 0:
            logger.info(
                "Episode %s recording progress: %d frames (%.1f s dataset time)",
                self._active_episode_id,
                self._episode_frame_count,
                self._episode_frame_count / self.cfg.dataset.fps,
            )
        self._last_written_sample_timestamp = sample_timestamp

    def _finish_episode(self, *, save: bool):
        episode_id = self._active_episode_id
        if self._writer is not None and self._writer.episode_open:
            if save and self._writer.has_pending_frames:
                self._writer.save_episode()
                logger.info(
                    "Episode %s saved: %d frames, %d stale frames dropped, root=%s",
                    episode_id,
                    self._episode_frame_count,
                    self._episode_stale_frames,
                    self.cfg.dataset.root.resolve(),
                )
            else:
                self._writer.discard_episode()
                logger.info("Episode %s discarded after %d frames", episode_id, self._episode_frame_count)
        elif episode_id is not None and save:
            logger.warning(
                "Episode %s was not saved because no synchronized camera/control frames were recorded",
                episode_id,
            )
        self._active_episode_id = None
        self._active_task = None
        self._last_written_sample_timestamp = -1
        self._episode_frame_count = 0
        self._episode_stale_frames = 0
        self._episode_unmatched_frames = 0

    def _update_camera_status(self, frames: dict[str, CameraFrame | None]):
        missing = [name for name, frame in frames.items() if frame is None]
        now_ns = time.monotonic_ns()
        if not missing:
            self._camera_missing_since_ns = None
            if not self._camera_stream_ready:
                shapes = ", ".join(
                    f"{cfg.name}={camera.shape}" for cfg, camera in zip(self.cfg.cameras, self.cameras)
                )
                logger.info("Camera stream ready: %s", shapes)
                self._camera_stream_ready = True
            return

        if self._camera_missing_since_ns is None:
            self._camera_missing_since_ns = now_ns
        missing_duration_ns = now_ns - self._camera_missing_since_ns
        if missing_duration_ns >= 2_000_000_000 and now_ns - self._last_camera_wait_log_ns >= 5_000_000_000:
            logger.warning("Waiting for camera frames: %s", ", ".join(missing))
            self._last_camera_wait_log_ns = now_ns

    def step(self):
        self._receive_messages()
        frames = {
            cfg.name: camera.read(self.cfg.sync.poll_timeout_ms)
            for cfg, camera in zip(self.cfg.cameras, self.cameras)
        }
        self._update_camera_status(frames)
        if all(frame is not None for frame in frames.values()):
            self._record_frames(frames)

    def run(self):
        connected = []
        try:
            for camera in self.cameras:
                camera.connect()
                connected.append(camera)
                cfg = self.cfg.cameras[len(connected) - 1]
                logger.info("Camera backend connected: name=%s type=%s", cfg.name, cfg.type)
        except Exception:
            for camera in reversed(connected):
                camera.close()
            raise
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
        for camera in self.cameras:
            camera.close()
        self._socket.close(linger=0)
        logger.info("Recorder closed (total stale frames dropped: %d)", self.dropped_stale_frames)
