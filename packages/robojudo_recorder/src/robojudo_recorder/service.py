"""Real-time phase of the two-stage RoboJuDo recorder."""

import logging
import time
from collections import deque

import msgpack
import zmq

from .cameras import CameraFrame, CameraSource, create_camera
from .config import RecorderConfig
from .profiles import NamedJointProfile
from .protocol import ControlSample
from .raw import RawEpisodeWriter

logger = logging.getLogger(__name__)


class RecorderService:
    """Spool timestamped controls and compressed camera frames without encoding video.

    This service is deliberately latency-oriented. It does no state interpolation, camera
    resampling, Parquet generation, or H.264 encoding; ``robojudo-finalize`` performs those
    operations after collection.
    """

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
        self._profile: NamedJointProfile | None = None
        self._raw_writer: RawEpisodeWriter | None = None
        self._active_episode_id: int | None = None
        self._active_episode_started_at_ns: int | None = None
        self._review_episode_id: int | None = None
        self._active_task: str | None = None
        self._running = False

        self._last_camera_sequences = {item.name: -1 for item in cfg.cameras}
        self._camera_stream_ready = False
        self._camera_missing_since_ns: int | None = None
        self._last_camera_wait_log_ns = 0
        self._episode_frame_counts = {item.name: 0 for item in cfg.cameras}
        self._throughput_window_started_ns: int | None = None
        self._throughput_input_frames = {item.name: 0 for item in cfg.cameras}
        self._throughput_written_frames = {item.name: 0 for item in cfg.cameras}
        self._throughput_sequence_gaps = {item.name: 0 for item in cfg.cameras}
        self._throughput_last_sequences: dict[str, int | None] = {item.name: None for item in cfg.cameras}
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

    def _start_episode(self, episode_id: int, task: str, source_timestamp_ns: int, receive_timestamp_ns: int):
        if self._active_episode_id == episode_id:
            return
        if self._active_episode_id is not None:
            self._finish_episode(save=True)
        self._raw_writer = RawEpisodeWriter(
            raw_root=self.cfg.dataset.raw_root,
            episode_id=episode_id,
            task=task,
            camera_names=tuple(item.name for item in self.cfg.cameras),
            started_source_ns=source_timestamp_ns,
            started_receive_ns=receive_timestamp_ns,
            jpeg_quality=self.cfg.dataset.jpeg_quality,
        )
        self._active_episode_id = episode_id
        self._active_episode_started_at_ns = (
            source_timestamp_ns if self.cfg.sync.clock == "source" else receive_timestamp_ns
        )
        self._active_task = task
        self._episode_frame_counts = {item.name: 0 for item in self.cfg.cameras}
        self._reset_throughput_metrics()
        logger.info("Episode %d raw capture armed: %s", episode_id, task)

    def _handle_message(self, message: dict, received_at: int):
        kind = message.get("kind")
        if kind == "sample":
            sample = ControlSample.from_message(message, received_at)
            if self._active_episode_id != sample.episode_id:
                self._start_episode(
                    sample.episode_id,
                    sample.task,
                    sample.source_timestamp_ns,
                    sample.receive_timestamp_ns,
                )
            if self._profile is None:
                self._profile = NamedJointProfile.from_sample(sample)
            else:
                self._profile.validate(sample)
            self._samples.append(sample)
            self._raw_writer.add_control(sample)
        elif kind == "episode_start":
            source_timestamp_ns = int(message.get("timestamp_ns", received_at))
            self._start_episode(int(message["episode_id"]), str(message["task"]), source_timestamp_ns, received_at)
        elif kind == "episode_end":
            if int(message["episode_id"]) == self._active_episode_id:
                self._finish_episode(save=bool(message.get("save", True)))
        elif kind == "episode_review":
            if int(message["episode_id"]) == self._active_episode_id:
                self._finish_episode(save=None)
        elif kind in {"episode_commit", "episode_discard"}:
            if int(message["episode_id"]) == self._review_episode_id:
                self._finish_episode(save=kind == "episode_commit")
        elif kind == "client_close":
            self._finish_episode(save=False)
        else:
            raise ValueError(f"unknown message kind {kind!r}")

    def _finish_episode(self, *, save: bool | None):
        if self._raw_writer is None or self._active_episode_id is None:
            return
        self._log_throughput(force=True)
        episode_id = self._active_episode_id
        frame_summary = ", ".join(f"{name}={count}" for name, count in self._episode_frame_counts.items())
        if save is None:
            self._raw_writer.close_for_review()
            self._review_episode_id = episode_id
            logger.warning("Episode %s raw capture pending operator review: %s", episode_id, frame_summary)
            return
        if save:
            path = self._raw_writer.commit()
            logger.info("Episode %s raw capture committed: %s, path=%s", episode_id, frame_summary, path)
        else:
            self._raw_writer.discard()
            logger.info("Episode %s raw capture discarded: %s", episode_id, frame_summary)
        self._raw_writer = None
        self._active_episode_id = None
        self._active_episode_started_at_ns = None
        self._review_episode_id = None
        self._active_task = None
        self._reset_throughput_metrics()

    def _record_frame(self, camera_name: str, frame: CameraFrame):
        if frame.sequence == self._last_camera_sequences[camera_name]:
            return
        self._last_camera_sequences[camera_name] = frame.sequence
        timestamp_ns = frame.source_timestamp_ns if self.cfg.sync.clock == "source" else frame.receive_timestamp_ns
        if self._active_episode_started_at_ns is not None and timestamp_ns < self._active_episode_started_at_ns:
            return
        self._raw_writer.add_frame(camera_name, frame)
        self._episode_frame_counts[camera_name] += 1
        self._throughput_written_frames[camera_name] += 1
        if self._episode_frame_counts[camera_name] == 1:
            logger.info(
                "Episode %s recording first raw frame: %s=%s encoding=%s",
                self._active_episode_id,
                camera_name,
                frame.shape,
                frame.encoding or "jpeg",
            )

    def _observe_camera_throughput(self, camera_name: str, frame: CameraFrame):
        previous = self._throughput_last_sequences[camera_name]
        if frame.sequence == previous:
            return
        if previous is not None and frame.sequence > previous + 1:
            self._throughput_sequence_gaps[camera_name] += frame.sequence - previous - 1
        self._throughput_last_sequences[camera_name] = frame.sequence
        self._throughput_input_frames[camera_name] += 1
        if self._throughput_window_started_ns is None:
            self._throughput_window_started_ns = time.monotonic_ns()

    def _reset_throughput_metrics(self):
        self._throughput_window_started_ns = None
        self._throughput_input_frames = {item.name: 0 for item in self.cfg.cameras}
        self._throughput_written_frames = {item.name: 0 for item in self.cfg.cameras}
        self._throughput_sequence_gaps = {item.name: 0 for item in self.cfg.cameras}
        self._throughput_last_sequences = {item.name: None for item in self.cfg.cameras}

    def _log_throughput(self, *, force: bool = False):
        if self._throughput_window_started_ns is None:
            return
        if (
            force
            and not any(self._throughput_input_frames.values())
            and not any(self._throughput_written_frames.values())
        ):
            return
        now_ns = time.monotonic_ns()
        elapsed_s = (now_ns - self._throughput_window_started_ns) / 1_000_000_000
        if elapsed_s <= 0 or (not force and elapsed_s < self.cfg.sync.throughput_log_interval_s):
            return
        input_fps = {name: count / elapsed_s for name, count in self._throughput_input_frames.items()}
        write_fps = {name: count / elapsed_s for name, count in self._throughput_written_frames.items()}
        input_summary = ", ".join(f"{name}={value:.1f}" for name, value in input_fps.items())
        write_summary = ", ".join(f"{name}={value:.1f}" for name, value in write_fps.items())
        gap_summary = ", ".join(f"{name}={value}" for name, value in self._throughput_sequence_gaps.items())
        expected_fps = {item.name: float(item.options.get("fps", self.cfg.dataset.fps)) for item in self.cfg.cameras}
        expected_summary = ", ".join(f"{name}={value:.1f}" for name, value in expected_fps.items())
        unhealthy = (
            any(input_fps[name] < expected * 0.9 for name, expected in expected_fps.items())
            or any(write_fps[name] < expected * 0.9 for name, expected in expected_fps.items())
            or any(self._throughput_sequence_gaps.values())
        )
        log = logger.warning if unhealthy else logger.info
        log(
            "Episode %s raw throughput (%.1f s): input_fps=[%s], write_fps=[%s], expected_fps=[%s], "
            "output_fps=%.1f, sequence_gaps=[%s]",
            self._active_episode_id,
            elapsed_s,
            input_summary,
            write_summary,
            expected_summary,
            self.cfg.dataset.fps,
            gap_summary,
        )
        self._throughput_window_started_ns = now_ns
        self._throughput_input_frames = {item.name: 0 for item in self.cfg.cameras}
        self._throughput_written_frames = {item.name: 0 for item in self.cfg.cameras}
        self._throughput_sequence_gaps = {item.name: 0 for item in self.cfg.cameras}

    def _update_camera_status(self, frames: dict[str, CameraFrame | None]):
        missing = [name for name, frame in frames.items() if frame is None]
        now_ns = time.monotonic_ns()
        if not missing:
            self._camera_missing_since_ns = None
            if not self._camera_stream_ready:
                shapes = ", ".join(
                    f"{item.name}={camera.shape}" for item, camera in zip(self.cfg.cameras, self.cameras, strict=True)
                )
                logger.info("Camera stream ready: %s", shapes)
                self._camera_stream_ready = True
            return
        if self._camera_missing_since_ns is None:
            self._camera_missing_since_ns = now_ns
        missing_long_enough = now_ns - self._camera_missing_since_ns >= 2_000_000_000
        log_interval_elapsed = now_ns - self._last_camera_wait_log_ns >= 5_000_000_000
        if missing_long_enough and log_interval_elapsed:
            logger.warning("Waiting for camera frames: %s", ", ".join(missing))
            self._last_camera_wait_log_ns = now_ns

    def step(self):
        self._receive_messages()
        if self._active_episode_id is None or self._review_episode_id is not None:
            return
        frames = {
            item.name: camera.read(self.cfg.sync.poll_timeout_ms)
            for item, camera in zip(self.cfg.cameras, self.cameras, strict=True)
        }
        self._update_camera_status(frames)
        for name, frame in frames.items():
            if frame is None:
                continue
            self._observe_camera_throughput(name, frame)
            self._record_frame(name, frame)
        self._log_throughput()

    def run(self):
        connected = []
        try:
            for item, camera in zip(self.cfg.cameras, self.cameras, strict=True):
                camera.connect()
                connected.append(camera)
                logger.info("Camera backend connected: name=%s type=%s", item.name, item.type)
        except Exception:
            for camera in reversed(connected):
                camera.close()
            raise
        self._running = True
        logger.info("Recorder connected to control endpoint %s", self.cfg.control_endpoint)
        logger.info(
            "Raw episodes will be written to %s; run robojudo-finalize after collection",
            self.cfg.dataset.raw_root,
        )
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
        for camera in self.cameras:
            camera.close()
        self._socket.close(linger=0)
        logger.info("Recorder closed; raw episodes remain available for offline finalization")
