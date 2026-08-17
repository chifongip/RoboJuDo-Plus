import logging
import time
from collections import deque

import msgpack
import numpy as np
import zmq

from .cameras import CameraFrame, CameraSource, create_camera
from .config import RecorderConfig
from .dataset import LeRobotV3Writer
from .profiles import NamedJointProfile
from .protocol import ControlSample

logger = logging.getLogger(__name__)


class RecorderService:
    """Pair camera groups with timestamped controls without dropping video frames."""

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
        self._active_episode_started_at_ns: int | None = None
        self._review_episode_id: int | None = None
        self._active_task: str | None = None
        self._writer: LeRobotV3Writer | None = None
        self._profile: NamedJointProfile | None = None
        self._last_camera_sequences = {camera.name: -1 for camera in cfg.cameras}
        self._pending_frames: deque[dict[str, CameraFrame]] = deque(maxlen=cfg.sync.pending_frame_capacity)
        self._running = False
        self.dropped_stale_frames = 0
        self._episode_frame_count = 0
        self._episode_stale_frames = 0
        self._episode_unmatched_frames = 0
        self._episode_interpolated_frames = 0
        self._episode_hold_last_frames = 0
        self._episode_over_age_frames = 0
        self._episode_max_control_age_ms = 0.0
        self._camera_stream_ready = False
        self._camera_missing_since_ns: int | None = None
        self._last_camera_wait_log_ns = 0
        self._throughput_window_started_ns: int | None = None
        self._throughput_input_frames = {camera.name: 0 for camera in cfg.cameras}
        self._throughput_sequence_gaps = {camera.name: 0 for camera in cfg.cameras}
        self._throughput_last_sequences: dict[str, int | None] = {camera.name: None for camera in cfg.cameras}
        self._throughput_written_frames = 0

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
                self._active_episode_started_at_ns = sample.timestamp_ns(self.cfg.sync.clock)
                self._active_task = sample.task
                self._episode_frame_count = 0
                self._episode_stale_frames = 0
                self._episode_unmatched_frames = 0
                self._episode_interpolated_frames = 0
                self._episode_hold_last_frames = 0
                self._episode_over_age_frames = 0
                self._episode_max_control_age_ms = 0.0
                self._reset_throughput_metrics()
                logger.info("Episode %d started from first control sample: %s", sample.episode_id, sample.task)
            self._samples.append(sample)
        elif kind == "episode_start":
            episode_id = int(message["episode_id"])
            if self._active_episode_id not in (None, episode_id):
                self._finish_episode(save=True)
            self._active_episode_id = episode_id
            self._active_episode_started_at_ns = (
                int(message.get("timestamp_ns", received_at)) if self.cfg.sync.clock == "source" else received_at
            )
            self._active_task = str(message["task"])
            self._episode_frame_count = 0
            self._episode_stale_frames = 0
            self._episode_unmatched_frames = 0
            self._episode_interpolated_frames = 0
            self._episode_hold_last_frames = 0
            self._episode_over_age_frames = 0
            self._episode_max_control_age_ms = 0.0
            self._reset_throughput_metrics()
            logger.info("Episode %d armed: %s", episode_id, self._active_task)
        elif kind == "episode_end":
            if int(message["episode_id"]) == self._active_episode_id:
                self._finish_episode(save=bool(message.get("save", True)))
        elif kind == "episode_review":
            if int(message["episode_id"]) == self._active_episode_id:
                self._finish_episode(save=None)
                self._review_episode_id = self._active_episode_id
                logger.warning("Episode %s pending operator review", message["episode_id"])
        elif kind in {"episode_commit", "episode_discard"}:
            if int(message["episode_id"]) == self._review_episode_id:
                self._finish_episode(save=kind == "episode_commit")
                self._review_episode_id = None
        elif kind == "client_close":
            self._finish_episode(save=False)
            self._review_episode_id = None
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
                camera_shapes={
                    cfg.name: camera.shape for cfg, camera in zip(self.cfg.cameras, self.cameras, strict=True)
                },
                codec=self.cfg.dataset.codec,
                resume=self.cfg.dataset.resume,
            )
            camera_schema = ", ".join(
                f"{cfg.name}={camera.shape}" for cfg, camera in zip(self.cfg.cameras, self.cameras, strict=True)
            )
            logger.info("Dataset initialized at %s with cameras: %s", self.cfg.dataset.root.resolve(), camera_schema)
        if not self._writer.episode_open:
            self._writer.start_episode(self._active_task or sample.task)

    def _matching_sample(self, frame: CameraFrame, *, force: bool = False):
        if self._active_episode_id is None:
            return None
        candidates = [
            sample
            for sample in self._samples
            if sample.episode_id == self._active_episode_id
        ]
        timestamp = frame.timestamp_ns
        previous = [sample for sample in candidates if sample.timestamp_ns(self.cfg.sync.clock) <= timestamp]
        following = [sample for sample in candidates if sample.timestamp_ns(self.cfg.sync.clock) > timestamp]
        if not previous:
            return None
        previous_sample = previous[-1]
        age_ms = (timestamp - previous_sample.timestamp_ns(self.cfg.sync.clock)) / 1_000_000
        self._episode_max_control_age_ms = max(self._episode_max_control_age_ms, age_ms)
        if age_ms > self.cfg.sync.max_control_age_ms:
            self._episode_over_age_frames += 1
        if not following:
            if not force:
                return None
            self._episode_hold_last_frames += 1
            if self._episode_hold_last_frames == 1:
                logger.warning(
                    "Episode %s ended before future joint state arrived; holding the last state for pending frames",
                    self._active_episode_id,
                )
            return previous_sample, previous_sample.joint_positions, previous_sample.action

        next_sample = following[0]
        t0 = previous_sample.timestamp_ns(self.cfg.sync.clock)
        t1 = next_sample.timestamp_ns(self.cfg.sync.clock)
        if t1 <= t0:
            return previous_sample, previous_sample.joint_positions, previous_sample.action
        alpha = np.float32((timestamp - t0) / (t1 - t0))
        state = previous_sample.joint_positions + alpha * (
            next_sample.joint_positions - previous_sample.joint_positions
        )
        self._episode_interpolated_frames += 1
        return previous_sample, state.astype(np.float32), previous_sample.action

    def _write_frame(self, frames: dict[str, CameraFrame], match):
        sample, state, action = match
        self._ensure_writer(sample)
        self._writer.add_frame(state, action, {name: frame.image for name, frame in frames.items()})
        self._episode_frame_count += 1
        self._throughput_written_frames += 1
        if self._episode_frame_count == 1:
            logger.info(
                "Episode %s recording first camera frame from cameras: %s",
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

    def _flush_pending_frames(self, *, force: bool = False):
        while self._pending_frames:
            frames = self._pending_frames[0]
            match = self._matching_sample(frames[self.cfg.cameras[0].name], force=force)
            if match is None:
                if force:
                    self._pending_frames.popleft()
                    self._episode_unmatched_frames += 1
                    continue
                break
            self._pending_frames.popleft()
            self._write_frame(frames, match)

    def _record_frames(self, frames: dict[str, CameraFrame]):
        if any(frame.sequence == self._last_camera_sequences[name] for name, frame in frames.items()):
            return
        for name, frame in frames.items():
            self._last_camera_sequences[name] = frame.sequence
        primary_frame = frames[self.cfg.cameras[0].name]
        if (
            self._active_episode_started_at_ns is not None
            and primary_frame.timestamp_ns < self._active_episode_started_at_ns
        ):
            return
        episode_samples = [sample for sample in self._samples if sample.episode_id == self._active_episode_id]
        if not episode_samples:
            self._episode_unmatched_frames += 1
            if self._episode_unmatched_frames == 1 or self._episode_unmatched_frames % 100 == 0:
                logger.warning(
                    "Episode %s dropped %d camera frames before the first control sample",
                    self._active_episode_id,
                    self._episode_unmatched_frames,
                )
            return
        if len(self._pending_frames) == self._pending_frames.maxlen:
            self._pending_frames.popleft()
            self._episode_unmatched_frames += 1
            if self._episode_unmatched_frames == 1 or self._episode_unmatched_frames % 100 == 0:
                logger.warning(
                    "Episode %s dropped %d camera frames from a full pending queue (capacity=%d)",
                    self._active_episode_id,
                    self._episode_unmatched_frames,
                    self._pending_frames.maxlen,
                )
        self._pending_frames.append(frames)
        self._flush_pending_frames()

    def _finish_episode(self, *, save: bool | None):
        self._flush_pending_frames(force=True)
        self._log_throughput(force=True)
        episode_id = self._active_episode_id
        if self._writer is not None and self._writer.episode_open:
            if save is None:
                logger.warning(
                    "Episode %s review: %d frames, interpolated=%d, hold_last=%d, over_age=%d, unmatched=%d",
                    episode_id,
                    self._episode_frame_count,
                    self._episode_interpolated_frames,
                    self._episode_hold_last_frames,
                    self._episode_over_age_frames,
                    self._episode_unmatched_frames,
                )
            elif save and self._writer.has_pending_frames:
                self._writer.save_episode()
                logger.info(
                    "Episode %s saved: %d frames, %d stale frames dropped, root=%s",
                    episode_id,
                    self._episode_frame_count,
                    self._episode_stale_frames,
                    self.cfg.dataset.root.resolve(),
                )
                if (
                    self._episode_interpolated_frames
                    or self._episode_hold_last_frames
                    or self._episode_over_age_frames
                    or self._episode_unmatched_frames
                ):
                    logger.warning(
                        "Episode %s sync quality: interpolated=%d, hold_last=%d, over_age=%d, "
                        "unmatched=%d, max_control_age=%.1f ms (warning threshold %.1f ms); "
                        "matched video frames were retained",
                        episode_id,
                        self._episode_interpolated_frames,
                        self._episode_hold_last_frames,
                        self._episode_over_age_frames,
                        self._episode_unmatched_frames,
                        self._episode_max_control_age_ms,
                        self.cfg.sync.max_control_age_ms,
                    )
            elif save is False:
                self._writer.discard_episode()
                logger.info("Episode %s discarded after %d frames", episode_id, self._episode_frame_count)
        elif episode_id is not None and save:
            logger.warning(
                "Episode %s was not saved because no synchronized camera/control frames were recorded",
                episode_id,
            )
        if save is None:
            return
        self._active_episode_id = None
        self._active_episode_started_at_ns = None
        self._active_task = None
        self._pending_frames.clear()
        self._episode_frame_count = 0
        self._episode_stale_frames = 0
        self._episode_unmatched_frames = 0
        self._episode_over_age_frames = 0
        self._episode_max_control_age_ms = 0.0
        self._reset_throughput_metrics()

    def _reset_throughput_metrics(self):
        self._throughput_window_started_ns = None
        self._throughput_input_frames = {camera.name: 0 for camera in self.cfg.cameras}
        self._throughput_sequence_gaps = {camera.name: 0 for camera in self.cfg.cameras}
        self._throughput_last_sequences = {camera.name: None for camera in self.cfg.cameras}
        self._throughput_written_frames = 0

    def _observe_camera_throughput(self, frames: dict[str, CameraFrame | None]):
        observed_new_frame = False
        for name, frame in frames.items():
            if frame is None:
                continue
            previous_sequence = self._throughput_last_sequences[name]
            if frame.sequence == previous_sequence:
                continue
            if previous_sequence is not None and frame.sequence > previous_sequence + 1:
                self._throughput_sequence_gaps[name] += frame.sequence - previous_sequence - 1
            self._throughput_last_sequences[name] = frame.sequence
            self._throughput_input_frames[name] += 1
            observed_new_frame = True
        if observed_new_frame and self._throughput_window_started_ns is None:
            self._throughput_window_started_ns = time.monotonic_ns()

    def _log_throughput(self, *, force: bool = False):
        if self._throughput_window_started_ns is None:
            return
        now_ns = time.monotonic_ns()
        elapsed_s = (now_ns - self._throughput_window_started_ns) / 1_000_000_000
        if not force and elapsed_s < self.cfg.sync.throughput_log_interval_s:
            return
        if elapsed_s <= 0:
            return
        if force and not any(self._throughput_input_frames.values()) and self._throughput_written_frames == 0:
            return

        input_fps = {name: count / elapsed_s for name, count in self._throughput_input_frames.items()}
        write_fps = self._throughput_written_frames / elapsed_s
        target_fps = float(self.cfg.dataset.fps)
        input_summary = ", ".join(f"{name}={fps:.1f}" for name, fps in input_fps.items())
        gap_summary = ", ".join(f"{name}={count}" for name, count in self._throughput_sequence_gaps.items())
        below_target = any(fps < target_fps * 0.9 for fps in input_fps.values()) or write_fps < target_fps * 0.9
        has_sequence_gaps = any(self._throughput_sequence_gaps.values())
        log = logger.warning if below_target or has_sequence_gaps else logger.info
        log(
            "Episode %s throughput (%.1f s): input_fps=[%s], write_fps=%.1f, target_fps=%.1f, "
            "sequence_gaps=[%s]",
            self._active_episode_id,
            elapsed_s,
            input_summary,
            write_fps,
            target_fps,
            gap_summary,
        )
        self._throughput_window_started_ns = now_ns
        self._throughput_input_frames = {camera.name: 0 for camera in self.cfg.cameras}
        self._throughput_sequence_gaps = {camera.name: 0 for camera in self.cfg.cameras}
        self._throughput_written_frames = 0

    def _update_camera_status(self, frames: dict[str, CameraFrame | None]):
        missing = [name for name, frame in frames.items() if frame is None]
        now_ns = time.monotonic_ns()
        if not missing:
            self._camera_missing_since_ns = None
            if not self._camera_stream_ready:
                shapes = ", ".join(
                    f"{cfg.name}={camera.shape}" for cfg, camera in zip(self.cfg.cameras, self.cameras, strict=True)
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
        if self._active_episode_id is None or self._review_episode_id is not None:
            return
        frames = {
            cfg.name: camera.read(self.cfg.sync.poll_timeout_ms)
            for cfg, camera in zip(self.cfg.cameras, self.cameras, strict=True)
        }
        self._update_camera_status(frames)
        self._observe_camera_throughput(frames)
        if all(frame is not None for frame in frames.values()):
            self._record_frames(frames)
        self._log_throughput()

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
