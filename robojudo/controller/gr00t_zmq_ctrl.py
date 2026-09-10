import logging
import math
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from numbers import Integral, Real

import msgpack
import numpy as np
import zmq

from robojudo.controller import ControllerHook, ctrl_registry
from robojudo.controller.casia_hand_runtime import (
    CASIA_JOINT_NAMES,
    CASIA_LEFT_JOINT_NAMES,
    CASIA_RIGHT_JOINT_NAMES,
    CasiaHandRuntime,
)
from robojudo.controller.ctrl_cfgs import Gr00tZmqCtrlCfg

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _EncodedCameraFrame:
    shape: tuple[int, int, int]
    jpeg: bytes
    timestamp_ns: int
    source_timestamp_ns: int
    receive_timestamp_ns: int
    sequence: int


class _JpegEncoderWorker:
    """Encode one camera independently while keeping latency bounded."""

    def __init__(self, name: str, cv2, jpeg_quality: int, capacity: int):
        self.name = name
        self._cv2 = cv2
        self._jpeg_quality = jpeg_quality
        self._capacity = capacity
        self._condition = threading.Condition()
        self._pending = deque()
        self._completed = deque()
        self._stopping = False
        self._error: Exception | None = None
        self.dropped_frames = 0
        self._thread = threading.Thread(target=self._run, name=f"Gr00tJpegEncoder-{name}", daemon=True)
        self._thread.start()

    def submit(self, frame) -> None:
        with self._condition:
            if len(self._pending) >= self._capacity:
                self._pending.popleft()
                self.dropped_frames += 1
            self._pending.append(frame)
            self._condition.notify()

    def drain(self) -> list[_EncodedCameraFrame]:
        with self._condition:
            if self._error is not None:
                raise RuntimeError(f"GR00T JPEG encoder failed: {self.name}") from self._error
            completed = list(self._completed)
            self._completed.clear()
            return completed

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._stopping or bool(self._pending))
                    if self._stopping and not self._pending:
                        return
                    frame = self._pending.popleft()
                shape, jpeg = Gr00tZmqCtrl._prepare_observation_jpeg(
                    frame,
                    self._cv2,
                    self._jpeg_quality,
                )
                timestamp_ns = int(frame.timestamp_ns)
                prepared = _EncodedCameraFrame(
                    shape=shape,
                    jpeg=jpeg,
                    timestamp_ns=timestamp_ns,
                    source_timestamp_ns=int(getattr(frame, "source_timestamp_ns", None) or timestamp_ns),
                    receive_timestamp_ns=int(getattr(frame, "receive_timestamp_ns", None) or timestamp_ns),
                    sequence=int(frame.sequence),
                )
                with self._condition:
                    if len(self._completed) >= self._capacity:
                        self._completed.popleft()
                        self.dropped_frames += 1
                    self._completed.append(prepared)
        except Exception as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._stopping = True
            self._pending.clear()
            self._condition.notify_all()
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            logger.warning("GR00T JPEG encoder %s did not stop within 2 seconds", self.name)


@ctrl_registry.register
class Gr00tZmqCtrl(ControllerHook):
    """Receive atomic GR00T commands and publish camera/joint observations.

    Threading and data flow:

    - Control thread (pipeline rate): ``get_data_with_hook`` snapshots measured
      upper-body joints from ``env_data`` and non-blockingly returns the latest
      GR00T command to the pipeline.
    - Observation worker: drains independent camera sources and JPEG workers,
      time-aligns their frames, combines them with the latest thread-safe joint
      snapshot and task, then publishes one atomic multipart message to deploy.

    The worker never reads robot state directly, and it never executes robot
    control; final targets are still applied synchronously by the pipeline.
    """

    cfg_ctrl: Gr00tZmqCtrlCfg

    def __init__(self, cfg_ctrl: Gr00tZmqCtrlCfg, env=None, device="cpu"):
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, device=device)
        self._joint_names = tuple(cfg_ctrl.joint_names)
        self._hand_joint_names = CASIA_JOINT_NAMES if cfg_ctrl.casia_hand is not None else ()
        self._policy_joint_names = (*self._joint_names, *self._hand_joint_names)
        self._joint_name_set = set(self._policy_joint_names)
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVHWM, 100)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.connect(cfg_ctrl.endpoint)
        self._joint_indices = (
            np.asarray(
                [env.joint_names.index(name) for name in self._joint_names],
                dtype=np.int32,
            )
            if cfg_ctrl.observation_enabled
            else np.asarray([], dtype=np.int32)
        )
        self._latest_positions: dict[str, float] = {}
        self._latest_locomotion_command: np.ndarray | None = None
        self._latest_sequence: int | None = None
        self._latest_command_stream_id: str | None = None
        self._latest_command_session: int | None = None
        self._last_received_at: float | None = None
        self._last_invalid_log_at = float("-inf")
        self._observation_snapshot_lock = threading.Lock()
        self._observation_snapshot: tuple[int, np.ndarray] | None = None
        self._observation_stream_id = uuid.uuid4().hex
        self._takeover_enabled = False
        self._control_session = 0
        self._observation_stop = threading.Event()
        self._observation_ready = threading.Event()
        self._observation_thread: threading.Thread | None = None
        self._observation_error: Exception | None = None
        self._published_observations = 0
        self._dropped_observations = 0
        self._camera_encoder_drops: dict[str, int] = {}
        self._hand_runtime = None
        logger.info("Gr00tZmqCtrl subscribed to %s", cfg_ctrl.endpoint)
        try:
            if cfg_ctrl.casia_hand is not None:
                self._hand_runtime = CasiaHandRuntime(cfg_ctrl.casia_hand)
            if cfg_ctrl.observation_enabled:
                self._start_observation_worker()
        except Exception:
            self.close()
            raise

    def reset(self):
        self._latest_positions.clear()
        self._latest_locomotion_command = None
        self._latest_sequence = None
        self._latest_command_stream_id = None
        self._latest_command_session = None
        self._last_received_at = None
        with self._observation_snapshot_lock:
            self._takeover_enabled = False
        hand_runtime = getattr(self, "_hand_runtime", None)
        if hand_runtime is not None:
            hand_runtime.reset()
        for _ in range(100):
            try:
                self._socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break

    def close(self):
        self._observation_stop.set()
        if self._observation_thread is not None:
            shutdown_timeout_s = max(3.0, 2.5 * len(self.cfg_ctrl.observation_cameras))
            self._observation_thread.join(timeout=shutdown_timeout_s)
            if self._observation_thread.is_alive():
                logger.warning(
                    "GR00T observation worker did not stop within %.1f seconds",
                    shutdown_timeout_s,
                )
            self._observation_thread = None
        hand_runtime = getattr(self, "_hand_runtime", None)
        if hand_runtime is not None:
            hand_runtime.close()
            self._hand_runtime = None
        self._socket.close(linger=0)

    def _start_observation_worker(self):
        self._observation_thread = threading.Thread(
            target=self._observation_loop,
            name="Gr00tObservationPublisher",
            daemon=True,
        )
        self._observation_thread.start()
        if not self._observation_ready.wait(self.cfg_ctrl.camera_startup_timeout_s):
            self._observation_stop.set()
            raise TimeoutError(
                f"GR00T camera did not start within {self.cfg_ctrl.camera_startup_timeout_s:.1f} seconds"
            )
        if self._observation_error is not None:
            raise RuntimeError("failed to start GR00T observation publisher") from self._observation_error

    def set_takeover_enabled(self, enabled: bool, *, return_hand_to_default: bool = False) -> bool:
        """Publish takeover state and advance the session on each enable edge."""
        enabled = bool(enabled)
        with self._observation_snapshot_lock:
            changed = enabled != self._takeover_enabled
            if enabled and not self._takeover_enabled:
                self._control_session += 1
            self._takeover_enabled = enabled
        if changed:
            self._latest_positions.clear()
            self._latest_locomotion_command = None
            self._latest_command_stream_id = None
            self._latest_command_session = None
            self._last_received_at = None
        hand_runtime = getattr(self, "_hand_runtime", None)
        if changed and hand_runtime is not None:
            hand_runtime.set_takeover_enabled(
                enabled,
                return_to_default=bool(not enabled and return_hand_to_default),
            )
        return changed

    @staticmethod
    def _prepare_observation_jpeg(frame, cv2, jpeg_quality: int) -> tuple[tuple[int, int, int], bytes]:
        """Return an RGB shape and JPEG payload for either CameraFrame representation.

        Camera backends may expose decoded RGB pixels in ``image`` or preserve an
        already-compressed payload in ``encoded_image``. JPEG input is forwarded
        byte-for-byte; other compressed formats are decoded and converted to JPEG.
        """
        encoded_image = getattr(frame, "encoded_image", None)
        encoding = str(getattr(frame, "encoding", "") or "").lower()
        if encoded_image is not None and encoding in {"jpeg", "jpg"}:
            shape = tuple(frame.shape)
            if len(shape) != 3 or shape[2] != 3:
                raise ValueError(f"GR00T camera returned invalid RGB shape {shape}")
            return shape, bytes(encoded_image)

        image = getattr(frame, "image", None)
        if image is None and encoded_image is not None:
            compressed = np.frombuffer(encoded_image, dtype=np.uint8)
            decoded = cv2.imdecode(compressed, cv2.IMREAD_UNCHANGED)
            if decoded is None:
                raise ValueError(f"GR00T camera could not decode {encoding or 'compressed'} image")
            if decoded.ndim == 2:
                image = cv2.cvtColor(decoded, cv2.COLOR_GRAY2RGB)
            elif decoded.ndim == 3 and decoded.shape[2] == 4:
                image = cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGB)
            elif decoded.ndim == 3 and decoded.shape[2] == 3:
                image = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError(f"GR00T camera returned invalid decoded shape {decoded.shape}")
        if image is None:
            raise ValueError("GR00T camera frame contains neither decoded pixels nor compressed bytes")

        image = np.asarray(image, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"GR00T camera returned invalid RGB shape {image.shape}")
        bgr = np.ascontiguousarray(image[:, :, ::-1])
        ok, encoded = cv2.imencode(
            ".jpg",
            bgr,
            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
        )
        if not ok:
            raise RuntimeError("failed to encode GR00T camera frame as JPEG")
        return tuple(image.shape), encoded.tobytes()

    @staticmethod
    def _read_camera_batch(camera, max_frames: int):
        read_batch = getattr(camera, "read_batch", None)
        if callable(read_batch):
            return read_batch(max_frames)
        frame = camera.read(timeout_ms=0)
        return [] if frame is None else [frame]

    @staticmethod
    def _camera_bundle_candidate(
        buffers: dict[str, deque],
    ) -> tuple[dict[str, _EncodedCameraFrame], int, int] | None:
        """Select frames nearest the newest timestamp available from every camera."""
        if any(not frames for frames in buffers.values()):
            return None
        target_ns = min(frames[-1].timestamp_ns for frames in buffers.values())
        selected = {
            name: min(
                frames,
                key=lambda frame: (abs(frame.timestamp_ns - target_ns), -frame.timestamp_ns),
            )
            for name, frames in buffers.items()
        }
        timestamps = [frame.timestamp_ns for frame in selected.values()]
        return selected, target_ns, max(timestamps) - min(timestamps)

    @staticmethod
    def _consume_camera_bundle(buffers: dict[str, deque], selected: dict[str, _EncodedCameraFrame]) -> None:
        for name, selected_frame in selected.items():
            while buffers[name]:
                frame = buffers[name].popleft()
                if frame is selected_frame:
                    break

    def _observation_loop(self):
        # Worker flow: independent cameras -> independent JPEG workers -> synchronized multipart PUB.
        cameras = []
        encoders: dict[str, _JpegEncoderWorker] = {}
        publisher = None
        try:
            try:
                import cv2
                from robojudo_recorder.cameras import create_camera
                from robojudo_recorder.config import CameraConfig
            except ImportError as exc:
                raise RuntimeError(
                    "GR00T observation publishing requires robojudo-recorder and OpenCV"
                ) from exc

            configured_cameras = self.cfg_ctrl.observation_cameras
            image_keys = tuple(camera.resolved_image_key for camera in configured_cameras)
            for configured in configured_cameras:
                camera_cfg = CameraConfig(
                    type=configured.type,
                    name=configured.name,
                    options=dict(configured.options),
                )
                camera = create_camera(camera_cfg)
                set_pending_capacity = getattr(camera, "set_pending_capacity", None)
                if callable(set_pending_capacity):
                    set_pending_capacity(self.cfg_ctrl.camera_pending_capacity)
                try:
                    camera.connect()
                except Exception:
                    camera.close()
                    raise
                cameras.append(camera)
                encoders[configured.resolved_image_key] = _JpegEncoderWorker(
                    configured.resolved_image_key,
                    cv2,
                    self.cfg_ctrl.observation_jpeg_quality,
                    self.cfg_ctrl.camera_encoder_queue_capacity,
                )
            publisher = self._context.socket(zmq.PUB)
            publisher.setsockopt(zmq.LINGER, 0)
            publisher.setsockopt(zmq.SNDHWM, 2)
            publisher.bind(self.cfg_ctrl.observation_endpoint)
            self._observation_ready.set()
            logger.info(
                "GR00T observations publishing to %s from cameras [%s]",
                self.cfg_ctrl.observation_endpoint,
                ", ".join(
                    f"{configured.resolved_image_key}={configured.type}:{configured.name}"
                    for configured in configured_cameras
                ),
            )

            minimum_period_ns = int(1_000_000_000 / self.cfg_ctrl.observation_fps)
            # Accept normal camera/scheduler jitter at the configured rate.
            minimum_interval_ns = minimum_period_ns * 9 // 10
            last_published_at = 0
            last_camera_sequences = {key: -1 for key in image_keys}
            observation_sequence = 0
            max_skew_ns = int(self.cfg_ctrl.max_camera_skew_ms * 1_000_000)
            buffer_capacity = max(2, self.cfg_ctrl.camera_pending_capacity * 2)
            encoded_buffers = {key: deque(maxlen=buffer_capacity) for key in image_keys}
            while not self._observation_stop.is_set():
                made_progress = False
                for configured, camera in zip(configured_cameras, cameras, strict=True):
                    image_key = configured.resolved_image_key
                    for frame in self._read_camera_batch(camera, self.cfg_ctrl.camera_pending_capacity):
                        if frame.sequence == last_camera_sequences[image_key]:
                            continue
                        last_camera_sequences[image_key] = frame.sequence
                        encoders[image_key].submit(frame)
                        made_progress = True
                for image_key, encoder in encoders.items():
                    completed = encoder.drain()
                    self._camera_encoder_drops[image_key] = encoder.dropped_frames
                    if completed:
                        encoded_buffers[image_key].extend(completed)
                        made_progress = True

                candidate = self._camera_bundle_candidate(encoded_buffers)
                while candidate is not None:
                    selected, target_ns, camera_skew_ns = candidate
                    if camera_skew_ns > max_skew_ns:
                        oldest_timestamp = min(frame.timestamp_ns for frame in selected.values())
                        for image_key, frame in selected.items():
                            if frame.timestamp_ns == oldest_timestamp:
                                self._consume_camera_bundle(
                                    {image_key: encoded_buffers[image_key]},
                                    {image_key: frame},
                                )
                        self._dropped_observations += 1
                        candidate = self._camera_bundle_candidate(encoded_buffers)
                        continue

                    now_ns = time.monotonic_ns()
                    if now_ns - last_published_at < minimum_interval_ns:
                        self._consume_camera_bundle(encoded_buffers, selected)
                        candidate = self._camera_bundle_candidate(encoded_buffers)
                        continue
                    with self._observation_snapshot_lock:
                        snapshot = self._observation_snapshot
                        takeover_enabled = self._takeover_enabled
                        control_session = self._control_session
                    if snapshot is None:
                        self._consume_camera_bundle(encoded_buffers, selected)
                        break

                    joint_timestamp_ns, joint_positions = snapshot
                    joint_timeout_ns = int(self.cfg_ctrl.observation_joint_timeout_s * 1_000_000_000)
                    if now_ns - joint_timestamp_ns > joint_timeout_ns:
                        self._consume_camera_bundle(encoded_buffers, selected)
                        break

                    observation_sequence += 1
                    header = {
                        "protocol_version": 1 if len(image_keys) == 1 else 2,
                        "stream_id": self._observation_stream_id,
                        "control_session": control_session,
                        "takeover_enabled": takeover_enabled,
                        "sequence": observation_sequence,
                        "timestamp_ns": int(target_ns),
                        "joint_timestamp_ns": joint_timestamp_ns,
                        "robot_type": self.cfg_ctrl.observation_profile.split("_", 1)[0],
                        "profile": self.cfg_ctrl.observation_profile,
                        "task": self.cfg_ctrl.observation_task,
                        "encoding": "jpeg",
                        "joint_names": list(self._policy_joint_names),
                        "joint_positions": joint_positions.tolist(),
                    }
                    if len(image_keys) == 1:
                        image_key = image_keys[0]
                        frame = selected[image_key]
                        configured = configured_cameras[0]
                        header.update(
                            camera_sequence=frame.sequence,
                            camera_name=configured.name,
                            shape=list(frame.shape),
                        )
                    else:
                        header.update(
                            image_keys=list(image_keys),
                            image_shapes={key: list(selected[key].shape) for key in image_keys},
                            image_timestamps_ns={key: selected[key].timestamp_ns for key in image_keys},
                            image_source_timestamps_ns={
                                key: selected[key].source_timestamp_ns for key in image_keys
                            },
                            image_receive_timestamps_ns={
                                key: selected[key].receive_timestamp_ns for key in image_keys
                            },
                            image_sequences={key: selected[key].sequence for key in image_keys},
                            camera_skew_ns=camera_skew_ns,
                        )
                    parts = [
                        msgpack.packb(header, use_bin_type=True),
                        *(selected[key].jpeg for key in image_keys),
                    ]
                    try:
                        publisher.send_multipart(parts, flags=zmq.NOBLOCK)
                        self._published_observations += 1
                        last_published_at = now_ns
                    except zmq.Again:
                        self._dropped_observations += 1
                    self._consume_camera_bundle(encoded_buffers, selected)
                    candidate = self._camera_bundle_candidate(encoded_buffers)

                if not made_progress:
                    self._observation_stop.wait(self.cfg_ctrl.camera_poll_timeout_ms / 1000)
        except Exception as exc:
            self._observation_error = exc
            logger.exception("GR00T observation publisher stopped: %s", exc)
            self._observation_ready.set()
        finally:
            for encoder in encoders.values():
                encoder.close()
            for camera in reversed(cameras):
                camera.close()
            if publisher is not None:
                publisher.close(linger=0)

    def _decode_positions(self, value) -> dict[str, float]:
        if not isinstance(value, dict) or not value:
            raise ValueError("positions must be a non-empty object")
        unknown = sorted(set(value) - self._joint_name_set)
        if unknown:
            raise ValueError(f"unknown GR00T joints: {unknown}")
        if self.cfg_ctrl.require_complete_positions:
            missing = sorted(self._joint_name_set - set(value))
            if missing:
                raise ValueError(f"GR00T message is missing joints: {missing}")

        positions = {}
        for name, position in value.items():
            if isinstance(position, bool) or not isinstance(position, Real):
                raise ValueError(f"position for {name} must be numeric")
            position = float(position)
            if not math.isfinite(position):
                raise ValueError(f"position for {name} must be finite")
            positions[name] = position
        return positions

    def _split_policy_positions(
        self, positions: dict[str, float]
    ) -> tuple[dict[str, float], np.ndarray | None, np.ndarray | None]:
        """Split one validated policy target into robot-arm and optional CASIA commands."""
        arm_positions = {name: positions[name] for name in self._joint_names}
        if getattr(self, "_hand_runtime", None) is None:
            return arm_positions, None, None
        left_hand = np.asarray(
            [positions[name] for name in CASIA_LEFT_JOINT_NAMES],
            dtype=np.float64,
        )
        right_hand = np.asarray(
            [positions[name] for name in CASIA_RIGHT_JOINT_NAMES],
            dtype=np.float64,
        )
        return arm_positions, left_hand, right_hand

    @staticmethod
    def _decode_locomotion_command(value) -> np.ndarray:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ValueError("locomotion_command must be a four-element array [vx, vy, yaw_rate, height]")
        if any(isinstance(item, bool) or not isinstance(item, Real) for item in value):
            raise ValueError("locomotion_command values must be numeric")
        command = np.asarray(value, dtype=np.float32)
        if not np.isfinite(command).all():
            raise ValueError("locomotion_command values must be finite")
        return command

    def _decode_message(
        self, message
    ) -> tuple[dict[str, float], np.ndarray, int | None, str, int]:
        if not isinstance(message, dict):
            raise ValueError("GR00T message must be an object")
        if "positions" not in message or "locomotion_command" not in message:
            raise ValueError("GR00T message must contain positions and locomotion_command")
        positions = self._decode_positions(message["positions"])
        locomotion_command = self._decode_locomotion_command(message["locomotion_command"])

        sequence = message.get("sequence")
        if sequence is not None:
            if isinstance(sequence, bool) or not isinstance(sequence, Integral) or sequence < 0:
                raise ValueError("sequence must be a non-negative integer")
            sequence = int(sequence)
        stream_id = message.get("stream_id")
        if not isinstance(stream_id, str) or not stream_id.strip():
            raise ValueError("stream_id must be a non-empty string")
        control_session = message.get("control_session")
        if (
            isinstance(control_session, bool)
            or not isinstance(control_session, Integral)
            or control_session < 0
        ):
            raise ValueError("control_session must be a non-negative integer")
        return positions, locomotion_command, sequence, stream_id, int(control_session)

    def _log_invalid_message(self, exc: Exception, now: float):
        if now - self._last_invalid_log_at >= 1.0:
            logger.warning("Rejected GR00T ZMQ message: %s", exc)
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
                positions, locomotion_command, sequence, stream_id, control_session = (
                    self._decode_message(message)
                )
            except ValueError as exc:
                self._log_invalid_message(exc, now)
                continue
            with self._observation_snapshot_lock:
                takeover_enabled = self._takeover_enabled
                expected_session = self._control_session
            if (
                not takeover_enabled
                or stream_id != self._observation_stream_id
                or control_session != expected_session
            ):
                self._log_invalid_message(
                    ValueError(
                        f"command session {stream_id}:{control_session} does not match "
                        f"active session {self._observation_stream_id}:{expected_session}"
                    ),
                    now,
                )
                continue
            stream_is_fresh = (
                self._last_received_at is not None and now - self._last_received_at <= self.cfg_ctrl.timeout_s
            )
            if (
                stream_is_fresh
                and sequence is not None
                and self._latest_sequence is not None
                and sequence <= self._latest_sequence
            ):
                self._log_invalid_message(
                    ValueError(f"sequence {sequence} is not newer than {self._latest_sequence}"),
                    now,
                )
                continue

            try:
                arm_positions, left_hand, right_hand = self._split_policy_positions(positions)
                hand_runtime = getattr(self, "_hand_runtime", None)
                if hand_runtime is not None:
                    hand_runtime.set_joint_commands(
                        left_hand,
                        right_hand,
                        time.monotonic_ns(),
                        sequence,
                    )
            except ValueError as exc:
                self._log_invalid_message(exc, now)
                continue
            self._latest_positions = arm_positions
            self._latest_locomotion_command = locomotion_command
            self._latest_sequence = sequence
            self._latest_command_stream_id = stream_id
            self._latest_command_session = control_session
            self._last_received_at = now

    def get_data(self):
        now = time.monotonic()
        self._receive_available(now)
        has_received = self._last_received_at is not None
        age_s = None if self._last_received_at is None else now - self._last_received_at
        with self._observation_snapshot_lock:
            takeover_enabled = self._takeover_enabled
            control_session = self._control_session
        fresh = bool(
            takeover_enabled
            and age_s is not None
            and age_s <= self.cfg_ctrl.timeout_s
            and self._latest_command_stream_id == self._observation_stream_id
            and self._latest_command_session == control_session
        )
        observation_error = getattr(self, "_observation_error", None)
        observation_ready = getattr(self, "_observation_ready", None)
        result = {
            "joint_positions": self._latest_positions.copy(),
            "locomotion_command": (
                None if self._latest_locomotion_command is None else self._latest_locomotion_command.copy()
            ),
            "sequence": self._latest_sequence,
            "stream_id": self._observation_stream_id,
            "control_session": control_session,
            "has_received": has_received,
            "fresh": fresh,
            "age_s": age_s,
            "observation_ready": bool(
                not self.cfg_ctrl.observation_enabled
                or (
                    observation_ready is not None
                    and observation_ready.is_set()
                    and observation_error is None
                )
            ),
            "observation_error": None if observation_error is None else str(observation_error),
            "published_observations": getattr(self, "_published_observations", 0),
            "dropped_observations": getattr(self, "_dropped_observations", 0),
            "camera_encoder_drops": getattr(self, "_camera_encoder_drops", {}).copy(),
        }
        hand_runtime = getattr(self, "_hand_runtime", None)
        if hand_runtime is not None:
            result["casia_hand"] = hand_runtime.get_data()
        return result

    def get_data_with_hook(self, prior_ctrl_data: dict, env_data: dict):
        # Control flow: env joints -> shared snapshot; latest GR00T command -> pipeline.
        del prior_ctrl_data
        if self.cfg_ctrl.observation_enabled:
            joint_positions = np.asarray(env_data["dof_pos"], dtype=np.float32)[self._joint_indices]
            hand_runtime = getattr(self, "_hand_runtime", None)
            if hand_runtime is not None:
                hand_data = hand_runtime.get_data()
                if not hand_data.get("joint_state_fresh", False):
                    return self.get_data()
                if tuple(hand_data.get("joint_names", ())) != self._hand_joint_names:
                    raise ValueError("CASIA hand joint names or order do not match the GR00T profile")
                hand_positions = np.asarray(hand_data.get("joint_positions"), dtype=np.float32)
                expected_shape = (len(self._hand_joint_names),)
                if hand_positions.shape != expected_shape:
                    raise ValueError(
                        f"CASIA hand joint positions have shape {hand_positions.shape}, "
                        f"expected {expected_shape}"
                    )
                joint_positions = np.concatenate((joint_positions, hand_positions))
            if not np.isfinite(joint_positions).all():
                raise FloatingPointError("GR00T observation joint positions contain non-finite values")
            with self._observation_snapshot_lock:
                self._observation_snapshot = (time.monotonic_ns(), joint_positions.copy())
        return self.get_data()

    def process_triggers(self, ctrl_data):
        return ctrl_data, []
