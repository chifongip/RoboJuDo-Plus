import logging
import math
import threading
import time
import uuid
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


@ctrl_registry.register
class Gr00tZmqCtrl(ControllerHook):
    """Receive atomic GR00T commands and publish camera/joint observations.

    Threading and data flow:

    - Control thread (pipeline rate): ``get_data_with_hook`` snapshots measured
      upper-body joints from ``env_data`` and non-blockingly returns the latest
      GR00T command to the pipeline.
    - Observation worker: reads camera frames, combines each frame with the
      latest thread-safe joint snapshot and task, then publishes to deploy.

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
            self._observation_thread.join(timeout=3.0)
            if self._observation_thread.is_alive():
                logger.warning("GR00T observation worker did not stop within 3 seconds")
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

    def set_takeover_enabled(self, enabled: bool) -> bool:
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
            hand_runtime.set_takeover_enabled(enabled)
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

    def _observation_loop(self):
        # Worker flow: camera -> latest joint snapshot + task -> observation PUB.
        camera = None
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

            camera_cfg = CameraConfig(
                type=self.cfg_ctrl.camera.type,
                name=self.cfg_ctrl.camera.name,
                options=dict(self.cfg_ctrl.camera.options),
            )
            camera = create_camera(camera_cfg)
            camera.connect()
            publisher = self._context.socket(zmq.PUB)
            publisher.setsockopt(zmq.LINGER, 0)
            publisher.setsockopt(zmq.SNDHWM, 2)
            publisher.bind(self.cfg_ctrl.observation_endpoint)
            self._observation_ready.set()
            logger.info(
                "GR00T observations publishing to %s from %s camera",
                self.cfg_ctrl.observation_endpoint,
                self.cfg_ctrl.camera.type,
            )

            minimum_period_ns = int(1_000_000_000 / self.cfg_ctrl.observation_fps)
            # Accept normal camera/scheduler jitter at the configured rate.
            minimum_interval_ns = minimum_period_ns * 9 // 10
            last_published_at = 0
            last_camera_sequence = -1
            observation_sequence = 0
            while not self._observation_stop.is_set():
                frame = camera.read(self.cfg_ctrl.camera_poll_timeout_ms)
                if frame is None:
                    continue
                if frame.sequence == last_camera_sequence:
                    continue
                last_camera_sequence = frame.sequence
                now_ns = time.monotonic_ns()
                if now_ns - last_published_at < minimum_interval_ns:
                    continue
                with self._observation_snapshot_lock:
                    snapshot = self._observation_snapshot
                    takeover_enabled = self._takeover_enabled
                    control_session = self._control_session
                if snapshot is None:
                    continue

                joint_timestamp_ns, joint_positions = snapshot
                joint_timeout_ns = int(self.cfg_ctrl.observation_joint_timeout_s * 1_000_000_000)
                if now_ns - joint_timestamp_ns > joint_timeout_ns:
                    continue
                image_shape, jpeg_payload = self._prepare_observation_jpeg(
                    frame,
                    cv2,
                    self.cfg_ctrl.observation_jpeg_quality,
                )

                observation_sequence += 1
                header = {
                    "protocol_version": 1,
                    "stream_id": self._observation_stream_id,
                    "control_session": control_session,
                    "takeover_enabled": takeover_enabled,
                    "sequence": observation_sequence,
                    "camera_sequence": int(frame.sequence),
                    "timestamp_ns": int(frame.timestamp_ns),
                    "joint_timestamp_ns": joint_timestamp_ns,
                    "robot_type": self.cfg_ctrl.observation_profile.split("_", 1)[0],
                    "profile": self.cfg_ctrl.observation_profile,
                    "task": self.cfg_ctrl.observation_task,
                    "camera_name": self.cfg_ctrl.camera.name,
                    "encoding": "jpeg",
                    "shape": list(image_shape),
                    "joint_names": list(self._policy_joint_names),
                    "joint_positions": joint_positions.tolist(),
                }
                try:
                    publisher.send_multipart(
                        [msgpack.packb(header, use_bin_type=True), jpeg_payload],
                        flags=zmq.NOBLOCK,
                    )
                    self._published_observations += 1
                    last_published_at = now_ns
                except zmq.Again:
                    self._dropped_observations += 1
        except Exception as exc:
            self._observation_error = exc
            logger.exception("GR00T observation publisher stopped: %s", exc)
            self._observation_ready.set()
        finally:
            if camera is not None:
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
