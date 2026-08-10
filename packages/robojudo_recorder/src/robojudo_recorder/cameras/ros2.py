import json
import subprocess
from pathlib import Path

import numpy as np
import zmq

from robojudo_recorder.config import CameraConfig

from . import register_camera
from .base import CameraFrame, CameraSource


@register_camera("ros2")
class Ros2CompressedCameraSource(CameraSource):
    """Receive ROS 2 ``CompressedImage`` frames through a native ROS Python helper."""

    def __init__(self, cfg: CameraConfig):
        self.topic = str(cfg.options.get("topic", "")).strip()
        if not self.topic:
            raise ValueError("ROS 2 camera requires a non-empty topic")

        self.node_name = str(cfg.options.get("node_name", "robojudo_recorder_camera")).strip()
        if not self.node_name:
            raise ValueError("ROS 2 camera node_name must not be empty")
        self.qos_depth = int(cfg.options.get("qos_depth", 1))
        if self.qos_depth <= 0:
            raise ValueError("ROS 2 camera qos_depth must be positive")
        self.qos_reliability = str(cfg.options.get("qos_reliability", "best_effort")).lower()
        if self.qos_reliability not in {"best_effort", "reliable"}:
            raise ValueError("ROS 2 camera qos_reliability must be 'best_effort' or 'reliable'")
        self.ros_python_executable = str(cfg.options.get("ros_python_executable", "/usr/bin/python3"))

        width = cfg.options.get("width")
        height = cfg.options.get("height")
        if (width is None) != (height is None):
            raise ValueError("ROS 2 camera width and height must be specified together")
        self._shape = None if width is None else (int(height), int(width), 3)
        if self._shape is not None and (self._shape[0] <= 0 or self._shape[1] <= 0):
            raise ValueError("ROS 2 camera width and height must be positive")

        self._context = zmq.Context.instance()
        self._socket = None
        self._process: subprocess.Popen | None = None
        self._cv2 = None

    @property
    def shape(self) -> tuple[int, int, int]:
        if self._shape is None:
            raise RuntimeError("ROS 2 camera shape is unavailable until the first frame arrives")
        return self._shape

    def connect(self) -> None:
        if self._socket is not None:
            raise RuntimeError("ROS 2 camera is already connected")
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("ROS 2 CompressedImage decoding requires robojudo-recorder[ros2]") from exc

        socket = self._context.socket(zmq.PULL)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVHWM, 2)
        socket.bind("tcp://127.0.0.1:*")
        endpoint = socket.getsockopt_string(zmq.LAST_ENDPOINT)
        helper = Path(__file__).with_name("ros2_bridge.py")
        command = [
            self.ros_python_executable,
            str(helper),
            "--endpoint",
            endpoint,
            "--topic",
            self.topic,
            "--node-name",
            self.node_name,
            "--qos-reliability",
            self.qos_reliability,
            "--qos-depth",
            str(self.qos_depth),
        ]
        try:
            process = subprocess.Popen(command)
        except Exception:
            socket.close(linger=0)
            raise
        self._cv2 = cv2
        self._socket = socket
        self._process = process

    def _decode_frame(self, header_bytes: bytes, payload: bytes) -> CameraFrame:
        header = json.loads(header_bytes)
        encoded = np.frombuffer(payload, dtype=np.uint8)
        bgr = self._cv2.imdecode(encoded, self._cv2.IMREAD_UNCHANGED)
        if bgr is None:
            raise ValueError(f"failed to decode CompressedImage from {self.topic}")
        if bgr.ndim == 2:
            image = self._cv2.cvtColor(bgr, self._cv2.COLOR_GRAY2RGB)
        elif bgr.shape[2] == 4:
            image = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGRA2RGB)
        elif bgr.shape[2] == 3:
            image = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"unsupported CompressedImage shape {bgr.shape}")

        frame_shape = tuple(image.shape)
        if self._shape is None:
            self._shape = frame_shape
        elif frame_shape != self._shape:
            raise ValueError(f"ROS 2 camera returned shape {frame_shape}, expected {self._shape}")
        return CameraFrame(
            image=np.asarray(image, dtype=np.uint8),
            timestamp_ns=int(header["timestamp_ns"]),
            sequence=int(header["sequence"]),
        )

    def read(self, timeout_ms: int) -> CameraFrame | None:
        if self._socket is None or self._process is None:
            raise RuntimeError("ROS 2 camera is not connected")
        if self._socket.poll(max(timeout_ms, 0), zmq.POLLIN) == 0:
            return_code = self._process.poll()
            if return_code is not None:
                raise RuntimeError(f"ROS 2 camera helper exited with status {return_code}")
            return None
        header, payload = self._socket.recv_multipart()
        return self._decode_frame(header, payload)

    def close(self) -> None:
        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=2)
            self._process = None
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        self._cv2 = None
