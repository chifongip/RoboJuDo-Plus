import json
import time

import numpy as np
import zmq

from robojudo_recorder.config import CameraConfig

from . import register_camera
from .base import CameraFrame, CameraSource


@register_camera("zmq")
class ZmqCameraSource(CameraSource):
    """Receive multipart ``[json header, image bytes]`` camera frames."""

    def __init__(self, cfg: CameraConfig):
        self.endpoint = str(cfg.options["endpoint"])
        self.encoding = str(cfg.options.get("encoding", "raw_rgb"))
        self.timestamp_mode = str(cfg.options.get("timestamp_mode", "receive"))
        if self.timestamp_mode not in {"receive", "source"}:
            raise ValueError("ZMQ camera timestamp_mode must be 'receive' or 'source'")
        self.height = int(cfg.options["height"])
        self.width = int(cfg.options["width"])
        self._context = zmq.Context.instance()
        self._socket = None

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.height, self.width, 3

    def connect(self):
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVHWM, 2)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.connect(self.endpoint)

    def read(self, timeout_ms: int) -> CameraFrame | None:
        if self._socket.poll(timeout_ms, zmq.POLLIN) == 0:
            return None
        header_bytes, payload = self._socket.recv_multipart()
        header = json.loads(header_bytes)
        if self.encoding == "raw_rgb":
            image = np.frombuffer(payload, dtype=np.uint8).reshape(self.shape).copy()
        elif self.encoding == "jpeg":
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError("JPEG ZMQ camera requires robojudo-recorder[opencv]") from exc
            bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"unsupported ZMQ camera encoding {self.encoding!r}")
        if image.shape != self.shape:
            raise ValueError(f"ZMQ camera returned shape {image.shape}, expected {self.shape}")
        timestamp_ns = (
            int(header["timestamp_ns"])
            if self.timestamp_mode == "source"
            else time.monotonic_ns()
        )
        return CameraFrame(
            image=image,
            timestamp_ns=timestamp_ns,
            sequence=int(header["sequence"]),
        )

    def close(self):
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
