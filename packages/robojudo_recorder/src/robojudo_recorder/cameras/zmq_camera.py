import json
import time

import msgpack
import numpy as np
import zmq

from robojudo_recorder.config import CameraConfig

from . import register_camera
from .base import CameraFrame, CameraSource


@register_camera("zmq")
class ZmqCameraSource(CameraSource):
    """Receive multipart ``[header, image bytes]`` camera frames."""

    def __init__(self, cfg: CameraConfig):
        self.endpoint = str(cfg.options["endpoint"])
        # Offline DAgger reuses GR00T's multi-camera observation multipart;
        # image_key selects one JPEG so multiple recorder sources can fan out it.
        self.image_key = str(cfg.options.get("image_key", cfg.name))
        self.encoding = str(cfg.options.get("encoding", "auto"))
        if self.encoding not in {"auto", "raw_rgb", "jpeg"}:
            raise ValueError("ZMQ camera encoding must be 'auto', 'raw_rgb', or 'jpeg'")
        self.timestamp_mode = str(cfg.options.get("timestamp_mode", "receive"))
        if self.timestamp_mode not in {"receive", "source"}:
            raise ValueError("ZMQ camera timestamp_mode must be 'receive' or 'source'")
        width = cfg.options.get("width")
        height = cfg.options.get("height")
        if (width is None) != (height is None):
            raise ValueError("ZMQ camera width and height must be specified together")
        self._shape = None if width is None else (int(height), int(width), 3)
        self._context = zmq.Context.instance()
        self._socket = None

    @property
    def shape(self) -> tuple[int, int, int]:
        if self._shape is None:
            raise RuntimeError("ZMQ camera shape is unavailable until the first frame arrives")
        return self._shape

    def connect(self):
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVHWM, 2)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.connect(self.endpoint)

    def read(self, timeout_ms: int) -> CameraFrame | None:
        if self._socket.poll(timeout_ms, zmq.POLLIN) == 0:
            return None
        parts = self._socket.recv_multipart()
        if len(parts) < 2:
            raise ValueError("ZMQ camera message must contain a header and image payload")
        header_bytes = parts[0]
        header = self._decode_header(header_bytes)
        image_keys = tuple(header.get("image_keys", ()))
        if image_keys:
            # Offline DAgger observation protocol v2 carries one payload per
            # image_key after the shared msgpack header.
            if len(parts) != len(image_keys) + 1:
                raise ValueError("ZMQ multi-camera payload count does not match image_keys")
            if self.image_key not in image_keys:
                raise ValueError(f"ZMQ observation does not contain image_key {self.image_key!r}")
            image_index = image_keys.index(self.image_key)
            payload = parts[image_index + 1]
            frame_shape = tuple(header.get("image_shapes", {}).get(self.image_key, ()))
            sequence = int(header.get("image_sequences", {}).get(self.image_key, header["sequence"]))
            source_timestamp_ns = int(
                header.get("image_source_timestamps_ns", {}).get(
                    self.image_key,
                    header.get("image_timestamps_ns", {}).get(
                        self.image_key, header.get("timestamp_ns", time.monotonic_ns())
                    ),
                )
            )
        else:
            if len(parts) != 2:
                raise ValueError("legacy ZMQ camera message must contain exactly one image")
            payload = parts[1]
            frame_shape = tuple(header.get("shape", ()))
            sequence = int(header["sequence"])
            source_timestamp_ns = int(
                header.get("source_timestamp_ns", header.get("timestamp_ns", time.monotonic_ns()))
            )
        if frame_shape:
            if len(frame_shape) != 3 or frame_shape[2] != 3:
                raise ValueError(f"invalid ZMQ camera shape {frame_shape}")
            if self._shape is None:
                self._shape = frame_shape
            elif frame_shape != self._shape:
                raise ValueError(f"ZMQ camera returned shape {frame_shape}, expected {self._shape}")

        encoding = str(header.get("encoding", self.encoding)) if self.encoding == "auto" else self.encoding
        if encoding == "raw_rgb":
            if self._shape is None:
                raise ValueError("raw_rgb ZMQ camera requires a shape (config width/height or header.shape)")
            image = np.frombuffer(payload, dtype=np.uint8).reshape(self._shape).copy()
        elif encoding == "jpeg":
            image = None
            if self._shape is None:
                try:
                    import cv2
                except ImportError as exc:
                    raise RuntimeError("JPEG ZMQ camera requires shape metadata or robojudo-recorder[opencv]") from exc
                bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError("failed to decode JPEG ZMQ camera frame")
                image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"unsupported ZMQ camera encoding {encoding!r}")
        if self._shape is None and image is not None:
            self._shape = tuple(image.shape)
        elif image is not None and image.shape != self.shape:
            raise ValueError(f"ZMQ camera returned shape {image.shape}, expected {self.shape}")
        receive_timestamp_ns = time.monotonic_ns()
        timestamp_ns = source_timestamp_ns if self.timestamp_mode == "source" else receive_timestamp_ns
        return CameraFrame(
            image=image,
            timestamp_ns=timestamp_ns,
            sequence=sequence,
            encoded_image=payload if encoding == "jpeg" else None,
            encoding=encoding,
            source_timestamp_ns=source_timestamp_ns,
            receive_timestamp_ns=receive_timestamp_ns,
            image_shape=self._shape,
        )

    @staticmethod
    def _decode_header(header_bytes: bytes) -> dict:
        try:
            header = msgpack.unpackb(header_bytes, raw=False)
            if isinstance(header, dict):
                return header
        except (msgpack.UnpackException, ValueError):
            pass
        header = json.loads(header_bytes)
        if not isinstance(header, dict):
            raise ValueError("ZMQ camera header must be an object")
        return header

    def close(self):
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
