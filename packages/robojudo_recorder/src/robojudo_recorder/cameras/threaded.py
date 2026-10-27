import threading
import time
from abc import abstractmethod
from collections import deque

import numpy as np

from .base import CameraFrame, CameraSource


class ThreadedCameraSource(CameraSource):
    def __init__(self, shape: tuple[int, int, int]):
        self._shape = shape
        self._condition = threading.Condition()
        self._pending_capacity = 32
        self._pending: deque[CameraFrame] = deque()
        self._sequence = 0
        self._stopping = False
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    def connect(self) -> None:
        self._open()
        with self._condition:
            self._pending.clear()
            self._stopping = False
            self._error = None
        self._thread = threading.Thread(target=self._capture_loop, name=type(self).__name__, daemon=True)
        self._thread.start()

    def set_pending_capacity(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("camera pending capacity must be positive")
        with self._condition:
            self._pending_capacity = capacity
            while len(self._pending) > capacity:
                self._pending.popleft()

    def _capture_loop(self):
        try:
            while not self._stopping:
                image = self._capture()
                if image is None:
                    continue
                image = np.asarray(image, dtype=np.uint8)
                if image.shape != self.shape:
                    raise RuntimeError(f"camera returned shape {image.shape}, expected {self.shape}")
                with self._condition:
                    self._sequence += 1
                    if len(self._pending) >= self._pending_capacity:
                        self._pending.popleft()
                    self._pending.append(CameraFrame(image.copy(), time.monotonic_ns(), self._sequence))
                    self._condition.notify_all()
        except Exception as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()

    def read(self, timeout_ms: int) -> CameraFrame | None:
        with self._condition:
            self._condition.wait_for(
                lambda: self._error is not None or bool(self._pending),
                timeout=timeout_ms / 1000,
            )
            if self._error is not None:
                raise RuntimeError("camera capture thread failed") from self._error
            return self._pending.popleft() if self._pending else None

    def read_batch(self, max_frames: int) -> list[CameraFrame]:
        with self._condition:
            if self._error is not None:
                raise RuntimeError("camera capture thread failed") from self._error
            frames = []
            for _ in range(min(max_frames, len(self._pending))):
                frames.append(self._pending.popleft())
            return frames

    def clear_pending(self) -> None:
        with self._condition:
            self._pending.clear()

    def close(self) -> None:
        self._stopping = True
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        self._close()

    @abstractmethod
    def _open(self) -> None: ...

    @abstractmethod
    def _capture(self) -> np.ndarray | None: ...

    @abstractmethod
    def _close(self) -> None: ...
