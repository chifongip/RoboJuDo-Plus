import threading
import time
from abc import abstractmethod

import numpy as np

from .base import CameraFrame, CameraSource


class ThreadedCameraSource(CameraSource):
    def __init__(self, shape: tuple[int, int, int]):
        self._shape = shape
        self._condition = threading.Condition()
        self._latest: CameraFrame | None = None
        self._sequence = 0
        self._stopping = False
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    def connect(self) -> None:
        self._open()
        self._stopping = False
        self._thread = threading.Thread(target=self._capture_loop, name=type(self).__name__, daemon=True)
        self._thread.start()

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
                    self._latest = CameraFrame(image.copy(), time.monotonic_ns(), self._sequence)
                    self._condition.notify_all()
        except Exception as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()

    def read(self, timeout_ms: int) -> CameraFrame | None:
        with self._condition:
            previous_sequence = self._latest.sequence if self._latest is not None else 0
            self._condition.wait_for(
                lambda: self._error is not None
                or (self._latest is not None and self._latest.sequence > previous_sequence),
                timeout=timeout_ms / 1000,
            )
            if self._error is not None:
                raise RuntimeError("camera capture thread failed") from self._error
            return self._latest

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
