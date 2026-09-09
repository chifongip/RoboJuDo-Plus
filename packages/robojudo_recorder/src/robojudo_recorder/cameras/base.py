from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraFrame:
    image: np.ndarray | None
    timestamp_ns: int
    sequence: int
    encoded_image: bytes | None = None
    encoding: str | None = None
    source_timestamp_ns: int | None = None
    receive_timestamp_ns: int | None = None
    image_shape: tuple[int, int, int] | None = None

    def __post_init__(self):
        if self.source_timestamp_ns is None:
            object.__setattr__(self, "source_timestamp_ns", self.timestamp_ns)
        if self.receive_timestamp_ns is None:
            object.__setattr__(self, "receive_timestamp_ns", self.timestamp_ns)

    @property
    def shape(self) -> tuple[int, int, int]:
        if self.image is not None:
            return tuple(self.image.shape)
        if self.image_shape is not None:
            return self.image_shape
        raise RuntimeError("encoded camera frame shape must be supplied by its camera source")


class CameraSource(ABC):
    @property
    @abstractmethod
    def shape(self) -> tuple[int, int, int]: ...

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def read(self, timeout_ms: int) -> CameraFrame | None: ...

    def read_batch(self, max_frames: int) -> list[CameraFrame]:
        """Return up to ``max_frames`` immediately, preserving source order."""
        if max_frames <= 0:
            return []
        frame = self.read(timeout_ms=0)
        return [] if frame is None else [frame]

    def set_pending_capacity(self, capacity: int) -> None:
        """Configure source buffering when the backend supports it."""
        del capacity
        return None

    def clear_pending(self) -> None:
        """Discard frames captured before a new episode is armed."""
        return None

    @abstractmethod
    def close(self) -> None: ...
