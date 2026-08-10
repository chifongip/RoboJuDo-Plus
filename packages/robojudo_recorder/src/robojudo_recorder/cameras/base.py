from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraFrame:
    image: np.ndarray
    timestamp_ns: int
    sequence: int


class CameraSource(ABC):
    @property
    @abstractmethod
    def shape(self) -> tuple[int, int, int]: ...

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def read(self, timeout_ms: int) -> CameraFrame | None: ...

    @abstractmethod
    def close(self) -> None: ...
