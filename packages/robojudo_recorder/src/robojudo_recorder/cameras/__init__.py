from collections.abc import Callable

from robojudo_recorder.config import CameraConfig

from .base import CameraFrame, CameraSource

CameraFactory = Callable[[CameraConfig], CameraSource]
_REGISTRY: dict[str, CameraFactory] = {}


def register_camera(name: str):
    def decorator(factory: CameraFactory):
        _REGISTRY[name] = factory
        return factory

    return decorator


def create_camera(cfg: CameraConfig) -> CameraSource:
    if cfg.type not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"unknown camera type {cfg.type!r}; available: {available}")
    return _REGISTRY[cfg.type](cfg)


from . import opencv, realsense, ros2, zmq_camera  # noqa: E402, F401

__all__ = ["CameraFrame", "CameraSource", "create_camera", "register_camera"]
