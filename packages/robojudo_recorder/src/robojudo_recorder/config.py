from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CameraConfig:
    type: str
    name: str = "head_rgb"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SyncConfig:
    clock: str = "receive"
    max_control_age_ms: float = 50.0
    poll_timeout_ms: int = 10
    pending_frame_capacity: int = 32
    throughput_log_interval_s: float = 5.0

    def __post_init__(self):
        if self.clock not in {"source", "receive"}:
            raise ValueError("sync.clock must be 'source' or 'receive'")
        if self.max_control_age_ms <= 0:
            raise ValueError("sync.max_control_age_ms must be positive")
        if self.pending_frame_capacity <= 0:
            raise ValueError("sync.pending_frame_capacity must be positive")
        if self.throughput_log_interval_s <= 0:
            raise ValueError("sync.throughput_log_interval_s must be positive")


@dataclass(frozen=True)
class DatasetConfig:
    root: Path
    repo_id: str
    fps: int
    codec: str = "libx264"
    resume: bool = False

    def __post_init__(self):
        if self.fps <= 0:
            raise ValueError("dataset.fps must be positive")
        if not self.repo_id:
            raise ValueError("dataset.repo_id must not be empty")


@dataclass(frozen=True)
class RecorderConfig:
    control_endpoint: str
    dataset: DatasetConfig
    camera: CameraConfig | None = None
    sync: SyncConfig = SyncConfig()
    cameras: tuple[CameraConfig, ...] = ()

    def __post_init__(self):
        # ``camera`` remains an alias for the first entry so existing Python callers keep working.
        if self.camera is not None and self.cameras and self.camera != self.cameras[0]:
            raise ValueError("configure either camera or cameras, not both")
        cameras = self.cameras or ((self.camera,) if self.camera is not None else ())
        if not cameras:
            raise ValueError("at least one camera must be configured")
        names = [camera.name for camera in cameras]
        if any(not name for name in names):
            raise ValueError("camera names must not be empty")
        if len(set(names)) != len(names):
            raise ValueError(f"camera names must be unique: {names}")
        object.__setattr__(self, "cameras", tuple(cameras))
        object.__setattr__(self, "camera", cameras[0])


def _load_camera(raw: dict[str, Any]) -> CameraConfig:
    camera_raw = dict(raw)
    return CameraConfig(
        type=camera_raw.pop("type"),
        name=camera_raw.pop("name", "head_rgb"),
        options=camera_raw,
    )


def load_config(path: str | Path) -> RecorderConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text())
    dataset_raw = dict(raw["dataset"])
    dataset_raw["root"] = Path(dataset_raw["root"]).expanduser()
    has_camera = "camera" in raw
    has_cameras = "cameras" in raw
    if has_camera == has_cameras:
        raise ValueError("configure exactly one of camera or cameras")
    cameras = (
        (_load_camera(raw["camera"]),)
        if has_camera
        else tuple(_load_camera(camera_raw) for camera_raw in raw["cameras"])
    )
    dataset = DatasetConfig(**dataset_raw)
    for camera in cameras:
        camera_fps = camera.options.get("fps")
        if camera_fps is not None and int(camera_fps) != dataset.fps:
            raise ValueError(f"camera {camera.name!r} fps ({camera_fps}) must equal dataset fps ({dataset.fps})")
    return RecorderConfig(
        control_endpoint=raw.get("control_endpoint", "tcp://127.0.0.1:8560"),
        dataset=dataset,
        cameras=cameras,
        sync=SyncConfig(**raw.get("sync", {})),
    )
