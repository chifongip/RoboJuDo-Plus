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

    def __post_init__(self):
        if self.clock not in {"source", "receive"}:
            raise ValueError("sync.clock must be 'source' or 'receive'")
        if self.max_control_age_ms <= 0:
            raise ValueError("sync.max_control_age_ms must be positive")


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
    camera: CameraConfig
    sync: SyncConfig = SyncConfig()


def load_config(path: str | Path) -> RecorderConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text())
    dataset_raw = dict(raw["dataset"])
    dataset_raw["root"] = Path(dataset_raw["root"]).expanduser()
    camera_raw = dict(raw["camera"])
    camera = CameraConfig(
        type=camera_raw.pop("type"),
        name=camera_raw.pop("name", "head_rgb"),
        options=camera_raw,
    )
    dataset = DatasetConfig(**dataset_raw)
    camera_fps = camera.options.get("fps")
    if camera_fps is not None and int(camera_fps) != dataset.fps:
        raise ValueError(f"camera fps ({camera_fps}) must equal dataset fps ({dataset.fps})")
    return RecorderConfig(
        control_endpoint=raw.get("control_endpoint", "tcp://127.0.0.1:8560"),
        dataset=dataset,
        camera=camera,
        sync=SyncConfig(**raw.get("sync", {})),
    )
