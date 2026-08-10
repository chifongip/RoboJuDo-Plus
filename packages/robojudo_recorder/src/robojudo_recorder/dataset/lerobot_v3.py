"""Small LeRobot v3 writer specialized for RoboJuDo recording.

The directory names, feature metadata, frame columns, and episode metadata follow
LeRobot v3.0 as implemented by Hugging Face LeRobot commit 95211b98. This module
is an independent implementation and does not import LeRobot at runtime.
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
}


def _write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=4, sort_keys=True) + "\n")
    temporary.replace(path)


@dataclass
class _VectorStats:
    count: int
    minimum: np.ndarray
    maximum: np.ndarray
    total: np.ndarray
    total_square: np.ndarray

    @classmethod
    def from_array(cls, values: np.ndarray) -> "_VectorStats":
        values = np.asarray(values, dtype=np.float64)
        return cls(
            count=len(values),
            minimum=values.min(axis=0),
            maximum=values.max(axis=0),
            total=values.sum(axis=0),
            total_square=np.square(values).sum(axis=0),
        )

    @classmethod
    def from_serialized(cls, value: dict) -> "_VectorStats":
        count = int(np.asarray(value["count"]).reshape(-1)[0])
        mean = np.asarray(value["mean"], dtype=np.float64)
        std = np.asarray(value["std"], dtype=np.float64)
        return cls(
            count=count,
            minimum=np.asarray(value["min"], dtype=np.float64),
            maximum=np.asarray(value["max"], dtype=np.float64),
            total=mean * count,
            total_square=(np.square(std) + np.square(mean)) * count,
        )

    def merge(self, other: "_VectorStats"):
        self.count += other.count
        self.minimum = np.minimum(self.minimum, other.minimum)
        self.maximum = np.maximum(self.maximum, other.maximum)
        self.total += other.total
        self.total_square += other.total_square

    def serialize(self) -> dict[str, Any]:
        mean = self.total / self.count
        variance = np.maximum(0.0, self.total_square / self.count - np.square(mean))
        return {
            "min": self.minimum.astype(np.float32).tolist(),
            "max": self.maximum.astype(np.float32).tolist(),
            "mean": mean.astype(np.float32).tolist(),
            "std": np.sqrt(variance).astype(np.float32).tolist(),
            "count": [self.count],
        }


class _EpisodeVideoWriter:
    def __init__(self, path: Path, fps: int, shape: tuple[int, int, int], codec: str):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.container = av.open(str(path), mode="w")
        try:
            self.stream = self.container.add_stream(codec, rate=fps)
        except Exception as exc:
            self.container.close()
            raise RuntimeError(f"PyAV/FFmpeg does not provide the requested encoder {codec!r}") from exc
        height, width, channels = shape
        if channels != 3:
            raise ValueError("RGB video shape must end in 3 channels")
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = "yuv420p"
        self.frame_count = 0

    def add(self, image: np.ndarray):
        frame = av.VideoFrame.from_ndarray(np.asarray(image, dtype=np.uint8), format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.frame_count += 1

    def close(self):
        if self.container is None:
            return
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
        self.container = None


class LeRobotV3Writer:
    """Write fixed-schema observation/action episodes in LeRobot v3 layout."""

    def __init__(
        self,
        *,
        root: str | Path,
        repo_id: str,
        robot_type: str,
        fps: int,
        state_names: list[str],
        action_names: list[str],
        camera_shapes: dict[str, tuple[int, int, int]] | None = None,
        camera_name: str | None = None,
        camera_shape: tuple[int, int, int] | None = None,
        codec: str = "libx264",
        resume: bool = False,
    ):
        self.root = Path(root)
        has_existing_data = self.root.exists() and any(self.root.iterdir())
        if has_existing_data and not resume:
            raise FileExistsError(f"dataset root is not empty: {self.root}; set resume=true to append")
        self.root.mkdir(parents=True, exist_ok=True)
        self.repo_id = repo_id
        self.robot_type = robot_type
        self.fps = fps
        self.state_names = list(state_names)
        self.action_names = list(action_names)
        if camera_shapes is not None and (camera_name is not None or camera_shape is not None):
            raise ValueError("provide camera_shapes or camera_name/camera_shape, not both")
        if camera_shapes is None:
            if camera_name is None or camera_shape is None:
                raise ValueError("at least one camera shape must be provided")
            camera_shapes = {camera_name: camera_shape}
        if not camera_shapes:
            raise ValueError("at least one camera shape must be provided")
        if any(not name for name in camera_shapes):
            raise ValueError("camera names must not be empty")
        self.camera_shapes = {name: tuple(shape) for name, shape in camera_shapes.items()}
        self.camera_keys = {name: f"observation.images.{name}" for name in self.camera_shapes}
        self.codec = codec
        self._episode_index = 0
        self._total_frames = 0
        self._episode_rows: list[dict] = []
        self._tasks: list[str] = []
        self._global_stats: dict[str, _VectorStats] = {}
        self._state_frames: list[np.ndarray] = []
        self._action_frames: list[np.ndarray] = []
        self._task: str | None = None
        self._videos: dict[str, _EpisodeVideoWriter] = {}
        if has_existing_data:
            self._load_existing()
        else:
            self._write_info()

    @property
    def has_pending_frames(self) -> bool:
        return bool(self._state_frames)

    @property
    def episode_open(self) -> bool:
        return bool(self._videos)

    def _features(self) -> dict:
        camera_features = {}
        for name, shape in self.camera_shapes.items():
            video_info = {
                "is_depth_map": False,
                "video.height": shape[0],
                "video.width": shape[1],
                "video.fps": self.fps,
                "video.channels": 3,
                "video.codec": "h264" if "264" in self.codec else self.codec,
                "video.pix_fmt": "yuv420p",
                "has_audio": False,
            }
            camera_features[self.camera_keys[name]] = {
                "dtype": "video",
                "shape": list(shape),
                "names": ["height", "width", "channels"],
                "info": video_info,
            }
        return {
            "observation.state": {
                "dtype": "float32",
                "shape": [len(self.state_names)],
                "names": self.state_names,
            },
            **camera_features,
            "action": {
                "dtype": "float32",
                "shape": [len(self.action_names)],
                "names": self.action_names,
            },
            **DEFAULT_FEATURES,
        }

    def _write_info(self):
        info = {
            "codebase_version": "v3.0",
            "fps": self.fps,
            "features": self._features(),
            "total_episodes": self._episode_index,
            "total_frames": self._total_frames,
            "total_tasks": len(self._tasks),
            "chunks_size": 1000,
            "data_files_size_in_mb": 100,
            "video_files_size_in_mb": 200,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "robot_type": self.robot_type,
            "splits": {"train": f"0:{self._episode_index}"} if self._episode_index else {},
        }
        _write_json(self.root / "meta/info.json", info)

    def _load_existing(self):
        info = json.loads((self.root / "meta/info.json").read_text())
        state_feature = info["features"]["observation.state"]
        action_feature = info["features"]["action"]
        existing_camera_features = {
            key: feature for key, feature in info["features"].items() if key.startswith("observation.images.")
        }
        expected = {
            "codebase_version": (info["codebase_version"], "v3.0"),
            "fps": (int(info["fps"]), self.fps),
            "robot_type": (info.get("robot_type"), self.robot_type),
            "state_names": (state_feature["names"], self.state_names),
            "action_names": (action_feature["names"], self.action_names),
            "camera_shapes": (
                {key: tuple(feature["shape"]) for key, feature in existing_camera_features.items()},
                {self.camera_keys[name]: shape for name, shape in self.camera_shapes.items()},
            ),
        }
        mismatches = {key: values for key, values in expected.items() if values[0] != values[1]}
        if mismatches:
            raise ValueError(f"cannot resume dataset with a different schema: {mismatches}")

        self._episode_index = int(info["total_episodes"])
        self._total_frames = int(info["total_frames"])
        tasks_path = self.root / "meta/tasks.parquet"
        if tasks_path.exists():
            tasks = pd.read_parquet(tasks_path)
            self._tasks = list(tasks.sort_values("task_index").index)
        episodes_path = self.root / "meta/episodes/chunk-000/file-000.parquet"
        if episodes_path.exists():
            self._episode_rows = pd.read_parquet(episodes_path).to_dict("records")
        stats_path = self.root / "meta/stats.json"
        if stats_path.exists():
            stats = json.loads(stats_path.read_text())
            self._global_stats = {key: _VectorStats.from_serialized(value) for key, value in stats.items()}

    def _episode_location(self) -> tuple[int, int]:
        return divmod(self._episode_index, 1000)

    def start_episode(self, task: str):
        if self.has_pending_frames:
            raise RuntimeError("cannot start a new episode while frames are pending")
        self._task = task
        chunk_index, file_index = self._episode_location()
        try:
            for name, shape in self.camera_shapes.items():
                video_path = (
                    self.root
                    / "videos"
                    / self.camera_keys[name]
                    / f"chunk-{chunk_index:03d}"
                    / f"file-{file_index:03d}.mp4"
                )
                self._videos[name] = _EpisodeVideoWriter(video_path, self.fps, shape, self.codec)
        except Exception:
            self.discard_episode()
            raise

    def add_frame(
        self,
        state: np.ndarray,
        action: np.ndarray,
        images: dict[str, np.ndarray] | np.ndarray | None = None,
        *,
        image: np.ndarray | None = None,
    ):
        if not self._videos or self._task is None:
            raise RuntimeError("start_episode() must be called before add_frame()")
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        if images is not None and image is not None:
            raise ValueError("provide images or image, not both")
        if image is not None:
            images = image
        if images is None:
            raise ValueError("camera images are required")
        if isinstance(images, np.ndarray):
            if len(self.camera_shapes) != 1:
                raise ValueError("a camera image mapping is required for a multi-camera dataset")
            images = {next(iter(self.camera_shapes)): images}
        if set(images) != set(self.camera_shapes):
            raise ValueError(f"image cameras {sorted(images)} do not match {sorted(self.camera_shapes)}")
        normalized_images = {name: np.asarray(image, dtype=np.uint8) for name, image in images.items()}
        if state.shape != (len(self.state_names),):
            raise ValueError(f"state shape {state.shape} does not match {(len(self.state_names),)}")
        if action.shape != (len(self.action_names),):
            raise ValueError(f"action shape {action.shape} does not match {(len(self.action_names),)}")
        for name, image in normalized_images.items():
            if image.shape != self.camera_shapes[name]:
                raise ValueError(f"camera {name!r} image shape {image.shape} does not match {self.camera_shapes[name]}")
        self._state_frames.append(state.copy())
        self._action_frames.append(action.copy())
        for name, image in normalized_images.items():
            self._videos[name].add(image)

    def _write_data(self, states: np.ndarray, actions: np.ndarray, task_index: int):
        length = len(states)
        chunk_index, file_index = self._episode_location()
        frame_indices = np.arange(length, dtype=np.int64)
        table = pa.table(
            {
                "observation.state": pa.array(states.tolist(), type=pa.list_(pa.float32(), len(self.state_names))),
                "action": pa.array(actions.tolist(), type=pa.list_(pa.float32(), len(self.action_names))),
                "timestamp": pa.array(frame_indices.astype(np.float32) / self.fps, type=pa.float32()),
                "frame_index": pa.array(frame_indices, type=pa.int64()),
                "episode_index": pa.array(np.full(length, self._episode_index), type=pa.int64()),
                "index": pa.array(np.arange(self._total_frames, self._total_frames + length), type=pa.int64()),
                "task_index": pa.array(np.full(length, task_index), type=pa.int64()),
            }
        )
        path = self.root / "data" / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path, compression="snappy", use_dictionary=True)

    @staticmethod
    def _episode_stats(states: np.ndarray, actions: np.ndarray) -> dict[str, _VectorStats]:
        return {
            "observation.state": _VectorStats.from_array(states),
            "action": _VectorStats.from_array(actions),
        }

    def _update_stats(self, episode_stats: dict[str, _VectorStats]):
        for key, stats in episode_stats.items():
            if key in self._global_stats:
                self._global_stats[key].merge(stats)
            else:
                self._global_stats[key] = stats
        _write_json(
            self.root / "meta/stats.json",
            {key: value.serialize() for key, value in self._global_stats.items()},
        )

    def _write_tasks(self):
        frame = pd.DataFrame(
            {"task_index": np.arange(len(self._tasks), dtype=np.int64)},
            index=pd.Index(self._tasks, name="task"),
        )
        path = self.root / "meta/tasks.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)

    def _write_episodes(self):
        frame = pd.DataFrame(self._episode_rows)
        path = self.root / "meta/episodes/chunk-000/file-000.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)

    def save_episode(self):
        if not self.has_pending_frames:
            raise ValueError("cannot save an empty episode")
        for video in self._videos.values():
            video.close()
        states = np.stack(self._state_frames).astype(np.float32)
        actions = np.stack(self._action_frames).astype(np.float32)
        task = self._task
        if task not in self._tasks:
            self._tasks.append(task)
        task_index = self._tasks.index(task)
        self._write_data(states, actions, task_index)
        episode_stats = self._episode_stats(states, actions)
        chunk_index, file_index = self._episode_location()
        length = len(states)
        row = {
            "episode_index": self._episode_index,
            "tasks": [task],
            "length": length,
            "data/chunk_index": chunk_index,
            "data/file_index": file_index,
            "dataset_from_index": self._total_frames,
            "dataset_to_index": self._total_frames + length,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        for camera_key in self.camera_keys.values():
            row[f"videos/{camera_key}/chunk_index"] = chunk_index
            row[f"videos/{camera_key}/file_index"] = file_index
            row[f"videos/{camera_key}/from_timestamp"] = 0.0
            row[f"videos/{camera_key}/to_timestamp"] = length / self.fps
        for feature_name, stats in episode_stats.items():
            for stat_name, value in stats.serialize().items():
                row[f"stats/{feature_name}/{stat_name}"] = value
        self._episode_rows.append(row)
        self._write_tasks()
        self._write_episodes()
        self._update_stats(episode_stats)
        self._total_frames += length
        self._episode_index += 1
        self._write_info()
        self._clear_buffer()

    def discard_episode(self):
        for video in self._videos.values():
            video.close()
            if video.path.exists():
                video.path.unlink()
        self._clear_buffer()

    def _clear_buffer(self):
        self._state_frames.clear()
        self._action_frames.clear()
        self._task = None
        self._videos = {}

    def finalize(self):
        if self.has_pending_frames:
            self.save_episode()
        elif self._videos:
            self.discard_episode()

    def abort(self):
        self.discard_episode()
        if self._episode_index == 0:
            shutil.rmtree(self.root, ignore_errors=True)
