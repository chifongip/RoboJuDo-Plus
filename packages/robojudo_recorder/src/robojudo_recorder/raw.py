"""Crash-tolerant raw episode storage for the recorder's real-time phase."""

import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .cameras import CameraFrame
from .protocol import ControlSample

RAW_FORMAT_VERSION = 1


def _append_json_line(handle, value: dict):
    handle.write(json.dumps(value, separators=(",", ":")) + "\n")
    handle.flush()


def _encode_jpeg(image: np.ndarray, quality: int) -> bytes:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "encoding raw camera images requires OpenCV; use a compressed camera topic or install "
            "robojudo-recorder[opencv]"
        ) from exc
    bgr = cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("failed to JPEG-encode camera frame")
    return encoded.tobytes()


@dataclass(frozen=True)
class RawFrameRecord:
    camera: str
    frame_index: int
    sequence: int
    source_timestamp_ns: int
    receive_timestamp_ns: int
    path: str
    encoding: str
    shape: tuple[int, int, int]


class RawEpisodeWriter:
    """Append camera and control events without performing video encoding or synchronization."""

    def __init__(
        self,
        *,
        raw_root: Path,
        episode_id: int,
        task: str,
        camera_names: tuple[str, ...],
        started_source_ns: int,
        started_receive_ns: int,
        jpeg_quality: int,
    ):
        self.raw_root = Path(raw_root)
        self.episode_id = episode_id
        self.task = task
        self.jpeg_quality = jpeg_quality
        capture_id = f"capture_{time.time_ns()}_episode_{episode_id:06d}"
        self.pending_path = self.raw_root / ".pending" / capture_id
        self.committed_path = self.raw_root / "episodes" / capture_id
        self.pending_path.mkdir(parents=True, exist_ok=False)
        self._control_handle = (self.pending_path / "controls.jsonl").open("a", encoding="utf-8")
        self._frame_handles = {}
        self._frame_counts = {name: 0 for name in camera_names}
        self._sequence_gaps = {name: 0 for name in camera_names}
        self._last_sequences: dict[str, int | None] = {name: None for name in camera_names}
        for name in camera_names:
            camera_dir = self.pending_path / "cameras" / name
            camera_dir.mkdir(parents=True)
            self._frame_handles[name] = (camera_dir / "frames.jsonl").open("a", encoding="utf-8")
        self._manifest = {
            "format_version": RAW_FORMAT_VERSION,
            "status": "recording",
            "episode_id": episode_id,
            "task": task,
            "started_source_timestamp_ns": started_source_ns,
            "started_receive_timestamp_ns": started_receive_ns,
            "camera_names": list(camera_names),
            "robot_type": None,
            "joint_names": None,
            "frame_counts": self._frame_counts,
            "sequence_gaps": self._sequence_gaps,
        }
        self._write_manifest()
        self._closed = False

    def _write_manifest(self):
        path = self.pending_path / "manifest.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self._manifest, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)

    def add_control(self, sample: ControlSample):
        if self._closed:
            raise RuntimeError("raw episode is closed")
        if self._manifest["robot_type"] is None:
            self._manifest["robot_type"] = sample.robot_type
            self._manifest["joint_names"] = list(sample.joint_names)
            self._write_manifest()
        elif self._manifest["robot_type"] != sample.robot_type or self._manifest["joint_names"] != list(
            sample.joint_names
        ):
            raise ValueError("control schema changed within a raw episode")
        _append_json_line(
            self._control_handle,
            {
                "source_timestamp_ns": sample.source_timestamp_ns,
                "receive_timestamp_ns": sample.receive_timestamp_ns,
                "joint_positions": sample.joint_positions.tolist(),
                "joint_position_commands": sample.joint_position_commands.tolist(),
                "velocity_height_command": sample.velocity_height_command.tolist(),
            },
        )

    def add_frame(self, camera_name: str, frame: CameraFrame):
        if self._closed:
            raise RuntimeError("raw episode is closed")
        index = self._frame_counts[camera_name]
        previous = self._last_sequences[camera_name]
        if previous is not None and frame.sequence > previous + 1:
            self._sequence_gaps[camera_name] += frame.sequence - previous - 1
        self._last_sequences[camera_name] = frame.sequence

        encoding = (frame.encoding or "jpeg").lower()
        extensions = {"jpeg": "jpg", "jpg": "jpg", "png": "png"}
        if frame.encoded_image is not None and encoding in extensions:
            payload = frame.encoded_image
            extension = extensions[encoding]
            encoding = "jpeg" if encoding == "jpg" else encoding
        elif frame.image is not None:
            payload = _encode_jpeg(frame.image, self.jpeg_quality)
            encoding = "jpeg"
            extension = "jpg"
        else:
            raise ValueError("camera frame contains neither an RGB image nor supported encoded bytes")
        relative_path = f"cameras/{camera_name}/frame_{index:06d}.{extension}"
        (self.pending_path / relative_path).write_bytes(payload)
        record = RawFrameRecord(
            camera=camera_name,
            frame_index=index,
            sequence=frame.sequence,
            source_timestamp_ns=frame.source_timestamp_ns,
            receive_timestamp_ns=frame.receive_timestamp_ns,
            path=relative_path,
            encoding=encoding,
            shape=frame.shape,
        )
        _append_json_line(self._frame_handles[camera_name], asdict(record))
        self._frame_counts[camera_name] += 1

    def close_for_review(self):
        if self._closed:
            return
        self._manifest["status"] = "review"
        self._manifest["frame_counts"] = dict(self._frame_counts)
        self._manifest["sequence_gaps"] = dict(self._sequence_gaps)
        self._write_manifest()
        self._close_handles()

    def commit(self) -> Path:
        self.close_for_review()
        self._manifest["status"] = "committed"
        self._write_manifest()
        self.committed_path.parent.mkdir(parents=True, exist_ok=True)
        self.pending_path.replace(self.committed_path)
        return self.committed_path

    def discard(self):
        self._close_handles()
        shutil.rmtree(self.pending_path, ignore_errors=True)

    def _close_handles(self):
        if self._closed:
            return
        self._control_handle.close()
        for handle in self._frame_handles.values():
            handle.close()
        self._closed = True
