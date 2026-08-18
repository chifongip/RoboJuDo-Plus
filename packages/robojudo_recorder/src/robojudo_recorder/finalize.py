"""Offline raw-episode synchronization and LeRobot v3 finalization."""

import argparse
import bisect
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np

from .config import RecorderConfig, load_config
from .dataset import LeRobotV3Writer
from .protocol import LOCOMOTION_COMMAND_NAMES
from .raw import RAW_FORMAT_VERSION

logger = logging.getLogger(__name__)


def _read_json_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _decode_image(path: Path) -> np.ndarray:
    with av.open(str(path), mode="r") as container:
        frame = next(container.decode(video=0), None)
    if frame is None:
        raise ValueError(f"could not decode image {path}")
    return frame.to_ndarray(format="rgb24")


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values else 0.0


@dataclass
class _ControlMatch:
    state: np.ndarray
    action: np.ndarray
    age_ms: float


class RawDatasetFinalizer:
    """Convert committed raw episodes into a uniformly sampled LeRobot v3 dataset."""

    def __init__(self, cfg: RecorderConfig):
        self.cfg = cfg
        self._writer: LeRobotV3Writer | None = None
        self._schema: tuple[str, tuple[str, ...], tuple[tuple[str, tuple[int, int, int]], ...]] | None = None

    @property
    def episodes_root(self) -> Path:
        return self.cfg.dataset.raw_root / "episodes"

    def _timestamp(self, record: dict) -> int:
        return int(record[f"{self.cfg.sync.clock}_timestamp_ns"])

    @staticmethod
    def _nearest_record(records: list[dict], timestamps: list[int], target_ns: int) -> tuple[dict, float]:
        index = bisect.bisect_left(timestamps, target_ns)
        candidates = []
        if index < len(records):
            candidates.append(records[index])
        if index > 0:
            candidates.append(records[index - 1])
        record = min(candidates, key=lambda item: abs(int(item["_timestamp_ns"]) - target_ns))
        delta_ms = abs(int(record["_timestamp_ns"]) - target_ns) / 1_000_000
        return record, delta_ms

    def _match_control(self, controls: list[dict], timestamps: list[int], target_ns: int) -> _ControlMatch | None:
        following_index = bisect.bisect_right(timestamps, target_ns)
        previous_index = following_index - 1
        if previous_index < 0:
            return None
        previous = controls[previous_index]
        previous_timestamp = timestamps[previous_index]
        state0 = np.asarray(previous["joint_positions"], dtype=np.float32)
        action = np.concatenate(
            (
                np.asarray(previous["joint_position_commands"], dtype=np.float32),
                np.asarray(previous["velocity_height_command"], dtype=np.float32),
            )
        ).astype(np.float32)
        age_ms = (target_ns - previous_timestamp) / 1_000_000
        if target_ns == previous_timestamp:
            return _ControlMatch(state0, action, age_ms)
        if following_index >= len(controls):
            return None
        following = controls[following_index]
        following_timestamp = timestamps[following_index]
        if following_timestamp <= previous_timestamp:
            return _ControlMatch(state0, action, age_ms)
        alpha = np.float32((target_ns - previous_timestamp) / (following_timestamp - previous_timestamp))
        state1 = np.asarray(following["joint_positions"], dtype=np.float32)
        state = state0 + alpha * (state1 - state0)
        return _ControlMatch(state.astype(np.float32), action, age_ms)

    def _load_episode(self, episode_path: Path):
        manifest = json.loads((episode_path / "manifest.json").read_text())
        if manifest.get("format_version") != RAW_FORMAT_VERSION or manifest.get("status") != "committed":
            raise ValueError(f"raw episode {episode_path.name} is not a committed format-v{RAW_FORMAT_VERSION} episode")
        controls = _read_json_lines(episode_path / "controls.jsonl")
        camera_records = {}
        for name in manifest["camera_names"]:
            records = _read_json_lines(episode_path / "cameras" / name / "frames.jsonl")
            camera_records[name] = records
        return manifest, controls, camera_records

    def _ensure_writer(self, manifest: dict, camera_records: dict[str, list[dict]]):
        joint_names = manifest.get("joint_names")
        robot_type = manifest.get("robot_type")
        if not joint_names or not robot_type:
            raise ValueError("raw episode contains no control schema")
        camera_shapes = {}
        for name, records in camera_records.items():
            if not records:
                raise ValueError(f"raw episode contains no frames for camera {name!r}")
            camera_shapes[name] = tuple(records[0]["shape"])
            if any(tuple(record["shape"]) != camera_shapes[name] for record in records):
                raise ValueError(f"camera {name!r} changed shape within the episode")
        schema = (robot_type, tuple(joint_names), tuple(sorted(camera_shapes.items())))
        if self._schema is not None and schema != self._schema:
            raise ValueError("raw episode schema differs from episodes already finalized in this run")
        if self._writer is None:
            self._writer = LeRobotV3Writer(
                root=self.cfg.dataset.root,
                repo_id=self.cfg.dataset.repo_id,
                robot_type=robot_type,
                fps=self.cfg.dataset.fps,
                state_names=[f"{name}.pos" for name in joint_names],
                action_names=[*[f"{name}.pos" for name in joint_names], *LOCOMOTION_COMMAND_NAMES],
                camera_shapes=camera_shapes,
                codec=self.cfg.dataset.codec,
                resume=self.cfg.dataset.resume,
            )
            self._schema = schema
        return camera_shapes

    def finalize_episode(self, episode_path: Path) -> dict:
        report_path = episode_path / "finalize_report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            output_file = self.cfg.dataset.root / report.get("data_file", "missing")
            if report.get("status") == "finalized" and output_file.is_file():
                logger.info("Skipping already finalized raw episode %s", episode_path.name)
                return report

        manifest, controls, camera_records = self._load_episode(episode_path)
        if not controls:
            raise ValueError(f"raw episode {episode_path.name} contains no controls")
        camera_shapes = self._ensure_writer(manifest, camera_records)
        for record in controls:
            record["_timestamp_ns"] = self._timestamp(record)
        controls.sort(key=lambda item: item["_timestamp_ns"])
        control_timestamps = [record["_timestamp_ns"] for record in controls]
        camera_timestamps = {}
        for name, records in camera_records.items():
            for record in records:
                record["_timestamp_ns"] = self._timestamp(record)
            records.sort(key=lambda item: item["_timestamp_ns"])
            camera_timestamps[name] = [record["_timestamp_ns"] for record in records]

        start_ns = max(control_timestamps[0], *(timestamps[0] for timestamps in camera_timestamps.values()))
        end_ns = min(control_timestamps[-1], *(timestamps[-1] for timestamps in camera_timestamps.values()))
        period_ns = round(1_000_000_000 / self.cfg.dataset.fps)
        primary_name = manifest["camera_names"][0]
        primary_times = camera_timestamps[primary_name]
        primary_start_index = bisect.bisect_left(primary_times, start_ns)
        if primary_start_index >= len(primary_times):
            raise ValueError(f"raw episode {episode_path.name} has no overlapping camera/control interval")
        grid_start_ns = primary_times[primary_start_index]
        target_timestamps = list(range(grid_start_ns, end_ns + 1, period_ns))

        report = {
            "status": "finalizing",
            "raw_episode": episode_path.name,
            "episode_id": manifest["episode_id"],
            "clock": self.cfg.sync.clock,
            "target_fps": self.cfg.dataset.fps,
            "raw_control_frames": len(controls),
            "raw_camera_frames": {name: len(records) for name, records in camera_records.items()},
            "raw_camera_fps": {
                name: (
                    (len(timestamps) - 1) * 1_000_000_000 / (timestamps[-1] - timestamps[0])
                    if len(timestamps) > 1 and timestamps[-1] > timestamps[0]
                    else 0.0
                )
                for name, timestamps in camera_timestamps.items()
            },
            "source_sequence_gaps": manifest.get("sequence_gaps", {}),
            "target_slots": len(target_timestamps),
            "written_frames": 0,
            "dropped_camera_slots": 0,
            "dropped_control_slots": 0,
            "over_age_frames": 0,
            "camera_delta_ms": {name: [] for name in camera_records},
            "control_age_ms": [],
            "camera_shapes": {name: list(shape) for name, shape in camera_shapes.items()},
        }
        self._writer.start_episode(manifest["task"])
        dataset_episode_index = self._writer.next_episode_index
        max_camera_delta_ms = self.cfg.sync.max_camera_delta_ms
        selected_frame_indices = {name: set() for name in camera_records}
        try:
            for target_ns in target_timestamps:
                selected = {}
                selected_deltas = {}
                camera_failed = False
                for name, records in camera_records.items():
                    record, delta_ms = self._nearest_record(records, camera_timestamps[name], target_ns)
                    if delta_ms > max_camera_delta_ms:
                        camera_failed = True
                        break
                    selected[name] = record
                    selected_deltas[name] = delta_ms
                if camera_failed:
                    report["dropped_camera_slots"] += 1
                    continue
                control = self._match_control(controls, control_timestamps, target_ns)
                if control is None:
                    report["dropped_control_slots"] += 1
                    continue
                for name, record in selected.items():
                    report["camera_delta_ms"][name].append(selected_deltas[name])
                    selected_frame_indices[name].add(int(record["frame_index"]))
                report["control_age_ms"].append(control.age_ms)
                if control.age_ms > self.cfg.sync.max_control_age_ms:
                    report["over_age_frames"] += 1
                images = {name: _decode_image(episode_path / record["path"]) for name, record in selected.items()}
                self._writer.add_frame(control.state, control.action, images)
                report["written_frames"] += 1
            if report["written_frames"] == 0:
                raise ValueError(f"raw episode {episode_path.name} produced no synchronized output frames")
            self._writer.save_episode()
        except Exception:
            self._writer.discard_episode()
            raise

        report["status"] = "finalized"
        chunk_index, file_index = divmod(dataset_episode_index, 1000)
        report["dataset_episode_index"] = dataset_episode_index
        report["data_file"] = f"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        report["selected_unique_camera_frames"] = {
            name: len(indices) for name, indices in selected_frame_indices.items()
        }
        report["duplicated_camera_slots"] = {
            name: report["written_frames"] - len(indices) for name, indices in selected_frame_indices.items()
        }
        report["unused_camera_frames"] = {
            name: len(records) - len(selected_frame_indices[name]) for name, records in camera_records.items()
        }
        report["control_age_summary_ms"] = {
            "mean": float(np.mean(report["control_age_ms"])) if report["control_age_ms"] else 0.0,
            "p95": _percentile(report["control_age_ms"], 95),
            "max": max(report["control_age_ms"], default=0.0),
        }
        report["camera_delta_summary_ms"] = {
            name: {
                "mean": float(np.mean(values)) if values else 0.0,
                "p95": _percentile(values, 95),
                "max": max(values, default=0.0),
            }
            for name, values in report["camera_delta_ms"].items()
        }
        del report["control_age_ms"]
        del report["camera_delta_ms"]
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        logger.info(
            "Finalized %s: written=%d/%d, camera_drops=%d, control_drops=%d, over_age=%d",
            episode_path.name,
            report["written_frames"],
            report["target_slots"],
            report["dropped_camera_slots"],
            report["dropped_control_slots"],
            report["over_age_frames"],
        )
        return report

    def run(self, episode_names: set[str] | None = None) -> list[dict]:
        if not self.episodes_root.exists():
            logger.warning("No committed raw episodes found at %s", self.episodes_root)
            return []
        paths = sorted(path for path in self.episodes_root.iterdir() if path.is_dir())
        if episode_names:
            paths = [path for path in paths if path.name in episode_names]
        reports = [self.finalize_episode(path) for path in paths]
        if self._writer is not None:
            self._writer.finalize()
        return reports


def parse_args():
    parser = argparse.ArgumentParser(description="Finalize raw RoboJuDo episodes into a LeRobot v3 dataset")
    parser.add_argument("--config", required=True, help="Recorder YAML configuration used during collection")
    parser.add_argument("--episode", action="append", default=[], help="Raw episode directory name; repeat as needed")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    reports = RawDatasetFinalizer(load_config(args.config)).run(set(args.episode) or None)
    logger.info("Finalization complete: %d raw episodes examined", len(reports))


if __name__ == "__main__":
    main()
