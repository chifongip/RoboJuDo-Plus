"""Offline raw-episode synchronization and LeRobot v3 finalization."""

import argparse
import bisect
import json
import logging
import multiprocessing
import shutil
import tempfile
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
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
    # Offline DAgger label associated with the zero-order-held action.
    dagger: dict | None = None


class RawDatasetFinalizer:
    """Convert committed raw episodes into a uniformly sampled LeRobot v3 dataset."""

    def __init__(self, cfg: RecorderConfig, *, encoder_threads: int = 0):
        self.cfg = cfg
        self.encoder_threads = encoder_threads
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
            return _ControlMatch(state0, action, age_ms, previous.get("dagger"))
        if following_index >= len(controls):
            return None
        following = controls[following_index]
        following_timestamp = timestamps[following_index]
        if following_timestamp <= previous_timestamp:
            return _ControlMatch(state0, action, age_ms, previous.get("dagger"))
        alpha = np.float32((target_ns - previous_timestamp) / (following_timestamp - previous_timestamp))
        state1 = np.asarray(following["joint_positions"], dtype=np.float32)
        state = state0 + alpha * (state1 - state0)
        return _ControlMatch(state.astype(np.float32), action, age_ms, previous.get("dagger"))

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
                dagger_features=self.cfg.dataset.preserve_dagger_labels,
                encoder_threads=self.encoder_threads,
            )
            self._schema = schema
        return camera_shapes

    def _existing_report(self, episode_path: Path) -> dict | None:
        report_path = episode_path / "finalize_report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            data_files = report.get("data_files") or [report.get("data_file", "missing")]
            output_files_exist = all((self.cfg.dataset.root / path).is_file() for path in data_files)
            if report.get("status") == "finalized" and output_files_exist:
                logger.info("Skipping already finalized raw episode %s", episode_path.name)
                return report
        return None

    def finalize_episode(self, episode_path: Path, *, write_report: bool = True) -> dict:
        if write_report and (report := self._existing_report(episode_path)) is not None:
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
            "writer_queue_drops": manifest.get("writer_queue_drops", {}),
            "target_slots": len(target_timestamps),
            "written_frames": 0,
            "dropped_camera_slots": 0,
            "dropped_control_slots": 0,
            "over_age_frames": 0,
            "camera_delta_ms": {name: [] for name in camera_records},
            "control_age_ms": [],
            "camera_shapes": {name: list(shape) for name, shape in camera_shapes.items()},
        }
        max_camera_delta_ms = self.cfg.sync.max_camera_delta_ms
        selected_frame_indices = {name: set() for name in camera_records}
        # Offline DAgger keeps the physical rollout boundary. Expert sessions
        # remain frame labels and never become synthetic LeRobot episodes.
        dataset_episode_index = self._writer.next_episode_index
        self._writer.start_episode(manifest["task"])
        episode_open = True
        report["dagger_labels"] = self.cfg.dataset.preserve_dagger_labels
        report["preserve_full_rollout"] = True
        report["legacy_expert_only_requested"] = self.cfg.dataset.expert_only
        report["expert_frames"] = 0
        report["policy_frames"] = 0
        report["missing_dagger_labels"] = 0
        report["missing_expert_frame_ids"] = 0
        intervention_sessions = set()
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
                dagger = control.dagger
                if self.cfg.dataset.preserve_dagger_labels:
                    if dagger is None:
                        report["missing_dagger_labels"] += 1
                    else:
                        if dagger.get("expert_intervention", False):
                            intervention_sessions.add(int(dagger["intervention_session"]))
                        if dagger.get("expert_applied", False):
                            report["expert_frames"] += 1
                            if dagger.get("expert_frame_id") is None:
                                report["missing_expert_frame_ids"] += 1
                        else:
                            report["policy_frames"] += 1
                for name, record in selected.items():
                    report["camera_delta_ms"][name].append(selected_deltas[name])
                    selected_frame_indices[name].add(int(record["frame_index"]))
                report["control_age_ms"].append(control.age_ms)
                if control.age_ms > self.cfg.sync.max_control_age_ms:
                    report["over_age_frames"] += 1
                images = {name: _decode_image(episode_path / record["path"]) for name, record in selected.items()}
                self._writer.add_frame(
                    control.state,
                    control.action,
                    images,
                    dagger=dagger if self.cfg.dataset.preserve_dagger_labels else None,
                )
                report["written_frames"] += 1
            if report["written_frames"] == 0:
                raise ValueError(f"raw episode {episode_path.name} produced no synchronized output frames")
            self._writer.save_episode()
            episode_open = False
        except Exception:
            if episode_open:
                self._writer.discard_episode()
            raise

        report["status"] = "finalized"
        chunk_index, file_index = divmod(dataset_episode_index, 1000)
        data_files = [f"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"]
        report["dataset_episode_indices"] = [dataset_episode_index]
        report["data_files"] = data_files
        report["intervention_sessions"] = sorted(intervention_sessions)
        # Preserve legacy singular report keys for downstream tooling.
        report["dataset_episode_index"] = dataset_episode_index
        report["data_file"] = data_files[0]
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
        if write_report:
            self._write_report(episode_path, report)
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

    @staticmethod
    def _write_report(episode_path: Path, report: dict):
        path = episode_path / "finalize_report.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)

    def _run_parallel(self, paths: list[Path], workers: int) -> list[dict]:
        reports = {}
        pending = []
        for path in paths:
            report = self._existing_report(path)
            if report is None:
                pending.append(path)
            else:
                reports[path.name] = report
        if not pending:
            return [reports[path.name] for path in paths]

        # Keep staging on the destination filesystem so videos can be renamed
        # into place. At most `workers` episodes are staged at any time.
        self.cfg.dataset.root.parent.mkdir(parents=True, exist_ok=True)
        workers = min(workers, len(pending))
        with tempfile.TemporaryDirectory(prefix=".robojudo-finalize-", dir=self.cfg.dataset.root.parent) as temporary:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                queue = deque()
                remaining = iter(pending)

                def submit(path):
                    stage_root = Path(temporary) / path.name
                    future = pool.submit(_finalize_worker, self.cfg, path, stage_root, self.encoder_threads)
                    queue.append((path, stage_root, future))

                for _ in range(workers):
                    submit(next(remaining))
                try:
                    while queue:
                        path, stage_root, future = queue.popleft()
                        report = future.result()
                        manifest, _, camera_records = self._load_episode(path)
                        self._ensure_writer(manifest, camera_records)
                        index = self._writer.next_episode_index
                        self._writer.append_preencoded_episode(stage_root)
                        chunk, file = divmod(index, 1000)
                        data_file = f"data/chunk-{chunk:03d}/file-{file:03d}.parquet"
                        report.update(
                            dataset_episode_index=index,
                            dataset_episode_indices=[index],
                            data_file=data_file,
                            data_files=[data_file],
                        )
                        self._write_report(path, report)
                        reports[path.name] = report
                        logger.info(
                            "Finalized %s: written=%d/%d, camera_drops=%d, control_drops=%d, over_age=%d",
                            path.name,
                            report["written_frames"],
                            report["target_slots"],
                            report["dropped_camera_slots"],
                            report["dropped_control_slots"],
                            report["over_age_frames"],
                        )
                        shutil.rmtree(stage_root)
                        if (next_path := next(remaining, None)) is not None:
                            submit(next_path)
                finally:
                    for _, _, future in queue:
                        future.cancel()
        return [reports[path.name] for path in paths]

    def run(self, episode_names: set[str] | None = None, *, workers: int = 1) -> list[dict]:
        if workers < 1:
            raise ValueError("workers must be at least 1")
        if not self.episodes_root.exists():
            logger.warning("No committed raw episodes found at %s", self.episodes_root)
            return []
        paths = sorted(path for path in self.episodes_root.iterdir() if path.is_dir())
        if episode_names:
            paths = [path for path in paths if path.name in episode_names]
        reports = self._run_parallel(paths, workers) if workers > 1 else [self.finalize_episode(path) for path in paths]
        if self._writer is not None:
            self._writer.finalize()
        return reports


def _finalize_worker(cfg: RecorderConfig, episode_path: Path, stage_root: Path, encoder_threads: int) -> dict:
    """Encode one episode in isolation; only the parent persists raw reports."""
    staged_cfg = replace(cfg, dataset=replace(cfg.dataset, root=stage_root, resume=False))
    finalizer = RawDatasetFinalizer(staged_cfg, encoder_threads=encoder_threads)
    report = finalizer.finalize_episode(episode_path, write_report=False)
    finalizer._writer.finalize()
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Finalize raw RoboJuDo episodes into a LeRobot v3 dataset")
    parser.add_argument("--config", required=True, help="Recorder YAML configuration used during collection")
    parser.add_argument("--episode", action="append", default=[], help="Raw episode directory name; repeat as needed")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent episode processes (default: 1)")
    parser.add_argument(
        "--encoder-threads",
        type=int,
        default=None,
        help="Threads per video encoder (default: 1 with parallel workers, otherwise FFmpeg auto)",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.encoder_threads is not None and args.encoder_threads < 0:
        parser.error("--encoder-threads must be non-negative")
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    threads = args.encoder_threads if args.encoder_threads is not None else (1 if args.workers > 1 else 0)
    reports = RawDatasetFinalizer(load_config(args.config), encoder_threads=threads).run(
        set(args.episode) or None, workers=args.workers
    )
    logger.info("Finalization complete: %d raw episodes examined", len(reports))


if __name__ == "__main__":
    main()
