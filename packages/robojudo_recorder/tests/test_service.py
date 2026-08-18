import json
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

PACKAGE_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PACKAGE_SRC))

from robojudo_recorder.cameras.base import CameraFrame, CameraSource  # noqa: E402
from robojudo_recorder.config import CameraConfig, DatasetConfig, RecorderConfig, SyncConfig  # noqa: E402
from robojudo_recorder.finalize import RawDatasetFinalizer  # noqa: E402
from robojudo_recorder.service import RecorderService  # noqa: E402


class FakeCamera(CameraSource):
    def __init__(self, shape=(24, 32, 3), offset=0):
        self._shape = shape
        self.offset = offset
        self.sequence = 0
        self.read_count = 0

    @property
    def shape(self):
        return self._shape

    def connect(self):
        pass

    def read(self, timeout_ms):
        del timeout_ms
        self.read_count += 1
        self.sequence += 1
        now = time.monotonic_ns()
        return CameraFrame(
            image=np.full(self.shape, self.sequence + self.offset, dtype=np.uint8),
            timestamp_ns=now,
            sequence=self.sequence,
        )

    def close(self):
        pass


class MissingCamera(FakeCamera):
    def read(self, timeout_ms):
        del timeout_ms
        return None


class TestRecorderService(unittest.TestCase):
    @staticmethod
    def _sample_message(timestamp_ns, positions=(0.0, 0.0), commands=(1.0, 1.0)):
        return {
            "kind": "sample",
            "episode_id": 1,
            "task": "test task",
            "robot_type": "g1",
            "timestamp_ns": timestamp_ns,
            "joint_names": ["left_arm", "right_arm"],
            "joint_positions": list(positions),
            "joint_position_commands": list(commands),
            "velocity_height_command": [0.5, 0.0, 0.1, 0.76],
        }

    @staticmethod
    def _config(temporary_dir, *, cameras=None, clock="source", fps=10):
        root = Path(temporary_dir) / "dataset"
        return RecorderConfig(
            control_endpoint=f"inproc://recorder-{uuid.uuid4()}",
            dataset=DatasetConfig(
                root=root,
                raw_root=Path(temporary_dir) / "raw",
                repo_id="local/service",
                fps=fps,
            ),
            cameras=cameras or (CameraConfig(type="fake", name="head_rgb"),),
            sync=SyncConfig(clock=clock, max_control_age_ms=60, max_camera_delta_ms=60),
        )

    def test_spools_raw_then_finalizes_with_interpolated_state(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            cfg = self._config(temporary_dir)
            camera = FakeCamera()
            service = RecorderService(cfg, camera=camera)
            service._handle_message(self._sample_message(0, (0.0, 0.0), (1.0, 1.0)), 0)
            service._handle_message(self._sample_message(100_000_000, (1.0, 2.0), (2.0, 2.0)), 100_000_000)
            service._handle_message(self._sample_message(200_000_000, (2.0, 4.0), (3.0, 3.0)), 200_000_000)
            for sequence, timestamp_ns in enumerate((50_000_000, 150_000_000), start=1):
                frame = CameraFrame(
                    image=np.full(camera.shape, sequence * 20, dtype=np.uint8),
                    timestamp_ns=timestamp_ns,
                    sequence=sequence,
                )
                service._record_frame("head_rgb", frame)
            service._finish_episode(save=True)
            service.close()

            raw_episodes = list((cfg.dataset.raw_root / "episodes").iterdir())
            self.assertEqual(len(raw_episodes), 1)
            self.assertFalse(cfg.dataset.root.exists())
            RawDatasetFinalizer(cfg).run()

            data = pq.read_table(cfg.dataset.root / "data/chunk-000/file-000.parquet")
            self.assertEqual(data.num_rows, 2)
            np.testing.assert_allclose(data["observation.state"].to_pylist(), [[0.5, 1.0], [1.5, 3.0]])
            np.testing.assert_allclose(data["action"].to_pylist()[0], [1.0, 1.0, 0.5, 0.0, 0.1, 0.76])
            report = json.loads((raw_episodes[0] / "finalize_report.json").read_text())
            self.assertEqual(report["written_frames"], 2)
            self.assertEqual(report["dropped_camera_slots"], 0)
            self.assertEqual(report["dropped_control_slots"], 0)
            self.assertEqual(report["selected_unique_camera_frames"], {"head_rgb": 2})
            self.assertIn("camera_delta_summary_ms", report)
            self.assertIn("control_age_summary_ms", report)

    def test_review_stops_camera_reads_and_discard_removes_pending_raw(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            cfg = self._config(temporary_dir, clock="receive")
            camera = FakeCamera()
            service = RecorderService(cfg, camera=camera)
            now = time.monotonic_ns()
            service._handle_message(self._sample_message(now), now)
            service.step()
            service._handle_message({"kind": "episode_review", "episode_id": 1}, time.monotonic_ns())
            reads_at_review = camera.read_count

            service.step()
            self.assertEqual(camera.read_count, reads_at_review)
            service._handle_message({"kind": "episode_discard", "episode_id": 1}, time.monotonic_ns())
            service.close()
            self.assertFalse(any((cfg.dataset.raw_root / ".pending").glob("*")))
            self.assertFalse((cfg.dataset.raw_root / "episodes").exists())

    def test_spools_each_camera_independently(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            cameras = (
                CameraConfig(type="fake", name="head_rgb"),
                CameraConfig(type="fake", name="wrist_rgb"),
            )
            cfg = self._config(temporary_dir, cameras=cameras, clock="receive")
            service = RecorderService(cfg, cameras=(FakeCamera(), FakeCamera(offset=10)))
            now = time.monotonic_ns()
            service._handle_message(self._sample_message(now), now)
            service.step()
            service._finish_episode(save=True)
            service.close()

            episode = next((cfg.dataset.raw_root / "episodes").iterdir())
            manifest = json.loads((episode / "manifest.json").read_text())
            self.assertEqual(manifest["frame_counts"], {"head_rgb": 1, "wrist_rgb": 1})
            self.assertTrue((episode / "cameras/head_rgb/frame_000000.jpg").exists())
            self.assertTrue((episode / "cameras/wrist_rgb/frame_000000.jpg").exists())

    def test_does_not_spool_frames_from_before_episode_start(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            cfg = self._config(temporary_dir)
            service = RecorderService(cfg, camera=FakeCamera())
            service._handle_message(
                {"kind": "episode_start", "episode_id": 1, "task": "test task", "timestamp_ns": 2_000},
                2_000,
            )
            service._record_frame(
                "head_rgb",
                CameraFrame(image=np.zeros((24, 32, 3), dtype=np.uint8), timestamp_ns=1_000, sequence=1),
            )
            service._finish_episode(save=False)
            self.assertEqual(service._episode_frame_counts["head_rgb"], 0)
            service.close()

    def test_logs_raw_input_and_write_fps_with_sequence_gaps(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            cfg = self._config(temporary_dir, fps=30)
            service = RecorderService(cfg, camera=FakeCamera())
            service._active_episode_id = 1
            service._throughput_window_started_ns = time.monotonic_ns() - 1_000_000_000
            service._throughput_input_frames["head_rgb"] = 20
            service._throughput_written_frames["head_rgb"] = 19
            service._throughput_sequence_gaps["head_rgb"] = 10

            with self.assertLogs("robojudo_recorder.service", level="WARNING") as logs:
                service._log_throughput(force=True)

            message = logs.output[-1]
            self.assertIn("input_fps=[head_rgb=20.0]", message)
            self.assertIn("write_fps=[head_rgb=19.0]", message)
            self.assertIn("sequence_gaps=[head_rgb=10]", message)
            service.close()

    def test_logs_when_camera_frames_are_missing(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            cfg = self._config(temporary_dir, clock="receive")
            service = RecorderService(cfg, camera=MissingCamera())
            now = time.monotonic_ns()
            service._handle_message(self._sample_message(now), now)
            service._camera_missing_since_ns = now - 3_000_000_000
            with self.assertLogs("robojudo_recorder.service", level="WARNING") as logs:
                service.step()
            self.assertTrue(any("Waiting for camera frames" in message for message in logs.output))
            service._finish_episode(save=False)
            service.close()


if __name__ == "__main__":
    unittest.main()
