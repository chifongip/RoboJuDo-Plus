import json
import shutil
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PACKAGE_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PACKAGE_SRC))

from robojudo_recorder.cameras.base import CameraFrame  # noqa: E402
from robojudo_recorder.config import CameraConfig, DatasetConfig, RecorderConfig  # noqa: E402
from robojudo_recorder.finalize import RawDatasetFinalizer  # noqa: E402
from robojudo_recorder.protocol import ControlSample  # noqa: E402
from robojudo_recorder.raw import RawEpisodeWriter  # noqa: E402


class TestParallelFinalize(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.cfg = RecorderConfig(
            control_endpoint="inproc://parallel-finalize-test",
            dataset=DatasetConfig(
                root=self.base / "parallel",
                raw_root=self.base / "raw",
                repo_id="local/test",
                fps=10,
                resume=True,
                dagger_labels=True,
            ),
            cameras=(CameraConfig(type="fake", name="head"), CameraConfig(type="fake", name="wrist")),
        )
        for episode_id, length in enumerate((5, 3, 4)):
            writer = RawEpisodeWriter(
                raw_root=self.cfg.dataset.raw_root,
                episode_id=episode_id,
                task=f"task {episode_id % 2}",
                camera_names=("head", "wrist"),
                started_source_ns=0,
                started_receive_ns=0,
                jpeg_quality=90,
            )
            for frame_id in range(length):
                timestamp = frame_id * 100_000_000
                expert = frame_id % 2 == 0
                writer.add_control(
                    ControlSample(
                        episode_id=episode_id,
                        task=f"task {episode_id % 2}",
                        robot_type="g1",
                        source_timestamp_ns=timestamp,
                        receive_timestamp_ns=timestamp,
                        joint_names=("left", "right"),
                        joint_positions=np.array([episode_id, frame_id], dtype=np.float32),
                        joint_position_commands=np.array([frame_id, episode_id], dtype=np.float32),
                        velocity_height_command=np.zeros(4, dtype=np.float32),
                        dagger={
                            "expert_intervention": expert,
                            "expert_applied": expert,
                            "action_source": "expert" if expert else "policy",
                            "intervention_session": 1,
                            "expert_frame_id": frame_id if expert else None,
                        },
                    )
                )
                for camera in ("head", "wrist"):
                    writer.add_frame(
                        camera,
                        CameraFrame(
                            image=np.full((24, 32, 3), episode_id * 30 + frame_id * 5, dtype=np.uint8),
                            timestamp_ns=timestamp,
                            sequence=frame_id,
                        ),
                    )
            path = writer.commit()
            path.rename(path.with_name(f"episode_{episode_id:03d}"))

    def serial_reference(self):
        raw_root = self.base / "serial_raw"
        shutil.copytree(self.cfg.dataset.raw_root, raw_root)
        for report in raw_root.glob("episodes/*/finalize_report.json"):
            report.unlink()
        cfg = replace(self.cfg, dataset=replace(self.cfg.dataset, root=self.base / "serial", raw_root=raw_root))
        reports = RawDatasetFinalizer(cfg, encoder_threads=1).run()
        return cfg.dataset.root, reports

    def assert_datasets_equal(self, reference):
        root = self.cfg.dataset.root
        for name in ("info.json", "stats.json"):
            self.assertEqual(
                json.loads((root / "meta" / name).read_text()), json.loads((reference / "meta" / name).read_text())
            )
        for name in ("tasks.parquet", "episodes/chunk-000/file-000.parquet"):
            pd.testing.assert_frame_equal(
                pd.read_parquet(root / "meta" / name), pd.read_parquet(reference / "meta" / name)
            )
        for path in reference.glob("data/*/*.parquet"):
            self.assertTrue(pq.read_table(path).equals(pq.read_table(root / path.relative_to(reference))))
        for path in reference.glob("videos/*/*/*.mp4"):
            with av.open(str(path)) as left, av.open(str(root / path.relative_to(reference))) as right:
                frames_left = [frame.to_ndarray(format="rgb24") for frame in left.decode(video=0)]
                frames_right = [frame.to_ndarray(format="rgb24") for frame in right.decode(video=0)]
            np.testing.assert_array_equal(frames_left, frames_right)
        self.assertFalse(list(self.base.glob(".robojudo-finalize-*")))

    def test_parallel_matches_serial_including_labels_videos_tasks_and_statistics(self):
        reference, serial_reports = self.serial_reference()
        reports = RawDatasetFinalizer(self.cfg, encoder_threads=1).run(workers=2)
        self.assertEqual(reports, serial_reports)
        self.assert_datasets_equal(reference)

    def test_resume_and_skip_preserve_global_indices_and_reports(self):
        reference, serial_reports = self.serial_reference()
        RawDatasetFinalizer(self.cfg, encoder_threads=1).run({"episode_000"}, workers=2)
        report = self.cfg.dataset.raw_root / "episodes/episode_000/finalize_report.json"
        modified = report.stat().st_mtime_ns
        reports = RawDatasetFinalizer(self.cfg, encoder_threads=1).run(workers=2)
        self.assertEqual(report.stat().st_mtime_ns, modified)
        self.assertEqual(reports, serial_reports)
        self.assert_datasets_equal(reference)
        self.assertEqual(RawDatasetFinalizer(self.cfg).run(workers=2), reports)

    def test_failed_worker_does_not_mark_unmerged_episodes_and_can_retry(self):
        reference, serial_reports = self.serial_reference()
        manifest = self.cfg.dataset.raw_root / "episodes/episode_001/manifest.json"
        original = manifest.read_text()
        damaged = json.loads(original)
        damaged["status"] = "recording"
        manifest.write_text(json.dumps(damaged))
        with self.assertRaisesRegex(ValueError, "not a committed"):
            RawDatasetFinalizer(self.cfg, encoder_threads=1).run(workers=2)
        self.assertTrue((manifest.parent.parent / "episode_000/finalize_report.json").exists())
        self.assertFalse((manifest.parent / "finalize_report.json").exists())
        self.assertFalse((manifest.parent.parent / "episode_002/finalize_report.json").exists())
        self.assertFalse(list(self.base.glob(".robojudo-finalize-*")))
        manifest.write_text(original)
        reports = RawDatasetFinalizer(self.cfg, encoder_threads=1).run(workers=2)
        self.assertEqual(reports, serial_reports)
        self.assert_datasets_equal(reference)

    def test_rejects_invalid_worker_count(self):
        with self.assertRaisesRegex(ValueError, "workers"):
            RawDatasetFinalizer(self.cfg).run(workers=0)


if __name__ == "__main__":
    unittest.main()
