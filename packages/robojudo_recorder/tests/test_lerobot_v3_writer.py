import json
import sys
import tempfile
import unittest
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PACKAGE_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PACKAGE_SRC))

from robojudo_recorder.dataset import LeRobotV3Writer  # noqa: E402


class TestLeRobotV3Writer(unittest.TestCase):
    def setUp(self):
        self.temporary_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_dir.name) / "dataset"
        self.writer = LeRobotV3Writer(
            root=self.root,
            repo_id="local/test",
            robot_type="x2",
            fps=10,
            state_names=["left.pos", "right.pos"],
            action_names=[
                "left.pos",
                "right.pos",
                "base.velocity.x",
                "base.velocity.y",
                "base.yaw_rate",
                "base.height",
            ],
            camera_name="head_rgb",
            camera_shape=(48, 64, 3),
        )

    def tearDown(self):
        self.temporary_dir.cleanup()

    def test_writes_v3_metadata_parquet_and_video(self):
        self.writer.start_episode("test task")
        for index in range(5):
            self.writer.add_frame(
                state=np.asarray([index, index + 1], dtype=np.float32),
                action=np.asarray([index, index + 1, 0.2, 0.0, 0.1, 0.62], dtype=np.float32),
                image=np.full((48, 64, 3), index * 10, dtype=np.uint8),
            )
        self.writer.save_episode()

        info = json.loads((self.root / "meta/info.json").read_text())
        self.assertEqual(info["codebase_version"], "v3.0")
        self.assertEqual(info["total_episodes"], 1)
        self.assertEqual(info["total_frames"], 5)
        self.assertEqual(info["features"]["action"]["shape"], [6])

        data = pq.read_table(self.root / "data/chunk-000/file-000.parquet")
        self.assertEqual(data.num_rows, 5)
        self.assertEqual(data["frame_index"].to_pylist(), [0, 1, 2, 3, 4])
        self.assertEqual(data["action"].type.list_size, 6)

        episodes = pd.read_parquet(self.root / "meta/episodes/chunk-000/file-000.parquet")
        self.assertEqual(episodes.loc[0, "dataset_from_index"], 0)
        self.assertEqual(episodes.loc[0, "dataset_to_index"], 5)
        self.assertEqual(episodes.loc[0, "videos/observation.images.head_rgb/from_timestamp"], 0.0)

        tasks = pd.read_parquet(self.root / "meta/tasks.parquet")
        self.assertEqual(tasks.loc["test task", "task_index"], 0)

        video_path = self.root / "videos/observation.images.head_rgb/chunk-000/file-000.mp4"
        with av.open(str(video_path)) as container:
            frames = list(container.decode(video=0))
        self.assertEqual(len(frames), 5)
        self.assertEqual((frames[0].height, frames[0].width), (48, 64))

    def test_discard_removes_unsaved_video(self):
        self.writer.start_episode("discard")
        self.writer.add_frame(np.zeros(2), np.zeros(6), np.zeros((48, 64, 3), dtype=np.uint8))
        video_path = self.root / "videos/observation.images.head_rgb/chunk-000/file-000.mp4"

        self.writer.discard_episode()

        self.assertFalse(video_path.exists())
        self.assertFalse(self.writer.has_pending_frames)

    def test_resume_appends_episode_and_preserves_global_indices(self):
        self.writer.start_episode("first task")
        self.writer.add_frame(np.zeros(2), np.zeros(6), np.zeros((48, 64, 3), dtype=np.uint8))
        self.writer.save_episode()

        resumed = LeRobotV3Writer(
            root=self.root,
            repo_id="local/test",
            robot_type="x2",
            fps=10,
            state_names=["left.pos", "right.pos"],
            action_names=[
                "left.pos",
                "right.pos",
                "base.velocity.x",
                "base.velocity.y",
                "base.yaw_rate",
                "base.height",
            ],
            camera_name="head_rgb",
            camera_shape=(48, 64, 3),
            resume=True,
        )
        resumed.start_episode("second task")
        resumed.add_frame(np.ones(2), np.ones(6), np.ones((48, 64, 3), dtype=np.uint8))
        resumed.save_episode()

        info = json.loads((self.root / "meta/info.json").read_text())
        self.assertEqual((info["total_episodes"], info["total_frames"], info["total_tasks"]), (2, 2, 2))
        second = pq.read_table(self.root / "data/chunk-000/file-001.parquet")
        self.assertEqual(second["index"].to_pylist(), [1])
        episodes = pd.read_parquet(self.root / "meta/episodes/chunk-000/file-000.parquet")
        self.assertEqual(episodes["dataset_to_index"].tolist(), [1, 2])

    def test_resume_rejects_schema_change(self):
        self.writer.start_episode("first task")
        self.writer.add_frame(np.zeros(2), np.zeros(6), np.zeros((48, 64, 3), dtype=np.uint8))
        self.writer.save_episode()

        with self.assertRaisesRegex(ValueError, "different schema"):
            LeRobotV3Writer(
                root=self.root,
                repo_id="local/test",
                robot_type="x2",
                fps=20,
                state_names=["left.pos", "right.pos"],
                action_names=self.writer.action_names,
                camera_name="head_rgb",
                camera_shape=(48, 64, 3),
                resume=True,
            )


if __name__ == "__main__":
    unittest.main()
