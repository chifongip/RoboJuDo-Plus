import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path

import msgpack
import numpy as np
import pyarrow.parquet as pq
import zmq

PACKAGE_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PACKAGE_SRC))

from robojudo_recorder.cameras.base import CameraFrame, CameraSource  # noqa: E402
from robojudo_recorder.config import CameraConfig, DatasetConfig, RecorderConfig, SyncConfig  # noqa: E402
from robojudo_recorder.service import RecorderService  # noqa: E402


class FakeCamera(CameraSource):
    def __init__(self, shape=(48, 64, 3), offset=0):
        self._shape = shape
        self.offset = offset
        self.sequence = 0
        self.connected = False

    @property
    def shape(self):
        return self._shape

    def connect(self):
        self.connected = True

    def read(self, timeout_ms):
        del timeout_ms
        self.sequence += 1
        return CameraFrame(
            image=np.full(self.shape, self.sequence + self.offset, dtype=np.uint8),
            timestamp_ns=time.monotonic_ns(),
            sequence=self.sequence,
        )

    def close(self):
        self.connected = False


class MissingCamera(FakeCamera):
    def read(self, timeout_ms):
        del timeout_ms
        return None


class TestRecorderService(unittest.TestCase):
    @staticmethod
    def _sample_message():
        return {
            "kind": "sample",
            "episode_id": 1,
            "task": "test task",
            "robot_type": "g1",
            "timestamp_ns": time.monotonic_ns(),
            "joint_names": ["left_arm", "right_arm"],
            "joint_positions": [0.1, 0.2],
            "joint_position_commands": [0.3, 0.4],
            "velocity_height_command": [0.5, 0.0, 0.1, 0.76],
        }

    def test_pairs_control_sample_with_camera_and_saves_episode(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            endpoint = f"inproc://recorder-{uuid.uuid4()}"
            root = Path(temporary_dir) / "dataset"
            cfg = RecorderConfig(
                control_endpoint=endpoint,
                dataset=DatasetConfig(root=root, repo_id="local/service", fps=10),
                camera=CameraConfig(type="fake", name="head_rgb"),
                sync=SyncConfig(clock="receive", max_control_age_ms=100),
            )
            context = zmq.Context.instance()
            sender = context.socket(zmq.PUSH)
            sender.setsockopt(zmq.LINGER, 0)
            sender.bind(endpoint)
            camera = FakeCamera()
            camera.connect()
            service = RecorderService(cfg, camera=camera)
            time.sleep(0.01)

            sender.send(
                msgpack.packb(
                    self._sample_message(),
                    use_bin_type=True,
                )
            )
            time.sleep(0.01)
            service.step()

            sender.send(
                msgpack.packb(
                    {"kind": "episode_end", "episode_id": 1, "save": True},
                    use_bin_type=True,
                )
            )
            time.sleep(0.01)
            service.step()
            service.close()
            sender.close(linger=0)

            data = pq.read_table(root / "data/chunk-000/file-000.parquet")
            self.assertEqual(data.num_rows, 1)
            np.testing.assert_allclose(
                data["action"].to_pylist()[0],
                [0.3, 0.4, 0.5, 0.0, 0.1, 0.76],
            )

    def test_records_all_configured_cameras_in_each_row(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir) / "dataset"
            cfg = RecorderConfig(
                control_endpoint=f"inproc://recorder-{uuid.uuid4()}",
                dataset=DatasetConfig(root=root, repo_id="local/service", fps=10),
                cameras=(
                    CameraConfig(type="fake", name="head_rgb"),
                    CameraConfig(type="fake", name="wrist_rgb"),
                ),
                sync=SyncConfig(clock="receive", max_control_age_ms=100),
            )
            cameras = (FakeCamera(), FakeCamera(shape=(24, 32, 3), offset=10))
            service = RecorderService(cfg, cameras=cameras)
            service._handle_message(self._sample_message(), time.monotonic_ns())

            service.step()
            with self.assertLogs("robojudo_recorder.service", level="INFO") as logs:
                service._finish_episode(save=True)
            service.close()

            self.assertTrue(any("Episode 1 saved: 1 frames" in message for message in logs.output))
            self.assertTrue(root.joinpath("videos/observation.images.head_rgb/chunk-000/file-000.mp4").exists())
            self.assertTrue(root.joinpath("videos/observation.images.wrist_rgb/chunk-000/file-000.mp4").exists())
            self.assertEqual(pq.read_table(root / "data/chunk-000/file-000.parquet").num_rows, 1)

    def test_interpolates_state_at_camera_timestamp_without_dropping_frame(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir) / "dataset"
            cfg = RecorderConfig(
                control_endpoint=f"inproc://recorder-{uuid.uuid4()}",
                dataset=DatasetConfig(root=root, repo_id="local/service", fps=10),
                camera=CameraConfig(type="fake", name="head_rgb"),
                sync=SyncConfig(clock="source", max_control_age_ms=1),
            )
            camera = FakeCamera()
            service = RecorderService(cfg, camera=camera)
            first = self._sample_message() | {
                "timestamp_ns": 1_000,
                "joint_positions": [0.0, 0.0],
                "joint_position_commands": [1.0, 1.0],
            }
            second = self._sample_message() | {
                "timestamp_ns": 3_000,
                "joint_positions": [2.0, 4.0],
                "joint_position_commands": [3.0, 3.0],
            }
            service._handle_message(first, 1_000)
            service._handle_message(second, 3_000)
            service._active_episode_id = 1
            service._active_task = "test task"
            frame = CameraFrame(
                image=np.zeros(camera.shape, dtype=np.uint8),
                timestamp_ns=2_000,
                sequence=1,
            )
            service._record_frames({"head_rgb": frame})
            service._finish_episode(save=True)
            service.close()

            data = pq.read_table(root / "data/chunk-000/file-000.parquet")
            np.testing.assert_allclose(data["observation.state"].to_pylist()[0], [1.0, 2.0])
            np.testing.assert_allclose(data["action"].to_pylist()[0], [1.0, 1.0, 0.5, 0.0, 0.1, 0.76])

    def test_logs_when_camera_frames_are_missing(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            cfg = RecorderConfig(
                control_endpoint=f"inproc://recorder-{uuid.uuid4()}",
                dataset=DatasetConfig(root=Path(temporary_dir) / "dataset", repo_id="local/service", fps=10),
                camera=CameraConfig(type="fake", name="head_rgb"),
                sync=SyncConfig(clock="receive", max_control_age_ms=100),
            )
            service = RecorderService(cfg, camera=MissingCamera())
            service._handle_message(self._sample_message(), time.monotonic_ns())
            service._camera_missing_since_ns = time.monotonic_ns() - 3_000_000_000

            with self.assertLogs("robojudo_recorder.service", level="WARNING") as logs:
                service.step()
                service._finish_episode(save=True)
            service.close()

            self.assertTrue(any("Waiting for camera frames: head_rgb" in message for message in logs.output))
            self.assertTrue(any("no synchronized camera/control frames" in message for message in logs.output))


if __name__ == "__main__":
    unittest.main()
