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
    def __init__(self):
        self.sequence = 0
        self.connected = False

    @property
    def shape(self):
        return 48, 64, 3

    def connect(self):
        self.connected = True

    def read(self, timeout_ms):
        del timeout_ms
        self.sequence += 1
        return CameraFrame(
            image=np.full(self.shape, self.sequence, dtype=np.uint8),
            timestamp_ns=time.monotonic_ns(),
            sequence=self.sequence,
        )

    def close(self):
        self.connected = False


class TestRecorderService(unittest.TestCase):
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
                    {
                        "kind": "sample",
                        "episode_id": 1,
                        "task": "test task",
                        "robot_type": "g1",
                        "timestamp_ns": time.monotonic_ns(),
                        "joint_names": ["left_arm", "right_arm"],
                        "joint_positions": [0.1, 0.2],
                        "joint_position_commands": [0.3, 0.4],
                        "velocity_height_command": [0.5, 0.0, 0.1, 0.76],
                    },
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


if __name__ == "__main__":
    unittest.main()
