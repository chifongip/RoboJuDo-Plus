import json
import sys
import tempfile
import threading
import time
import unittest
import uuid
from collections import deque
from pathlib import Path
from unittest.mock import patch

PACKAGE_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PACKAGE_SRC))

from robojudo_recorder.cameras.base import CameraFrame, CameraSource  # noqa: E402
from robojudo_recorder.cameras.threaded import ThreadedCameraSource  # noqa: E402
from robojudo_recorder.config import CameraConfig, DatasetConfig, RecorderConfig, SyncConfig  # noqa: E402
from robojudo_recorder.raw import RawEpisodeWriter  # noqa: E402
from robojudo_recorder.service import RecorderService  # noqa: E402


def encoded_frame(sequence: int, camera_value: int = 0) -> CameraFrame:
    now = time.monotonic_ns()
    return CameraFrame(
        image=None,
        timestamp_ns=now,
        sequence=sequence,
        encoded_image=f"jpeg-{camera_value}-{sequence}".encode(),
        encoding="jpeg",
        image_shape=(4, 6, 3),
    )


class BatchCamera(CameraSource):
    def __init__(self, frames=()):
        self.frames = deque(frames)

    @property
    def shape(self):
        return (4, 6, 3)

    def connect(self):
        pass

    def read(self, timeout_ms):
        del timeout_ms
        return self.frames.popleft() if self.frames else None

    def read_batch(self, max_frames):
        frames = []
        while self.frames and len(frames) < max_frames:
            frames.append(self.frames.popleft())
        return frames

    def close(self):
        pass


class ManualThreadedCamera(ThreadedCameraSource):
    def __init__(self):
        super().__init__((1, 1, 3))

    def _open(self):
        pass

    def _capture(self):
        return None

    def _close(self):
        pass


class TestAsyncCameraPipeline(unittest.TestCase):
    @staticmethod
    def config(temporary_dir, names=("head_rgb",), capacity=8):
        return RecorderConfig(
            control_endpoint=f"inproc://async-camera-{uuid.uuid4()}",
            dataset=DatasetConfig(
                root=Path(temporary_dir) / "dataset",
                raw_root=Path(temporary_dir) / "raw",
                repo_id="local/async-camera",
                fps=30,
            ),
            cameras=tuple(CameraConfig(type="fake", name=name) for name in names),
            sync=SyncConfig(clock="receive", poll_timeout_ms=1, pending_frame_capacity=capacity),
        )

    def test_threaded_source_preserves_fifo_and_drops_oldest_at_capacity(self):
        camera = ManualThreadedCamera()
        camera.set_pending_capacity(2)
        with camera._condition:
            camera._pending.extend([encoded_frame(1), encoded_frame(2), encoded_frame(3)])
            while len(camera._pending) > camera._pending_capacity:
                camera._pending.popleft()

        self.assertEqual([frame.sequence for frame in camera.read_batch(10)], [2, 3])

    def test_camera_readiness_does_not_require_same_poll_cycle(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            names = ("head_rgb", "wrist_rgb")
            service = RecorderService(self.config(temporary_dir, names), cameras=(BatchCamera(), BatchCamera()))

            service._update_camera_status({"head_rgb": encoded_frame(1), "wrist_rgb": None})
            self.assertFalse(service._camera_stream_ready)
            service._update_camera_status({"head_rgb": None, "wrist_rgb": encoded_frame(1)})
            self.assertTrue(service._camera_stream_ready)
            service.close()

    def test_flushes_each_camera_worker_before_commit(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            names = ("head_rgb", "wrist_rgb")
            cameras = (
                BatchCamera([encoded_frame(index, 1) for index in range(1, 5)]),
                BatchCamera([encoded_frame(index, 2) for index in range(1, 5)]),
            )
            cfg = self.config(temporary_dir, names)
            service = RecorderService(cfg, cameras=cameras)
            service._handle_message(
                {"kind": "episode_start", "episode_id": 1, "task": "test", "timestamp_ns": 0},
                0,
            )
            service.step()
            service._finish_episode(save=True)
            service.close()

            episode = next((cfg.dataset.raw_root / "episodes").iterdir())
            manifest = json.loads((episode / "manifest.json").read_text())
            self.assertEqual(manifest["frame_counts"], {"head_rgb": 4, "wrist_rgb": 4})
            self.assertEqual(len(list((episode / "cameras/head_rgb").glob("*.jpg"))), 4)
            self.assertEqual(len(list((episode / "cameras/wrist_rgb").glob("*.jpg"))), 4)

    def test_writer_failure_discards_episode(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            cfg = self.config(temporary_dir)
            service = RecorderService(cfg, camera=BatchCamera([encoded_frame(1)]))
            service._handle_message(
                {"kind": "episode_start", "episode_id": 1, "task": "test", "timestamp_ns": 0},
                0,
            )
            with patch("robojudo_recorder.raw.RawEpisodeWriter.add_frame", side_effect=OSError("disk full")):
                service.step()
                with self.assertRaisesRegex(RuntimeError, "camera writer failed"):
                    service._finish_episode(save=True)
            service.close()

            self.assertFalse(any((cfg.dataset.raw_root / ".pending").glob("*")))
            self.assertFalse((cfg.dataset.raw_root / "episodes").exists())

    def test_camera_writers_run_in_parallel_and_preserve_commit(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            names = ("head_rgb", "wrist_rgb")
            cameras = (BatchCamera([encoded_frame(1, 1)]), BatchCamera([encoded_frame(1, 2)]))
            cfg = self.config(temporary_dir, names)
            service = RecorderService(cfg, cameras=cameras)
            service._handle_message(
                {"kind": "episode_start", "episode_id": 1, "task": "test", "timestamp_ns": 0},
                0,
            )
            barrier = threading.Barrier(2)
            original_add_frame = RawEpisodeWriter.add_frame

            def synchronized_add_frame(writer, camera_name, frame):
                barrier.wait(timeout=1)
                original_add_frame(writer, camera_name, frame)

            with patch.object(RawEpisodeWriter, "add_frame", new=synchronized_add_frame):
                service.step()
                service._finish_episode(save=True)
            service.close()

            episode = next((cfg.dataset.raw_root / "episodes").iterdir())
            manifest = json.loads((episode / "manifest.json").read_text())
            self.assertEqual(manifest["frame_counts"], {"head_rgb": 1, "wrist_rgb": 1})

    def test_empty_episode_is_not_committed(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            cfg = self.config(temporary_dir)
            service = RecorderService(cfg, camera=BatchCamera())
            now = time.monotonic_ns()
            service._handle_message(
                {"kind": "episode_start", "episode_id": 1, "task": "test", "timestamp_ns": now},
                now,
            )
            service._finish_episode(save=True)
            service.close()

            self.assertFalse((cfg.dataset.raw_root / "episodes").exists())


if __name__ == "__main__":
    unittest.main()
