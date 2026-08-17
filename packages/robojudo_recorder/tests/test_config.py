import sys
import tempfile
import unittest
from pathlib import Path

PACKAGE_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PACKAGE_SRC))

from robojudo_recorder.config import load_config  # noqa: E402


class TestRecorderConfig(unittest.TestCase):
    def _load(self, contents: str):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "recorder.yaml"
            path.write_text(contents)
            return load_config(path)

    def test_loads_legacy_single_camera(self):
        cfg = self._load(
            """
dataset:
  root: ~/record_data/test
  repo_id: local/test
  fps: 30
camera:
  type: opencv
  name: head_rgb
  device: 0
  fps: 30
"""
        )

        self.assertEqual(len(cfg.cameras), 1)
        self.assertEqual(cfg.camera, cfg.cameras[0])
        self.assertEqual(cfg.cameras[0].options["device"], 0)

    def test_loads_multiple_cameras(self):
        cfg = self._load(
            """
dataset:
  root: ~/record_data/test
  repo_id: local/test
  fps: 30
cameras:
  - type: ros2
    name: head_rgb
    topic: /head/image/compressed
    fps: 30
  - type: ros2
    name: wrist_rgb
    topic: /wrist/image/compressed
    fps: 30
"""
        )

        self.assertEqual([camera.name for camera in cfg.cameras], ["head_rgb", "wrist_rgb"])
        self.assertEqual(cfg.cameras[1].options["topic"], "/wrist/image/compressed")

    def test_rejects_duplicate_camera_names(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            self._load(
                """
dataset:
  root: ~/record_data/test
  repo_id: local/test
  fps: 30
cameras:
  - type: opencv
    name: rgb
  - type: opencv
    name: rgb
"""
            )

if __name__ == "__main__":
    unittest.main()
