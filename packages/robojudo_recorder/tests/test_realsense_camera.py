import sys
import unittest
from pathlib import Path

PACKAGE_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PACKAGE_SRC))

from robojudo_recorder.cameras.realsense import RealSenseCameraSource  # noqa: E402
from robojudo_recorder.config import CameraConfig  # noqa: E402


class TimeoutPipeline:
    def wait_for_frames(self, timeout_ms):
        del timeout_ms
        raise RuntimeError("Frame didn't arrive within 1000")


class TestRealSenseCameraSource(unittest.TestCase):
    def test_retries_transient_frame_timeouts(self):
        camera = RealSenseCameraSource(CameraConfig(type="realsense", options={"serial_number": "123"}))
        camera._pipeline = TimeoutPipeline()

        for _ in range(4):
            self.assertIsNone(camera._capture())
        with self.assertRaisesRegex(RuntimeError, "123 timed out 5 consecutive times"):
            camera._capture()


if __name__ == "__main__":
    unittest.main()
