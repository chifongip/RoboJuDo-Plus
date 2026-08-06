import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PACKAGE_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PACKAGE_SRC))

from robojudo_recorder.cameras.ros2 import Ros2CompressedCameraSource  # noqa: E402
from robojudo_recorder.config import CameraConfig  # noqa: E402


class FakeCv2:
    IMREAD_UNCHANGED = 1
    COLOR_GRAY2RGB = 2
    COLOR_BGRA2RGB = 3
    COLOR_BGR2RGB = 4

    def __init__(self, decoded):
        self.decoded = decoded

    def imdecode(self, payload, flags):
        self.payload = payload
        self.flags = flags
        return self.decoded.copy()

    def cvtColor(self, image, conversion):
        if conversion == self.COLOR_GRAY2RGB:
            return np.repeat(image[:, :, None], 3, axis=2)
        if conversion == self.COLOR_BGRA2RGB:
            return image[:, :, [2, 1, 0]]
        if conversion == self.COLOR_BGR2RGB:
            return image[:, :, ::-1]
        raise AssertionError(f"unexpected conversion {conversion}")


class FakeSocket:
    def __init__(self):
        self.options = []
        self.endpoint = None
        self.messages = []
        self.closed = False

    def setsockopt(self, option, value):
        self.options.append((option, value))

    def bind(self, endpoint):
        self.endpoint = endpoint

    def getsockopt_string(self, option):
        del option
        return "tcp://127.0.0.1:54321"

    def poll(self, timeout_ms, event):
        del timeout_ms, event
        return bool(self.messages)

    def recv_multipart(self):
        return self.messages.pop(0)

    def close(self, *, linger):
        del linger
        self.closed = True


class FakeContext:
    def __init__(self, socket):
        self._socket = socket

    def socket(self, socket_type):
        del socket_type
        return self._socket


class FakeProcess:
    def __init__(self):
        self.return_code = None
        self.terminated = False

    def poll(self):
        return self.return_code

    def terminate(self):
        self.terminated = True
        self.return_code = -15

    def wait(self, *, timeout):
        del timeout
        return self.return_code


class TestRos2CompressedCameraSource(unittest.TestCase):
    def test_launches_bridge_and_returns_rgb_frame(self):
        bgr = np.asarray([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
        cv2 = FakeCv2(bgr)
        socket = FakeSocket()
        process = FakeProcess()
        camera = Ros2CompressedCameraSource(
            CameraConfig(
                type="ros2",
                options={
                    "topic": "/camera/rgb/compressed",
                    "qos_reliability": "reliable",
                    "qos_depth": 3,
                },
            )
        )
        camera._context = FakeContext(socket)

        with (
            patch.dict(sys.modules, {"cv2": cv2}),
            patch("robojudo_recorder.cameras.ros2.subprocess.Popen", return_value=process) as popen,
        ):
            camera.connect()

        command = popen.call_args.args[0]
        self.assertEqual(command[0], "/usr/bin/python3")
        self.assertIn("ros2_bridge.py", command[1])
        self.assertEqual(command[command.index("--topic") + 1], "/camera/rgb/compressed")
        self.assertEqual(command[command.index("--qos-reliability") + 1], "reliable")
        self.assertEqual(command[command.index("--qos-depth") + 1], "3")

        socket.messages.append(
            [json.dumps({"sequence": 7, "timestamp_ns": 123456}).encode(), b"compressed image"]
        )
        frame = camera.read(timeout_ms=25)
        self.assertEqual(camera.shape, (1, 2, 3))
        np.testing.assert_array_equal(frame.image, bgr[:, :, ::-1])
        self.assertEqual(frame.timestamp_ns, 123456)
        self.assertEqual(frame.sequence, 7)

        camera.close()
        self.assertTrue(process.terminated)
        self.assertTrue(socket.closed)

    def test_reports_bridge_exit(self):
        camera = Ros2CompressedCameraSource(
            CameraConfig(type="ros2", options={"topic": "/camera/rgb/compressed"})
        )
        camera._socket = FakeSocket()
        camera._process = FakeProcess()
        camera._process.return_code = 1
        with self.assertRaisesRegex(RuntimeError, "status 1"):
            camera.read(timeout_ms=0)

    def test_rejects_unexpected_frame_shape(self):
        camera = Ros2CompressedCameraSource(
            CameraConfig(
                type="ros2",
                options={"topic": "/camera/rgb/compressed", "width": 640, "height": 480},
            )
        )
        camera._cv2 = FakeCv2(np.zeros((240, 320, 3), dtype=np.uint8))
        header = json.dumps({"sequence": 1, "timestamp_ns": 123}).encode()
        with self.assertRaisesRegex(ValueError, "expected"):
            camera._decode_frame(header, b"compressed image")

    def test_validates_configuration(self):
        with self.assertRaisesRegex(ValueError, "topic"):
            Ros2CompressedCameraSource(CameraConfig(type="ros2"))
        with self.assertRaisesRegex(ValueError, "specified together"):
            Ros2CompressedCameraSource(
                CameraConfig(type="ros2", options={"topic": "/camera", "width": 640})
            )
        with self.assertRaisesRegex(ValueError, "qos_reliability"):
            Ros2CompressedCameraSource(
                CameraConfig(type="ros2", options={"topic": "/camera", "qos_reliability": "unknown"})
            )


if __name__ == "__main__":
    unittest.main()
