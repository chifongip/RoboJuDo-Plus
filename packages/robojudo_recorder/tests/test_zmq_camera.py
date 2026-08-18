import sys
import unittest
from pathlib import Path

import cv2
import msgpack
import numpy as np

PACKAGE_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PACKAGE_SRC))

from robojudo_recorder.cameras.zmq_camera import ZmqCameraSource  # noqa: E402
from robojudo_recorder.config import CameraConfig  # noqa: E402


class FakeSocket:
    def __init__(self, message):
        self.message = message

    def poll(self, timeout_ms, event):
        del timeout_ms, event
        return self.message is not None

    def recv_multipart(self):
        message = self.message
        self.message = None
        return message


class TestZmqCameraSource(unittest.TestCase):
    def test_reads_gr00t_msgpack_jpeg_and_infers_shape(self):
        image = np.full((8, 12, 3), [20, 80, 140], dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image[:, :, ::-1])
        self.assertTrue(ok)
        header = {
            "sequence": 7,
            "timestamp_ns": 123456,
            "encoding": "jpeg",
            "shape": list(image.shape),
            "joint_names": ["left_arm", "right_arm"],
            "joint_positions": [0.1, -0.2],
        }
        camera = ZmqCameraSource(
            CameraConfig(
                type="zmq",
                options={"endpoint": "tcp://127.0.0.1:8561", "timestamp_mode": "source"},
            )
        )
        camera._socket = FakeSocket([msgpack.packb(header, use_bin_type=True), encoded.tobytes()])

        frame = camera.read(timeout_ms=0)

        self.assertEqual(camera.shape, (8, 12, 3))
        self.assertEqual(frame.sequence, 7)
        self.assertEqual(frame.timestamp_ns, 123456)
        self.assertIsNone(frame.image)
        self.assertEqual(frame.encoded_image, encoded.tobytes())
        self.assertEqual(frame.shape, image.shape)


if __name__ == "__main__":
    unittest.main()
