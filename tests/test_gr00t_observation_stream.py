import threading
import time
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import msgpack
import numpy as np
import zmq
from robojudo_recorder.cameras.base import CameraFrame

from robojudo.controller.ctrl_cfgs import Gr00tCameraCfg, Gr00tZmqCtrlCfg
from robojudo.controller.gr00t_zmq_ctrl import Gr00tZmqCtrl


class FakeCamera:
    def __init__(self):
        self.sequence = 0
        self.closed = False

    def connect(self):
        return None

    def read(self, timeout_ms):
        del timeout_ms
        self.sequence += 1
        return SimpleNamespace(
            image=np.full((8, 12, 3), [20, 80, 140], dtype=np.uint8),
            timestamp_ns=time.monotonic_ns(),
            sequence=self.sequence,
        )

    def close(self):
        self.closed = True


class TestGr00tObservationStream(unittest.TestCase):
    def test_reuses_an_existing_jpeg_without_reencoding(self):
        image = np.full((8, 12, 3), [20, 80, 140], dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", np.ascontiguousarray(image[:, :, ::-1]))
        self.assertTrue(ok)
        payload = encoded.tobytes()
        frame = CameraFrame(
            image=None,
            encoded_image=payload,
            encoding="jpeg",
            image_shape=image.shape,
            timestamp_ns=1,
            sequence=1,
        )

        with patch.object(cv2, "imencode", side_effect=AssertionError("JPEG must not be re-encoded")):
            shape, prepared = Gr00tZmqCtrl._prepare_observation_jpeg(frame, cv2, jpeg_quality=20)

        self.assertEqual(shape, image.shape)
        self.assertEqual(prepared, payload)

    def test_converts_non_jpeg_compressed_frames_to_jpeg(self):
        image = np.full((8, 12, 3), [20, 80, 140], dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", np.ascontiguousarray(image[:, :, ::-1]))
        self.assertTrue(ok)
        frame = CameraFrame(
            image=None,
            encoded_image=encoded.tobytes(),
            encoding="png",
            image_shape=image.shape,
            timestamp_ns=1,
            sequence=1,
        )

        shape, prepared = Gr00tZmqCtrl._prepare_observation_jpeg(frame, cv2, jpeg_quality=90)
        decoded = cv2.imdecode(np.frombuffer(prepared, dtype=np.uint8), cv2.IMREAD_COLOR)

        self.assertEqual(shape, image.shape)
        self.assertEqual(decoded.shape, image.shape)

    def test_publishes_camera_and_current_named_joint_positions(self):
        context = zmq.Context.instance()
        observation_endpoint = f"inproc://gr00t-observation-{uuid.uuid4()}"
        command_endpoint = f"inproc://gr00t-command-{uuid.uuid4()}"
        camera = FakeCamera()
        cfg = Gr00tZmqCtrlCfg(
            endpoint="tcp://127.0.0.1:18559",
            joint_names=["left_arm", "right_arm"],
            observation_enabled=True,
            observation_endpoint="tcp://*:18561",
            observation_profile="test_robot",
            observation_task="test task",
            observation_fps=100,
            camera=Gr00tCameraCfg(type="fake", name="ego_view"),
        )
        cfg.endpoint = command_endpoint
        cfg.observation_endpoint = observation_endpoint
        env = SimpleNamespace(joint_names=["leg", "left_arm", "right_arm"])

        with patch("robojudo_recorder.cameras.create_camera", return_value=camera):
            controller = Gr00tZmqCtrl(cfg, env=env)

        subscriber = context.socket(zmq.SUB)
        subscriber.setsockopt(zmq.LINGER, 0)
        subscriber.setsockopt(zmq.SUBSCRIBE, b"")
        subscriber.connect(observation_endpoint)
        try:
            deadline = time.monotonic() + 1.0
            parts = None
            while time.monotonic() < deadline:
                controller.get_data_with_hook({}, {"dof_pos": np.asarray([0.0, 0.25, -0.5])})
                if subscriber.poll(20, zmq.POLLIN):
                    parts = subscriber.recv_multipart()
                    break
            self.assertIsNotNone(parts)
            header = msgpack.unpackb(parts[0], raw=False)
            image = cv2.cvtColor(
                cv2.imdecode(np.frombuffer(parts[1], dtype=np.uint8), cv2.IMREAD_COLOR),
                cv2.COLOR_BGR2RGB,
            )

            self.assertEqual(header["protocol_version"], 1)
            self.assertIsInstance(header["stream_id"], str)
            self.assertTrue(header["stream_id"])
            self.assertEqual(header["control_session"], 0)
            self.assertFalse(header["takeover_enabled"])
            self.assertEqual(header["profile"], "test_robot")
            self.assertEqual(header["task"], "test task")
            self.assertEqual(header["camera_name"], "ego_view")
            self.assertEqual(header["encoding"], "jpeg")
            self.assertEqual(header["shape"], [8, 12, 3])
            self.assertEqual(header["joint_names"], ["left_arm", "right_arm"])
            np.testing.assert_allclose(header["joint_positions"], [0.25, -0.5])
            self.assertEqual(image.shape, (8, 12, 3))
        finally:
            controller.close()
            subscriber.close(linger=0)

        self.assertTrue(camera.closed)

    def test_takeover_enable_edges_advance_control_session(self):
        controller = Gr00tZmqCtrl.__new__(Gr00tZmqCtrl)
        controller._observation_snapshot_lock = threading.Lock()
        controller._takeover_enabled = False
        controller._control_session = 0
        controller._latest_positions = {}
        controller._latest_locomotion_command = None
        controller._latest_command_stream_id = None
        controller._latest_command_session = None
        controller._last_received_at = None

        controller.set_takeover_enabled(False)
        self.assertTrue(controller.set_takeover_enabled(True))
        self.assertFalse(controller.set_takeover_enabled(True))
        self.assertTrue(controller._takeover_enabled)
        self.assertEqual(controller._control_session, 1)

        self.assertTrue(controller.set_takeover_enabled(False))
        controller.set_takeover_enabled(True)
        self.assertEqual(controller._control_session, 2)


if __name__ == "__main__":
    unittest.main()
