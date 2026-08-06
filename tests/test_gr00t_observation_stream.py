import time
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import msgpack
import numpy as np
import zmq

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


if __name__ == "__main__":
    unittest.main()
