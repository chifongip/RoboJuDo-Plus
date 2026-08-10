import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import zmq

from robojudo.controller.ctrl_cfgs import VelocityZmqCtrlCfg
from robojudo.controller.velocity_zmq_ctrl import VelocityZmqCtrl


class FakeZmqSocket:
    def __init__(self, messages=None):
        self.messages = list(messages or [])

    def recv_json(self, flags=0):
        del flags
        if not self.messages:
            raise zmq.Again()
        message = self.messages.pop(0)
        if isinstance(message, Exception):
            raise message
        return message


def twist(x=0.0, y=0.0, yaw=0.0):
    return {
        "linear": {"x": x, "y": y, "z": 0.0},
        "angular": {"x": 0.0, "y": 0.0, "z": yaw},
    }


class TestVelocityZmqCtrl(unittest.TestCase):
    def make_controller(self, messages=None):
        controller = VelocityZmqCtrl.__new__(VelocityZmqCtrl)
        controller.cfg_ctrl = SimpleNamespace(timeout_s=0.25)
        controller._socket = FakeZmqSocket(messages)
        controller._linear_velocity = np.zeros(3, dtype=np.float32)
        controller._angular_velocity = np.zeros(3, dtype=np.float32)
        controller._last_received_at = None
        controller._last_invalid_log_at = float("-inf")
        return controller

    def test_config_defaults_and_validation(self):
        cfg = VelocityZmqCtrlCfg()
        self.assertEqual(cfg.endpoint, "tcp://127.0.0.1:8558")
        self.assertEqual(cfg.timeout_s, 0.25)
        with self.assertRaises(ValueError):
            VelocityZmqCtrlCfg(endpoint="ipc:///tmp/velocity")
        with self.assertRaises(ValueError):
            VelocityZmqCtrlCfg(endpoint="tcp://")
        with self.assertRaises(ValueError):
            VelocityZmqCtrlCfg(timeout_s=0.0)

    def test_latest_valid_twist_is_returned(self):
        controller = self.make_controller([twist(x=0.2), twist(x=0.4, y=-0.1, yaw=0.3)])
        with patch("robojudo.controller.velocity_zmq_ctrl.time.monotonic", return_value=10.0):
            data = controller.get_data()

        np.testing.assert_allclose(data["linear_velocity"], [0.4, -0.1, 0.0])
        np.testing.assert_allclose(data["angular_velocity"], [0.0, 0.0, 0.3])
        self.assertTrue(data["has_received"])
        self.assertTrue(data["fresh"])

    def test_invalid_message_is_atomic_and_does_not_refresh_timeout(self):
        controller = self.make_controller([twist(x=0.2), {"linear": {"x": float("nan"), "y": 0, "z": 0}}])
        with patch("robojudo.controller.velocity_zmq_ctrl.time.monotonic", return_value=2.0):
            first = controller.get_data()
        np.testing.assert_allclose(first["linear_velocity"], [0.2, 0.0, 0.0])
        self.assertEqual(controller._last_received_at, 2.0)

        controller._socket = FakeZmqSocket([twist(x=True)])
        with patch("robojudo.controller.velocity_zmq_ctrl.time.monotonic", return_value=2.251):
            stale = controller.get_data()
        self.assertFalse(stale["fresh"])
        self.assertEqual(controller._last_received_at, 2.0)

    def test_timeout_boundary_is_fresh(self):
        controller = self.make_controller([twist(y=0.3)])
        with patch("robojudo.controller.velocity_zmq_ctrl.time.monotonic", return_value=4.0):
            self.assertTrue(controller.get_data()["fresh"])
        with patch("robojudo.controller.velocity_zmq_ctrl.time.monotonic", return_value=4.25):
            self.assertTrue(controller.get_data()["fresh"])
        with patch("robojudo.controller.velocity_zmq_ctrl.time.monotonic", return_value=4.251):
            self.assertFalse(controller.get_data()["fresh"])


if __name__ == "__main__":
    unittest.main()
