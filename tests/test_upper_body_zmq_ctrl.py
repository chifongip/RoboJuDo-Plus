import unittest
from types import SimpleNamespace
from unittest.mock import patch

import zmq

from robojudo.config.x2.env.x2_env_cfg import X2_ARM_JOINT_NAMES
from robojudo.controller.ctrl_cfgs import UpperBodyZmqCtrlCfg
from robojudo.controller.upper_body_zmq_ctrl import UpperBodyZmqCtrl


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


class TestUpperBodyZmqCtrl(unittest.TestCase):
    def make_controller(self, messages=None):
        controller = UpperBodyZmqCtrl.__new__(UpperBodyZmqCtrl)
        controller.cfg_ctrl = SimpleNamespace(timeout_s=0.25)
        controller._joint_names = set(X2_ARM_JOINT_NAMES)
        controller._socket = FakeZmqSocket(messages)
        controller._latest_positions = {}
        controller._last_received_at = None
        controller._last_invalid_log_at = float("-inf")
        return controller

    def test_config_requires_unique_named_joints(self):
        cfg = UpperBodyZmqCtrlCfg(joint_names=X2_ARM_JOINT_NAMES)
        self.assertEqual(cfg.endpoint, "tcp://127.0.0.1:8559")
        self.assertEqual(cfg.timeout_s, 0.25)
        self.assertEqual(cfg.ema_alpha, 0.95)

        with self.assertRaises(ValueError):
            UpperBodyZmqCtrlCfg(joint_names=[])
        with self.assertRaises(ValueError):
            UpperBodyZmqCtrlCfg(joint_names=[X2_ARM_JOINT_NAMES[0]] * 2)

    def test_named_partial_updates_are_merged(self):
        controller = self.make_controller(
            [
                {"positions": {"left_shoulder_pitch_joint": 0.2}, "source": "test"},
                {"positions": {"right_elbow_joint": -0.6}},
            ]
        )
        with patch("robojudo.controller.upper_body_zmq_ctrl.time.monotonic", return_value=10.0):
            data = controller.get_data()

        self.assertTrue(data["has_received"])
        self.assertTrue(data["fresh"])
        self.assertEqual(
            data["joint_positions"],
            {"left_shoulder_pitch_joint": 0.2, "right_elbow_joint": -0.6},
        )

    def test_invalid_message_is_atomic_and_does_not_refresh_timeout(self):
        controller = self.make_controller(
            [
                {"positions": {"left_elbow_joint": -0.4}},
                {"positions": {"left_elbow_joint": float("nan")}},
                {"positions": {"unknown_joint": 0.1}},
            ]
        )
        controller._latest_positions = {"right_elbow_joint": -0.3}
        controller._last_received_at = 1.0
        with patch("robojudo.controller.upper_body_zmq_ctrl.time.monotonic", return_value=2.0):
            data = controller.get_data()

        self.assertTrue(data["fresh"])
        self.assertEqual(
            data["joint_positions"],
            {"right_elbow_joint": -0.3, "left_elbow_joint": -0.4},
        )
        self.assertEqual(controller._last_received_at, 2.0)

        controller._socket = FakeZmqSocket([{"positions": {"left_elbow_joint": True}}])
        with patch("robojudo.controller.upper_body_zmq_ctrl.time.monotonic", return_value=2.5):
            stale = controller.get_data()
        self.assertFalse(stale["fresh"])
        self.assertEqual(controller._last_received_at, 2.0)

    def test_stream_becomes_stale_after_timeout(self):
        controller = self.make_controller([{"positions": {"left_wrist_yaw_joint": 0.1}}])
        with patch("robojudo.controller.upper_body_zmq_ctrl.time.monotonic", return_value=4.0):
            self.assertTrue(controller.get_data()["fresh"])
        with patch("robojudo.controller.upper_body_zmq_ctrl.time.monotonic", return_value=4.251):
            data = controller.get_data()
        self.assertTrue(data["has_received"])
        self.assertFalse(data["fresh"])
        self.assertAlmostEqual(data["age_s"], 0.251)


if __name__ == "__main__":
    unittest.main()
