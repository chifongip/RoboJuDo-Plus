import json
import unittest
from unittest.mock import patch

import zmq

from robojudo.config.x2.env.x2_env_cfg import X2_ARM_JOINT_NAMES
from robojudo.controller.ctrl_cfgs import UpperBodyZmqCtrlCfg
from robojudo.controller.upper_body_zmq_ctrl import UpperBodyZmqCtrl


class FakeZmqSocket:
    def __init__(self, messages=None):
        self.messages = list(messages or [])
        self.more = False
        self.close_count = 0

    def setsockopt(self, *args):
        pass

    def getsockopt(self, option):
        self.assert_option = option
        return self.more

    def connect(self, endpoint):
        self.endpoint = endpoint

    def close(self, linger=0):
        self.close_count += 1

    def recv(self, flags=0):
        if not self.messages:
            raise zmq.Again()
        message = self.messages.pop(0)
        if isinstance(message, Exception):
            raise message
        if isinstance(message, tuple):
            message, self.more = message
        else:
            self.more = False
        return message if isinstance(message, bytes) else json.dumps(message).encode()


class TestUpperBodyZmqCtrl(unittest.TestCase):
    def make_controller(self, messages=None):
        socket = FakeZmqSocket(messages)
        with patch("robojudo.controller.upper_body_zmq_ctrl.zmq.Context.instance") as context:
            context.return_value.socket.return_value = socket
            return UpperBodyZmqCtrl(UpperBodyZmqCtrlCfg(joint_names=X2_ARM_JOINT_NAMES))

    def test_config_requires_unique_named_joints(self):
        cfg = UpperBodyZmqCtrlCfg(joint_names=X2_ARM_JOINT_NAMES)
        self.assertEqual(cfg.endpoint, "tcp://127.0.0.1:8559")
        self.assertEqual(cfg.timeout_s, 0.25)
        self.assertEqual(cfg.ema_alpha, 0.95)
        self.assertIsNone(cfg.upper_body_default_pose)

        for timeout in (float("inf"), float("nan"), 0.0):
            with self.assertRaises(ValueError):
                UpperBodyZmqCtrlCfg(joint_names=X2_ARM_JOINT_NAMES, timeout_s=timeout)
        with self.assertRaises(ValueError):
            UpperBodyZmqCtrlCfg(joint_names=X2_ARM_JOINT_NAMES, endpoint="tcp://")
        with self.assertRaises(ValueError):
            UpperBodyZmqCtrlCfg(joint_names=[])
        with self.assertRaises(ValueError):
            UpperBodyZmqCtrlCfg(joint_names=[X2_ARM_JOINT_NAMES[0]] * 2)
        with self.assertRaises(ValueError):
            UpperBodyZmqCtrlCfg(
                joint_names=["left", "right"],
                upper_body_default_pose=[0.0],
            )

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
        controller._joint_received_at = {"right_elbow_joint": 1.0}
        with patch("robojudo.controller.upper_body_zmq_ctrl.time.monotonic", return_value=2.0):
            data = controller.get_data()

        self.assertTrue(data["fresh"])
        self.assertEqual(
            data["joint_positions"],
            {"left_elbow_joint": -0.4},
        )
        self.assertEqual(controller._last_received_at, 2.0)

        controller._socket = FakeZmqSocket([{"positions": {"left_elbow_joint": True}}])
        with patch("robojudo.controller.upper_body_zmq_ctrl.time.monotonic", return_value=2.2):
            controller.get_data()
        with patch("robojudo.controller.upper_body_zmq_ctrl.time.monotonic", return_value=2.26):
            stale = controller.get_data()
        self.assertFalse(stale["fresh"])
        self.assertEqual(controller._last_received_at, 2.0)

    def test_stream_becomes_stale_after_timeout(self):
        controller = self.make_controller([{"positions": {"left_wrist_yaw_joint": 0.1}}])
        with patch("robojudo.controller.upper_body_zmq_ctrl.time.monotonic", return_value=4.0):
            self.assertTrue(controller.get_data()["fresh"])
        with patch("robojudo.controller.upper_body_zmq_ctrl.time.monotonic", return_value=4.251):
            data = controller.get_data()
        self.assertFalse(data["fresh"])
        self.assertEqual(data["joint_positions"], {})

    def test_partial_joint_updates_have_independent_expiry(self):
        controller = self.make_controller([{"positions": {"left_elbow_joint": -0.4}}])
        with patch("robojudo.controller.upper_body_zmq_ctrl.time.monotonic", return_value=1.0):
            controller.get_data()
        controller._socket.messages.append({"positions": {"right_elbow_joint": -0.2}})
        with patch("robojudo.controller.upper_body_zmq_ctrl.time.monotonic", return_value=1.2):
            self.assertEqual(len(controller.get_data()["joint_positions"]), 2)
        with patch("robojudo.controller.upper_body_zmq_ctrl.time.monotonic", return_value=1.26):
            data = controller.get_data()
        self.assertTrue(data["fresh"])
        self.assertEqual(data["joint_positions"], {"right_elbow_joint": -0.2})

    def test_malformed_frames_never_refresh_the_lease(self):
        good = b'{"positions":{"left_elbow_joint":-0.4}}'
        invalid = (
            b'{"positions":{"left_elbow_joint":-0.2,"left_elbow_joint":-0.3}}',
            b'{"positions":{},"positions":{"left_elbow_joint":-0.3}}',
            b'{"positions":{"left_elbow_joint":NaN}}',
            b'{"positions":{"left_elbow_joint":' + b"9" * 400 + b"}}",
            b"\xff",
            b"[" * 1100 + b"]" * 1100,
            b"x" * 16385,
            (b"topic", True),
            (good, False),
        )
        controller = self.make_controller([good])
        with patch("robojudo.controller.upper_body_zmq_ctrl.time.monotonic", return_value=1.0):
            controller.get_data()
        for frame in invalid:
            controller._socket.messages.append(frame)
        with patch("robojudo.controller.upper_body_zmq_ctrl.time.monotonic", return_value=1.1):
            data = controller.get_data()
        self.assertEqual(data["joint_positions"], {"left_elbow_joint": -0.4})
        self.assertEqual(controller._last_received_at, 1.0)

    def test_backlog_is_bounded_and_discarded_until_new_packet_after_empty_queue(self):
        controller = self.make_controller([{"positions": {"left_elbow_joint": -0.4}}] * 250)
        with patch("robojudo.controller.upper_body_zmq_ctrl.time.monotonic", return_value=1.0):
            self.assertFalse(controller.get_data()["fresh"])
        self.assertEqual(len(controller._socket.messages), 150)
        with patch("robojudo.controller.upper_body_zmq_ctrl.time.monotonic", return_value=1.02):
            self.assertFalse(controller.get_data()["fresh"])
        with patch("robojudo.controller.upper_body_zmq_ctrl.time.monotonic", return_value=1.04):
            self.assertFalse(controller.get_data()["fresh"])
        controller._socket.messages.append({"positions": {"left_elbow_joint": -0.1}})
        with patch("robojudo.controller.upper_body_zmq_ctrl.time.monotonic", return_value=1.06):
            self.assertEqual(controller.get_data()["joint_positions"], {"left_elbow_joint": -0.1})

    def test_control_stall_and_reset_discard_previously_queued_commands(self):
        controller = self.make_controller([{"positions": {"left_elbow_joint": -0.4}}])
        with patch("robojudo.controller.upper_body_zmq_ctrl.time.monotonic", return_value=1.0):
            controller.get_data()
        controller._socket.messages.append({"positions": {"left_elbow_joint": -0.8}})
        with patch("robojudo.controller.upper_body_zmq_ctrl.time.monotonic", return_value=2.0):
            self.assertFalse(controller.get_data()["fresh"])
        controller._socket.messages.append({"positions": {"left_elbow_joint": -0.1}})
        with patch("robojudo.controller.upper_body_zmq_ctrl.time.monotonic", return_value=2.02):
            self.assertTrue(controller.get_data()["fresh"])
        controller.reset()
        controller._socket.messages.append({"positions": {"left_elbow_joint": -0.9}})
        with patch("robojudo.controller.upper_body_zmq_ctrl.time.monotonic", return_value=2.04):
            self.assertFalse(controller.get_data()["fresh"])

    def test_multipart_tail_cannot_become_a_command_across_receive_budgets(self):
        good = b'{"positions":{"left_elbow_joint":-0.4}}'
        controller = self.make_controller([(b"part", True)] * 100 + [(good, False)])
        with patch("robojudo.controller.upper_body_zmq_ctrl.time.monotonic", return_value=1.0):
            self.assertFalse(controller.get_data()["fresh"])
        with patch("robojudo.controller.upper_body_zmq_ctrl.time.monotonic", return_value=1.02):
            self.assertFalse(controller.get_data()["fresh"])
        controller._socket.messages.append(good)
        with patch("robojudo.controller.upper_body_zmq_ctrl.time.monotonic", return_value=1.04):
            self.assertTrue(controller.get_data()["fresh"])

    def test_transport_fault_and_close_clear_targets_without_refreshing(self):
        controller = self.make_controller([{"positions": {"left_elbow_joint": -0.4}}])
        with patch("robojudo.controller.upper_body_zmq_ctrl.time.monotonic", return_value=1.0):
            controller.get_data()
        controller._socket.messages.append(zmq.ZMQError(zmq.ETERM))
        with patch("robojudo.controller.upper_body_zmq_ctrl.time.monotonic", return_value=1.02):
            data = controller.get_data()
        self.assertFalse(data["fresh"])
        self.assertIsNotNone(data["transport_error"])
        controller.close()
        controller.close()
        self.assertEqual(controller._socket.close_count, 1)
        self.assertFalse(controller.get_data()["fresh"])


if __name__ == "__main__":
    unittest.main()
