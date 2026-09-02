import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import zmq

from robojudo.config.x2.env.x2_env_cfg import X2_ARM_JOINT_NAMES
from robojudo.controller.ctrl_cfgs import UpperBodyHandZmqCtrlCfg
from robojudo.controller.omnihand_runtime import (
    OMNIHAND_LEFT_JOINT_NAMES,
    OMNIHAND_RIGHT_JOINT_NAMES,
)
from robojudo.controller.upper_body_hand_zmq_ctrl import UpperBodyHandZmqCtrl


class FakeZmqSocket:
    def __init__(self, messages=None):
        self.messages = list(messages or [])

    def recv_json(self, flags=0):
        del flags
        if not self.messages:
            raise zmq.Again()
        return self.messages.pop(0)


class FakeOmniHandRuntime:
    def __init__(self):
        self.enabled = []
        self.queued = []

    def set_takeover_enabled(self, enabled):
        self.enabled.append(enabled)

    def get_data(self):
        return {
            "joint_state_fresh": True,
            "enabled": True,
            "joint_names": [*OMNIHAND_LEFT_JOINT_NAMES, *OMNIHAND_RIGHT_JOINT_NAMES],
            "joint_positions": np.zeros(24, dtype=np.float32),
        }

    def set_joint_commands(self, left_command, right_command, source_timestamp_ns, frame_id):
        self.queued.append((left_command.copy(), right_command.copy(), source_timestamp_ns, frame_id))
        return np.concatenate((left_command, right_command)).astype(np.float32)


class TestUpperBodyHandZmqCtrl(unittest.TestCase):
    def make_controller(self, messages=None):
        controller = UpperBodyHandZmqCtrl.__new__(UpperBodyHandZmqCtrl)
        controller.cfg_ctrl = SimpleNamespace(timeout_s=0.25)
        controller._arm_joint_names = tuple(X2_ARM_JOINT_NAMES)
        controller._socket = FakeZmqSocket(messages)
        controller._latest_arm_joint_positions = {}
        controller._latest_hand_joint_commands = None
        controller._latest_sync_frame_id = None
        controller._latest_source_timestamp_ns = None
        controller._last_received_at = None
        controller._last_invalid_log_at = float("-inf")
        controller._omnihand = FakeOmniHandRuntime()
        return controller

    @staticmethod
    def synchronized_frame(frame_id=7, *, right_valid=True):
        return {
            "schema_version": 1,
            "type": "synchronized_teleop_frame",
            "mode": "sim2real",
            "frame_id": frame_id,
            "timestamp_ns": 10_000_000_000,
            "robot": "x2",
            "hand_type": "omnihand",
            "arm": {
                "valid": True,
                "joint_names": list(X2_ARM_JOINT_NAMES),
                "qpos": [index / 10 for index in range(len(X2_ARM_JOINT_NAMES))],
            },
            "left_hand": {
                "valid": True,
                "joint_names": list(OMNIHAND_LEFT_JOINT_NAMES),
                "qpos": [0.1] * 12,
            },
            "right_hand": {
                "valid": right_valid,
                "joint_names": list(OMNIHAND_RIGHT_JOINT_NAMES),
                "qpos": [0.2] * 12 if right_valid else [],
            },
        }

    def test_config_uses_dedicated_controller_and_port(self):
        cfg = UpperBodyHandZmqCtrlCfg(joint_names=X2_ARM_JOINT_NAMES)
        self.assertEqual(cfg.ctrl_type, "UpperBodyHandZmqCtrl")
        self.assertEqual(cfg.endpoint, "tcp://127.0.0.1:8560")
        self.assertEqual(cfg.omnihand.transport, "hcan")

    def test_synchronized_frame_drives_arm_and_both_hands_atomically(self):
        controller = self.make_controller([self.synchronized_frame()])
        controller.set_takeover_enabled(True)
        with patch("robojudo.controller.upper_body_hand_zmq_ctrl.time.monotonic", return_value=10.0):
            data = controller.get_data()

        self.assertEqual(controller._omnihand.enabled, [True])
        self.assertEqual(data["frame_id"], 7)
        self.assertEqual(data["joint_positions"]["left_elbow_joint"], 0.3)
        self.assertEqual(controller._omnihand.queued[0][3], 7)
        np.testing.assert_allclose(data["omnihand"]["joint_position_commands"][:12], 0.1)
        self.assertTrue(data["omnihand"]["fresh"])

    def test_incomplete_frame_is_rejected_as_a_whole(self):
        controller = self.make_controller([self.synchronized_frame(right_valid=False)])
        with patch("robojudo.controller.upper_body_hand_zmq_ctrl.time.monotonic", return_value=10.0):
            data = controller.get_data()

        self.assertFalse(data["has_received"])
        self.assertEqual(data["joint_positions"], {})
        self.assertEqual(controller._omnihand.queued, [])

    def test_repeated_frame_id_is_rejected_while_stream_is_fresh(self):
        frame = self.synchronized_frame()
        controller = self.make_controller([frame, frame])
        with patch("robojudo.controller.upper_body_hand_zmq_ctrl.time.monotonic", return_value=10.0):
            self.assertTrue(controller.get_data()["fresh"])
            self.assertTrue(controller.get_data()["fresh"])
        self.assertEqual(len(controller._omnihand.queued), 1)


if __name__ == "__main__":
    unittest.main()
