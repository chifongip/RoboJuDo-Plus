import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import zmq

from robojudo.config.g1.policy.g1_locomanipulation_policy_cfg import G1Locomanipulation23ObsDoF
from robojudo.controller.casia_hand_runtime import CASIA_LEFT_JOINT_NAMES, CASIA_RIGHT_JOINT_NAMES
from robojudo.controller.ctrl_cfgs import UpperBodyCasiaHandZmqCtrlCfg
from robojudo.controller.upper_body_casia_hand_zmq_ctrl import UpperBodyCasiaHandZmqCtrl

ARM_JOINT_NAMES = tuple(G1Locomanipulation23ObsDoF().joint_names[13:])


class FakeZmqSocket:
    def __init__(self, messages=None):
        self.messages = list(messages or [])

    def recv_json(self, flags=0):
        del flags
        if not self.messages:
            raise zmq.Again()
        return self.messages.pop(0)


class FakeCasiaHandRuntime:
    def __init__(self):
        self.enabled = []
        self.queued = []

    def set_takeover_enabled(self, enabled):
        self.enabled.append(enabled)

    def get_data(self):
        return {
            "joint_state_fresh": True,
            "enabled": True,
            "joint_names": [*CASIA_LEFT_JOINT_NAMES, *CASIA_RIGHT_JOINT_NAMES],
            "joint_positions": np.zeros(20, dtype=np.float32),
        }

    def set_joint_commands(self, left_command, right_command, source_timestamp_ns, frame_id):
        self.queued.append((left_command.copy(), right_command.copy(), source_timestamp_ns, frame_id))
        return np.concatenate((left_command, right_command)).astype(np.float32)


class TestUpperBodyCasiaHandZmqCtrl(unittest.TestCase):
    def make_controller(self, messages=None):
        controller = UpperBodyCasiaHandZmqCtrl.__new__(UpperBodyCasiaHandZmqCtrl)
        controller.cfg_ctrl = SimpleNamespace(timeout_s=0.25)
        controller._arm_joint_names = ARM_JOINT_NAMES
        controller._socket = FakeZmqSocket(messages)
        controller._latest_arm_joint_positions = {}
        controller._latest_hand_joint_commands = None
        controller._latest_sync_frame_id = None
        controller._latest_source_timestamp_ns = None
        controller._last_received_at = None
        controller._last_invalid_log_at = float("-inf")
        controller._casia_hand = FakeCasiaHandRuntime()
        return controller

    @staticmethod
    def synchronized_frame(frame_id=7, *, hand_type="casia", right_valid=True, physical_schema=True):
        left_names = list(CASIA_LEFT_JOINT_NAMES)
        right_names = list(CASIA_RIGHT_JOINT_NAMES)
        if not physical_schema:
            left_names += ["left_distal_0", "left_distal_1", "left_distal_2", "left_distal_3"]
            right_names += ["right_distal_0", "right_distal_1", "right_distal_2", "right_distal_3"]
        return {
            "schema_version": 1,
            "type": "synchronized_teleop_frame",
            "mode": "sim2real",
            "frame_id": frame_id,
            "timestamp_ns": 10_000_000_000,
            "robot": "g1_23",
            "hand_type": hand_type,
            "arm": {
                "valid": True,
                "joint_names": list(ARM_JOINT_NAMES),
                "qpos": [index / 10 for index in range(len(ARM_JOINT_NAMES))],
            },
            "left_hand": {
                "valid": True,
                "joint_names": left_names,
                "qpos": [0.1] * len(left_names),
            },
            "right_hand": {
                "valid": right_valid,
                "joint_names": right_names,
                "qpos": [0.2] * len(right_names) if right_valid else [],
            },
        }

    def test_config_uses_dedicated_controller_and_port(self):
        cfg = UpperBodyCasiaHandZmqCtrlCfg(joint_names=list(ARM_JOINT_NAMES))
        self.assertEqual(cfg.ctrl_type, "UpperBodyCasiaHandZmqCtrl")
        self.assertEqual(cfg.endpoint, "tcp://127.0.0.1:8560")
        self.assertEqual(cfg.casia_hand.left_hand_id, 2)
        self.assertEqual(cfg.casia_hand.right_hand_id, 0x20)

    def test_synchronized_frame_drives_arm_and_both_hands_atomically(self):
        controller = self.make_controller([self.synchronized_frame()])
        controller.set_takeover_enabled(True)
        with patch("robojudo.controller.upper_body_casia_hand_zmq_ctrl.time.monotonic", return_value=10.0):
            data = controller.get_data()

        self.assertEqual(controller._casia_hand.enabled, [True])
        self.assertEqual(data["frame_id"], 7)
        self.assertEqual(data["joint_positions"]["left_elbow_joint"], 0.3)
        self.assertEqual(controller._casia_hand.queued[0][3], 7)
        np.testing.assert_allclose(data["casia_hand"]["joint_position_commands"][:10], 0.1)
        self.assertTrue(data["casia_hand"]["fresh"])

    def test_incomplete_frame_is_rejected_as_a_whole(self):
        controller = self.make_controller([self.synchronized_frame(right_valid=False)])
        with patch("robojudo.controller.upper_body_casia_hand_zmq_ctrl.time.monotonic", return_value=10.0):
            data = controller.get_data()

        self.assertFalse(data["has_received"])
        self.assertEqual(controller._casia_hand.queued, [])

    def test_simulation_joint_schema_and_other_hand_type_are_rejected(self):
        controller = self.make_controller(
            [self.synchronized_frame(physical_schema=False), self.synchronized_frame(hand_type="omnihand")]
        )
        with patch("robojudo.controller.upper_body_casia_hand_zmq_ctrl.time.monotonic", return_value=10.0):
            self.assertFalse(controller.get_data()["has_received"])
            self.assertFalse(controller.get_data()["has_received"])
        self.assertEqual(controller._casia_hand.queued, [])

    def test_repeated_frame_id_is_rejected_while_stream_is_fresh(self):
        frame = self.synchronized_frame()
        controller = self.make_controller([frame, frame])
        with patch("robojudo.controller.upper_body_casia_hand_zmq_ctrl.time.monotonic", return_value=10.0):
            self.assertTrue(controller.get_data()["fresh"])
            self.assertTrue(controller.get_data()["fresh"])
        self.assertEqual(len(controller._casia_hand.queued), 1)


if __name__ == "__main__":
    unittest.main()
