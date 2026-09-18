import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import zmq

from robojudo.controller.ctrl_cfgs import LocomanipulationPostureZmqCtrlCfg
from robojudo.controller.locomanipulation_posture_zmq_ctrl import LocomanipulationPostureZmqCtrl
from robojudo.policy.locomanipulation_policy import LocomanipulationPolicyBase


def posture_entry(height=0.64, waist_yaw=0.0, fresh=True):
    return {"height": height, "waist_yaw": waist_yaw, "fresh": fresh}


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


def policy_with_posture_controls():
    policy = LocomanipulationPolicyBase.__new__(LocomanipulationPolicyBase)
    policy.commands_map = [
        [-0.5, 0.0, 1.0],
        [0.5, 0.0, -0.5],
        [1.0, 0.0, -1.0],
        [0.3, 0.64, 0.64],
        [-1.5708, 0.0, 1.5708],
    ]
    policy.cmd = np.asarray([0.0, 0.0, 0.0, 0.64, 0.0], dtype=np.float32)
    policy.current_vel_cmd = np.zeros(3, dtype=np.float32)
    policy._target_height = 0.64
    policy._target_waist_yaw = 0.0
    policy._held_keys = set()
    policy.base_height_default = 0.64
    policy.cfg_policy = SimpleNamespace(
        command_decay=0.0,
        standing_command_threshold=0.1,
        height_step=0.02,
        waist_yaw_step=0.1,
    )
    return policy


class TestLocomanipulationPostureZmq(unittest.TestCase):
    @staticmethod
    def make_controller(messages=None):
        controller = LocomanipulationPostureZmqCtrl.__new__(LocomanipulationPostureZmqCtrl)
        controller.cfg_ctrl = SimpleNamespace(timeout_s=0.25)
        controller._socket = FakeZmqSocket(messages)
        controller._height = None
        controller._waist_yaw = None
        controller._last_received_at = None
        controller._last_invalid_log_at = float("-inf")
        return controller

    def test_config_and_decoder_require_an_atomic_finite_setpoint(self):
        cfg = LocomanipulationPostureZmqCtrlCfg()
        self.assertEqual(cfg.endpoint, "tcp://127.0.0.1:8557")
        self.assertEqual(cfg.timeout_s, 0.25)
        self.assertIsNone(cfg.posture_priority)
        self.assertEqual(
            LocomanipulationPostureZmqCtrl._decode_message({"height": 0.64, "waist_yaw": -0.2}),
            (0.64, -0.2),
        )

        for message in (
            {},
            {"height": 0.64},
            {"height": 0.64, "waist_yaw": 0.0, "extra": 1.0},
            {"height": True, "waist_yaw": 0.0},
            {"height": float("nan"), "waist_yaw": 0.0},
        ):
            with self.subTest(message=message), self.assertRaises(ValueError):
                LocomanipulationPostureZmqCtrl._decode_message(message)

    def test_latest_valid_setpoint_is_returned(self):
        controller = self.make_controller(
            [
                {"height": 0.5, "waist_yaw": -0.1},
                {"height": 0.6, "waist_yaw": 0.2},
            ]
        )
        with patch("robojudo.controller.locomanipulation_posture_zmq_ctrl.time.monotonic", return_value=10.0):
            data = controller.get_data()

        self.assertEqual(data["height"], 0.6)
        self.assertEqual(data["waist_yaw"], 0.2)
        self.assertTrue(data["has_received"])
        self.assertTrue(data["fresh"])

    def test_invalid_setpoint_does_not_replace_or_refresh_the_last_valid_one(self):
        controller = self.make_controller(
            [
                {"height": 0.5, "waist_yaw": -0.4},
                {"height": 0.6},
            ]
        )
        with patch("robojudo.controller.locomanipulation_posture_zmq_ctrl.time.monotonic", return_value=2.0):
            data = controller.get_data()

        self.assertEqual((data["height"], data["waist_yaw"]), (0.5, -0.4))
        self.assertEqual(controller._last_received_at, 2.0)

        with patch("robojudo.controller.locomanipulation_posture_zmq_ctrl.time.monotonic", return_value=2.251):
            self.assertFalse(controller.get_data()["fresh"])
        self.assertEqual(controller._last_received_at, 2.0)

    def test_timeout_boundary_is_fresh(self):
        controller = self.make_controller([{"height": 0.5, "waist_yaw": 0.1}])
        with patch("robojudo.controller.locomanipulation_posture_zmq_ctrl.time.monotonic", return_value=4.0):
            self.assertTrue(controller.get_data()["fresh"])
        with patch("robojudo.controller.locomanipulation_posture_zmq_ctrl.time.monotonic", return_value=4.25):
            self.assertTrue(controller.get_data()["fresh"])
        with patch("robojudo.controller.locomanipulation_posture_zmq_ctrl.time.monotonic", return_value=4.251):
            self.assertFalse(controller.get_data()["fresh"])

    def test_fresh_posture_clips_both_axes_without_affecting_velocity(self):
        policy = policy_with_posture_controls()
        commands = policy._get_commands({"LocomanipulationPostureZmqCtrl": posture_entry(height=1.0, waist_yaw=2.0)})

        np.testing.assert_allclose(commands[:3], np.zeros(3))
        np.testing.assert_allclose(commands[3:], [0.64, 1.5708])

    def test_posture_and_velocity_zmq_use_independent_sources(self):
        policy = policy_with_posture_controls()
        velocity = {
            "linear_velocity": np.asarray([2.0, -2.0, 0.0], dtype=np.float32),
            "angular_velocity": np.asarray([0.0, 0.0, 2.0], dtype=np.float32),
            "fresh": True,
        }
        commands = policy._get_commands(
            {
                "VelocityZmqCtrl": velocity,
                "VELOCITY_SOURCE": "VelocityZmqCtrl",
                "LocomanipulationPostureZmqCtrl": posture_entry(height=0.3, waist_yaw=-1.0),
            }
        )

        np.testing.assert_allclose(commands, [1.0, -0.5, 1.0, 0.3, -1.0])

    def test_stale_posture_holds_the_last_accepted_setpoint(self):
        policy = policy_with_posture_controls()
        policy._get_commands({"LocomanipulationPostureZmqCtrl": posture_entry(height=0.5, waist_yaw=-0.4)})
        commands = policy._get_commands(
            {"LocomanipulationPostureZmqCtrl": posture_entry(height=0.64, waist_yaw=0.0, fresh=False)}
        )

        np.testing.assert_allclose(commands[3:], [0.5, -0.4])

    def test_manual_posture_command_temporarily_overrides_fresh_zmq(self):
        policy = policy_with_posture_controls()
        manual_commands = policy._get_commands(
            {
                "KeyboardCtrl": {"keyboard_event": [{"type": "keyboard", "name": "r", "pressed": True}]},
                "LocomanipulationPostureZmqCtrl": posture_entry(height=0.3, waist_yaw=-1.0),
                "POSTURE_SOURCE": "KeyboardCtrl",
            }
        )
        resumed_commands = policy._get_commands(
            {
                "LocomanipulationPostureZmqCtrl": posture_entry(height=0.3, waist_yaw=-1.0),
                "POSTURE_SOURCE": "LocomanipulationPostureZmqCtrl",
            }
        )

        np.testing.assert_allclose(manual_commands[3:], [0.64, 0.0])
        np.testing.assert_allclose(resumed_commands[3:], [0.3, -1.0])

    def test_manual_velocity_owner_blocks_fresh_zmq_posture(self):
        policy = policy_with_posture_controls()
        commands = policy._get_commands(
            {
                "KeyboardCtrl": {"keyboard_event": [], "pressed_keys": ["w"]},
                "VELOCITY_SOURCE": "KeyboardCtrl",
                "LocomanipulationPostureZmqCtrl": posture_entry(height=0.3, waist_yaw=-1.0),
                "POSTURE_SOURCE": "KeyboardCtrl",
            }
        )

        np.testing.assert_allclose(commands, [1.0, 0.0, 0.0, 0.64, 0.0])

    def test_standard_manual_posture_control_works_with_empty_posture_arbitration(self):
        policy = policy_with_posture_controls()
        commands = policy._get_commands(
            {
                "KeyboardCtrl": {"keyboard_event": [{"type": "keyboard", "name": "z", "pressed": True}]},
                "POSTURE_SOURCE": None,
            }
        )

        np.testing.assert_allclose(commands[3:], [0.64, 0.1])

    def test_existing_g1_and_x2_presets_do_not_enable_the_posture_source(self):
        from robojudo.config.g1.g1_cfg import (
            g1_23_locomanipulation_default,
            g1_23_locomanipulation_default_real,
            g1_23_locomanipulation_stiff,
            g1_23_locomanipulation_stiff_real,
            g1_29_locomanipulation_stiff,
            g1_29_locomanipulation_stiff_real,
        )
        from robojudo.config.x2 import x2_locomanipulation, x2_locomanipulation_real

        configs = [
            g1_23_locomanipulation_default(),
            g1_23_locomanipulation_stiff(),
            g1_29_locomanipulation_stiff(),
            g1_23_locomanipulation_default_real(),
            g1_23_locomanipulation_stiff_real(),
            g1_29_locomanipulation_stiff_real(),
            x2_locomanipulation(),
            x2_locomanipulation_real(),
        ]
        for cfg in configs:
            with self.subTest(config=type(cfg).__name__):
                self.assertFalse(any(ctrl.ctrl_type == "LocomanipulationPostureZmqCtrl" for ctrl in cfg.ctrl))


if __name__ == "__main__":
    unittest.main()
