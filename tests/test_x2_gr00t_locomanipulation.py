import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import zmq
from box import Box

from robojudo.controller.ctrl_cfgs import Gr00tZmqCtrlCfg
from robojudo.controller.gr00t_zmq_ctrl import Gr00tZmqCtrl
from robojudo.pipeline.four_mode_pipeline import ControlMode
from robojudo.pipeline.x2_gr00t_locomanipulation_pipeline import X2Gr00tLocomanipulationPipeline
from robojudo.policy.x2_gr00t_locomanipulation_policy import X2Gr00tLocomanipulationPolicy


class FakeZmqSocket:
    def __init__(self, messages=None):
        self.messages = list(messages or [])

    def recv_json(self, flags=0):
        del flags
        if not self.messages:
            raise zmq.Again()
        return self.messages.pop(0)


class TestGr00tZmqCtrl(unittest.TestCase):
    joint_names = ["left_arm", "right_arm"]

    def make_controller(self, messages=None):
        controller = Gr00tZmqCtrl.__new__(Gr00tZmqCtrl)
        controller.cfg_ctrl = Gr00tZmqCtrlCfg(joint_names=self.joint_names)
        controller._joint_names = tuple(self.joint_names)
        controller._joint_name_set = set(self.joint_names)
        controller._socket = FakeZmqSocket(messages)
        controller._latest_positions = {}
        controller._latest_locomotion_command = None
        controller._latest_sequence = None
        controller._last_received_at = None
        controller._last_invalid_log_at = float("-inf")
        return controller

    def message(self, *, sequence=1, command=None):
        return {
            "sequence": sequence,
            "positions": {"left_arm": 0.2, "right_arm": -0.3},
            "locomotion_command": command or [0.5, -0.1, 0.2, 0.64],
        }

    def test_receives_atomic_arm_and_locomotion_command(self):
        controller = self.make_controller([self.message()])
        with patch("robojudo.controller.gr00t_zmq_ctrl.time.monotonic", return_value=10.0):
            data = controller.get_data()

        self.assertTrue(data["fresh"])
        self.assertEqual(data["sequence"], 1)
        self.assertEqual(data["joint_positions"], {"left_arm": 0.2, "right_arm": -0.3})
        np.testing.assert_allclose(data["locomotion_command"], [0.5, -0.1, 0.2, 0.64])

    def test_invalid_or_replayed_message_does_not_refresh_state(self):
        controller = self.make_controller([self.message(sequence=2)])
        with patch("robojudo.controller.gr00t_zmq_ctrl.time.monotonic", return_value=10.0):
            controller.get_data()

        invalid = self.message(sequence=3, command=[0.0, 0.0, 0.0])
        replayed = self.message(sequence=1)
        controller._socket = FakeZmqSocket([invalid, replayed])
        with patch("robojudo.controller.gr00t_zmq_ctrl.time.monotonic", return_value=10.1):
            data = controller.get_data()

        self.assertTrue(data["fresh"])
        self.assertEqual(data["sequence"], 2)
        self.assertEqual(data["joint_positions"], {"left_arm": 0.2, "right_arm": -0.3})
        np.testing.assert_allclose(data["locomotion_command"], [0.5, -0.1, 0.2, 0.64])

        with patch("robojudo.controller.gr00t_zmq_ctrl.time.monotonic", return_value=10.3):
            self.assertFalse(controller.get_data()["fresh"])

    def test_accepts_restarted_sequence_after_timeout(self):
        controller = self.make_controller([self.message(sequence=20)])
        with patch("robojudo.controller.gr00t_zmq_ctrl.time.monotonic", return_value=10.0):
            controller.get_data()
        controller._socket = FakeZmqSocket([self.message(sequence=0)])
        with patch("robojudo.controller.gr00t_zmq_ctrl.time.monotonic", return_value=10.3):
            data = controller.get_data()
        self.assertTrue(data["fresh"])
        self.assertEqual(data["sequence"], 0)

    def test_requires_complete_positions(self):
        controller = self.make_controller()
        with self.assertRaisesRegex(ValueError, "missing joints"):
            controller._decode_message(
                {
                    "positions": {"left_arm": 0.2},
                    "locomotion_command": [0.0, 0.0, 0.0, 0.64],
                }
            )


class TestX2Gr00tLocomanipulationPolicy(unittest.TestCase):
    def make_policy(self):
        policy = X2Gr00tLocomanipulationPolicy.__new__(X2Gr00tLocomanipulationPolicy)
        policy.commands_map = [
            [-0.5, 0.0, 1.0],
            [0.5, 0.0, -0.5],
            [1.0, 0.0, -1.0],
            [0.40, 0.64, 0.66],
            [-1.5708, 0.0, 1.5708],
        ]
        policy._command_defaults = np.asarray([0.0, 0.0, 0.0, 0.64, 0.0], dtype=np.float32)
        policy.cmd = policy._command_defaults.copy()
        policy.current_vel_cmd = np.zeros(3, dtype=np.float32)
        policy._target_height = 0.64
        policy._target_waist_yaw = 0.0
        policy._held_keys = set()
        policy.cfg_policy = SimpleNamespace(
            command_decay=0.95,
            standing_command_threshold=0.1,
            height_step=0.02,
            waist_yaw_step=0.1,
        )
        return policy

    def test_uses_clipped_gr00t_command_only_while_takeover_is_active(self):
        policy = self.make_policy()
        active = Box(
            {
                "Gr00tZmqCtrl": {
                    "takeover_enabled": True,
                    "fresh": True,
                    "locomotion_command": [2.0, 2.0, -2.0, 0.9],
                }
            }
        )
        command = policy._get_commands(active)
        np.testing.assert_allclose(command, [1.0, 0.5, -1.0, 0.66, 0.0])

        stale = Box({"Gr00tZmqCtrl": {"takeover_enabled": True, "fresh": False}})
        command = policy._get_commands(stale)
        np.testing.assert_allclose(command, [0.0, 0.0, 0.0, 0.66, 0.0])

    def test_uses_manual_joystick_commands_after_takeover_is_disabled(self):
        policy = self.make_policy()
        manual = Box(
            {
                "JoystickCtrl": {
                    "axes": {"LeftX": 0.4, "LeftY": 0.6, "RightX": -0.5},
                    "button_event": [],
                },
                "Gr00tZmqCtrl": {"takeover_enabled": False, "fresh": True},
            }
        )

        command = policy._get_commands(manual)

        np.testing.assert_allclose(command, [0.6, -0.2, 0.5, 0.64, 0.0])

    def test_stopping_takeover_clears_the_previous_vla_velocity(self):
        policy = self.make_policy()
        active = Box(
            {
                "Gr00tZmqCtrl": {
                    "takeover_enabled": True,
                    "fresh": True,
                    "locomotion_command": [0.8, -0.2, 0.4, 0.65],
                }
            }
        )
        policy._get_commands(active)
        stopped = Box(
            {
                "JoystickCtrl": {
                    "axes": {"LeftX": 0.0, "LeftY": 0.0, "RightX": 0.0},
                    "button_event": [],
                },
                "Gr00tZmqCtrl": {"takeover_enabled": False, "fresh": True},
            }
        )

        command = policy._get_commands(stopped)

        np.testing.assert_allclose(command[:3], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(command[3], 0.65)

    def test_real_lower_body_model_accepts_gr00t_commands(self):
        from robojudo.config.x2.policy.x2_gr00t_locomanipulation_policy_cfg import (
            X2Gr00tLocomanipulationPolicyCfg,
        )

        cfg = X2Gr00tLocomanipulationPolicyCfg()
        policy = X2Gr00tLocomanipulationPolicy(cfg, "cpu")
        env_data = Box(
            {
                "base_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                "base_ang_vel": np.zeros(3, dtype=np.float32),
                "dof_pos": np.asarray(cfg.obs_dof.default_pos, dtype=np.float32),
                "dof_vel": np.zeros(cfg.obs_dof.num_dofs, dtype=np.float32),
            }
        )
        ctrl_data = Box(
            {
                "Gr00tZmqCtrl": {
                    "takeover_enabled": True,
                    "fresh": True,
                    "locomotion_command": [0.3, -0.1, 0.2, 0.62],
                }
            }
        )

        observation, extras = policy.get_observation(env_data, ctrl_data)
        action = policy.get_action(observation)

        self.assertEqual(observation.shape, (430,))
        self.assertEqual(action.shape, (15,))
        self.assertTrue(np.isfinite(action).all())
        np.testing.assert_allclose(extras["locomotion_command"], [0.3, -0.1, 0.2, 0.62, 0.0])


class TestX2Gr00tLocomanipulationPipeline(unittest.TestCase):
    def test_takeover_gate_is_added_without_changing_controller_data(self):
        pipeline = X2Gr00tLocomanipulationPipeline.__new__(X2Gr00tLocomanipulationPipeline)
        pipeline.mode = ControlMode.RL_DEFAULT
        pipeline._upper_body_enabled = True
        pipeline._upper_body_control_available = lambda: True
        stream = {"fresh": True, "joint_positions": {"left_arm": 0.2}}
        ctrl_data = {"Gr00tZmqCtrl": stream}

        prepared = pipeline._prepare_gr00t_stream(ctrl_data)
        self.assertTrue(prepared["takeover_enabled"])
        self.assertIs(ctrl_data["UpperBodyZmqCtrl"], stream)

        pipeline.mode = ControlMode.JOINT_DEFAULT
        pipeline._prepare_gr00t_stream(ctrl_data)
        self.assertFalse(stream["takeover_enabled"])

    def test_arm_target_is_rate_limited(self):
        pipeline = X2Gr00tLocomanipulationPipeline.__new__(X2Gr00tLocomanipulationPipeline)
        pipeline._upper_body_cfg = SimpleNamespace(
            joint_names=["left_arm", "right_arm"],
            ema_alpha=0.0,
            max_joint_velocity_rad_s=4.0,
        )
        pipeline._upper_body_indices = np.asarray([0, 1], dtype=np.int32)
        pipeline._upper_body_default = np.zeros(2, dtype=np.float32)
        pipeline._upper_body_filtered = np.zeros(2, dtype=np.float32)
        pipeline._upper_body_enabled = True
        pipeline._upper_body_stream_was_fresh = False
        pipeline.dt = 0.02
        pipeline.env = SimpleNamespace(position_limits=np.asarray([[-2.0, 2.0], [-2.0, 2.0]]))
        ctrl_data = {
            "UpperBodyZmqCtrl": {
                "fresh": True,
                "joint_positions": {"left_arm": 1.0, "right_arm": -1.0},
            }
        }

        target = pipeline._apply_pd_target_override(np.zeros(2, dtype=np.float32), ctrl_data)
        np.testing.assert_allclose(target, [0.08, -0.08])
        np.testing.assert_allclose(pipeline._upper_body_filtered, target)

    def test_configs_are_isolated_from_existing_locomanipulation(self):
        from robojudo.config.x2 import x2_gr00t_locomanipulation, x2_gr00t_locomanipulation_real

        sim = x2_gr00t_locomanipulation()
        real = x2_gr00t_locomanipulation_real()
        self.assertEqual(sim.pipeline_type, "X2Gr00tLocomanipulationPipeline")
        self.assertEqual(sim.policy.policy_type, "X2Gr00tLocomanipulationPolicy")
        self.assertEqual([cfg.ctrl_type for cfg in sim.ctrl], ["JoystickCtrl", "KeyboardCtrl", "Gr00tZmqCtrl"])
        self.assertEqual([cfg.ctrl_type for cfg in real.ctrl], ["JoystickCtrl", "Gr00tZmqCtrl"])
        for cfg in (sim.ctrl[-1], real.ctrl[-1]):
            self.assertTrue(cfg.observation_enabled)
            self.assertEqual(cfg.observation_profile, "x2")
        for cfg in (sim.ctrl[-1], real.ctrl[-1]):
            self.assertEqual(cfg.camera.type, "ros2")
            self.assertEqual(
                cfg.camera.options["topic"],
                "/aima/hal/sensor/stereo_head_front_right/rgb_image/compressed",
            )


if __name__ == "__main__":
    unittest.main()
