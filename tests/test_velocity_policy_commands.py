import unittest
from types import SimpleNamespace

import numpy as np

from robojudo.config.g1.g1_cfg import g1_real_zmq, g1_zmq
from robojudo.policy.amo_policy import AMOPolicy
from robojudo.policy.asap_policy import AsapLocoPolicy
from robojudo.policy.locomanipulation_policy import LocomanipulationPolicyBase
from robojudo.policy.smooth_policy import SmoothPolicy
from robojudo.policy.unitree_policy import UnitreePolicy


def velocity_entry(x=0.0, y=0.0, yaw=0.0, fresh=True):
    return {
        "linear_velocity": np.asarray([x, y, 0.0], dtype=np.float32),
        "angular_velocity": np.asarray([0.0, 0.0, yaw], dtype=np.float32),
        "fresh": fresh,
    }


class TestVelocityPolicyCommands(unittest.TestCase):
    def test_opt_in_configs_keep_local_fallback_and_real_safety(self):
        sim_cfg = g1_zmq()
        real_cfg = g1_real_zmq()
        self.assertEqual([ctrl.ctrl_type for ctrl in sim_cfg.ctrl], ["VelocityZmqCtrl", "JoystickCtrl"])
        self.assertEqual([ctrl.ctrl_type for ctrl in real_cfg.ctrl], ["VelocityZmqCtrl", "UnitreeCtrl"])
        self.assertEqual([ctrl.velocity_priority for ctrl in sim_cfg.ctrl], [100, 300])
        self.assertEqual([ctrl.velocity_priority for ctrl in real_cfg.ctrl], [100, 300])
        self.assertTrue(real_cfg.do_safety_check)

    def test_unitree_converts_si_values_to_normalized_commands_and_clamps(self):
        policy = UnitreePolicy.__new__(UnitreePolicy)
        policy.max_cmd = [0.8, 0.5, 1.57]
        policy.commands_map = [[-1.0, 0.0, 1.0]] * 3
        commands = policy._get_commands({"VelocityZmqCtrl": velocity_entry(0.4, -1.0, 3.14)})
        np.testing.assert_allclose(commands, [0.5, -1.0, 1.0])

    def test_stale_zmq_falls_through_to_local_controller(self):
        policy = UnitreePolicy.__new__(UnitreePolicy)
        policy.max_cmd = [0.8, 0.5, 1.57]
        policy.commands_map = [[-1.0, 0.0, 1.0]] * 3
        joystick = {
            "axes": {"LeftX": 0.0, "LeftY": 1.0, "RightX": 0.0, "RightY": 0.0},
            "button_event": [],
        }
        commands = policy._get_commands(
            {
                "VelocityZmqCtrl": velocity_entry(0.7, fresh=False),
                "JoystickCtrl": joystick,
                "VELOCITY_SOURCE": "JoystickCtrl",
            }
        )
        self.assertGreater(commands[0], 0.0)

    def test_fresh_zero_zmq_keeps_priority_over_local_controller(self):
        policy = UnitreePolicy.__new__(UnitreePolicy)
        policy.max_cmd = [0.8, 0.5, 1.57]
        policy.commands_map = [[-1.0, 0.0, 1.0]] * 3
        joystick = {
            "axes": {"LeftX": 1.0, "LeftY": 1.0, "RightX": 1.0, "RightY": 0.0},
            "button_event": [],
        }
        commands = policy._get_commands(
            {
                "VelocityZmqCtrl": velocity_entry(),
                "JoystickCtrl": joystick,
                "VELOCITY_SOURCE": "VelocityZmqCtrl",
            }
        )
        np.testing.assert_array_equal(commands, np.zeros(3))

    def test_explicit_selected_source_overrides_mapping_order(self):
        policy = UnitreePolicy.__new__(UnitreePolicy)
        policy.max_cmd = [0.8, 0.5, 1.57]
        policy.commands_map = [[-1.0, 0.0, 1.0]] * 3
        joystick = {
            "axes": {"LeftX": 0.0, "LeftY": 1.0, "RightX": 0.0, "RightY": 0.0},
            "button_event": [],
        }
        ctrl_data = {
            "VelocityZmqCtrl": velocity_entry(-0.4),
            "JoystickCtrl": joystick,
            "VELOCITY_SOURCE": "JoystickCtrl",
        }
        self.assertGreater(policy._get_commands(ctrl_data)[0], 0.0)

    def test_smooth_clamps_physical_velocity_and_stale_only_is_zero(self):
        policy = SmoothPolicy.__new__(SmoothPolicy)
        policy.commands_map = [[-0.5, 0.0, 1.0], [0.5, 0.0, -0.5], [1.0, 0.0, -1.0]]
        np.testing.assert_allclose(
            policy._get_commands({"VelocityZmqCtrl": velocity_entry(2.0, -2.0, 0.25)}),
            [1.0, -0.5, 0.25],
        )
        np.testing.assert_array_equal(
            policy._get_commands({"VelocityZmqCtrl": velocity_entry(fresh=False)}), np.zeros(3)
        )

    def test_smooth_consumes_selected_keyboard_velocity(self):
        policy = SmoothPolicy.__new__(SmoothPolicy)
        policy.commands_map = [[-0.5, 0.0, 1.0], [0.5, 0.0, -0.5], [1.0, 0.0, -1.0]]
        commands = policy._get_commands(
            {
                "KeyboardCtrl": {"keyboard_event": [], "pressed_keys": ["w", "a", "q"]},
                "VELOCITY_SOURCE": "KeyboardCtrl",
            }
        )
        self.assertFalse(np.allclose(commands, np.zeros(3)))

    def test_amo_reorders_ros_planar_velocity(self):
        policy = AMOPolicy.__new__(AMOPolicy)
        policy.cmd = np.zeros(8, dtype=np.float32)
        policy.commands_map = [[-1.0, 0.0, 1.0], [0.2, 0.0, -0.2], [0.8, 0.0, -0.8], [0.3, 0.75, 0.9]]
        commands = policy._get_commands({"VelocityZmqCtrl": velocity_entry(0.4, -0.3, 0.1)})
        np.testing.assert_allclose(commands[:3], [-0.3, 0.1, 0.4])

    def test_non_selected_joystick_buttons_are_still_processed(self):
        policy = AMOPolicy.__new__(AMOPolicy)
        policy.cmd = np.zeros(8, dtype=np.float32)
        policy.commands_map = [[-1.0, 0.0, 1.0], [0.2, 0.0, -0.2], [0.8, 0.0, -0.8], [0.3, 0.75, 0.9]]
        ctrl_data = {
            "VelocityZmqCtrl": velocity_entry(),
            "JoystickCtrl": {
                "axes": {"LeftX": 0.0, "LeftY": 0.0, "RightX": 0.0, "RightY": 0.0},
                "button_event": [{"type": "button", "name": "Y", "pressed": True}],
            },
            "VELOCITY_SOURCE": "VelocityZmqCtrl",
        }
        self.assertEqual(policy._get_commands(ctrl_data)[7], 1.0)

    def test_amo_consumes_selected_keyboard_velocity(self):
        policy = AMOPolicy.__new__(AMOPolicy)
        policy.cmd = np.zeros(8, dtype=np.float32)
        policy.commands_map = [[-1.0, 0.0, 1.0], [0.2, 0.0, -0.2], [0.8, 0.0, -0.8], [0.3, 0.75, 0.9]]
        commands = policy._get_commands(
            {
                "KeyboardCtrl": {"keyboard_event": [], "pressed_keys": ["w", "a", "q"]},
                "VELOCITY_SOURCE": "KeyboardCtrl",
            }
        )
        self.assertFalse(np.allclose(commands[:3], np.zeros(3)))

    def test_asap_derives_stand_and_clamps(self):
        policy = AsapLocoPolicy.__new__(AsapLocoPolicy)
        policy.lin_vel_command = np.zeros(2)
        policy.ang_vel_command = np.zeros(1)
        policy.stand_command = np.zeros(1, dtype=int)
        policy.ref_upper_dof_pos = np.zeros(0)
        policy._update_commands({"VelocityZmqCtrl": velocity_entry(0.8, -0.2, 2.0)})
        np.testing.assert_allclose(policy.lin_vel_command, [0.5, -0.2])
        np.testing.assert_allclose(policy.ang_vel_command, [1.0])
        np.testing.assert_array_equal(policy.stand_command, [1])
        policy._update_commands({"VelocityZmqCtrl": velocity_entry()})
        np.testing.assert_array_equal(policy.stand_command, [0])

    def test_asap_respects_local_controller_before_zmq(self):
        policy = AsapLocoPolicy.__new__(AsapLocoPolicy)
        policy.lin_vel_command = np.zeros(2)
        policy.ang_vel_command = np.zeros(1)
        policy.stand_command = np.ones(1, dtype=int)
        policy.ref_upper_dof_pos = np.zeros(0)
        keyboard = {"keyboard_event": [{"type": "keyboard", "name": "w", "pressed": True}]}
        policy._update_commands(
            {
                "KeyboardCtrl": keyboard,
                "VelocityZmqCtrl": velocity_entry(-0.5),
                "VELOCITY_SOURCE": "KeyboardCtrl",
            }
        )
        np.testing.assert_allclose(policy.lin_vel_command, [0.1, 0.0])

    def test_asap_stale_zmq_without_fallback_clears_stand(self):
        policy = AsapLocoPolicy.__new__(AsapLocoPolicy)
        policy.lin_vel_command = np.ones(2)
        policy.ang_vel_command = np.ones(1)
        policy.stand_command = np.ones(1, dtype=int)
        policy.ref_upper_dof_pos = np.zeros(0)
        policy._update_commands(
            {
                "VelocityZmqCtrl": velocity_entry(fresh=False),
                "VELOCITY_SOURCE": None,
            }
        )
        np.testing.assert_array_equal(policy.lin_vel_command, np.zeros(2))
        np.testing.assert_array_equal(policy.ang_vel_command, np.zeros(1))
        np.testing.assert_array_equal(policy.stand_command, np.zeros(1))

    def test_asap_non_selected_manual_stop_overrides_zmq_for_current_step(self):
        policy = AsapLocoPolicy.__new__(AsapLocoPolicy)
        policy.lin_vel_command = np.zeros(2)
        policy.ang_vel_command = np.zeros(1)
        policy.stand_command = np.zeros(1, dtype=int)
        policy.ref_upper_dof_pos = np.zeros(0)
        policy.base_height_command = np.zeros(1)
        policy._update_commands(
            {
                "VelocityZmqCtrl": velocity_entry(0.5),
                "JoystickCtrl": {
                    "axes": {"LeftX": 0.0, "LeftY": 0.0, "RightX": 0.0, "RightY": 0.0},
                    "button_event": [{"type": "button", "name": "Left", "pressed": True}],
                },
                "VELOCITY_SOURCE": "VelocityZmqCtrl",
            }
        )
        np.testing.assert_array_equal(policy.lin_vel_command, np.zeros(2))
        np.testing.assert_array_equal(policy.ang_vel_command, np.zeros(1))
        np.testing.assert_array_equal(policy.stand_command, np.zeros(1))

    def test_locomanipulation_uses_existing_smoothing(self):
        policy = LocomanipulationPolicyBase.__new__(LocomanipulationPolicyBase)
        policy.commands_map = [
            [-0.5, 0.0, 1.0],
            [0.5, 0.0, -0.5],
            [1.0, 0.0, -1.0],
            [0.3, 0.64, 0.64],
            [-1.57, 0.0, 1.57],
        ]
        policy.cmd = np.asarray([0.0, 0.0, 0.0, 0.64, 0.0], dtype=np.float32)
        policy.current_vel_cmd = np.zeros(3, dtype=np.float32)
        policy._target_height = 0.64
        policy._target_waist_yaw = 0.0
        policy._held_keys = set()
        policy.cfg_policy = SimpleNamespace(command_decay=0.5, standing_command_threshold=0.1)
        commands = policy._get_commands({"VelocityZmqCtrl": velocity_entry(2.0, -2.0, 2.0)})
        np.testing.assert_allclose(commands[:3], [1.0, -0.5, 1.0])


if __name__ == "__main__":
    unittest.main()
