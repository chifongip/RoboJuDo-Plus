import unittest
from unittest.mock import patch

import numpy as np

from robojudo.config.g1.policy.g1_gr00t_locomanipulation_policy_cfg import G1Gr00tLocomanipulation23PolicyCfg
from robojudo.config.g1.policy.g1_locomanipulation_policy_cfg import (
    G1Locomanipulation23PolicyCfg,
    G1Locomanipulation29PolicyCfg,
)
from robojudo.config.x2.policy.x2_gr00t_locomanipulation_policy_cfg import X2Gr00tLocomanipulationPolicyCfg
from robojudo.config.x2.policy.x2_locomanipulation_policy_cfg import X2LocomanipulationPolicyCfg
from robojudo.controller.velocity_source import JOYSTICK_SOURCE_TYPES
from robojudo.policy.locomanipulation_policy import LocomanipulationPolicyBase
from robojudo.policy.x2_gr00t_locomanipulation_policy import X2Gr00tLocomanipulationPolicy


def make_policy(policy_type=LocomanipulationPolicyBase, rate=0.10):
    """Initialize command state without loading a checkpoint or robot SDK."""
    policy = policy_type.__new__(policy_type)
    policy.cfg_policy = X2LocomanipulationPolicyCfg(max_height_command_rate_m_s=rate, command_status_interval_s=0.0)
    policy.commands_map = policy.cfg_policy.commands_map
    policy._command_defaults = np.asarray([entry[1] for entry in policy.commands_map], dtype=np.float32)
    policy.cmd = policy._command_defaults.copy()
    policy.base_height_default = float(policy.cmd[3])
    policy.num_actions = policy.cfg_policy.action_dof.num_dofs
    policy.reset()
    return policy


def posture(height, fresh=True):
    return {"LocomanipulationPostureZmqCtrl": {"height": height, "waist_yaw": 1.0, "fresh": fresh}}


def keyboard(*keys):
    return {"KeyboardCtrl": {"keyboard_event": [{"type": "keyboard", "name": key, "pressed": True} for key in keys]}}


def joystick(source, *buttons):
    return {
        source: {
            "axes": {"LeftX": 0.0, "LeftY": 0.6, "RightX": 0.0},
            "button_event": [{"type": "button", "name": button, "pressed": True} for button in buttons],
        }
    }


def gr00t(height=0.3, fresh=True, enabled=True):
    return {
        "Gr00tZmqCtrl": {"takeover_enabled": enabled, "fresh": fresh, "locomotion_command": [2.0, -2.0, 2.0, height]}
    }


class TestLocomanipulationHeightRate(unittest.TestCase):
    def setUp(self):
        clock = patch("robojudo.policy.locomanipulation_policy.time.monotonic")
        self.clock = clock.start()
        self.addCleanup(clock.stop)
        self.print_patch = patch("builtins.print")
        self.print_patch.start()
        self.addCleanup(self.print_patch.stop)

    def step(self, policy, now, data):
        self.clock.return_value = now
        return policy._get_commands(data)

    def test_rate_config_defaults_overrides_and_validation(self):
        for cfg_type in (
            G1Locomanipulation23PolicyCfg,
            G1Locomanipulation29PolicyCfg,
            X2LocomanipulationPolicyCfg,
            G1Gr00tLocomanipulation23PolicyCfg,
            X2Gr00tLocomanipulationPolicyCfg,
        ):
            with self.subTest(config=cfg_type.__name__):
                self.assertEqual(cfg_type().max_height_command_rate_m_s, 0.20)
                self.assertEqual(cfg_type(max_height_command_rate_m_s=0.05).max_height_command_rate_m_s, 0.05)
                for value in (0.0, -0.1, float("nan"), float("inf"), float("-inf")):
                    with self.subTest(value=value), self.assertRaises(ValueError):
                        cfg_type(max_height_command_rate_m_s=value)

    def test_elapsed_time_reversal_clipping_and_arrival_without_overshoot(self):
        policy = make_policy(rate=0.05)
        initial = float(policy.cmd[3])
        self.assertEqual(self.step(policy, 10.0, posture(-1.0))[3], initial)
        previous = initial
        for now in (10.02, 10.10, 10.35, 11.0):
            height = float(self.step(policy, now, posture(-1.0))[3])
            self.assertAlmostEqual(height, initial - 0.05 * (now - 10.0), places=6)
            self.assertLess(height, previous)
            previous = height
        # Changing direction starts from the emitted command, not the old target.
        self.assertAlmostEqual(self.step(policy, 11.2, posture(9.0))[3], previous + 0.01, places=6)
        self.assertAlmostEqual(self.step(policy, 20.0, posture(9.0))[3], initial, places=6)
        self.assertAlmostEqual(self.step(policy, 30.0, posture(-1.0))[3], 0.3, places=6)
        self.assertAlmostEqual(self.step(policy, 30.02, {})[3], 0.3, places=6)

    def test_repeated_or_backward_clock_does_not_move_height(self):
        policy = make_policy()
        self.step(policy, 10.0, posture(0.3))
        initial = float(policy.cmd[3])
        for now in (10.0, 9.0, 9.0):
            self.assertEqual(self.step(policy, now, posture(0.3))[3], initial)

    def test_small_remaining_distance_does_not_snap_past_rate_bound(self):
        policy = make_policy()
        initial = float(policy.cmd[3])
        target = initial - 0.0005
        self.step(policy, 0.0, posture(target))
        self.assertAlmostEqual(self.step(policy, 0.001, posture(target))[3], initial - 0.0001, places=7)
        self.assertAlmostEqual(self.step(policy, 0.01, posture(target))[3], target, places=7)

    def test_small_rates_accumulate_without_stalling_or_cumulative_rate_bias(self):
        for policy_type, command in ((LocomanipulationPolicyBase, posture), (X2Gr00tLocomanipulationPolicy, gr00t)):
            for rate in (1e-6, 2e-6):
                for direction in (-1.0, 1.0):
                    with self.subTest(policy=policy_type.__name__, rate=rate, direction=direction):
                        policy = make_policy(policy_type, rate=rate)
                        policy.cmd[3] = 0.5
                        policy.reset()
                        initial = float(policy.cmd[3])
                        data = command(initial + direction * 0.1)
                        tolerance = float(np.spacing(policy.cmd[3])) / 2 + 1e-12
                        for step in range(1001):
                            elapsed = step * 0.02
                            height = float(self.step(policy, elapsed, data)[3])
                            ideal = initial + direction * rate * elapsed
                            self.assertLessEqual(abs(height - ideal), tolerance)
                        self.assertGreater(direction * (height - initial), 0.0)
                        self.assertEqual(policy.cmd.dtype, np.dtype(np.float32))

    def test_source_changes_preserve_fractional_progress(self):
        policy = make_policy(X2Gr00tLocomanipulationPolicy, rate=1e-6)
        initial = float(policy.cmd[3])
        self.step(policy, 0.0, posture(0.3))
        self.step(policy, 0.01, posture(0.3))
        self.step(policy, 0.02, gr00t())
        self.step(policy, 0.03, gr00t(enabled=False))
        self.assertAlmostEqual(policy._height_command, initial - 3e-8, places=12)
        height = float(policy.cmd[3])
        self.assertLess(height, initial)

    def test_stale_gr00t_and_reset_discard_fractional_progress_and_hold(self):
        policy = make_policy(X2Gr00tLocomanipulationPolicy, rate=1e-6)
        initial = float(policy.cmd[3])
        self.step(policy, 0.0, gr00t())
        self.step(policy, 0.01, gr00t())
        self.assertLess(policy._height_command, initial)
        self.assertEqual(float(policy.cmd[3]), initial)
        self.assertEqual(self.step(policy, 1.0, gr00t(fresh=False))[3], initial)
        self.assertEqual(policy._height_command, initial)
        self.assertEqual(self.step(policy, 2.0, gr00t(enabled=False))[3], initial)
        # Resume, then reset while there is fractional progress again.
        self.step(policy, 2.01, gr00t())
        self.assertLess(policy._height_command, initial)
        policy.reset()
        self.assertEqual(policy._height_command, initial)
        self.assertEqual(self.step(policy, 100.0, {})[3], initial)

    def test_manual_sources_share_limiter_and_preserve_waist_and_velocity(self):
        sources = {"KeyboardCtrl", *JOYSTICK_SOURCE_TYPES}
        for source in sources:
            with self.subTest(source=source):
                policy = make_policy()
                initial = float(policy.cmd[3])
                data = keyboard("f", "z", "w") if source == "KeyboardCtrl" else joystick(source, "Down", "Left")
                first = self.step(policy, 10.0, data)
                second = self.step(policy, 10.02, {})
                self.assertEqual(first[3], initial)
                self.assertAlmostEqual(second[3], initial - 0.002, places=6)
                self.assertAlmostEqual(policy._target_height, initial - 0.02, places=6)
                self.assertAlmostEqual(first[4], 0.005, places=6)
                self.assertAlmostEqual(second[4], 0.00975, places=6)
                expected_vx = 1.0 if source == "KeyboardCtrl" else 0.6
                self.assertAlmostEqual(first[0], expected_vx, places=6)
                self.assertAlmostEqual(second[0], expected_vx * 0.95, places=6)

    def test_zmq_timeout_and_manual_source_handoffs_are_continuous(self):
        policy = make_policy()
        initial = float(policy.cmd[3])
        self.step(policy, 0.0, posture(0.3))
        self.assertAlmostEqual(self.step(policy, 0.1, posture(0.64, fresh=False))[3], initial - 0.01, places=6)
        manual = {**keyboard("r"), **posture(0.3), "POSTURE_SOURCE": "KeyboardCtrl"}
        self.assertAlmostEqual(self.step(policy, 0.2, manual)[3], initial - 0.02, places=6)
        resumed = {**posture(0.3), "POSTURE_SOURCE": "LocomanipulationPostureZmqCtrl"}
        self.assertAlmostEqual(self.step(policy, 0.3, resumed)[3], initial - 0.03, places=6)

    def test_command_resets_ramp_to_default_for_every_manual_source(self):
        resets = [keyboard("x")]
        resets.extend(
            joystick(source, button) for source in JOYSTICK_SOURCE_TYPES for button in ("Back", "Select", "F1")
        )
        for data in resets:
            with self.subTest(data=data):
                policy = make_policy()
                self.step(policy, 0.0, posture(0.3))
                self.step(policy, 1.0, posture(0.3))
                current = float(policy.cmd[3])
                data["VELOCITY_SOURCE"] = None
                self.assertAlmostEqual(self.step(policy, 1.02, data)[3], current + 0.002, places=6)
                self.assertEqual(policy._target_height, policy.base_height_default)
                np.testing.assert_array_equal(policy.current_vel_cmd, np.zeros(3))

    def test_full_reset_preserves_height_and_restarts_timing(self):
        for policy_type in (LocomanipulationPolicyBase, X2Gr00tLocomanipulationPolicy):
            with self.subTest(policy=policy_type.__name__):
                policy = make_policy(policy_type)
                self.step(policy, 0.0, posture(0.3))
                self.step(policy, 1.0, posture(0.3))
                current = float(policy.cmd[3])
                policy.reset()
                self.assertEqual(policy.cmd[3], current)
                self.assertIsNone(policy._last_height_command_at)
                self.assertEqual(self.step(policy, 100.0, {})[3], current)
                self.assertAlmostEqual(self.step(policy, 100.02, {})[3], current + 0.002, places=6)

    def test_gr00t_takeover_stale_reconnect_and_release_share_limiter(self):
        policy = make_policy(X2Gr00tLocomanipulationPolicy)
        initial = float(policy.cmd[3])
        self.step(policy, 0.0, {})
        active = self.step(policy, 0.02, gr00t())
        self.assertAlmostEqual(active[3], initial - 0.002, places=6)
        np.testing.assert_allclose(active[[0, 1, 2, 4]], [1.0, -0.5, 1.0, 0.0])
        stale = self.step(policy, 1.0, gr00t(fresh=False))
        np.testing.assert_allclose(stale, [0.0, 0.0, 0.0, active[3], 0.0])
        resumed = self.step(policy, 1.02, gr00t())
        self.assertAlmostEqual(resumed[3], active[3] - 0.002, places=6)
        released = self.step(policy, 1.04, {**gr00t(enabled=False), **keyboard("r")})
        self.assertAlmostEqual(released[3], resumed[3] - 0.002, places=6)
        np.testing.assert_array_equal(released[:3], np.zeros(3))
        # A stale GR00T request must not keep ramping after manual control resumes.
        self.step(policy, 1.06, gr00t(fresh=False))
        held = float(policy.cmd[3])
        self.assertEqual(self.step(policy, 2.0, gr00t(enabled=False))[3], held)


if __name__ == "__main__":
    unittest.main()
