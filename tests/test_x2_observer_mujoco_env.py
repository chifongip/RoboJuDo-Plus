"""Headless coverage of the interactive observer preset, without a robot SDK."""

import unittest
from types import SimpleNamespace

import numpy as np

from robojudo.config.config_manager import ConfigManager
from robojudo.config.x2 import x2_locomimic, x2_locomimic_observer
from robojudo.config.x2.observer_parameters import JOINT_NAMES
from robojudo.environment import env_registry
from robojudo.pipeline.four_mode_pipeline import ControlMode, FourModePipelineMixin
from robojudo.pipeline.rl_loco_mimic_pipeline import PolicyInterpManager
from robojudo.policy.x2_locomanipulation_policy import X2LocomanipulationPolicy


class InteractiveObserverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = ConfigManager("x2_locomimic_observer").get_cfg().env
        cfg.headless, cfg.visualize_extras = True, False
        cls.env = env_registry.get(cfg.env_type)(cfg)
        cls.addClassCleanup(cls.env.shutdown)

    def setUp(self):
        self.env.cfg_env = x2_locomimic_observer().env
        self.env.set_gains(self.env.cfg_env.dof.stiffness, self.env.cfg_env.dof.damping)
        self.env.set_control_joint_names(self.env.joint_names)
        self.env.reborn()
        self.env.set_observer_eligible(True)
        self.arms = self.env._observer.indices
        self.target = self.env.dof_pos.copy()

    def advance(self, count=1):
        for _ in range(count):
            self.env.step(self.target)

    def test_preset_preserves_policies_controls_and_base(self):
        base = x2_locomimic().model_dump()
        cfg = x2_locomimic_observer()
        actual = cfg.model_dump()
        actual.pop("env")
        base.pop("env")
        self.assertEqual(actual, base)
        self.assertTrue(cfg.env.is_sim)
        self.assertEqual(cfg.env.observer.mode, "ACTIVE")
        self.assertEqual(self.env.get_observer_status()["algorithm"], "nonlinear")
        self.assertEqual(cfg.env.sim_dt * cfg.env.sim_decimation, 0.02)
        self.assertEqual(set(cfg.env.feedforward_effort_limits), set(JOINT_NAMES))
        cfg.env.observer.correction_limits_nm[0] = 0.01
        self.assertNotEqual(x2_locomimic_observer().env.observer.correction_limits_nm[0], 0.01)
        self.assertEqual(x2_locomimic().env.env_type, "MujocoEnv")
        self.assertFalse(x2_locomimic().env.feedforward_effort_limits)

    def test_active_effort_bounds_and_mask_reconciliation(self):
        self.env.cfg_env.feedforward_effort_limits = dict.fromkeys(JOINT_NAMES, 0.01)
        self.env.set_control_joint_names([n for n in self.env.joint_names if n != JOINT_NAMES[0]])
        self.advance()
        np.testing.assert_array_equal(self.env.last_feedforward_effort, 0)
        peak = 0.0
        for _ in range(30):
            self.advance()
            effort = self.env.last_feedforward_effort
            peak = max(peak, float(np.max(abs(effort))))
            np.testing.assert_array_equal(np.delete(effort, self.arms), 0)
            np.testing.assert_array_equal(self.env._observer.issued_nm, effort[self.arms])
            np.testing.assert_array_equal(self.env._observer.observer.effort, effort[self.arms])
            self.assertEqual(effort[self.arms[0]], 0)
            self.assertTrue(np.all(abs(effort) <= 0.01))
            self.assertTrue(np.all(abs(self.env.last_applied_torque) <= self.env.torque_limits))
        self.assertGreater(peak, 0)

    def test_off_shadow_and_ineligible_apply_zero(self):
        self.env.set_observer_eligible(False)
        self.advance(3)
        np.testing.assert_array_equal(self.env.last_feedforward_effort, 0)
        self.assertIsNone(self.env._observer.last_time)
        for mode in ("OFF", "SHADOW"):
            with self.subTest(mode=mode):
                self.env.cfg_env.observer.mode = mode
                self.env.set_gains(self.env.stiffness, self.env.damping)
                self.env.set_observer_eligible(True)
                self.advance(10)
                np.testing.assert_array_equal(self.env.last_feedforward_effort, 0)
                self.assertEqual(self.env.get_observer_status()["mode"], mode)
                if mode == "SHADOW":
                    self.assertGreater(np.max(abs(self.env._observer.proposed_nm)), 0)

    def test_lifecycle_resets_and_gain_refresh(self):
        for operation in (
            self.env.reset,
            self.env.reborn,
            lambda: self.env.set_observer_eligible(False),
            lambda: self.env.set_control_joint_names(self.env.joint_names),
            lambda: self.env.set_gains(self.env.stiffness * 0.9, self.env.damping),
            self.env.command_passive,
            self.env.command_damping,
        ):
            self.env.set_observer_eligible(True)
            self.advance(3)
            operation()
            self.assertIsNone(self.env._observer.last_time)
            np.testing.assert_array_equal(self.env._observer.issued_nm, 0)
            self.env.set_observer_eligible(True)
            self.advance()
            np.testing.assert_array_equal(self.env.last_feedforward_effort, 0)
        np.testing.assert_array_equal(self.env._observer.observer.kp, self.env.stiffness[self.arms])

    def test_pipeline_policy_and_mode_gates(self):
        pipeline = FourModePipelineMixin.__new__(FourModePipelineMixin)
        pipeline.env = self.env
        inner = X2LocomanipulationPolicy.__new__(X2LocomanipulationPolicy)
        pipeline._inner_policy = lambda: inner
        manager = SimpleNamespace(
            current_policy_id=0,
            policy_loco_id=0,
            interp_state=PolicyInterpManager.InterpState.IDLE,
            InterpState=PolicyInterpManager.InterpState,
        )
        pipeline.policy_manager = manager
        for mode, policy_id, state, expected in (
            (ControlMode.RL_DEFAULT, 0, manager.InterpState.IDLE, True),
            (ControlMode.RL_DEFAULT, 1, manager.InterpState.IDLE, False),
            (ControlMode.RL_DEFAULT, 0, manager.InterpState.START, False),
            (ControlMode.JOINT_DEFAULT, 0, manager.InterpState.IDLE, False),
            (ControlMode.RL_DEFAULT, 0, manager.InterpState.IDLE, True),
        ):
            pipeline.mode, manager.current_policy_id, manager.interp_state = mode, policy_id, state
            pipeline._send_policy_target(self.target)
            self.assertEqual(self.env._observer_eligible, expected)
            np.testing.assert_array_equal(self.env.last_feedforward_effort, 0)

    def test_invalid_input_does_not_advance_physics(self):
        self.advance(3)
        for bad in (np.full(self.env.num_dofs, np.nan), np.zeros(1)):
            now = self.env.data.time
            with self.assertRaises(ValueError):
                self.env.step(bad)
            self.assertEqual(self.env.data.time, now)
            self.assertIsNone(self.env._observer.last_time)
        now = self.env.data.time
        self.env._dof_vel[self.arms[0]] = np.nan
        with self.assertRaises((ValueError, FloatingPointError)):
            self.advance()
        self.assertEqual(self.env.data.time, now)


if __name__ == "__main__":
    unittest.main()
