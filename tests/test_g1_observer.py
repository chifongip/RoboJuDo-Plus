import unittest
from types import SimpleNamespace

import numpy as np
from pydantic import ValidationError

from robojudo.config.config_manager import ConfigManager
from robojudo.environment import env_registry
from robojudo.pipeline.four_mode_pipeline import ControlMode
from robojudo.pipeline.g1_locomanipulation_loco_mimic_pipeline import G1LocomanipulationLocoMimicPipeline
from robojudo.pipeline.rl_loco_mimic_pipeline import PolicyInterpManager
from robojudo.policy.g1_locomanipulation_policy import G1LocomanipulationPolicy
from robojudo.policy.x2_locomanipulation_policy import X2LocomanipulationPolicy


class G1ObserverTests(unittest.TestCase):
    def make_config(self):
        return ConfigManager("g1_23_locomanipulation_locomimic_observer").get_cfg()

    def make_environment(self, mode):
        cfg = self.make_config().env
        cfg.headless = True
        cfg.observer.mode = mode
        env = env_registry.get(cfg.env_type)(cfg)
        self.addCleanup(env.shutdown)
        return env

    def test_observer_preset_uses_stiff_baseline_and_arm_joints(self):
        from robojudo.config.g1.observer_parameters_23 import JOINT_NAMES, selected_observer_parameters

        cfg = self.make_config()
        baseline = ConfigManager("g1_23_locomanipulation_locomimic").get_cfg()
        self.assertEqual(cfg.model_dump(exclude={"env"}), baseline.model_dump(exclude={"env"}))
        self.assertEqual(cfg.loco_policy.policy_name, "policy_23dof_stiff")
        self.assertEqual(cfg.env.dof, baseline.env.dof)
        self.assertEqual(cfg.env.xml, baseline.env.xml)
        self.assertEqual(cfg.env.sim_dt * cfg.env.sim_decimation, 0.02)
        self.assertEqual(cfg.env.observer.joint_names, cfg.env.dof.joint_names[-10:])
        self.assertEqual(cfg.env.observer.joint_names, list(JOINT_NAMES))
        self.assertEqual(cfg.env.observer.inertia, selected_observer_parameters()["inertia"])
        self.assertEqual(cfg.env.observer.friction_viscosity, selected_observer_parameters()["friction_viscosity"])
        self.assertEqual(set(cfg.env.feedforward_effort_limits), set(cfg.env.observer.joint_names))
        self.assertEqual(
            cfg.env.observer.actuator_limits_nm,
            [cfg.env.feedforward_effort_limits[name] for name in cfg.env.observer.joint_names],
        )
        self.assertEqual(baseline.env.env_type, "MujocoEnv")

        invalid = cfg.env.model_dump()
        invalid["observer"]["joint_names"][0] = "waist_yaw_joint"
        with self.assertRaisesRegex(ValidationError, "Observer joints must be allowed feedforward joints"):
            type(cfg.env).model_validate(invalid)

        for env_cfg in (baseline.env, cfg.env):
            with self.subTest(env=type(env_cfg).__name__):
                self.assertEqual(env_cfg.feedforward_joint_names, cfg.env.observer.joint_names)
                invalid = env_cfg.model_dump()
                invalid["feedforward_effort_limits"]["waist_yaw_joint"] = 1.0
                with self.assertRaisesRegex(ValidationError, "feedforward is restricted to the 10 arm joints"):
                    type(env_cfg).model_validate(invalid)

    def test_base_simulation_rejects_nonarm_effort_after_config_mutation(self):
        cfg = ConfigManager("g1_23_locomanipulation_locomimic").get_cfg().env
        cfg.headless = True
        env = env_registry.get(cfg.env_type)(cfg)
        self.addCleanup(env.shutdown)
        cfg.feedforward_effort_limits = {"left_knee_joint": 1.0}
        with self.assertRaisesRegex(ValueError, "disallowed"):
            env.step(env.dof_pos, feedforward_effort=np.ones(env.num_dofs))

    def test_29_dof_preset_uses_stiff_baseline_and_limits_feedforward_to_arms(self):
        from robojudo.config.g1.observer_parameters import JOINT_NAMES, selected_observer_parameters

        cfg = ConfigManager("g1_29_locomanipulation_locomimic_observer").get_cfg()
        baseline = ConfigManager("g1_29_locomanipulation_locomimic").get_cfg()
        self.assertEqual(cfg.model_dump(exclude={"env"}), baseline.model_dump(exclude={"env"}))
        self.assertEqual(cfg.env.dof, baseline.env.dof)
        self.assertEqual(cfg.env.xml, baseline.env.xml)
        self.assertEqual(cfg.loco_policy.policy_name, "policy_29dof_stiff")
        self.assertEqual(cfg.env.observer.joint_names, list(JOINT_NAMES))
        self.assertEqual(cfg.env.feedforward_joint_names, list(JOINT_NAMES))
        self.assertEqual(cfg.env.observer.inertia, selected_observer_parameters()["inertia"])
        self.assertEqual(cfg.env.observer.friction_viscosity, selected_observer_parameters()["friction_viscosity"])
        limits = dict(zip(cfg.env.dof.joint_names, cfg.env.dof.torque_limits, strict=True))
        self.assertEqual(cfg.env.observer.actuator_limits_nm, [limits[name] for name in JOINT_NAMES])
        self.assertEqual(cfg.env.feedforward_effort_limits, {name: limits[name] for name in JOINT_NAMES})
        self.assertEqual(cfg.env.observer.mode, "ACTIVE")

        invalid = cfg.env.model_dump()
        invalid["observer"]["joint_names"][0] = "waist_yaw_joint"
        with self.assertRaisesRegex(ValidationError, "Observer joints must be allowed feedforward joints"):
            type(cfg.env).model_validate(invalid)
        limits = {"waist_yaw_joint": 1.0, "left_knee_joint": 1.0}
        baseline_data = baseline.env.model_dump()
        baseline_data["feedforward_effort_limits"] = limits
        self.assertEqual(type(baseline.env).model_validate(baseline_data).feedforward_effort_limits, limits)

        observer_data = cfg.env.model_dump()
        observer_data["feedforward_effort_limits"] = limits
        with self.assertRaisesRegex(ValidationError, "feedforward is restricted to the 14 arm joints"):
            type(cfg.env).model_validate(observer_data)

    def test_29_dof_observer_issues_arm_only_effort_in_simulation(self):
        cfg = ConfigManager("g1_29_locomanipulation_locomimic_observer").get_cfg().env
        cfg.headless = True
        env = env_registry.get(cfg.env_type)(cfg)
        self.addCleanup(env.shutdown)
        env.set_observer_eligible(True)
        for _ in range(15):
            env.step(np.asarray(env.default_pos))
        np.testing.assert_array_equal(env.last_feedforward_effort[:-14], 0.0)
        self.assertGreater(np.max(np.abs(env.last_feedforward_effort[-14:])), 0.0)
        np.testing.assert_array_equal(env.get_observer_status()["issued_nm"], env.last_feedforward_effort[-14:])

    def test_applied_correction_is_arm_only_and_shadow_issues_zero(self):
        for mode in ("OFF", "SHADOW", "ACTIVE"):
            with self.subTest(mode=mode):
                env = self.make_environment(mode)
                target = np.asarray(env.default_pos)
                env.set_observer_eligible(True)
                for _ in range(15):
                    env.step(target)
                np.testing.assert_array_equal(env.last_feedforward_effort[:-10], 0.0)
                peak = np.max(np.abs(env.last_feedforward_effort))
                if mode == "ACTIVE":
                    self.assertGreater(peak, 0.0)
                    self.assertLessEqual(peak, max(env.cfg_env.observer.resolve_correction_limits()))
                else:
                    self.assertEqual(peak, 0.0)
                if mode != "OFF":
                    np.testing.assert_array_equal(
                        env.get_observer_status()["issued_nm"], env.last_feedforward_effort[-10:]
                    )
                    env.reborn()
                    np.testing.assert_array_equal(env.get_observer_status()["issued_nm"], 0.0)
                    env.step(target)
                    env.set_gains(env.stiffness, env.damping)
                    self.assertIsNone(env._observer.last_time)

    def test_g1_policy_gate_resets_on_interpolation_or_policy_change(self):
        env = self.make_environment("ACTIVE")
        pipeline = G1LocomanipulationLocoMimicPipeline.__new__(G1LocomanipulationLocoMimicPipeline)
        pipeline.env = env
        pipeline.mode = ControlMode.RL_DEFAULT
        manager = SimpleNamespace(policy_loco_id=0, current_policy_id=0, InterpState=PolicyInterpManager.InterpState)
        pipeline.policy_manager = manager
        manager.interp_state = manager.InterpState.IDLE
        policy = G1LocomanipulationPolicy.__new__(G1LocomanipulationPolicy)
        pipeline._inner_policy = lambda: policy
        pipeline._sync_observer_eligibility()
        self.assertTrue(env.get_observer_status()["eligible"])
        env.step(env.default_pos)
        self.assertIsNotNone(env._observer.last_time)

        manager.interp_state = manager.InterpState.START
        pipeline._sync_observer_eligibility()
        self.assertFalse(env.get_observer_status()["eligible"])
        self.assertIsNone(env._observer.last_time)

        manager.interp_state = manager.InterpState.IDLE
        pipeline._sync_observer_eligibility()
        env.step(env.default_pos)
        policy = G1LocomanipulationPolicy.__new__(G1LocomanipulationPolicy)
        pipeline._sync_observer_eligibility()
        self.assertTrue(env.get_observer_status()["eligible"])
        self.assertIsNone(env._observer.last_time)

        policy = X2LocomanipulationPolicy.__new__(X2LocomanipulationPolicy)
        pipeline._sync_observer_eligibility()
        self.assertFalse(env.get_observer_status()["eligible"])


if __name__ == "__main__":
    unittest.main()
