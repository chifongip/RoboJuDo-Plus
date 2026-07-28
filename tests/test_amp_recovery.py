import hashlib
import re
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from box import Box


class TestAmpRecovery(unittest.TestCase):
    @staticmethod
    def _resolve_patterns(joint_names, patterns, default=0.0):
        values = []
        for name in joint_names:
            value = default
            for pattern, pattern_value in patterns.items():
                if re.fullmatch(pattern, name):
                    value = pattern_value
            values.append(value)
        return values

    def test_presets_match_model_dimensions_and_control_timing(self):
        from robojudo.config.g1.g1_cfg import g1_23_amp_recovery, g1_amp_recovery
        from robojudo.config.x2.x2_cfg import x2_amp_recovery

        cases = (
            (g1_amp_recovery(), 29, 29, 384),
            (g1_23_amp_recovery(), 23, 23, 312),
            (x2_amp_recovery(), 31, 29, 384),
        )
        for cfg, env_dofs, policy_dofs, num_obs in cases:
            with self.subTest(config=type(cfg).__name__):
                self.assertEqual(cfg.pipeline_type, "RlPipeline")
                self.assertEqual(cfg.env.dof.num_dofs, env_dofs)
                self.assertEqual(cfg.policy.action_dof.num_dofs, policy_dofs)
                self.assertEqual(cfg.policy.num_obs, num_obs)
                self.assertEqual(cfg.policy.freq, 50)
                self.assertEqual(cfg.env.sim_dt, 0.005)
                self.assertEqual(cfg.env.sim_decimation, 4)

    def test_policy_parameters_match_training_env_yaml(self):
        from robojudo.config.g1.policy.g1_amp_recovery_policy_cfg import (
            G1_23AmpRecoveryPolicyCfg,
            G1AmpRecovery23DoF,
            G1AmpRecovery29DoF,
            G1AmpRecoveryPolicyCfg,
        )
        from robojudo.config.x2.policy.x2_amp_recovery_policy_cfg import (
            X2AmpRecoveryDoF,
            X2AmpRecoveryPolicyCfg,
        )

        g1_29_default = {
            ".*_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_shoulder_pitch_joint": 0.35,
            ".*_elbow_joint": 0.87,
            "left_shoulder_roll_joint": 0.18,
            "right_shoulder_roll_joint": -0.18,
        }
        g1_23_default = {
            ".*_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_shoulder_pitch_joint": 0.35,
            ".*_elbow_joint": 0.87,
            "left_shoulder_roll_joint": 0.18,
            "right_shoulder_roll_joint": -0.18,
        }
        x2_default = {
            ".*_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_shoulder_pitch_joint": 0.35,
            ".*_elbow_joint": -0.87,
            "left_shoulder_roll_joint": 0.1,
            "right_shoulder_roll_joint": -0.1,
        }

        g1_29_actuators = {
            ".*_elbow_joint|.*_shoulder_pitch_joint|.*_shoulder_roll_joint|.*_shoulder_yaw_joint|.*_wrist_roll_joint": (
                14.25062309787429,
                0.907222843292423,
                25.0,
                0.43857731392336724,
            ),
            ".*_hip_yaw_joint|waist_yaw_joint": (
                40.17923863450712,
                2.557889775413375,
                88.0,
                0.5475464629911068,
            ),
            ".*_hip_pitch_joint|.*_hip_roll_joint|.*_knee_joint": (
                99.09842777666111,
                6.308801853496639,
                139.0,
                0.35066146637882434,
            ),
            ".*_wrist_pitch_joint|.*_wrist_yaw_joint": (
                8.611032447370201,
                0.548195351665136,
                10.0,
                0.2903252328080005,
            ),
            "waist_pitch_joint|waist_roll_joint|.*_ankle_pitch_joint|.*_ankle_roll_joint": (
                28.50124619574858,
                1.814445686584846,
                50.0,
                0.43857731392336724,
            ),
        }
        g1_23_actuators = {
            ".*_elbow_joint|.*_shoulder_pitch_joint|.*_shoulder_roll_joint|.*_shoulder_yaw_joint|.*_wrist_roll_joint": (
                14.25062309787429,
                0.907222843292423,
                25.0,
                0.43857731392336724,
            ),
            ".*_hip_pitch_joint|.*_hip_yaw_joint|waist_yaw_joint": (
                40.17923863450712,
                2.557889775413375,
                88.0,
                0.5475464629911068,
            ),
            ".*_hip_roll_joint|.*_knee_joint": (
                99.09842777666111,
                6.308801853496639,
                139.0,
                0.35066146637882434,
            ),
            ".*_ankle_pitch_joint|.*_ankle_roll_joint": (
                28.50124619574858,
                1.814445686584846,
                50.0,
                0.43857731392336724,
            ),
        }
        x2_actuators = {
            ".*_hip_pitch_joint": (120.0, 5.0, 118.0, 0.24583333333333332),
            ".*_hip_roll_joint": (100.0, 4.0, 118.0, 0.295),
            ".*_hip_yaw_joint": (100.0, 4.0, 118.0, 0.295),
            ".*_knee_joint": (150.0, 5.0, 118.0, 0.19666666666666666),
            ".*_ankle_pitch_joint": (40.0, 2.0, 36.0, 0.225),
            ".*_ankle_roll_joint": (40.0, 2.0, 24.0, 0.15),
            "waist_yaw_joint": (40.18, 2.56, 118.0, 0.7341961174713788),
            "waist_pitch_joint|waist_roll_joint": (200.0, 2.0, 48.0, 0.06),
            ".*_shoulder_pitch_joint|.*_shoulder_roll_joint": (50.0, 3.0, 36.0, 0.18),
            ".*_shoulder_yaw_joint|.*_elbow_joint": (50.0, 3.0, 24.0, 0.12),
            ".*_wrist_yaw_joint": (20.0, 2.0, 24.0, 0.3),
            ".*_wrist_pitch_joint|.*_wrist_roll_joint": (20.0, 2.0, 2.2, 0.027500000000000004),
        }

        cases = (
            (G1AmpRecoveryPolicyCfg(), G1AmpRecovery29DoF, g1_29_default, g1_29_actuators),
            (G1_23AmpRecoveryPolicyCfg(), G1AmpRecovery23DoF, g1_23_default, g1_23_actuators),
            (X2AmpRecoveryPolicyCfg(), X2AmpRecoveryDoF, x2_default, x2_actuators),
        )
        for cfg, dof_type, default_patterns, actuator_patterns in cases:
            with self.subTest(policy=type(cfg).__name__):
                self.assertIs(type(cfg.obs_dof), dof_type)
                names = cfg.obs_dof.joint_names
                expected_default = self._resolve_patterns(names, default_patterns)
                expected_actuators = self._resolve_patterns(names, actuator_patterns, default=None)
                self.assertFalse(any(value is None for value in expected_actuators))
                expected_stiffness, expected_damping, expected_effort, expected_scale = zip(
                    *expected_actuators,
                    strict=True,
                )
                np.testing.assert_array_equal(cfg.obs_dof.default_pos, expected_default)
                np.testing.assert_array_equal(cfg.obs_dof.stiffness, expected_stiffness)
                np.testing.assert_array_equal(cfg.obs_dof.damping, expected_damping)
                np.testing.assert_array_equal(cfg.obs_dof.torque_limits, expected_effort)
                np.testing.assert_array_equal(cfg.action_scales, expected_scale)

    def test_observation_history_is_backfilled_term_major_and_chronological(self):
        from robojudo.config.g1.policy.g1_amp_recovery_policy_cfg import G1_23AmpRecoveryPolicyCfg
        from robojudo.policy.amp_recovery_policy import AmpRecoveryPolicy

        policy = AmpRecoveryPolicy(G1_23AmpRecoveryPolicyCfg(), "cpu")
        dof_pos = policy.default_dof_pos + np.arange(policy.num_dofs, dtype=np.float32)
        dof_vel = np.arange(policy.num_dofs, dtype=np.float32) + 100.0
        env_data = Box(
            {
                "base_ang_vel": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "base_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                "dof_pos": dof_pos,
                "dof_vel": dof_vel,
            }
        )

        first_obs, extras = policy.get_observation(env_data, Box({}))
        self.assertEqual(extras, {})
        np.testing.assert_array_equal(first_obs[:12], np.tile(env_data.base_ang_vel, 4))
        np.testing.assert_array_equal(first_obs[12:24], np.tile([0.0, 0.0, -1.0], 4))
        np.testing.assert_array_equal(first_obs[24:36], np.zeros(12, dtype=np.float32))
        np.testing.assert_array_equal(
            first_obs[36 : 36 + 4 * policy.num_dofs],
            np.tile(dof_pos - policy.default_dof_pos, 4),
        )

        env_data.base_ang_vel = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        second_obs, _ = policy.get_observation(env_data, Box({}))
        np.testing.assert_array_equal(
            second_obs[:12],
            np.concatenate(
                [
                    np.tile([1.0, 2.0, 3.0], 3),
                    [4.0, 5.0, 6.0],
                ]
            ),
        )

        policy.reset()
        reset_obs, _ = policy.get_observation(env_data, Box({}))
        np.testing.assert_array_equal(reset_obs[:12], np.tile([4.0, 5.0, 6.0], 4))

    def test_actual_models_load_and_apply_recorded_action_scales(self):
        from robojudo.config.g1.policy.g1_amp_recovery_policy_cfg import (
            G1_23AmpRecoveryPolicyCfg,
            G1AmpRecoveryPolicyCfg,
        )
        from robojudo.config.x2.policy.x2_amp_recovery_policy_cfg import X2AmpRecoveryPolicyCfg
        from robojudo.policy.amp_recovery_policy import AmpRecoveryPolicy

        for cfg in (G1AmpRecoveryPolicyCfg(), G1_23AmpRecoveryPolicyCfg(), X2AmpRecoveryPolicyCfg()):
            with self.subTest(policy=cfg.policy_file):
                policy = AmpRecoveryPolicy(cfg, "cpu")
                obs = np.zeros(cfg.num_obs, dtype=np.float32)
                raw_action = policy.session.run(["actions"], {"obs": obs.reshape(1, -1)})[0].reshape(-1)
                scaled_action = policy.get_action(obs)
                np.testing.assert_allclose(scaled_action, raw_action * np.asarray(cfg.action_scales), rtol=1e-6)
                np.testing.assert_array_equal(policy.last_action, raw_action)
                self.assertTrue(np.isfinite(scaled_action).all())

    def test_models_are_final_20000_exports(self):
        from robojudo.config.g1.policy.g1_amp_recovery_policy_cfg import (
            G1_23AmpRecoveryPolicyCfg,
            G1AmpRecoveryPolicyCfg,
        )
        from robojudo.config.x2.policy.x2_amp_recovery_policy_cfg import X2AmpRecoveryPolicyCfg

        expected_hashes = {
            "policy_29dof": "4789db4ae4751d351bece02f71cf63e924fb2e6ccd2d9ef4504ccc3e0b381709",
            "policy_23dof": "cf8d16ae60afd8e17e49366c5fdf4c16987e642c17dc1b5e98549a728e464548",
            "policy": "55244750dd59a14db72e7ec86f932d9961c4ddfb55d03228689524cf2babe1b4",
        }
        for cfg in (G1AmpRecoveryPolicyCfg(), G1_23AmpRecoveryPolicyCfg(), X2AmpRecoveryPolicyCfg()):
            with self.subTest(policy=cfg.policy_name):
                with open(cfg.policy_file, "rb") as model_file:
                    digest = hashlib.sha256(model_file.read()).hexdigest()
                self.assertEqual(digest, expected_hashes[cfg.policy_name])

    def test_policies_hold_training_standing_keyframes(self):
        from robojudo.config.g1.g1_cfg import g1_23_amp_recovery, g1_amp_recovery
        from robojudo.config.x2.x2_cfg import x2_amp_recovery
        from robojudo.environment.mujoco_env import MujocoEnv
        from robojudo.policy.amp_recovery_policy import AmpRecoveryPolicy
        from robojudo.tools.dof import DoFAdapter

        class HeadlessViewer:
            is_alive = False

            def __init__(self, *args, **kwargs):
                del args, kwargs
                self.cam = SimpleNamespace(distance=0.0, elevation=0.0, azimuth=0.0)

            def close(self):
                pass

        cases = (
            (g1_amp_recovery, 0.7),
            (g1_23_amp_recovery, 0.7),
            (x2_amp_recovery, 0.6),
        )
        with patch("robojudo.environment.mujoco_env.mujoco_viewer.MujocoViewer", HeadlessViewer):
            for make_cfg, minimum_height in cases:
                with self.subTest(config=make_cfg.__name__):
                    cfg = make_cfg()
                    env = MujocoEnv(cfg.env)
                    policy = AmpRecoveryPolicy(cfg.policy, "cpu")
                    env.update_dof_cfg(policy.cfg_action_dof)
                    obs_adapter = DoFAdapter(env.joint_names, policy.cfg_obs_dof.joint_names)
                    action_adapter = DoFAdapter(policy.cfg_action_dof.joint_names, env.joint_names)
                    heights = []
                    for _ in range(100):
                        env.update()
                        env_data = env.get_data()
                        env_data.dof_pos = obs_adapter.fit(env_data.dof_pos)
                        env_data.dof_vel = obs_adapter.fit(env_data.dof_vel)
                        obs, _ = policy.get_observation(env_data, Box({}))
                        target = action_adapter.fit(
                            policy.get_action(obs) + policy.default_pos,
                            template=env.default_pos,
                        )
                        env.step(target)
                        heights.append(float(env.data.qpos[2]))

                    self.assertGreater(min(heights), minimum_height)
                    self.assertLess(np.linalg.norm(env._base_rpy[:2]), 0.1)

    def test_x2_policy_mapping_holds_head_defaults(self):
        from robojudo.config.x2.policy.x2_amp_recovery_policy_cfg import X2AmpRecoveryPolicyCfg
        from robojudo.config.x2.policy.x2_locomanipulation_policy_cfg import X2LocomanipulationEnvDoF
        from robojudo.tools.dof import DoFAdapter

        env_dof = X2LocomanipulationEnvDoF()
        policy_dof = X2AmpRecoveryPolicyCfg().action_dof
        adapter = DoFAdapter(policy_dof.joint_names, env_dof.joint_names)
        action = np.arange(policy_dof.num_dofs, dtype=np.float32)
        target = adapter.fit(action, template=np.asarray(env_dof.default_pos, dtype=np.float32))

        np.testing.assert_array_equal(target[: policy_dof.num_dofs], action)
        np.testing.assert_array_equal(target[-2:], env_dof.default_pos[-2:])


if __name__ == "__main__":
    unittest.main()
