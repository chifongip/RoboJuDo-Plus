import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from box import Box


class _FakeValue:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class _FakeTrackingSession:
    def __init__(self, joint_names, observation_names):
        self.joint_names = list(joint_names)
        self.observation_names = list(observation_names)
        self.num_dofs = len(self.joint_names)
        self.body_names = ["pelvis", "torso_link"]
        observation_sizes = {
            "command": 2 * self.num_dofs,
            "motion_anchor_pos_b": 3,
            "motion_anchor_ori_b": 6,
            "base_lin_vel": 3,
            "base_ang_vel": 3,
            "joint_pos": self.num_dofs,
            "joint_vel": self.num_dofs,
            "actions": self.num_dofs,
        }
        self.num_obs = sum(observation_sizes[name] for name in self.observation_names)
        self._inputs = [
            _FakeValue("obs", [1, self.num_obs]),
            _FakeValue("time_step", [1, 1]),
        ]
        self._outputs = [
            _FakeValue("actions", [1, self.num_dofs]),
            _FakeValue("joint_pos", [1, self.num_dofs]),
            _FakeValue("joint_vel", [1, self.num_dofs]),
            _FakeValue("body_pos_w", [1, len(self.body_names), 3]),
            _FakeValue("body_quat_w", [1, len(self.body_names), 4]),
        ]
        metadata_values = ",".join(["0.000"] * self.num_dofs)
        gain_values = ",".join(["1.000"] * self.num_dofs)
        self._metadata = {
            "action_scale": gain_values,
            "anchor_body_name": "torso_link",
            "body_names": ",".join(self.body_names),
            "command_names": "motion",
            "default_joint_pos": metadata_values,
            "joint_damping": gain_values,
            "joint_names": ",".join(self.joint_names),
            "joint_stiffness": gain_values,
            "observation_names": ",".join(self.observation_names),
        }

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def get_modelmeta(self):
        return SimpleNamespace(custom_metadata_map=self._metadata)

    def run(self, output_names, inputs):
        self.assert_valid_inputs(inputs)
        body_quat = np.zeros((1, len(self.body_names), 4), dtype=np.float32)
        body_quat[..., 0] = 1.0
        values = {
            "actions": np.zeros((1, self.num_dofs), dtype=np.float32),
            "joint_pos": np.zeros((1, self.num_dofs), dtype=np.float32),
            "joint_vel": np.zeros((1, self.num_dofs), dtype=np.float32),
            "body_pos_w": np.zeros((1, len(self.body_names), 3), dtype=np.float32),
            "body_quat_w": body_quat,
        }
        return [values[name] for name in output_names]

    def assert_valid_inputs(self, inputs):
        if inputs["obs"].shape != (1, self.num_obs):
            raise AssertionError(f"unexpected obs shape: {inputs['obs'].shape}")
        if inputs["time_step"].shape != (1, 1):
            raise AssertionError(f"unexpected time_step shape: {inputs['time_step'].shape}")


class TestBeyondMimicCrossRobot(unittest.TestCase):
    state_observations = [
        "command",
        "motion_anchor_pos_b",
        "motion_anchor_ori_b",
        "base_lin_vel",
        "base_ang_vel",
        "joint_pos",
        "joint_vel",
        "actions",
    ]
    no_state_observations = [
        "command",
        "motion_anchor_ori_b",
        "base_ang_vel",
        "joint_pos",
        "joint_vel",
        "actions",
    ]

    def test_configs_register_robot_specific_shared_policy_entries(self):
        from robojudo.config.g1.g1_cfg import (
            g1_23_beyondmimic,
            g1_23_beyondmimic_real,
        )
        from robojudo.config.x2 import x2_beyondmimic, x2_beyondmimic_real
        from robojudo.config.x2.env.x2_env_cfg import X2_HEAD_JOINT_NAMES

        g1_sim = g1_23_beyondmimic()
        g1_real = g1_23_beyondmimic_real()
        x2_sim = x2_beyondmimic()
        x2_real = x2_beyondmimic_real()

        self.assertEqual(g1_sim.policy.policy_type, "G1BeyondMimicPolicy")
        self.assertEqual(g1_sim.policy.obs_dof.num_dofs, 23)
        self.assertFalse(g1_sim.policy.without_state_estimator)
        self.assertEqual(g1_real.env.env_type, "UnitreeCppEnv")
        self.assertTrue(g1_real.do_safety_check)

        self.assertEqual(x2_sim.pipeline_type, "X2DeployPipeline")
        self.assertEqual(x2_sim.policy.policy_type, "X2BeyondMimicPolicy")
        self.assertEqual(x2_sim.policy.obs_dof.num_dofs, 29)
        self.assertEqual(x2_sim.env.dof.num_dofs, 31)
        self.assertTrue(x2_sim.policy.without_state_estimator)
        self.assertTrue(set(X2_HEAD_JOINT_NAMES).isdisjoint(x2_sim.policy.obs_dof.joint_names))
        self.assertEqual(x2_real.env.env_type, "AgiBotCppEnv")
        self.assertTrue(x2_real.do_safety_check)

    def test_robot_specific_policies_share_the_base_runtime(self):
        from robojudo.policy.beyondmimic_policy import BeyondMimicPolicyBase
        from robojudo.policy.g1_beyondmimic_policy import G1BeyondMimicPolicy
        from robojudo.policy.x2_beyondmimic_policy import X2BeyondMimicPolicy

        self.assertTrue(issubclass(G1BeyondMimicPolicy, BeyondMimicPolicyBase))
        self.assertTrue(issubclass(X2BeyondMimicPolicy, BeyondMimicPolicyBase))

    def test_g1_23_state_estimator_contract_and_inference(self):
        from robojudo.config.g1.policy.g1_beyondmimic_policy_cfg import G1_23BeyondMimicPolicyCfg
        from robojudo.policy.g1_beyondmimic_policy import G1BeyondMimicPolicy

        cfg = G1_23BeyondMimicPolicyCfg()
        session = _FakeTrackingSession(cfg.obs_dof.joint_names, self.state_observations)
        with patch("robojudo.policy.beyondmimic_policy.ort.InferenceSession", return_value=session):
            policy = G1BeyondMimicPolicy(cfg, "cpu")

        env_data = Box(
            {
                "base_ang_vel": np.zeros(3, dtype=np.float32),
                "base_lin_vel": np.zeros(3, dtype=np.float32),
                "dof_pos": np.zeros(23, dtype=np.float32),
                "dof_vel": np.zeros(23, dtype=np.float32),
                "torso_pos": np.asarray([0.0, 0.0, 0.8], dtype=np.float32),
                "torso_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            }
        )
        obs, _ = policy.get_observation(env_data, Box({}))
        action = policy.get_action(obs)

        self.assertEqual(obs.shape, (130,))
        self.assertEqual(action.shape, (23,))
        self.assertTrue(np.isfinite(obs).all())
        policy.close_progress()

    def test_x2_no_state_uses_base_quaternion_and_metadata_order(self):
        from robojudo.config.x2.policy.x2_beyondmimic_policy_cfg import X2BeyondMimicPolicyCfg
        from robojudo.policy.x2_beyondmimic_policy import X2BeyondMimicPolicy

        cfg = X2BeyondMimicPolicyCfg()
        observation_names = [
            "base_ang_vel",
            "command",
            "motion_anchor_ori_b",
            "joint_pos",
            "joint_vel",
            "actions",
        ]
        session = _FakeTrackingSession(cfg.obs_dof.joint_names, observation_names)
        with patch("robojudo.policy.beyondmimic_policy.ort.InferenceSession", return_value=session):
            policy = X2BeyondMimicPolicy(cfg, "cpu")

        env_data = Box(
            {
                "base_ang_vel": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
                "base_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                "base_lin_vel": None,
                "dof_pos": np.zeros(29, dtype=np.float32),
                "dof_vel": np.zeros(29, dtype=np.float32),
                "torso_pos": None,
                "torso_quat": None,
            }
        )
        obs, _ = policy.get_observation(env_data, Box({}))
        action = policy.get_action(obs)

        self.assertEqual(obs.shape, (154,))
        np.testing.assert_array_equal(obs[:3], env_data.base_ang_vel)
        self.assertEqual(action.shape, (29,))
        policy.close_progress()

    def test_x2_policy_wrapper_holds_head_defaults(self):
        from robojudo.config.x2.env.x2_env_cfg import X2_31DoF
        from robojudo.config.x2.policy.x2_beyondmimic_policy_cfg import X2BeyondMimicPolicyCfg
        from robojudo.pipeline.rl_pipeline import PolicyWrapper

        cfg = X2BeyondMimicPolicyCfg()
        env_dof = X2_31DoF()
        env_dof.default_pos[-2:] = [0.12, -0.08]
        session = _FakeTrackingSession(cfg.obs_dof.joint_names, self.no_state_observations)
        with patch("robojudo.policy.beyondmimic_policy.ort.InferenceSession", return_value=session):
            wrapper = PolicyWrapper(cfg, env_dof, "cpu")

        env_data = Box(
            {
                "base_ang_vel": np.zeros(3, dtype=np.float32),
                "base_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                "base_lin_vel": None,
                "dof_pos": np.asarray(env_dof.default_pos, dtype=np.float32),
                "dof_vel": np.zeros(31, dtype=np.float32),
                "torso_pos": None,
                "torso_quat": None,
            }
        )
        obs, _ = wrapper.get_observation(env_data, Box({}))
        target = wrapper.get_pd_target(obs)

        self.assertEqual(obs.shape, (154,))
        self.assertEqual(target.shape, (31,))
        np.testing.assert_allclose(target[-2:], [0.12, -0.08])
        wrapper.close_progress()


if __name__ == "__main__":
    unittest.main()
