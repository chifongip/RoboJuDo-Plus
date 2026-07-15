import unittest
from types import SimpleNamespace

import numpy as np
from box import Box


class TestX2Locomanipulation(unittest.TestCase):
    def test_configs_use_recorded_training_parameters(self):
        from robojudo.config.x2 import x2_locomanipulation, x2_locomanipulation_real
        from robojudo.config.x2.policy.x2_locomanipulation_policy_cfg import (
            X2_LOCOMANIPULATION_ACTION_SCALES,
            X2_LOCOMANIPULATION_DAMPING,
            X2_LOCOMANIPULATION_DEFAULT_POS,
            X2_LOCOMANIPULATION_EFFORT_LIMITS,
            X2_LOCOMANIPULATION_JOINT_NAMES,
            X2_LOCOMANIPULATION_STIFFNESS,
        )

        sim_cfg = x2_locomanipulation()
        real_cfg = x2_locomanipulation_real()
        policy = sim_cfg.policy

        self.assertEqual(policy.obs_dof.joint_names, X2_LOCOMANIPULATION_JOINT_NAMES)
        self.assertEqual(policy.action_dof.joint_names, X2_LOCOMANIPULATION_JOINT_NAMES[:15])
        self.assertEqual(policy.obs_dof.default_pos, X2_LOCOMANIPULATION_DEFAULT_POS)
        self.assertEqual(policy.obs_dof.stiffness, X2_LOCOMANIPULATION_STIFFNESS)
        self.assertEqual(policy.obs_dof.damping, X2_LOCOMANIPULATION_DAMPING)
        self.assertEqual(policy.obs_dof.torque_limits, X2_LOCOMANIPULATION_EFFORT_LIMITS)
        self.assertEqual(policy.action_scales, X2_LOCOMANIPULATION_ACTION_SCALES)
        self.assertEqual(policy.commands_map[0], [-0.5, 0.0, 1.0])
        self.assertEqual(policy.commands_map[1], [0.5, 0.0, -0.5])
        self.assertEqual(policy.commands_map[2], [1.0, 0.0, -1.0])
        self.assertEqual(policy.commands_map[3], [0.4, 0.64, 0.66])
        self.assertEqual(policy.commands_map[4], [-1.5708, 0.0, 1.5708])

        self.assertEqual(sim_cfg.env.sim_dt, 0.005)
        self.assertEqual(sim_cfg.env.sim_decimation, 4)
        self.assertEqual(
            [ctrl.ctrl_type for ctrl in sim_cfg.ctrl],
            ["JoystickCtrl", "KeyboardCtrl", "UpperBodyZmqCtrl"],
        )
        self.assertEqual(sim_cfg.ctrl[-1].joint_names, sim_cfg.env.dof.joint_names[15:29])
        self.assertEqual(sim_cfg.ctrl[0].triggers["Start"], "[UPPER_BODY_TOGGLE]")
        self.assertEqual(sim_cfg.ctrl[1].triggers["t"], "[UPPER_BODY_TOGGLE]")
        self.assertEqual(real_cfg.env.env_type, "AgiBotCppEnv")
        self.assertEqual([ctrl.ctrl_type for ctrl in real_cfg.ctrl], ["JoystickCtrl", "UpperBodyZmqCtrl"])
        self.assertTrue(real_cfg.do_safety_check)

    def test_upper_body_zmq_override_changes_only_arms(self):
        from robojudo.config.x2 import x2_locomanipulation
        from robojudo.pipeline.x2_deploy_pipeline import X2DeployPipeline

        cfg = x2_locomanipulation()
        pipeline = X2DeployPipeline.__new__(X2DeployPipeline)
        pipeline._upper_body_cfg = cfg.ctrl[-1].model_copy(update={"ema_alpha": 0.0})
        pipeline._upper_body_enabled = True
        pipeline._upper_body_stream_was_fresh = False
        pipeline._upper_body_indices = np.asarray(
            [cfg.env.dof.joint_names.index(name) for name in pipeline._upper_body_cfg.joint_names]
        )
        pipeline._upper_body_default = np.asarray(cfg.env.dof.default_pos)[pipeline._upper_body_indices]
        pipeline._upper_body_filtered = pipeline._upper_body_default.copy()
        pipeline.env = SimpleNamespace(position_limits=np.asarray(cfg.env.dof.position_limits))

        policy_target = np.asarray(cfg.env.dof.default_pos, dtype=np.float32)
        command = {
            "UpperBodyZmqCtrl": {
                "fresh": True,
                "joint_positions": {
                    "left_shoulder_pitch_joint": 0.8,
                    "right_elbow_joint": -0.5,
                },
            }
        }
        result = pipeline._apply_upper_body_override(policy_target, command)

        left_shoulder = cfg.env.dof.joint_names.index("left_shoulder_pitch_joint")
        right_elbow = cfg.env.dof.joint_names.index("right_elbow_joint")
        self.assertEqual(result[left_shoulder], 0.8)
        self.assertEqual(result[right_elbow], -0.5)
        np.testing.assert_array_equal(result[:15], policy_target[:15])
        np.testing.assert_array_equal(result[-2:], policy_target[-2:])

        command["UpperBodyZmqCtrl"]["joint_positions"] = {"right_shoulder_roll_joint": 100.0}
        clamped = pipeline._apply_upper_body_override(policy_target, command)
        right_shoulder_roll = cfg.env.dof.joint_names.index("right_shoulder_roll_joint")
        self.assertAlmostEqual(clamped[right_shoulder_roll], 0.061)

        pipeline._upper_body_cfg = cfg.ctrl[-1]
        pipeline._upper_body_stream_was_fresh = True
        before_timeout = pipeline._upper_body_filtered.copy()
        returned = pipeline._apply_upper_body_override(policy_target, {"UpperBodyZmqCtrl": {"fresh": False}})
        expected = 0.95 * before_timeout + 0.05 * pipeline._upper_body_default
        np.testing.assert_allclose(returned[pipeline._upper_body_indices], expected, rtol=1e-6)

    def test_upper_body_toggle_requires_rl_mode(self):
        from robojudo.config.x2 import x2_locomanipulation
        from robojudo.pipeline.x2_deploy_pipeline import X2ControlMode, X2DeployPipeline

        pipeline = X2DeployPipeline.__new__(X2DeployPipeline)
        pipeline._upper_body_cfg = x2_locomanipulation().ctrl[-1]
        pipeline._upper_body_enabled = False
        pipeline._upper_body_stream_was_fresh = False
        pipeline.mode = X2ControlMode.JOINT_DEFAULT
        pipeline._toggle_upper_body()
        self.assertFalse(pipeline._upper_body_enabled)

        pipeline.mode = X2ControlMode.RL_DEFAULT
        pipeline._toggle_upper_body()
        self.assertTrue(pipeline._upper_body_enabled)
        pipeline._toggle_upper_body()
        self.assertFalse(pipeline._upper_body_enabled)

    def test_full_dof_holds_recorded_upper_body_and_zero_head(self):
        from robojudo.config.x2.policy.x2_locomanipulation_policy_cfg import (
            X2LocomanipulationEnvDoF,
            X2LocomanipulationPolicyCfg,
        )
        from robojudo.pipeline.rl_pipeline import PolicyWrapper
        from robojudo.tools.dof import DoFAdapter

        cfg = X2LocomanipulationPolicyCfg()
        env_dof = X2LocomanipulationEnvDoF()
        wrapper = PolicyWrapper.__new__(PolicyWrapper)
        wrapper.env_dof_cfg = env_dof
        wrapper.policy = SimpleNamespace(
            default_pos=np.asarray(cfg.action_dof.default_pos),
            get_action=lambda obs: np.zeros(cfg.action_dof.num_dofs),
        )
        wrapper.actions_adapter = DoFAdapter(cfg.action_dof.joint_names, env_dof.joint_names)

        target = wrapper.get_pd_target(None)

        np.testing.assert_allclose(target, env_dof.default_pos)
        self.assertEqual(target[env_dof.joint_names.index("left_elbow_joint")], -0.87)
        self.assertEqual(target[env_dof.joint_names.index("right_shoulder_roll_joint")], -0.1)
        np.testing.assert_array_equal(target[-2:], np.zeros(2))

    def test_observation_matches_mjlab_term_history_layout(self):
        from robojudo.config.x2.policy.x2_locomanipulation_policy_cfg import X2LocomanipulationPolicyCfg
        from robojudo.policy.x2_locomanipulation_policy import X2LocomanipulationPolicy

        cfg = X2LocomanipulationPolicyCfg()
        policy = X2LocomanipulationPolicy(cfg, "cpu")
        default_pos = np.asarray(cfg.obs_dof.default_pos, dtype=np.float32)
        env_data = Box(
            {
                "base_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                "base_ang_vel": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "dof_pos": default_pos,
                "dof_vel": np.zeros(29, dtype=np.float32),
            }
        )

        first, _ = policy.get_observation(env_data, Box({}))
        self.assertEqual(first.shape, (430,))
        np.testing.assert_array_equal(first[:15], np.tile(env_data.base_ang_vel, 5))
        np.testing.assert_array_equal(first[15:30], np.tile([0.0, 0.0, -1.0], 5))

        env_data.base_ang_vel = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        second, _ = policy.get_observation(env_data, Box({}))
        np.testing.assert_array_equal(
            second[:15],
            np.concatenate([np.tile([1.0, 2.0, 3.0], 4), [4.0, 5.0, 6.0]]),
        )
        # Standing commands zero the complete five-sample phase term.
        np.testing.assert_array_equal(second[55:65], np.zeros(10))

    def test_action_scale_is_applied_once_and_raw_action_is_remembered(self):
        from robojudo.config.x2.policy.x2_locomanipulation_policy_cfg import X2LocomanipulationPolicyCfg
        from robojudo.policy.x2_locomanipulation_policy import X2LocomanipulationPolicy

        cfg = X2LocomanipulationPolicyCfg()
        policy = X2LocomanipulationPolicy(cfg, "cpu")
        raw_action = np.linspace(-1.0, 1.0, 15, dtype=np.float32)
        policy.session = SimpleNamespace(run=lambda outputs, inputs: [raw_action.reshape(1, -1)])

        action = policy.get_action(np.zeros(430, dtype=np.float32))

        np.testing.assert_allclose(action, raw_action * np.asarray(cfg.action_scales), rtol=1e-6)
        np.testing.assert_array_equal(policy.last_action, raw_action)

    def test_keyboard_commands_use_recorded_limits_and_reset(self):
        from robojudo.config.x2.policy.x2_locomanipulation_policy_cfg import X2LocomanipulationPolicyCfg
        from robojudo.policy.x2_locomanipulation_policy import X2LocomanipulationPolicy

        policy = X2LocomanipulationPolicy(X2LocomanipulationPolicyCfg(), "cpu")
        pressed = [
            {"type": "keyboard", "name": name, "pressed": True}
            for name in ("w", "a", "q", "r", "z")
        ]
        commands = policy._get_commands(Box({"KeyboardCtrl": {"keyboard_event": pressed}}))

        np.testing.assert_allclose(commands[:3], [1.0, 0.5, 1.0])
        self.assertGreater(commands[3], 0.64)
        self.assertGreater(commands[4], 0.0)

        reset = [
            *[
                {"type": "keyboard", "name": name, "pressed": False}
                for name in ("w", "a", "q", "r", "z")
            ],
            {"type": "keyboard", "name": "x", "pressed": True},
        ]
        policy._get_commands(Box({"KeyboardCtrl": {"keyboard_event": reset}}))
        np.testing.assert_array_equal(policy.current_vel_cmd, np.zeros(3))
        self.assertAlmostEqual(policy._target_height, 0.64)
        self.assertEqual(policy._target_waist_yaw, 0.0)

    def test_onnx_contract_and_inference_are_finite(self):
        from robojudo.config.x2.policy.x2_locomanipulation_policy_cfg import X2LocomanipulationPolicyCfg
        from robojudo.policy.x2_locomanipulation_policy import X2LocomanipulationPolicy

        cfg = X2LocomanipulationPolicyCfg()
        policy = X2LocomanipulationPolicy(cfg, "cpu")
        action = policy.get_action(np.zeros(cfg.num_obs, dtype=np.float32))

        self.assertEqual({value.name: value.shape for value in policy.session.get_inputs()}, {"obs": [1, 430]})
        self.assertEqual({value.name: value.shape for value in policy.session.get_outputs()}, {"actions": [1, 15]})
        self.assertEqual(action.shape, (15,))
        self.assertTrue(np.isfinite(action).all())

if __name__ == "__main__":
    unittest.main()
