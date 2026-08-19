import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
from box import Box


class FakeAimdkController:
    def __init__(self):
        self.positions = None
        self.shutdown_called = False
        self.passive_called = False
        self.damping = None
        self.position_armed = False

    def step(self, positions):
        self.positions = positions

    def set_passive(self):
        self.passive_called = True

    def set_damping(self, damping):
        self.damping = damping

    def arm_position_control(self):
        self.position_armed = True

    def shutdown(self):
        self.shutdown_called = True


class TestX2Integration(unittest.TestCase):
    def test_aimdk_state_executor_is_configured_for_isolated_latest_sample_callbacks(self):
        package = Path(__file__).parents[1] / "packages" / "aimdk_cpp" / "src"
        header = (package / "aimdk_controller.hpp").read_text(encoding="utf-8")
        source = (package / "aimdk_controller.cpp").read_text(encoding="utf-8")

        self.assertIn("MultiThreadedExecutor", header)
        self.assertIn("joint_callback_group_", header)
        self.assertIn("imu_callback_group_", header)
        self.assertIn("odometry_callback_group_", header)
        self.assertIn("joint_state_mutex_", header)
        self.assertIn("imu_state_mutex_", header)
        self.assertIn("odometry_state_mutex_", header)
        self.assertNotIn("std::mutex state_mutex_", header)
        self.assertIn("rclcpp::SensorDataQoS()", source)
        self.assertIn("state_qos.keep_last(1)", source)
        self.assertIn("executor_->spin()", source)
        self.assertIn("executor_->cancel()", source)
        self.assertNotIn("spin_some()", source)

    def test_aimdk_safety_state_machine_uses_hold_before_latched_damping(self):
        package = Path(__file__).parents[1] / "packages" / "aimdk_cpp" / "src"
        header = (package / "aimdk_controller.hpp").read_text(encoding="utf-8")
        source = (package / "aimdk_controller.cpp").read_text(encoding="utf-8")

        self.assertIn("enum class AimdkSafetyState { ACTIVE, HOLD, DAMPING }", header)
        self.assertIn("command_damping_timeout", header)
        self.assertIn("state_damping_timeout", header)
        self.assertIn("odometry_damping_timeout", header)
        self.assertIn("enter_hold_locked", source)
        self.assertIn("enter_damping_locked", source)
        self.assertIn("publish_hold_commands", source)
        self.assertIn("state_recovery_confirmed_", source)
        self.assertIn("command_generation_ <= recovery_command_generation_", source)
        self.assertIn("command_mode_ != AimdkCommandMode::POSITION", source)
        self.assertIn("Check the previous command's age", source)
        self.assertIn("Ignoring invalid AimDK joint sample", source)
        self.assertIn("Ignoring invalid AimDK IMU sample", source)
        self.assertIn("command_joint_names_ = std::move(validated_joint_names)", source)

    def test_x2_configs_construct(self):
        from robojudo.config.x2 import x2, x2_real
        from robojudo.config.x2.env.x2_real_env_cfg import X2AimdkCfg

        sim_cfg = x2()
        real_cfg = x2_real()

        self.assertEqual(sim_cfg.robot, "x2")
        self.assertEqual(sim_cfg.pipeline_type, "X2LocomanipulationPipeline")
        self.assertEqual(sim_cfg.env.dof.num_dofs, 31)
        self.assertEqual(sim_cfg.policy.action_dof.num_dofs, 29)
        self.assertEqual(sim_cfg.policy.num_obs, 151)
        self.assertEqual(sim_cfg.policy.max_timestep, -1)
        self.assertEqual(real_cfg.policy.max_timestep, -1)
        self.assertEqual(real_cfg.env.env_type, "AgiBotCppEnv")
        self.assertEqual(real_cfg.env.aimdk.node_name, "robojudo_aimdk_cpp")
        self.assertEqual(real_cfg.env.aimdk.control_dt, 0.02)
        self.assertEqual(real_cfg.env.aimdk.publish_dt, 0.002)
        self.assertEqual(real_cfg.env.aimdk.command_timeout, 0.1)
        self.assertEqual(real_cfg.env.aimdk.command_damping_timeout, 0.5)
        self.assertEqual(real_cfg.env.aimdk.state_timeout, 0.1)
        self.assertEqual(real_cfg.env.aimdk.state_damping_timeout, 0.5)
        self.assertEqual(real_cfg.env.aimdk.odometry_damping_timeout, 0.5)
        self.assertEqual(real_cfg.env.odometry_type, "NONE")
        self.assertEqual(real_cfg.env.aimdk.odometry_topic, "/aima/mc/leg_odometry")
        self.assertTrue(real_cfg.env.update_with_fk)
        self.assertIsNotNone(real_cfg.env.forward_kinematic)
        self.assertEqual(sim_cfg.env.elastic_band.body_name, "torso_link")
        self.assertTrue(sim_cfg.env.elastic_band.active)
        self.assertEqual(
            sim_cfg.ctrl[1].triggers,
            {
                "k": "[PASSIVE_DEFAULT]",
                "l": "[DAMPING_DEFAULT]",
                "i": "[JOINT_DEFAULT]",
                "j": "[RL_DEFAULT]",
                "7": "[ELASTIC_BAND_LOWER]",
                "8": "[ELASTIC_BAND_LIFT]",
                "9": "[ELASTIC_BAND_TOGGLE]",
            },
        )
        self.assertEqual(len(real_cfg.ctrl), 1)
        self.assertEqual(real_cfg.ctrl[0].ctrl_type, "RosJoystickCtrl")
        self.assertEqual(real_cfg.ctrl[0].profile, "xbox_bluetooth")
        with self.assertRaisesRegex(ValueError, "damping timeout must not be shorter"):
            X2AimdkCfg(command_timeout=0.2, command_damping_timeout=0.1)
        timeout_fields = (
            "command_timeout",
            "command_damping_timeout",
            "state_timeout",
            "state_damping_timeout",
            "odometry_timeout",
            "odometry_damping_timeout",
        )
        for field in timeout_fields:
            with self.subTest(field=field), self.assertRaises(ValueError):
                X2AimdkCfg(**{field: float("nan")})
        self.assertEqual(real_cfg.ctrl[0].topic, "/joy")

    def test_x2_real_odometry_configuration_rejects_an_empty_aimdk_topic(self):
        from pydantic import ValidationError

        from robojudo.config.x2.env.x2_real_env_cfg import X2AimdkCfg, X2RealEnvCfg

        for odometry_type in ("NONE", "DUMMY", "AIMDK"):
            with self.subTest(odometry_type=odometry_type):
                self.assertEqual(X2RealEnvCfg(odometry_type=odometry_type).odometry_type, odometry_type)

        with self.assertRaisesRegex(ValidationError, "odometry_topic must be set"):
            X2RealEnvCfg(odometry_type="AIMDK", aimdk=X2AimdkCfg(odometry_topic=""))

    def test_x2_joint_default_matches_working_controller(self):
        from robojudo.config.x2.env.x2_env_cfg import X2JointDefaultDoF

        dof = X2JointDefaultDoF()
        values = {
            name: (dof.default_pos[i], dof.stiffness[i], dof.damping[i]) for i, name in enumerate(dof.joint_names)
        }
        self.assertEqual(values["left_knee_joint"], (0.1, 80.0, 8.0))
        self.assertEqual(values["right_elbow_joint"], (-1.2, 50.0, 1.0))
        self.assertEqual(values["waist_pitch_joint"], (0.0, 300.0, 3.0))
        self.assertEqual(values["left_shoulder_roll_joint"], (0.0, 20.0, 1.0))
        self.assertEqual(values["head_yaw_joint"], (0.0, 20.0, 1.0))

    def test_x2_mode_requires_completed_joint_default_before_rl(self):
        from robojudo.pipeline.four_mode_pipeline import ControlMode
        from robojudo.pipeline.x2_locomanipulation_pipeline import X2LocomanipulationPipeline

        class FakeEnv:
            def __init__(self):
                self.joint_names = ["joint_a", "joint_b"]
                self.dof_pos = np.array([0.0, 0.3], dtype=np.float32)
                self.control_joint_names = None
                self.stiffness = None
                self.damping = None
                self.targets = []
                self.position_armed = False

            def set_control_joint_names(self, names):
                self.control_joint_names = list(names)

            def set_gains(self, stiffness, damping):
                self.stiffness = np.asarray(stiffness)
                self.damping = np.asarray(damping)

            def step(self, target):
                self.targets.append(np.asarray(target))

            def arm_position_control(self):
                self.position_armed = True

        class FakePolicy:
            def __init__(self):
                self.cfg_action_dof = type("ActionDof", (), {"joint_names": ["joint_a"]})()
                self.reset_count = 0
                self.default_pose_mode = None

            def reset(self):
                self.reset_count += 1

            def set_default_pose_mode(self, enabled):
                self.default_pose_mode = enabled

        pipeline = X2LocomanipulationPipeline.__new__(X2LocomanipulationPipeline)
        pipeline.mode = ControlMode.PASSIVE_DEFAULT
        pipeline.env = FakeEnv()
        pipeline.policy = FakePolicy()
        pipeline._joint_default_start = None
        pipeline._joint_default_step = 0
        pipeline._joint_default_steps = 3
        pipeline._joint_default_complete = False
        pipeline._joint_default_target = np.array([0.6, -0.3], dtype=np.float32)
        pipeline._joint_default_stiffness = np.array([40.0, 50.0], dtype=np.float32)
        pipeline._joint_default_damping = np.array([4.0, 5.0], dtype=np.float32)
        pipeline._rl_stiffness = np.array([120.0, 20.0], dtype=np.float32)
        pipeline._rl_damping = np.array([5.0, 2.0], dtype=np.float32)

        self.assertFalse(pipeline._enter_mode(ControlMode.RL_DEFAULT))
        self.assertTrue(pipeline._enter_mode(ControlMode.JOINT_DEFAULT))
        self.assertEqual(pipeline.env.control_joint_names, ["joint_a", "joint_b"])
        self.assertTrue(pipeline.env.position_armed)
        for _ in range(3):
            pipeline._step_joint_default()
        np.testing.assert_allclose(pipeline.env.targets[-1], pipeline._joint_default_target)
        self.assertTrue(pipeline._joint_default_complete)

        self.assertTrue(pipeline._enter_mode(ControlMode.RL_DEFAULT))
        self.assertEqual(pipeline.env.control_joint_names, ["joint_a", "joint_b"])
        self.assertEqual(pipeline.policy.reset_count, 1)
        self.assertFalse(pipeline.policy.default_pose_mode)

        pipeline._enter_mode(ControlMode.DAMPING_DEFAULT)
        self.assertFalse(pipeline._joint_default_complete)

    def test_four_mode_dry_run_infers_while_passive_without_robot_commands(self):
        from robojudo.pipeline.four_mode_pipeline import ControlMode
        from robojudo.pipeline.x2_locomanipulation_pipeline import X2LocomanipulationPipeline

        calls = []

        class FakeEnv:
            dof_pos = np.zeros(2, dtype=np.float32)

            def update(self):
                calls.append("update")

            def get_data(self):
                return Box({})

            def command_passive(self):
                calls.append("passive")

            def command_damping(self, damping):
                calls.append(("damping", damping))

            def step(self, target):
                calls.append(("step", target))

        pipeline = X2LocomanipulationPipeline.__new__(X2LocomanipulationPipeline)
        pipeline.mode = ControlMode.PASSIVE_DEFAULT
        pipeline.env = FakeEnv()
        pipeline.ctrl_manager = SimpleNamespace(
            get_ctrl_data=lambda env_data: Box({"COMMANDS": []}),
            post_step_callback=lambda ctrl_data: calls.append("ctrl_post"),
        )
        pipeline.policy = SimpleNamespace(post_step_callback=lambda commands: calls.append("policy_post"))
        pipeline.cfg = SimpleNamespace(debug=SimpleNamespace(log_obs=False))
        pipeline.visualizer = None
        pipeline.timestep = 0
        pipeline.do_safety_check = False
        pipeline._shutdown_requested = False
        pipeline._step_rl_policy = lambda env_data, ctrl_data, dry_run: (
            calls.append(("inference", dry_run)) or np.zeros(2, dtype=np.float32),
            {},
        )

        pipeline.step(dry_run=True)

        self.assertIn(("inference", True), calls)
        self.assertNotIn("passive", calls)
        self.assertFalse(any(isinstance(call, tuple) and call[0] in {"damping", "step"} for call in calls))

    def test_four_mode_dry_run_propagates_inference_errors_without_forcing_damping(self):
        from robojudo.pipeline.four_mode_pipeline import ControlMode
        from robojudo.pipeline.x2_locomanipulation_pipeline import X2LocomanipulationPipeline

        pipeline = X2LocomanipulationPipeline.__new__(X2LocomanipulationPipeline)
        pipeline.mode = ControlMode.PASSIVE_DEFAULT
        pipeline.env = SimpleNamespace(
            dof_pos=np.zeros(2, dtype=np.float32),
            update=lambda: None,
            get_data=lambda: Box({}),
        )
        pipeline.ctrl_manager = SimpleNamespace(get_ctrl_data=lambda env_data: Box({"COMMANDS": []}))
        pipeline.do_safety_check = False
        pipeline._shutdown_requested = False

        def fail_inference(env_data, ctrl_data, dry_run):
            raise ValueError("invalid policy observation")

        pipeline._step_rl_policy = fail_inference
        pipeline._force_damping = lambda reason: self.fail(f"unexpected damping fallback: {reason}")

        with self.assertRaisesRegex(ValueError, "invalid policy observation"):
            pipeline.step(dry_run=True)

    def test_x2_policy_and_environment_joint_orders_round_trip(self):
        from robojudo.config.x2.env.x2_env_cfg import X2_31DoF
        from robojudo.config.x2.policy.x2_deploy_policy_cfg import X2DeployDoF
        from robojudo.tools.dof import DoFAdapter

        env_dof = X2_31DoF()
        policy_dof = X2DeployDoF()
        policy_values = np.arange(1, 30, dtype=np.float32)

        policy_to_env = DoFAdapter(policy_dof.joint_names, env_dof.joint_names)
        env_values = policy_to_env.fit(policy_values, template=np.asarray(env_dof.default_pos, dtype=np.float32))
        expected_by_name = dict(zip(policy_dof.joint_names, policy_values, strict=True))

        for env_index, name in enumerate(env_dof.joint_names):
            if name in expected_by_name:
                self.assertEqual(env_values[env_index], expected_by_name[name], name)
            else:
                self.assertIn(name, {"head_yaw_joint", "head_pitch_joint"})
                self.assertEqual(env_values[env_index], env_dof.default_pos[env_index], name)

        env_to_policy = DoFAdapter(env_dof.joint_names, policy_dof.joint_names)
        np.testing.assert_array_equal(env_to_policy.fit(env_values), policy_values)

    def test_x2_joint_mapping_rejects_duplicates_and_missing_names(self):
        from robojudo.pipeline.rl_pipeline import PolicyWrapper

        with self.assertRaisesRegex(ValueError, "Duplicate policy action"):
            PolicyWrapper._validate_joint_names("policy action", ["joint_a", "joint_a"])
        with self.assertRaisesRegex(ValueError, "missing from environment"):
            PolicyWrapper._require_joint_subset("policy action", ["joint_b"], ["joint_a"])

    def test_partial_policy_target_holds_environment_default_pose(self):
        from robojudo.pipeline.rl_pipeline import PolicyWrapper
        from robojudo.tools.dof import DoFAdapter

        wrapper = PolicyWrapper.__new__(PolicyWrapper)
        wrapper.env_dof_cfg = type("EnvDof", (), {"default_pos": [0.1, 0.2, 0.3]})()
        wrapper.policy = type(
            "Policy",
            (),
            {
                "default_pos": np.array([1.0]),
                "get_action": lambda self, obs: np.array([0.5]),
            },
        )()
        wrapper.actions_adapter = DoFAdapter(["lower_joint"], ["lower_joint", "upper_a", "upper_b"])

        target = wrapper.get_pd_target(obs=None)

        np.testing.assert_allclose(target, np.array([1.5, 0.2, 0.3]))

    def test_multi_policy_switch_restores_full_joint_control(self):
        from robojudo.pipeline.rl_multi_policy_pipeline import PolicyManager

        class FakeEnv:
            joint_names = ["lower_joint", "upper_joint"]

            def __init__(self):
                self.control_joint_names = None
                self.override_cfg = None

            def reset(self):
                return

            def update_dof_cfg(self, override_cfg):
                self.override_cfg = override_cfg

            def set_control_joint_names(self, joint_names):
                self.control_joint_names = list(joint_names)

        action_dof = type("ActionDof", (), {"joint_names": ["lower_joint"]})()
        policy = type("Policy", (), {"cfg_action_dof": action_dof, "name": "lower_body"})()
        manager = PolicyManager.__new__(PolicyManager)
        manager.env = FakeEnv()
        manager.policies = [policy]
        manager._current_policy_id = 0
        manager.warmup_policy_indices = {0}

        manager.set_policy(0)

        self.assertIs(manager.env.override_cfg, action_dof)
        self.assertEqual(manager.env.control_joint_names, manager.env.joint_names)

    def test_x2_policy_inference_shape(self):
        from robojudo.config.x2.policy.x2_deploy_policy_cfg import X2DeployPolicyCfg
        from robojudo.policy.x2_deploy_policy import X2DeployPolicy

        cfg = X2DeployPolicyCfg()
        policy = X2DeployPolicy(cfg, "cpu")
        env_data = Box(
            {
                "base_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                "base_ang_vel": np.zeros(3, dtype=np.float32),
                "dof_pos": np.asarray(cfg.obs_dof.default_pos, dtype=np.float32),
                "dof_vel": np.zeros(cfg.obs_dof.num_dofs, dtype=np.float32),
            }
        )

        obs, _ = policy.get_observation(env_data, Box({}))
        action = policy.get_action(obs)

        self.assertEqual(obs.shape, (151,))
        self.assertEqual(action.shape, (29,))
        self.assertEqual(policy.heart_count, cfg.phase_start_count - 1.0 + cfg.warmup_frames)
        self.assertIsNone(policy.pbar)
        self.assertTrue(np.isfinite(action).all())

    def test_x2_policy_onnx_io_contract(self):
        from robojudo.config.x2.policy.x2_deploy_policy_cfg import X2DeployPolicyCfg
        from robojudo.policy.x2_deploy_policy import X2DeployPolicy

        cfg = X2DeployPolicyCfg()
        policy = X2DeployPolicy(cfg, "cpu")
        inputs = {inp.name: inp.shape for inp in policy.session.get_inputs()}
        outputs = {out.name: out.shape for out in policy.session.get_outputs()}

        self.assertEqual(inputs["obs"], [1, 151])
        self.assertEqual(inputs["time_step"], [1, 1])
        self.assertEqual(outputs["actions"], [1, 29])
        self.assertEqual(outputs["joint_pos"], [1, 29])
        self.assertEqual(outputs["joint_vel"], [1, 29])

    def test_x2_observation_segments_match_aimdk_order(self):
        from robojudo.config.x2.policy.x2_deploy_policy_cfg import X2DeployPolicyCfg
        from robojudo.policy.x2_deploy_policy import X2DeployPolicy

        cfg = X2DeployPolicyCfg(warmup_frames=0)
        policy = X2DeployPolicy(cfg, "cpu")
        pos_delta = np.linspace(-0.2, 0.2, cfg.obs_dof.num_dofs, dtype=np.float32)
        dof_vel = np.linspace(-1.0, 1.0, cfg.obs_dof.num_dofs, dtype=np.float32)
        env_data = Box(
            {
                "base_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                "base_ang_vel": np.array([4.0, -8.0, 12.0], dtype=np.float32),
                "dof_pos": np.asarray(cfg.obs_dof.default_pos, dtype=np.float32) + pos_delta,
                "dof_vel": dof_vel,
            }
        )

        obs, _ = policy.get_observation(env_data, Box({}))
        segments = {
            "command_dof_pos": obs[0:29],
            "command_dof_vel": obs[29:58],
            "projected_gravity": obs[58:61],
            "base_ang_vel": obs[61:64],
            "dof_pos": obs[64:93],
            "dof_vel": obs[93:122],
            "actions": obs[122:151],
        }

        self.assertEqual(obs.shape, (151,))
        np.testing.assert_allclose(segments["command_dof_pos"], np.zeros(29, dtype=np.float32))
        np.testing.assert_allclose(segments["command_dof_vel"], np.zeros(29, dtype=np.float32))
        np.testing.assert_allclose(segments["projected_gravity"], np.array([0.0, -0.0, -1.0], dtype=np.float32))
        np.testing.assert_allclose(segments["base_ang_vel"], np.array([1.0, -2.0, 3.0], dtype=np.float32))
        np.testing.assert_allclose(segments["dof_pos"], pos_delta, atol=1e-6)
        np.testing.assert_allclose(segments["dof_vel"], dof_vel * cfg.obs_scales["dof_vel"], atol=1e-6)
        np.testing.assert_allclose(segments["actions"], np.zeros(29, dtype=np.float32))

    def test_x2_policy_warmup_primes_reference_outputs(self):
        from robojudo.config.x2.policy.x2_deploy_policy_cfg import X2DeployPolicyCfg
        from robojudo.policy.x2_deploy_policy import X2DeployPolicy

        cfg = X2DeployPolicyCfg()
        policy = X2DeployPolicy(cfg, "cpu")
        env_data = Box(
            {
                "base_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                "base_ang_vel": np.zeros(3, dtype=np.float32),
                "dof_pos": np.asarray(cfg.obs_dof.default_pos, dtype=np.float32),
                "dof_vel": np.zeros(cfg.obs_dof.num_dofs, dtype=np.float32),
            }
        )

        obs, _ = policy.get_observation(env_data, Box({}))

        self.assertEqual(obs.shape, (151,))
        self.assertEqual(policy.heart_count, cfg.phase_start_count - 1.0 + cfg.warmup_frames)
        self.assertFalse(policy._needs_warmup)
        self.assertEqual(policy.mimic_ref_pos.shape, (29,))
        self.assertEqual(policy.mimic_ref_vel.shape, (29,))
        self.assertEqual(policy.last_action.shape, (29,))
        self.assertTrue(np.isfinite(policy.mimic_ref_pos).all())
        self.assertTrue(np.isfinite(policy.mimic_ref_vel).all())

    def test_x2_first_action_uses_final_warmup_output(self):
        from robojudo.config.x2.policy.x2_deploy_policy_cfg import X2DeployPolicyCfg
        from robojudo.policy.x2_deploy_policy import X2DeployPolicy

        cfg = X2DeployPolicyCfg()
        policy = X2DeployPolicy(cfg, "cpu")
        env_data = Box(
            {
                "base_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                "base_ang_vel": np.zeros(3, dtype=np.float32),
                "dof_pos": np.asarray(cfg.obs_dof.default_pos, dtype=np.float32),
                "dof_vel": np.zeros(cfg.obs_dof.num_dofs, dtype=np.float32),
            }
        )

        obs, _ = policy.get_observation(env_data, Box({}))
        warmup_action = policy.last_action.copy()
        first_action = policy.get_action(obs)

        np.testing.assert_allclose(first_action, warmup_action * cfg.action_scale)
        self.assertEqual(policy.heart_count, cfg.phase_start_count - 1.0 + cfg.warmup_frames)

        policy.get_action(obs)
        self.assertEqual(policy.heart_count, cfg.phase_start_count + cfg.warmup_frames)

    def test_x2_default_pose_mode_holds_without_advancing_policy(self):
        from robojudo.config.x2.policy.x2_deploy_policy_cfg import X2DeployPolicyCfg
        from robojudo.policy.x2_deploy_policy import X2DeployPolicy

        cfg = X2DeployPolicyCfg()
        policy = X2DeployPolicy(cfg, "cpu")
        env_data = Box(
            {
                "base_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                "base_ang_vel": np.zeros(3, dtype=np.float32),
                "dof_pos": np.asarray(cfg.obs_dof.default_pos, dtype=np.float32),
                "dof_vel": np.zeros(cfg.obs_dof.num_dofs, dtype=np.float32),
            }
        )

        policy.set_default_pose_mode(True)
        for _ in range(20):
            obs, _ = policy.get_observation(env_data, Box({}))
            np.testing.assert_array_equal(policy.get_action(obs), np.zeros(29, dtype=np.float32))
        self.assertEqual(policy.heart_count, cfg.phase_start_count - 1.0)
        self.assertTrue(policy._needs_warmup)

        policy.set_default_pose_mode(False)
        obs, _ = policy.get_observation(env_data, Box({}))
        first_action = policy.get_action(obs)
        self.assertEqual(policy.heart_count, cfg.phase_start_count - 1.0 + cfg.warmup_frames)
        self.assertTrue(np.isfinite(first_action).all())

    def test_x2_policy_emits_motion_done_at_max_timestep(self):
        from robojudo.config.x2.policy.x2_deploy_policy_cfg import X2DeployPolicyCfg
        from robojudo.policy.x2_deploy_policy import X2DeployPolicy

        cfg = X2DeployPolicyCfg(warmup_frames=0, max_timestep=3)
        progress_patcher = patch("robojudo.policy.x2_deploy_policy.ProgressBar")
        progress_cls = progress_patcher.start()
        self.addCleanup(progress_patcher.stop)
        policy = X2DeployPolicy(cfg, "cpu")
        progress = progress_cls.return_value
        progress_cls.assert_called_once_with("X2Deploy kuailechongbai", 3)
        env_data = Box(
            {
                "base_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                "base_ang_vel": np.zeros(3, dtype=np.float32),
                "dof_pos": np.asarray(cfg.obs_dof.default_pos, dtype=np.float32),
                "dof_vel": np.zeros(cfg.obs_dof.num_dofs, dtype=np.float32),
            }
        )

        for expected_count in (1.0, 2.0, 3.0):
            obs, extras = policy.get_observation(env_data, Box({}))
            self.assertEqual(extras["CALLBACK"], [])
            policy.get_action(obs)
            self.assertEqual(policy.heart_count, expected_count)

        obs, extras = policy.get_observation(env_data, Box({}))
        self.assertEqual(extras["CALLBACK"], ["[MOTION_DONE]"])
        policy.get_action(obs)
        self.assertEqual(policy.heart_count, 3.0)
        self.assertEqual([entry.args[0] for entry in progress.set.call_args_list], [1.0, 2.0, 3.0])
        progress.close.assert_called_once()

        policy.reset()
        self.assertEqual(progress_cls.call_count, 2)
        policy.set_default_pose_mode(True)
        for _ in range(5):
            obs, extras = policy.get_observation(env_data, Box({}))
            policy.get_action(obs)
            self.assertEqual(extras["CALLBACK"], [])
        self.assertEqual(policy.heart_count, cfg.phase_start_count - 1.0)
        self.assertFalse(policy.flag_motion_done)

    def test_x2_policy_rejects_unreachable_max_timestep(self):
        from pydantic import ValidationError

        from robojudo.config.x2.policy.x2_deploy_policy_cfg import X2DeployPolicyCfg

        with self.assertRaises(ValidationError):
            X2DeployPolicyCfg(max_timestep=2821)
        with self.assertRaises(ValidationError):
            X2DeployPolicyCfg(max_timestep=0)

    def test_x2_mujoco_actuators_are_mapped_by_joint_name(self):
        from robojudo.config.x2.env.x2_mujuco_env_cfg import X2MujocoEnvCfg
        from robojudo.environment.mujoco_env import MujocoEnv

        cfg = X2MujocoEnvCfg()
        model = mujoco.MjModel.from_xml_path(cfg.xml)
        actuator_indices = MujocoEnv._resolve_actuator_indices(model, cfg.dof.joint_names)

        shoulder_index = cfg.dof.joint_names.index("left_shoulder_pitch_joint")
        shoulder_actuator = int(actuator_indices[shoulder_index])
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, shoulder_actuator)
        self.assertEqual(actuator_name, "motor_left_shoulder_pitch_joint")
        self.assertEqual(len(np.unique(actuator_indices)), cfg.dof.num_dofs)

        torque = np.arange(cfg.dof.num_dofs, dtype=np.float64)
        ctrl = np.zeros(model.nu, dtype=np.float64)
        ctrl[actuator_indices] = torque
        self.assertEqual(ctrl[shoulder_actuator], torque[shoulder_index])

    def test_mujoco_mode_torques_and_control_mask(self):
        from robojudo.environment.mujoco_env import MujocoEnv

        env = MujocoEnv.__new__(MujocoEnv)
        env.num_dofs = 3
        env._dof_pos = np.array([0.1, -0.2, 0.3], dtype=np.float64)
        env._dof_vel = np.array([1.0, -2.0, 3.0], dtype=np.float64)
        env.stiffness = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        env.damping = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        env._control_mask = np.array([True, False, True])
        captured = []
        env._simulate_torque = lambda torque_fn: captured.append(np.asarray(torque_fn()))

        env.command_passive()
        np.testing.assert_array_equal(captured[-1], np.zeros(3))

        env.command_damping(5.0)
        np.testing.assert_array_equal(captured[-1], np.array([-5.0, 10.0, -15.0]))

        env.step(np.array([0.2, 0.4, -0.1]))
        expected_pd = np.array([0.0, 0.0, -21.0])
        np.testing.assert_allclose(captured[-1], expected_pd)

        env.joint_names = ["joint_a", "joint_b", "joint_c"]
        env.set_control_joint_names(env.joint_names)
        env.step(np.array([0.2, 0.4, -0.1]))
        expected_full_pd = np.array([0.0, 16.0, -21.0])
        np.testing.assert_allclose(captured[-1], expected_full_pd)

    def test_elastic_band_config_rejects_invalid_values(self):
        from pydantic import ValidationError

        from robojudo.environment.env_cfgs import ElasticBandCfg

        for field, value in (
            ("stiffness", -1.0),
            ("damping", -1.0),
            ("rest_length", -1.0),
            ("length_step", 0.0),
            ("visual_radius", 0.0),
            ("anchor_radius", 0.0),
        ):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                ElasticBandCfg(body_name="torso_link", **{field: value})

    def test_elastic_band_force_is_tension_only_and_finite(self):
        from robojudo.environment.utils.elastic_band import ElasticBand

        kwargs = {
            "position": np.array([0.0, 0.0, 1.0]),
            "velocity": np.zeros(3),
            "anchor_point": np.array([0.0, 0.0, 3.0]),
            "rest_length": 1.0,
            "stiffness": 200.0,
            "damping": 100.0,
        }
        np.testing.assert_allclose(ElasticBand.compute_force(**kwargs), np.array([0.0, 0.0, 200.0]))

        kwargs["velocity"] = np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(ElasticBand.compute_force(**kwargs), np.array([0.0, 0.0, 100.0]))

        kwargs["rest_length"] = 2.0
        np.testing.assert_array_equal(ElasticBand.compute_force(**kwargs), np.zeros(3))

        kwargs["position"] = kwargs["anchor_point"]
        np.testing.assert_array_equal(ElasticBand.compute_force(**kwargs), np.zeros(3))

        kwargs["position"] = np.array([np.nan, 0.0, 0.0])
        np.testing.assert_array_equal(ElasticBand.compute_force(**kwargs), np.zeros(3))

    def test_elastic_band_body_controls_and_reset(self):
        from robojudo.config.x2.env.x2_mujuco_env_cfg import X2MujocoEnvCfg
        from robojudo.environment.env_cfgs import ElasticBandCfg
        from robojudo.environment.utils.elastic_band import ElasticBand

        cfg = X2MujocoEnvCfg()
        model = mujoco.MjModel.from_xml_path(cfg.xml)
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        band = ElasticBand(cfg.elastic_band, model, data)

        self.assertGreater(np.linalg.norm(band.apply()), 0.0)
        self.assertEqual(band.lower(), 0.1)
        self.assertEqual(band.lift(), 0.0)
        self.assertEqual(band.lift(), 0.0)
        self.assertFalse(band.toggle())
        np.testing.assert_array_equal(data.xfrc_applied[band.body_id, :3], np.zeros(3))

        band.reset()
        self.assertTrue(band.active)
        self.assertEqual(band.rest_length, 0.0)

        with self.assertRaisesRegex(ValueError, "was not found"):
            ElasticBand(ElasticBandCfg(body_name="missing_body"), model, data)

    def test_elastic_band_visualization_tracks_body_and_toggle(self):
        from robojudo.config.x2.env.x2_mujuco_env_cfg import X2MujocoEnvCfg
        from robojudo.environment.utils.elastic_band import ElasticBand

        class FakeViewer:
            def __init__(self):
                self.markers = {}

            def add_marker(self, **marker):
                self.markers[marker["id"]] = marker

        cfg = X2MujocoEnvCfg()
        model = mujoco.MjModel.from_xml_path(cfg.xml)
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        band = ElasticBand(cfg.elastic_band, model, data)
        viewer = FakeViewer()

        band.update_visualization(viewer)
        line = viewer.markers[ElasticBand.BAND_MARKER_ID]
        anchor = viewer.markers[ElasticBand.ANCHOR_MARKER_ID]
        body_position = data.xpos[band.body_id]
        distance = np.linalg.norm(band.anchor_point - body_position)

        self.assertEqual(line["type"], mujoco.mjtGeom.mjGEOM_CAPSULE)
        np.testing.assert_allclose(line["pos"], (body_position + band.anchor_point) * 0.5)
        np.testing.assert_allclose(line["size"], [cfg.elastic_band.visual_radius] * 2 + [distance * 0.5])
        np.testing.assert_allclose(line["mat"][:, 2], (band.anchor_point - body_position) / distance)
        np.testing.assert_allclose(anchor["pos"], band.anchor_point)
        self.assertEqual(line["rgba"][3], 1.0)

        band.toggle()
        band.update_visualization(viewer)
        self.assertEqual(viewer.markers[ElasticBand.BAND_MARKER_ID]["rgba"][3], 0.0)
        self.assertEqual(viewer.markers[ElasticBand.ANCHOR_MARKER_ID]["rgba"][3], 0.0)

    def test_elastic_band_is_applied_each_mujoco_substep(self):
        from robojudo.environment.mujoco_env import MujocoEnv

        class FakeBand:
            def __init__(self):
                self.apply_count = 0

            def apply(self):
                self.apply_count += 1

        env = MujocoEnv.__new__(MujocoEnv)
        env.sim_decimation = 4
        env.num_dofs = 2
        env.torque_limits = np.full(2, 10.0)
        env._dof_actuator_indices = np.array([0, 1])
        env.data = SimpleNamespace(ctrl=np.zeros(2))
        env.model = object()
        env.elastic_band = FakeBand()
        env._render = lambda: None
        env.update = lambda simple=False: None

        with patch("robojudo.environment.mujoco_env.mujoco.mj_step") as mj_step:
            env._simulate_torque(lambda: np.zeros(2))

        self.assertEqual(env.elastic_band.apply_count, env.sim_decimation)
        self.assertEqual(mj_step.call_count, env.sim_decimation)

    def test_x2_pipeline_routes_elastic_band_commands(self):
        from robojudo.pipeline.x2_locomanipulation_pipeline import X2LocomanipulationPipeline

        class FakeEnv:
            def __init__(self):
                self.calls = []

            def lower_elastic_band(self):
                self.calls.append("lower")

            def lift_elastic_band(self):
                self.calls.append("lift")

            def toggle_elastic_band(self):
                self.calls.append("toggle")

        pipeline = X2LocomanipulationPipeline.__new__(X2LocomanipulationPipeline)
        pipeline.env = FakeEnv()
        pipeline._process_commands(["[ELASTIC_BAND_LOWER]", "[ELASTIC_BAND_LIFT]", "[ELASTIC_BAND_TOGGLE]"])
        self.assertEqual(pipeline.env.calls, ["lower", "lift", "toggle"])

    def test_x2_default_keyframe_matches_policy_pose(self):
        from robojudo.config.x2.env.x2_mujuco_env_cfg import X2MujocoEnvCfg
        from robojudo.config.x2.policy.x2_deploy_policy_cfg import X2DeployDoF
        from robojudo.tools.dof import merge_dof_cfgs

        cfg = X2MujocoEnvCfg()
        model = mujoco.MjModel.from_xml_path(cfg.xml)
        data = mujoco.MjData(model)
        expected = merge_dof_cfgs(cfg.dof, X2DeployDoF()).default_pos

        self.assertGreater(model.nkey, 0)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        np.testing.assert_allclose(data.qpos[-cfg.dof.num_dofs :], expected, atol=1e-7)

        mujoco.mj_forward(model, data)
        collision_geom_ids = np.flatnonzero(model.geom_group == 3)
        foot_clearances = []
        for side in ("left", "right"):
            foot_geom_ids = [
                geom_id
                for geom_id in collision_geom_ids
                if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id]))
                == f"{side}_ankle_roll_link"
            ]
            self.assertEqual(len(foot_geom_ids), 12)
            self.assertTrue(np.all(model.geom_type[foot_geom_ids] == mujoco.mjtGeom.mjGEOM_SPHERE))
            self.assertTrue(np.allclose(model.geom_size[foot_geom_ids, 0], 0.005))
            foot_clearances.extend(data.geom_xpos[foot_geom_ids, 2] - model.geom_size[foot_geom_ids, 0])
        self.assertGreaterEqual(min(foot_clearances), 0.0)
        self.assertLess(min(foot_clearances), 0.01)

    def test_x2_collision_geometry_matches_full_reference(self):
        from robojudo.config.x2.env.x2_mujuco_env_cfg import X2MujocoEnvCfg

        model = mujoco.MjModel.from_xml_path(X2MujocoEnvCfg().xml)
        collision_geom_ids = np.flatnonzero(model.geom_group == 3)
        self.assertEqual(len(collision_geom_ids), 49)
        self.assertEqual(np.count_nonzero(model.geom_type[collision_geom_ids] == mujoco.mjtGeom.mjGEOM_MESH), 24)
        self.assertEqual(np.count_nonzero(model.geom_type[collision_geom_ids] == mujoco.mjtGeom.mjGEOM_SPHERE), 24)
        self.assertEqual(np.count_nonzero(model.geom_type[collision_geom_ids] == mujoco.mjtGeom.mjGEOM_CYLINDER), 1)
        self.assertTrue(np.all(model.geom_condim[collision_geom_ids] == 3))
        self.assertTrue(np.all(model.geom_priority[collision_geom_ids] == 0))
        self.assertEqual(model.nexclude, 0)
        self.assertEqual((model.nq, model.nv, model.nu, model.nkey), (38, 37, 31, 1))

    def test_agibot_real_targets_are_clamped_by_hardware_joint_name(self):
        from robojudo.config.x2.env.x2_env_cfg import X2_31DoF
        from robojudo.config.x2.policy.x2_deploy_policy_cfg import X2_POLICY_JOINT_NAMES
        from robojudo.environment.agibot_cpp_env import AgiBotCppEnv

        dof = X2_31DoF()
        env = AgiBotCppEnv.__new__(AgiBotCppEnv)
        env.enabled = True
        env.aimdk = FakeAimdkController()
        env.num_dofs = dof.num_dofs
        env.joint_names = dof.joint_names
        env.position_limits = np.asarray(dof.position_limits, dtype=np.float64)
        env._control_joint_names = set(X2_POLICY_JOINT_NAMES)
        env._last_clamp_log_time = 0.0

        targets = np.asarray(dof.default_pos, dtype=np.float64)
        left_knee_index = dof.joint_names.index("left_knee_joint")
        targets[left_knee_index] = 10.0
        env.step(targets)

        self.assertEqual(env.aimdk.positions[left_knee_index], dof.position_limits[left_knee_index][1])
        for index, name in enumerate(dof.joint_names):
            if index != left_knee_index:
                self.assertEqual(env.aimdk.positions[index], targets[index], name)

    def test_agibot_passive_and_damping_commands_reach_backend(self):
        from robojudo.environment.agibot_cpp_env import AgiBotCppEnv

        env = AgiBotCppEnv.__new__(AgiBotCppEnv)
        env.enabled = True
        env.aimdk = FakeAimdkController()
        env.command_passive()
        env.command_damping(6.0)
        self.assertTrue(env.aimdk.passive_called)
        self.assertEqual(env.aimdk.damping, 6.0)

    def test_agibot_non_finite_target_enters_damping(self):
        from robojudo.config.x2.env.x2_env_cfg import X2_31DoF
        from robojudo.environment.agibot_cpp_env import AgiBotCppEnv

        dof = X2_31DoF()
        env = AgiBotCppEnv.__new__(AgiBotCppEnv)
        env.enabled = True
        env.aimdk = FakeAimdkController()
        env.num_dofs = dof.num_dofs
        env.joint_names = dof.joint_names
        env.position_limits = np.asarray(dof.position_limits, dtype=np.float64)
        env._control_joint_names = set(dof.joint_names)
        env._last_clamp_log_time = 0.0
        env.cfg_env = type("Cfg", (), {"aimdk": type("Aimdk", (), {"shutdown_damping": 5.0})()})()
        targets = np.asarray(dof.default_pos, dtype=np.float64)
        targets[0] = np.nan

        with self.assertRaisesRegex(FloatingPointError, "non-finite PD target"):
            env.step(targets)
        self.assertTrue(env.enabled)
        self.assertEqual(env.aimdk.damping, 5.0)

    def test_agibot_rejects_invalid_gains(self):
        from robojudo.environment.agibot_cpp_env import AgiBotCppEnv

        env = AgiBotCppEnv.__new__(AgiBotCppEnv)
        env.num_dofs = 2
        env.enabled = False
        env.aimdk = None

        valid = np.ones(2, dtype=np.float64)
        for stiffness, damping in (
            (np.array([np.nan, 1.0]), valid),
            (valid, np.array([np.inf, 1.0])),
            (np.array([-1.0, 1.0]), valid),
            (valid, np.array([-1.0, 1.0])),
        ):
            with self.subTest(stiffness=stiffness, damping=damping):
                with self.assertRaisesRegex(ValueError, "finite|non-negative"):
                    env.set_gains(stiffness, damping)

    def test_aimdk_cpp_optional_backend(self):
        try:
            from aimdk_cpp import AimdkController
        except ImportError as exc:
            if os.environ.get("ROBOJUDO_REQUIRE_AIMDK") == "1":
                self.fail(f"aimdk_cpp is required but could not be imported: {exc!r}")
            self.skipTest("aimdk_cpp is not built in this environment")

        from robojudo.config.x2.env.x2_real_env_cfg import X2RealEnvCfg
        from robojudo.config.x2.policy.x2_deploy_policy_cfg import X2_POLICY_JOINT_NAMES

        cfg_env = X2RealEnvCfg(act=False)
        cfg = cfg_env.aimdk.to_dict()
        cfg.update(
            {
                "act": False,
                "joint_names": cfg_env.dof.joint_names,
                "leg_joint_names": cfg_env.leg_joint_names,
                "waist_joint_names": cfg_env.waist_joint_names,
                "arm_joint_names": cfg_env.arm_joint_names,
                "head_joint_names": cfg_env.head_joint_names,
                "stiffness": cfg_env.dof.stiffness,
                "damping": cfg_env.dof.damping,
            }
        )
        invalid_cfg = dict(cfg)
        invalid_cfg["stiffness"] = list(cfg["stiffness"])
        invalid_cfg["stiffness"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "stiffness must contain only finite"):
            AimdkController(invalid_cfg)

        invalid_odometry_cfg = dict(cfg, enable_odometry=True, odometry_topic="")
        with self.assertRaisesRegex(ValueError, "odometry_topic must not be empty"):
            AimdkController(invalid_odometry_cfg)

        invalid_timeout_cfg = dict(cfg, command_damping_timeout=cfg["command_timeout"] / 2)
        with self.assertRaisesRegex(ValueError, "damping timeouts"):
            AimdkController(invalid_timeout_cfg)

        controller = AimdkController(cfg)
        try:
            self.assertTrue(controller.self_check(0.0))
            self.assertTrue(controller.state_is_fresh(cfg_env.aimdk.state_timeout))
            safety_status = controller.get_safety_status()
            self.assertEqual(safety_status.state, "ACTIVE")
            self.assertEqual(safety_status.fault, "NONE")
            self.assertFalse(safety_status.latched)
            report = controller.get_state_freshness_report(cfg_env.aimdk.state_timeout)
            self.assertFalse(report.required_streams_fresh)
            self.assertIn("imu_missing", report.reasons)
            self.assertIn("joints_missing", report.reasons)
            self.assertEqual(report.missing_joint_names, cfg_env.dof.joint_names)
            self.assertIsNone(report.imu_age_sec)
            self.assertEqual(report.stream_telemetry["leg"].topic, cfg_env.aimdk.leg_state_topic)
            self.assertEqual(report.stream_telemetry["imu"].received_count, 0)
            self.assertIsNone(report.stream_telemetry["imu"].receive_rate_hz)
            state = controller.get_robot_state()
            self.assertEqual(len(state.motor_state.q), 31)
            self.assertFalse(state.odometry_state.valid)
            self.assertEqual(state.odometry_state.position, [0.0, 0.0, 0.0])
            self.assertEqual(state.odometry_state.quaternion, [0.0, 0.0, 0.0, 1.0])
            self.assertEqual(state.odometry_state.linear_velocity, [0.0, 0.0, 0.0])
            controller.set_control_joint_names(X2_POLICY_JOINT_NAMES)
            controller.arm_position_control()
            controller.set_passive()
            controller.set_damping(5.0)
            invalid_damping = list(cfg["damping"])
            invalid_damping[0] = -1.0
            with self.assertRaisesRegex(ValueError, "damping must contain only finite"):
                controller.set_gains(cfg["stiffness"], invalid_damping)
        finally:
            controller.shutdown()


if __name__ == "__main__":
    unittest.main()
