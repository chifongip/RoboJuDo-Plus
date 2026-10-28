import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from box import Box
from scipy.spatial.transform import Rotation

import torch
from robojudo.config.g1.g1_supermimic_cfg import (
    g1_23_maniloco_supermimic,
    g1_23_maniloco_supermimic_real,
    g1_23_supermimic_motion,
    g1_23_supermimic_velocity,
    g1_23_supermimic_velocity_locomimic,
    g1_23_supermimic_velocity_locomimic_real,
    g1_supermimic_motion,
    g1_supermimic_velocity,
    g1_supermimic_velocity_locomimic,
)
from robojudo.config.g1.policy.g1_supermimic_policy_cfg import (
    G1_23SuperMimicMotionPolicyCfg,
    G1_23SuperMimicVelocityPolicyCfg,
    G1SuperMimicMotionPolicyCfg,
    G1SuperMimicVelocityPolicyCfg,
)
from robojudo.policy.supermimic_policy import (
    TRACKED_BODIES,
    SuperMimicActor,
    SuperMimicMotionPolicy,
    SuperMimicVelocityPolicy,
    orientation_columns,
    tracked_bodies,
)


class TestSuperMimic(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        torch.manual_seed(7)

    def make_checkpoint(self, velocity=False, reverse=False, num_joints=29):
        if velocity:
            cfg_class = G1_23SuperMimicVelocityPolicyCfg if num_joints == 23 else G1SuperMimicVelocityPolicyCfg
        else:
            cfg_class = G1_23SuperMimicMotionPolicyCfg if num_joints == 23 else G1SuperMimicMotionPolicyCfg
        options = {} if velocity else {"motion_file": self.root / "walk.npz"}
        cfg = cfg_class(checkpoint=self.root / ("velocity.pt" if velocity else "motion.pt"), **options)
        names = cfg.obs_dof.joint_names
        order = list(reversed(range(num_joints))) if reverse else list(range(num_joints))
        default = torch.tensor(cfg.obs_dof.default_pos)[order]
        config = {
            "actor_dims": (16, 8),
            "reference_dim": cfg.reference_dim,
            "motion_anchor_ori_b": False if velocity else cfg.motion_anchor_ori_b,
            "relative_orientation": False,
            "actor_input_normalization": True,
        }
        model = SuperMimicActor(config, default, velocity=velocity)
        # Nontrivial frozen statistics exercise normalization and dtype handling.
        model.obs_norm.mean.copy_(torch.linspace(-0.2, 0.3, model.observation_dim))
        model.obs_norm.std.fill_(1.3)
        saved = {
            "kind": "supermimic_velocity_actor" if velocity else "supermimic_actor",
            "schema_version": 1,
            "config": config,
            "actor": model.state_dict(),
            "robot_metadata": {"default_q": default, "joint_names": [names[i] for i in order]},
            "simulation": {"control_dt": 0.02},
        }
        if velocity:
            saved.update(
                observation_schema=f"proprioceptive_{6 + 3 * num_joints}_v1",
                history_length=10,
                velocity_config={"command_scale": (2.0, 1.32, 3.35)},
            )
        else:
            saved.update(reference_schema="full_body_256_v1" if num_joints == 23 else "full_body_268", clips=["walk"])
            frames = 12
            positions = np.zeros((frames, 14, 3), dtype=np.float32)
            positions[:, :, 2] = 0.8
            positions[:, :, 0] = np.arange(frames)[:, None] * 0.02
            quats = np.zeros((frames, 14, 4), dtype=np.float32)
            quats[:, :, 0] = 1.0
            np.savez(
                self.root / "walk.npz",
                fps=50.0,
                quaternion_order="wxyz",
                joint_names=np.asarray(names),
                body_names=np.asarray(tracked_bodies(num_joints)),
                joint_pos=np.tile(np.asarray(cfg.obs_dof.default_pos), (frames, 1)),
                joint_vel=np.zeros((frames, num_joints)),
                body_pos_w=positions,
                body_quat_w=quats,
                body_lin_vel_w=np.zeros((frames, 14, 3)),
                body_ang_vel_w=np.zeros((frames, 14, 3)),
            )
        torch.save(saved, cfg.checkpoint)
        return cfg, saved

    @staticmethod
    def env_data(policy, yaw=0):
        quat = Rotation.from_euler("z", yaw).as_quat().astype(np.float32)
        return Box(
            dof_pos=policy.default_dof_pos.copy(),
            dof_vel=np.zeros(policy.num_dofs, dtype=np.float32),
            base_ang_vel=np.asarray([0.1, -0.2, 0.3], dtype=np.float32),
            base_quat=quat,
            torso_quat=quat,
        )

    def test_configs_use_original_environment_and_registry_imports(self):
        from robojudo.config import cfg_registry
        from robojudo.environment import env_registry
        from robojudo.environment.mujoco_env import MujocoEnv
        from robojudo.policy import policy_registry

        self.assertNotIn("SuperMimicMujocoEnv", env_registry.types)
        for cls in (
            g1_supermimic_motion,
            g1_supermimic_velocity,
            g1_23_supermimic_motion,
            g1_23_supermimic_velocity,
        ):
            cfg = cfg_registry.get(cls.__name__)()
            self.assertIs(env_registry.get(cfg.env.env_type), MujocoEnv)
            self.assertEqual(cfg.env.dof.num_dofs, cfg.policy.action_dof.num_dofs)
            self.assertAlmostEqual(cfg.env.sim_dt * cfg.env.sim_decimation, 1 / cfg.policy.freq)
            self.assertIsNotNone(policy_registry.get(cfg.policy.policy_type))
        self.assertEqual(g1_supermimic_motion().policy.reference_dim, 2680)
        self.assertEqual(g1_supermimic_velocity().policy.reference_dim, 730)

    def test_native_23_motion_uses_75_observations_and_rubber_hand_bodies(self):
        cfg, saved = self.make_checkpoint(num_joints=23, reverse=True)
        policy = SuperMimicMotionPolicy(cfg)
        self.assertEqual(policy.num_actions, 23)
        self.assertEqual(policy.observation_dim, 75)
        self.assertEqual(policy.model.reference_dim, 2560)
        self.assertEqual(policy.cfg_action_dof.joint_names, saved["robot_metadata"]["joint_names"])
        np.testing.assert_allclose(policy.action_scales, cfg.action_scales[::-1])
        data = self.env_data(policy)
        del data["torso_quat"]  # This checkpoint has no torso-anchor observation.
        obs, extras = policy.get_observation(data, {})
        self.assertEqual(obs.shape, (3310,))
        self.assertEqual(extras["frame"], 0)
        target = policy.get_action(obs)
        self.assertEqual(target.shape, (23,))
        self.assertTrue(np.isfinite(target).all())
        self.assertIn("left_wrist_roll_rubber_hand", policy.motion.body_names)
        self.assertNotIn("left_wrist_yaw_link", policy.motion.body_names)

    def test_cross_robot_checkpoint_is_rejected(self):
        cfg, _ = self.make_checkpoint(num_joints=23)
        with self.assertRaisesRegex(ValueError, "29-DoF"):
            SuperMimicMotionPolicy(G1SuperMimicMotionPolicyCfg(checkpoint=cfg.checkpoint, motion_file=cfg.motion_file))
        cfg, _ = self.make_checkpoint()
        with self.assertRaisesRegex(ValueError, "23-DoF"):
            SuperMimicMotionPolicy(
                G1_23SuperMimicMotionPolicyCfg(checkpoint=cfg.checkpoint, motion_file=cfg.motion_file)
            )

    def test_23_motion_rejects_29_dof_tracked_body_names(self):
        cfg, _ = self.make_checkpoint(num_joints=23)
        clip = self.root / "walk.npz"
        with np.load(clip, allow_pickle=False) as source:
            arrays = {key: source[key].copy() for key in source.files}
        arrays["body_names"] = np.asarray(TRACKED_BODIES)
        np.savez(clip, **arrays)
        with self.assertRaisesRegex(ValueError, "tracked bodies"):
            SuperMimicMotionPolicy(cfg)

    def test_motion_file_can_be_renamed_independently_of_checkpoint_clips(self):
        for num_joints in (23, 29):
            with self.subTest(num_joints=num_joints):
                cfg, _ = self.make_checkpoint(num_joints=num_joints)
                policy = SuperMimicMotionPolicy(cfg)
                env = self.env_data(policy)
                expected_obs, _ = policy.get_observation(env, {})
                expected_action = policy.get_action(expected_obs)
                cfg.motion_file = self.root / "new_motion.npz"
                cfg.motion_file.write_bytes((self.root / "walk.npz").read_bytes())
                direct_policy = SuperMimicMotionPolicy(cfg)
                obs, _ = direct_policy.get_observation(env, {})
                self.assertEqual(direct_policy.motion.name, "new_motion")
                np.testing.assert_array_equal(obs, expected_obs)
                np.testing.assert_array_equal(direct_policy.get_action(obs), expected_action)

    def test_direct_motion_file_keeps_npz_validation(self):
        cfg, _ = self.make_checkpoint(num_joints=23)
        cfg.motion_file = self.root / "invalid.npz"
        with np.load(self.root / "walk.npz", allow_pickle=False) as source:
            original = {key: source[key].copy() for key in source.files}
        for field, value, message in (
            ("fps", 30.0, "50 Hz"),
            ("quaternion_order", "xyzw", "wxyz"),
            ("joint_names", original["joint_names"][:-1], "joints"),
            ("joint_vel", None, "Missing SuperMimic motion fields"),
        ):
            with self.subTest(field=field):
                arrays = dict(original)
                if value is None:
                    del arrays[field]
                else:
                    arrays[field] = value
                np.savez(cfg.motion_file, **arrays)
                with self.assertRaisesRegex(ValueError, message):
                    SuperMimicMotionPolicy(cfg)
        cfg.motion_file = self.root / "raw_motion.csv"
        with self.assertRaisesRegex(ValueError, "NPZ"):
            SuperMimicMotionPolicy(cfg)

    def test_23_locomimic_preserves_locomanipulation_waist_control(self):
        from robojudo.config import cfg_registry
        from robojudo.policy import policy_registry

        cfg = cfg_registry.get("g1_23_maniloco_supermimic")()
        self.assertIsInstance(cfg, g1_23_maniloco_supermimic)
        self.assertEqual(cfg.env.dof.num_dofs, 23)
        self.assertEqual(cfg.loco_policy.obs_dof.num_dofs, 23)
        self.assertEqual(cfg.loco_policy.action_dof.num_dofs, 13)
        self.assertEqual(cfg.loco_policy.policy_type, "G1LocomanipulationPolicy")
        self.assertIsNotNone(policy_registry.get(cfg.loco_policy.policy_type))
        self.assertEqual(cfg.mimic_policies[0].action_dof.num_dofs, 23)
        self.assertEqual(cfg.upper_dof_num, 10)
        self.assertEqual(cfg.upper_dof_override_indices, list(range(-10, 0)))
        self.assertEqual(len(cfg.upper_dof_pos_default), 10)
        override_names = [cfg.env.dof.joint_names[i] for i in cfg.upper_dof_override_indices]
        self.assertNotIn("waist_yaw_joint", override_names)
        self.assertTrue(set(override_names).isdisjoint(cfg.loco_policy.action_dof.joint_names))
        np.testing.assert_allclose(cfg.env.dof.default_pos, cfg.loco_policy.obs_dof.default_pos)
        np.testing.assert_allclose(cfg.env.dof.stiffness, cfg.loco_policy.obs_dof.stiffness)
        np.testing.assert_allclose(cfg.env.dof.damping, cfg.loco_policy.obs_dof.damping)
        np.testing.assert_allclose(cfg.upper_dof_pos_default, cfg.loco_policy.obs_dof.default_pos[-10:])
        self.assertEqual(cfg.env.env_type, "MujocoEnv")

    def test_velocity_history_physical_commands_and_clipped_previous_action(self):
        cfg, _ = self.make_checkpoint(velocity=True)
        cfg.action_clip = 0.001
        policy = SuperMimicVelocityPolicy(cfg)
        env = self.env_data(policy)
        obs, _ = policy.get_observation(env, {"KeyboardCtrl": {"pressed_keys": ["w", "a", "q"]}})
        self.assertEqual(obs.shape, (933,))
        np.testing.assert_allclose(obs[-3:], cfg.keyboard_command)
        history = obs[:-3].reshape(10, 93)
        np.testing.assert_array_equal(history, np.repeat(history[:1], 10, axis=0))
        np.testing.assert_allclose(history[0, 58:64], [0.1, -0.2, 0.3, 0, 0, -1])
        before = policy.model.obs_norm.count.clone()
        action = policy.get_action(obs)
        np.testing.assert_allclose(action, policy.last_action * policy.action_scales)
        self.assertLessEqual(np.abs(policy.last_action).max(), cfg.action_clip + 1e-9)
        env.dof_vel = np.ones(29, dtype=np.float32)
        next_obs, _ = policy.get_observation(env, {})
        next_history = next_obs[:-3].reshape(10, 93)
        np.testing.assert_array_equal(next_history[:-1], history[1:])
        np.testing.assert_allclose(next_history[-1, -29:], policy.last_action)
        np.testing.assert_array_equal(policy.model.obs_norm.count, before)
        policy.reset()
        self.assertEqual(len(policy.history), 0)
        np.testing.assert_array_equal(policy.last_action, 0)

    def test_locomimic_config_preserves_full_body_control(self):
        from robojudo.config import cfg_registry
        from robojudo.pipeline import pipeline_registry

        cfg = cfg_registry.get("g1_supermimic_velocity_locomimic")()
        self.assertIsNotNone(pipeline_registry.get(cfg.pipeline_type))
        self.assertEqual(cfg.env.env_type, "MujocoEnv")
        self.assertEqual(cfg.loco_policy.policy_type, "SuperMimicVelocityPolicy")
        self.assertEqual(cfg.mimic_policies[0].policy_type, "SuperMimicMotionPolicy")
        self.assertEqual(cfg.upper_dof_num, 0)
        self.assertFalse(cfg.upper_dof_override_indices)
        self.assertFalse(cfg.upper_dof_pos_default)
        self.assertTrue(cfg.realign_on_policy_switch)
        self.assertEqual(cfg.ctrl[0].triggers["["], "[POLICY_MIMIC]")
        self.assertEqual(cfg.ctrl[0].triggers["]"], "[POLICY_LOCO]")

    def test_23_velocity_locomimic_preserves_full_body_control(self):
        from robojudo.config import cfg_registry

        cfg = cfg_registry.get("g1_23_supermimic_velocity_locomimic")()
        self.assertEqual(cfg.env.env_type, "MujocoEnv")
        self.assertEqual(cfg.env.dof.num_dofs, 23)
        self.assertIsInstance(cfg.loco_policy, G1_23SuperMimicVelocityPolicyCfg)
        self.assertIsInstance(cfg.mimic_policies[0], G1_23SuperMimicMotionPolicyCfg)
        self.assertEqual(cfg.loco_policy.obs_dof.num_dofs, 23)
        self.assertEqual(cfg.loco_policy.action_dof.num_dofs, 23)
        self.assertEqual(cfg.mimic_policies[0].action_dof.num_dofs, 23)
        self.assertEqual(cfg.upper_dof_num, 0)
        self.assertFalse(cfg.upper_dof_override_indices)
        self.assertFalse(cfg.upper_dof_pos_default)
        self.assertTrue(cfg.realign_on_policy_switch)
        self.assertEqual(cfg.loco_policy.checkpoint, g1_23_supermimic_velocity().policy.checkpoint)

    def test_23_switch_configs_are_independent_of_29_dof_configs(self):
        from robojudo.pipeline.pipeline_cfgs import RlLocoMimicPipelineCfg

        for cfg_class in (g1_23_supermimic_velocity_locomimic, g1_23_maniloco_supermimic):
            with self.subTest(config=cfg_class.__name__):
                self.assertEqual(cfg_class.__bases__, (RlLocoMimicPipelineCfg,))
                self.assertEqual(cfg_class().env.dof.num_dofs, 23)
                cfg = cfg_class()
                cfg.ctrl[0].triggers["["] = "[TEST]"
                self.assertEqual(g1_supermimic_velocity_locomimic().ctrl[0].triggers["["], "[POLICY_MIMIC]")
                self.assertEqual(cfg_class().ctrl[0].triggers["["], "[POLICY_MIMIC]")

    def test_real_23_configs_use_four_mode_runtime_and_logical_motor_mapping(self):
        from robojudo.config import cfg_registry
        from robojudo.config.g1.env.g1_env_cfg import G1_23_DOF_INDICES
        from robojudo.controller.ctrl_cfgs import UpperBodyZmqCtrlCfg
        from robojudo.pipeline import pipeline_registry
        from robojudo.pipeline.four_mode_pipeline import FourModePipelineMixin
        from robojudo.pipeline.rl_loco_mimic_pipeline import RlLocoMimicPipeline

        for cfg_class, sim_class in (
            (g1_23_supermimic_velocity_locomimic_real, g1_23_supermimic_velocity_locomimic),
            (g1_23_maniloco_supermimic_real, g1_23_maniloco_supermimic),
        ):
            with self.subTest(config=cfg_class.__name__):
                cfg = cfg_registry.get(cfg_class.__name__)()
                self.assertEqual(cfg_class.__bases__, (sim_class,))
                runtime = pipeline_registry.get(cfg.pipeline_type)
                self.assertTrue(issubclass(runtime, FourModePipelineMixin))
                self.assertTrue(issubclass(runtime, RlLocoMimicPipeline))
                self.assertFalse(cfg.env.is_sim)
                self.assertEqual(cfg.env.env_type, "UnitreeCppEnv")
                self.assertEqual(cfg.env.dof.num_dofs, 23)
                self.assertEqual(cfg.env.motor_dof_count, 29)
                self.assertEqual(cfg.env.joint2motor_idx, G1_23_DOF_INDICES)
                self.assertEqual(cfg.env.unitree.command_timeout, 0.1)
                self.assertEqual(cfg.env.unitree.state_timeout, 0.1)
                self.assertEqual(cfg.env.unitree.shutdown_damping, 5.0)
                self.assertTrue(cfg.do_safety_check)
                self.assertEqual(cfg.joint_default_duration, 1.5)
                self.assertEqual(cfg.default_damping, 5.0)
                self.assertEqual(cfg.joint_default_dof.joint_names, cfg.env.dof.joint_names)
                for field in ("default_pos", "stiffness", "damping"):
                    self.assertEqual(getattr(cfg.joint_default_dof, field), getattr(cfg.loco_policy.obs_dof, field))
                triggers = cfg.ctrl[0].triggers
                self.assertEqual(cfg.ctrl[0].ctrl_type, "UnitreeCtrl")
                for button, command in {
                    "A": "[PASSIVE_DEFAULT]",
                    "B": "[DAMPING_DEFAULT]",
                    "Y": "[JOINT_DEFAULT]",
                    "X": "[RL_DEFAULT]",
                    "Start": "[POLICY_MIMIC]",
                    "Select": "[POLICY_LOCO]",
                    "L1+R1+A": "[SHUTDOWN]",
                }.items():
                    self.assertEqual(triggers[button], command)
                self.assertEqual(cfg.mimic_policies[0].checkpoint, sim_class().mimic_policies[0].checkpoint)
                self.assertEqual(cfg.mimic_policies[0].motion_file, sim_class().mimic_policies[0].motion_file)
                upper_controllers = [ctrl for ctrl in cfg.ctrl if isinstance(ctrl, UpperBodyZmqCtrlCfg)]
                if cfg_class is g1_23_maniloco_supermimic_real:
                    self.assertEqual(upper_controllers[0].joint_names, cfg.env.dof.joint_names[13:])
                    self.assertEqual(cfg.loco_policy.action_dof.num_dofs, 13)
                else:
                    self.assertFalse(upper_controllers)
                    self.assertEqual(cfg.loco_policy.action_dof.num_dofs, 23)
                    self.assertFalse(cfg.upper_dof_override_indices)

    def test_native_23_velocity_history_and_action_mapping(self):
        cfg, _ = self.make_checkpoint(velocity=True, num_joints=23, reverse=True)
        policy = SuperMimicVelocityPolicy(cfg)
        self.assertEqual(policy.observation_dim, 75)
        self.assertEqual(policy.num_actions, 23)
        np.testing.assert_allclose(policy.action_scales, cfg.action_scales[::-1])
        env = self.env_data(policy)
        obs, _ = policy.get_observation(env, {})
        self.assertEqual(obs.shape, (753,))
        np.testing.assert_array_equal(obs[-3:], 0)
        history = obs[:-3].reshape(10, 75)
        np.testing.assert_array_equal(history, np.repeat(history[:1], 10, axis=0))
        action = policy.get_action(obs)
        self.assertEqual(action.shape, (23,))
        self.assertTrue(np.isfinite(action).all())
        env.dof_vel = np.ones(23, dtype=np.float32)
        obs, _ = policy.get_observation(env, {"KeyboardCtrl": {"pressed_keys": ["w"]}})
        np.testing.assert_allclose(obs[-3:], [cfg.keyboard_command[0], 0, 0])
        np.testing.assert_allclose(obs[:-3].reshape(10, 75)[-1, -23:], policy.last_action)
        policy.reset()
        self.assertEqual(len(policy.history), 0)

    def test_checkpoint_and_motion_paths_are_set_by_pipeline_configs(self):
        for single_cls, combined_cls in (
            (g1_supermimic_motion, g1_supermimic_velocity_locomimic),
            (g1_23_supermimic_motion, g1_23_maniloco_supermimic),
            (g1_23_supermimic_motion, g1_23_supermimic_velocity_locomimic),
        ):
            single, combined = single_cls(), combined_cls()
            motion = combined.mimic_policies[0]
            self.assertEqual(single.policy.checkpoint, motion.checkpoint)
            self.assertEqual(single.policy.motion_file, motion.motion_file)
            self.assertEqual(motion.motion_file.suffix, ".npz")

    def test_supermimic_switches_both_directions_and_restarts_motion(self):
        from robojudo.pipeline.rl_loco_mimic_pipeline import PolicyInterpManager

        for num_joints in (23, 29):
            with self.subTest(num_joints=num_joints):
                velocity, _ = self.make_checkpoint(velocity=True, num_joints=num_joints)
                motion, _ = self.make_checkpoint(num_joints=num_joints)
                dof = velocity.obs_dof
                env = SimpleNamespace(
                    dof_cfg=dof,
                    default_pos=np.asarray(dof.default_pos),
                    dof_pos=np.asarray(dof.default_pos),
                    update_dof_cfg=Mock(),
                )
                manager = PolicyInterpManager(velocity, [motion], env, realign_on_policy_switch=True)
                manager.DURATIONS_LOCO_MIMIC = [0, 2, 2]
                manager.DURATIONS_MIMIC_LOCO = [0, 2, 2]
                data = self.env_data(manager.policy)
                self.assertTrue(manager.switch_to_mimic())
                for _ in range(6):
                    manager.step(data, {})
                self.assertEqual(manager.current_policy_id, 1)
                self.assertEqual(manager.policy.timestep, 0)
                self.assertIsNone(manager.policy.heading_offset)
                obs, _ = manager.policy.get_observation(data, {})
                self.assertTrue(np.isfinite(manager.policy.get_pd_target(obs)).all())
                self.assertTrue(manager.switch_to_loco())
                for _ in range(6):
                    manager.step(data, {})
                self.assertEqual(manager.current_policy_id, 0)
                self.assertEqual(manager.interp_state, manager.InterpState.IDLE)
                self.assertTrue(manager.switch_to_mimic())
                for _ in range(6):
                    manager.step(data, {})
                self.assertEqual(manager.current_policy_id, 1)
                self.assertEqual(manager.policy.timestep, 0)

    def test_joint_order_mapping_and_metadata_rejection(self):
        cfg, saved = self.make_checkpoint(velocity=True, reverse=True)
        policy = SuperMimicVelocityPolicy(cfg)
        self.assertEqual(policy.cfg_action_dof.joint_names, saved["robot_metadata"]["joint_names"])
        np.testing.assert_allclose(policy.action_scales, cfg.action_scales[::-1])
        saved["actor"]["default_q"] = saved["actor"]["default_q"] + 0.1
        torch.save(saved, cfg.checkpoint)
        with self.assertRaisesRegex(ValueError, "default_q"):
            SuperMimicVelocityPolicy(cfg)

    def test_velocity_arbitration_and_si_limits(self):
        cfg, _ = self.make_checkpoint(velocity=True)
        policy = SuperMimicVelocityPolicy(cfg)
        data = {
            "VELOCITY_SOURCE": "VelocityZmqCtrl",
            "VelocityZmqCtrl": {"fresh": True, "linear_velocity": [3, -4, 0], "angular_velocity": [0, 0, 9]},
            "KeyboardCtrl": {"pressed_keys": ["s"]},
        }
        np.testing.assert_allclose(policy._get_command(data), [1, -0.5, 1])
        data["VelocityZmqCtrl"]["fresh"] = False
        np.testing.assert_array_equal(policy._get_command(data), 0)
        data["VELOCITY_SOURCE"] = "KeyboardCtrl"
        np.testing.assert_allclose(policy._get_command(data), [-0.5, 0, 0])

    def test_motion_heading_invariance_anchor_layout_and_end(self):
        cfg, _ = self.make_checkpoint()
        policy = SuperMimicMotionPolicy(cfg)
        observations = []
        for yaw in (0, 1.3, -2):
            policy.reset()
            obs, extras = policy.get_observation(self.env_data(policy, yaw), {})
            self.assertEqual(obs.shape, (3670,))
            np.testing.assert_allclose(obs[:990].reshape(10, 99)[0, 64:70], [1, 0, 0, 1, 0, 0], atol=1e-6)
            observations.append(obs)
            self.assertEqual(extras["frame"], 0)
        for obs in observations[1:]:
            np.testing.assert_allclose(obs, observations[0], atol=1e-6)
        self.assertTrue(np.isfinite(policy.get_action(obs)).all())
        policy.timestep = policy.motion.length
        _, extras = policy.get_observation(self.env_data(policy), {})
        self.assertEqual(extras["CALLBACK"], ["[MOTION_DONE]"])
        future = policy.motion.future(policy.motion.length - 1)["body_pos_w"]
        np.testing.assert_array_equal(future, np.repeat(future[:1], 10, axis=0))
        policy.post_step_callback(["[MOTION_RESET]"])
        self.assertEqual(policy.timestep, 0)
        self.assertIsNone(policy.heading_offset)
        policy.post_step_callback(["[MOTION_FADE_OUT]"])
        policy.post_step_callback()
        self.assertEqual(policy.timestep, 0)

    def test_motion_file_required_and_wrong_motion_architecture(self):
        cfg, saved = self.make_checkpoint()
        motion_file = cfg.motion_file
        cfg.motion_file = None
        with self.assertRaisesRegex(ValueError, "motion file"):
            SuperMimicMotionPolicy(cfg)
        cfg.motion_file = motion_file
        saved["config"]["motion_anchor_ori_b"] = False
        torch.save(saved, cfg.checkpoint)
        with self.assertRaisesRegex(ValueError, "motion_anchor_ori_b"):
            SuperMimicMotionPolicy(cfg)

    def test_anchor_and_reference_rotation_layouts_are_distinct(self):
        quat = torch.tensor([np.cos(0.3), 0, 0, np.sin(0.3)], dtype=torch.float32)
        matrix = Rotation.from_quat(quat.numpy()[[1, 2, 3, 0]]).as_matrix()
        np.testing.assert_allclose(orientation_columns(quat, interleaved=True), matrix[:, :2].flatten(), atol=1e-6)
        np.testing.assert_allclose(orientation_columns(quat), matrix[:, :2].T.flatten(), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
