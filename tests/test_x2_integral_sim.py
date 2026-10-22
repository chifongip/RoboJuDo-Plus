"""Arm-only simulation effort, hand-center load and visualization regressions."""

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from robojudo.config.x2.env.x2_env_cfg import X2_ARM_JOINT_NAMES
from robojudo.config.x2.env.x2_mujuco_env_cfg import X2MujocoEnvCfg
from robojudo.config.x2.x2_cfg import x2_locomimic
from robojudo.environment.mujoco_env import MujocoEnv

if "x2_integral_sim" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "x2_integral_sim", Path(__file__).parents[1] / "scripts/x2_integral_sim.py"
    )
    evaluation = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = evaluation
    spec.loader.exec_module(evaluation)
evaluation = sys.modules["x2_integral_sim"]


class TestSimulationFeedforward(unittest.TestCase):
    def make_env(self, limits=None):
        cfg = x2_locomimic().env
        cfg.headless = True
        cfg.visualize_extras = False
        cfg.elastic_band.active = False
        cfg.feedforward_effort_limits = limits or {}
        env = MujocoEnv(cfg)
        self.addCleanup(env.shutdown)
        return env

    def test_config_rejects_nonarm_unknown_and_invalid_limits(self):
        for name, value in (
            ("left_knee_joint", 1.0),
            ("missing", 1.0),
            ("left_elbow_joint", -1.0),
            ("left_elbow_joint", float("inf")),
        ):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                X2MujocoEnvCfg(feedforward_effort_limits={name: value})

    def test_default_limits_disable_effort_and_omission_clears_it(self):
        env = self.make_env()
        elbow = env.joint_names.index("left_elbow_joint")
        effort = np.zeros(env.num_dofs)
        effort[elbow] = 0.1
        env.step(env.dof_pos, feedforward_effort=effort)
        np.testing.assert_array_equal(env.last_feedforward_effort, 0.0)
        env.cfg_env.feedforward_effort_limits = {"left_elbow_joint": 0.2}
        env.step(env.dof_pos, feedforward_effort=effort)
        self.assertEqual(env.last_feedforward_effort[elbow], 0.1)
        env.step(env.dof_pos)
        np.testing.assert_array_equal(env.last_feedforward_effort, 0.0)
        env.command_damping(2.0)
        np.testing.assert_array_equal(env.last_feedforward_effort, 0.0)

    def test_clipping_mask_and_final_total_torque_limit(self):
        env = self.make_env({"left_elbow_joint": 0.2, "right_elbow_joint": 0.2})
        elbow = env.joint_names.index("left_elbow_joint")
        env.set_control_joint_names(["left_elbow_joint"])
        target = env.dof_pos.copy()
        target[elbow] += 0.1
        env.torque_limits[elbow] = 0.25
        effort = np.ones(env.num_dofs)  # Unspecified joints cannot receive effort.
        env.step(target, feedforward_effort=effort)
        self.assertEqual(env.last_feedforward_effort[elbow], 0.2)
        np.testing.assert_array_equal(np.delete(env.last_feedforward_effort, elbow), 0)
        self.assertEqual(env.last_applied_torque[elbow], 0.25)
        self.assertTrue(env.last_saturation_mask[elbow])
        np.testing.assert_array_equal(np.delete(env.last_applied_torque, elbow), 0)

    def test_invalid_effort_rejected_without_advancing_physics(self):
        env = self.make_env({"left_elbow_joint": 0.2})
        start = env.data.time
        for effort in (
            np.zeros(env.num_dofs - 1),
            np.full(env.num_dofs, np.nan),
            np.full(env.num_dofs, np.inf),
            np.zeros((env.num_dofs, 1)),
        ):
            with self.subTest(shape=effort.shape), self.assertRaises(ValueError):
                env.step(env.dof_pos, feedforward_effort=effort)
            self.assertEqual(env.data.time, start)

    def test_x2_runtime_cannot_enable_leg_effort_by_mutating_config(self):
        env = self.make_env()
        env.cfg_env.feedforward_effort_limits = {"left_knee_joint": 1.0}
        with self.assertRaisesRegex(ValueError, "disallowed"):
            env.step(env.dof_pos, feedforward_effort=np.ones(env.num_dofs))

    def test_callback_updates_at_each_physics_substep(self):
        env = self.make_env({"left_elbow_joint": 0.2})
        calls = []

        def effort():
            calls.append(env.data.time)
            value = np.zeros(env.num_dofs)
            value[env.joint_names.index("left_elbow_joint")] = 0.1
            return value

        env.step(env.dof_pos, feedforward_effort=effort)
        self.assertEqual(len(calls), env.sim_decimation)
        np.testing.assert_allclose(np.diff(calls), env.sim_dt)

    def test_zero_effort_preserves_position_only_dynamics(self):
        first, second = self.make_env(), self.make_env()
        target = first.dof_pos.copy()
        for _ in range(5):
            first.step(target)
            second.step(target, feedforward_effort=np.zeros(second.num_dofs))
        np.testing.assert_array_equal(first.data.qpos, second.data.qpos)
        np.testing.assert_array_equal(first.data.qvel, second.data.qvel)


class TestHandCenterLoad(unittest.TestCase):
    def test_benchmark_selects_fine_tune_without_global_config_change(self):
        runtime = x2_locomimic()
        runtime.loco_policy.policy_name = "policy"
        with patch.object(evaluation, "x2_locomimic", side_effect=lambda: runtime.model_copy(deep=True)):
            cfg = evaluation.evaluation_config()
        self.assertEqual(cfg.loco_policy.policy_name, "policy_fine_tune")
        self.assertEqual(runtime.loco_policy.policy_name, "policy")

    def make_env(self):
        cfg = evaluation.evaluation_config()
        env = evaluation.X2IntegralEvaluationEnv(cfg.env)
        self.addCleanup(env.shutdown)
        for name, angle in evaluation.POSTURE.items():
            env.data.qpos[env.model.jnt_qposadr[env.model.joint(name).id]] = angle
        mujoco.mj_forward(env.model, env.data)
        env.update()
        return env

    def test_force_arrows_follow_wrist_force_and_clear_persistent_markers(self):
        class MarkerViewer:
            def __init__(self):
                self.markers = {}

            def add_marker(self, **marker):
                self.markers[marker["id"]] = marker

            def close(self):
                pass

        env = self.make_env()
        env.viewer = MarkerViewer()
        force = np.asarray([5.0, 0.0, -20.0])
        env.data.xfrc_applied[env.ee_ids, :3] = force
        before = env.data.xfrc_applied.copy()
        env._update_force_visualization()
        self.assertEqual(len(env.viewer.markers), 2)
        for index, body_id in enumerate(env.ee_ids):
            marker = env.viewer.markers[4100 + index]
            np.testing.assert_allclose(marker["pos"], env.data.xipos[body_id])
            self.assertGreater(np.linalg.norm(marker["pos"] - env.data.xpos[body_id]), 0.08)
            np.testing.assert_allclose(marker["mat"][:, 2] * marker["size"][2], 0.01 * force)
            self.assertIn("20.6 N", marker["label"])
            self.assertGreater(marker["rgba"][3], 0)
        np.testing.assert_array_equal(env.data.xfrc_applied, before)
        env.clear_load()
        self.assertEqual(len(env.viewer.markers), 2)  # Same IDs replace persistent arrows.
        for marker in env.viewer.markers.values():
            self.assertEqual(marker["rgba"][3], 0)
            self.assertEqual(marker["label"], "")

    def test_posture_model_and_limits_match_specification(self):
        cfg = evaluation.evaluation_config()
        self.assertEqual(cfg.loco_policy.policy_name, "policy_fine_tune")
        self.assertEqual(cfg.loco_policy.num_obs, 430)
        self.assertEqual(cfg.loco_policy.action_dof.num_dofs, 15)
        self.assertEqual(set(cfg.env.feedforward_effort_limits), set(X2_ARM_JOINT_NAMES))
        self.assertFalse(cfg.env.elastic_band.active)
        self.assertEqual(cfg.ctrl, [])
        self.assertEqual(evaluation.POSTURE["left_elbow_joint"], -2.188)
        self.assertEqual(evaluation.POSTURE["right_wrist_pitch_joint"], 0.326)

    def test_force_points_apply_expected_hand_center_wrench(self):
        env = self.make_env()
        env.force[:] = [0, 0, -20]
        env.update_load()
        expected = np.zeros(env.model.nv)
        force = np.asarray([0.0, 0.0, -20.0])
        for body_id in env.ee_ids:
            np.testing.assert_array_equal(env.data.xfrc_applied[body_id, :3], force)
            mujoco.mj_applyFT(env.model, env.data, force, np.zeros(3), env.data.xipos[body_id], body_id, expected)
            at_com = np.zeros(env.model.nv)
            wrench = env.data.xfrc_applied[body_id]
            mujoco.mj_applyFT(env.model, env.data, wrench[:3], wrench[3:], env.data.xipos[body_id], body_id, at_com)
            at_point = np.zeros(env.model.nv)
            mujoco.mj_applyFT(env.model, env.data, force, np.zeros(3), env.data.xipos[body_id], body_id, at_point)
            np.testing.assert_allclose(at_com, at_point, atol=1e-12)
            np.testing.assert_array_equal(wrench[3:], 0)  # Applied at COM: no extra COM moment.
            origin_load = np.zeros(env.model.nv)
            mujoco.mj_applyFT(env.model, env.data, force, np.zeros(3), env.data.xpos[body_id], body_id, origin_load)
            self.assertGreater(np.linalg.norm(origin_load - at_point), 0.1)
        np.testing.assert_array_equal(env.end_effector_positions()[0], env.data.xipos[env.ee_ids])
        nonarm = [i for i, name in enumerate(env.joint_names) if name not in X2_ARM_JOINT_NAMES]
        np.testing.assert_array_equal(env.raw_effort[nonarm], 0)
        self.assertGreater(np.linalg.norm(expected[:6]), 0)  # Floating-base load is not cancelled.
        env.force[:] = 0
        env.update_load()
        np.testing.assert_array_equal(env.raw_effort, 0)
        np.testing.assert_array_equal(env.data.xfrc_applied[env.ee_ids], 0)

    def test_real_environment_only_arm_effort_and_feedback_inputs(self):
        cfg = evaluation.evaluation_config().env
        cfg.headless = True
        env = evaluation.X2IntegralEvaluationEnv(cfg)
        self.addCleanup(env.shutdown)
        env.integral_enabled = True
        target = env.dof_pos.copy()
        target[env.arm_indices] += 0.05
        env.step(target)
        nonarm = np.ones(env.num_dofs, dtype=bool)
        nonarm[env.arm_indices] = False
        np.testing.assert_array_equal(env.last_feedforward_effort[nonarm], 0)
        self.assertGreater(np.max(env.last_feedforward_effort[env.arm_indices]), 0)
        # Turning on a load changes physics, but cannot inject a known-force compensation term.
        env.clear_load()
        np.testing.assert_array_equal(env.integral.effort, 0)
        self.assertFalse(env.integral_enabled)
        np.testing.assert_array_equal(env.data.xfrc_applied[env.ee_ids], 0)

    def test_synthetic_command_uses_actual_policy_selection(self):
        from robojudo.policy.x2_locomanipulation_policy import X2LocomanipulationPolicy

        policy = X2LocomanipulationPolicy(evaluation.evaluation_config().loco_policy, device="cpu")
        policy.close_progress() if hasattr(policy, "close_progress") else None
        self.assertAlmostEqual(policy._get_commands(evaluation.synthetic_velocity(0.5))[0], 0.5)


if __name__ == "__main__":
    unittest.main()
