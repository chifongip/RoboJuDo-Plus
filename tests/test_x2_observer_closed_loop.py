"""Deployment environment and observer wrapper driving a headless MuJoCo plant."""

import unittest
from unittest.mock import patch

import numpy as np
from test_x2_observer import FakeAdapter, config, environment

from robojudo.config.x2.observer_parameters import JOINT_NAMES
from robojudo.config.x2.x2_cfg import x2_locomimic
from robojudo.environment.mujoco_env import MujocoEnv
from robojudo.utils.observers import JointObserver


class PlantAdapter(FakeAdapter):
    def __init__(self, plant, indices):
        super().__init__()
        self.plant, self.indices = plant, indices

    def step(self, positions, efforts=None, expected=None):
        receipt = super().step(positions, efforts, expected)
        # Exercise a real downstream correction limiter, then acknowledge what
        # was admitted to the plant. Total PD+effort clipping remains in MuJoCo.
        full = np.zeros(self.plant.num_dofs)
        full[self.indices] = self.effort
        admitted = self.plant._bounded_feedforward(full)
        self.effort = admitted[self.indices].copy()
        receipt.arm_efforts = self.effort.tolist()
        self.plant.step(positions, feedforward_effort=admitted)
        return receipt


class DeploymentClosedLoopTests(unittest.TestCase):
    def test_modes_saturation_reconciliation_freshness_and_reentry(self):
        cfg = x2_locomimic().env
        cfg.headless, cfg.visualize_extras = True, False
        cfg.feedforward_effort_limits = dict.fromkeys(JOINT_NAMES, 0.05)
        plant = MujocoEnv(cfg)
        self.addCleanup(plant.shutdown)
        arms = np.array([plant.joint_names.index(name) for name in JOINT_NAMES])
        target = plant.dof_pos.copy()
        for mode in ("OFF", "SHADOW", "ACTIVE"):
            env = environment()
            env.joint_names, env.num_dofs = plant.joint_names, plant.num_dofs
            env.position_limits = plant.position_limits
            env.stiffness, env.damping = plant.stiffness, plant.damping
            env._control_joint_names = set(plant.joint_names)
            observer_cfg = config(mode).model_copy(update={"actuator_limits_nm": plant.torque_limits[arms].tolist()})
            env._observer = (
                None
                if mode == "OFF"
                else JointObserver(
                    observer_cfg,
                    plant.joint_names,
                    plant.stiffness,
                    plant.damping,
                    actuator_limits=plant.torque_limits,
                    max_dt=0.1,
                )
            )
            if env._observer is not None:
                env._configure_observer_order(list(JOINT_NAMES))
            env.aimdk = PlantAdapter(plant, arms)
            peak, limited = 0.0, False

            def advance(goal, stale=False, env=env, mode=mode):
                now = 1 + plant.data.time
                env._dof_pos, env._dof_vel = plant.dof_pos.copy(), plant.dof_vel.copy()
                env._observer_feedback_time = now - (1 if stale else 0)
                env._observer_sample_times = np.full(plant.num_dofs, now)
                with patch("robojudo.environment.agibot_cpp_env.time.monotonic", return_value=now):
                    env.step(goal)
                np.testing.assert_array_equal(np.delete(plant.last_feedforward_effort, arms), 0)
                self.assertTrue(np.all(abs(plant.last_applied_torque) <= plant.torque_limits))
                if env._observer is not None:
                    np.testing.assert_array_equal(env._observer.issued_nm, env.aimdk.effort)
                    np.testing.assert_array_equal(env._observer.observer.effort, env.aimdk.effort)
                if mode != "ACTIVE":
                    np.testing.assert_array_equal(env.aimdk.effort, 0)

            for _ in range(30):
                advance(target)
                peak = max(peak, float(np.max(abs(env.aimdk.effort))))
                if env._observer is not None:
                    limited |= np.any(abs(env._observer.limited_nm) > abs(env.aimdk.effort) + 1e-10)
            if mode == "ACTIVE":
                self.assertGreater(peak, 0)
                self.assertTrue(limited)
                saturated = target.copy()
                saturated[arms] += 3.0
                advance(saturated)
                pd = env.stiffness[arms] * (np.clip(saturated, *env.position_limits.T)[arms] - env._dof_pos[arms])
                pd -= env.damping[arms] * env._dof_vel[arms]
                self.assertTrue(np.all(env.aimdk.effort[pd > plant.torque_limits[arms]] <= 0))
                advance(target)
            if mode != "OFF":
                env.set_observer_eligible(False)
                advance(target)
                np.testing.assert_array_equal(env.aimdk.effort, 0)
                env.set_observer_eligible(True)
                advance(target)
                np.testing.assert_array_equal(env.aimdk.effort, 0)
                advance(target, stale=True)
                self.assertIsNone(env._observer.sample_times)
                advance(target)
                np.testing.assert_array_equal(env.aimdk.effort, 0)


if __name__ == "__main__":
    unittest.main()
