"""The unloading harness must preserve acquired samples during delay/dropout."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_x2_observer_release import SimulationAdapter, X2ReleaseEvaluationEnv, local_trial

from robojudo.config.x2.observer_parameters import JOINT_NAMES


class ReleaseFeedbackTests(unittest.TestCase):
    def test_external_effort_is_rejected_before_simulation(self):
        env = X2ReleaseEvaluationEnv.__new__(X2ReleaseEvaluationEnv)
        with self.assertRaisesRegex(ValueError, "owns feedforward effort"):
            env.step(np.zeros(14), feedforward_effort=np.zeros(14))

    def assert_repeats_identical(self, records):
        seen, repeated = {}, 0
        for stamp, q, v in records:
            if stamp in seen:
                np.testing.assert_array_equal(q, seen[stamp][0])
                np.testing.assert_array_equal(v, seen[stamp][1])
                repeated += 1
            seen[stamp] = q, v
        self.assertGreaterEqual(repeated, 3)

    def test_local_delay_and_dropout_hold_complete_acquired_sample(self):
        records = []

        def capture(adapter, target, q, v, now, stamps):
            records.append((stamps[0], q.copy(), v.copy()))
            return np.zeros(14), 0

        with patch.object(SimulationAdapter, "issue", capture):
            local_trial("improved", 16, -4, "impaired", 1)
        self.assert_repeats_identical(records)

    def test_mujoco_feedback_hold_preserves_timestamp_noise_and_position(self):
        env = X2ReleaseEvaluationEnv.__new__(X2ReleaseEvaluationEnv)
        env.num_dofs, env.joint_names = 14, list(JOINT_NAMES)
        env.arm_indices = np.arange(14)
        env.stiffness, env.damping, env.torque_limits = np.full(14, 20), np.full(14, 2), np.full(14, 10)
        env.position_limits = np.tile([-1, 1], (14, 1))
        env.data = SimpleNamespace(time=0)
        env.configure("improved", 16, -20, "impaired", 1)
        records = []

        def capture(adapter, target, q, v, now, stamps):
            records.append((stamps[0], q.copy(), v.copy()))
            return np.zeros(14), 0

        with patch.object(SimulationAdapter, "issue", capture), patch("evaluate_x2_observer_release.MujocoEnv.step"):
            for elapsed in np.arange(6.94, 7.12, 0.02):
                env.data.time = elapsed
                env._dof_pos = np.full(14, elapsed / 100)
                env._dof_vel = np.full(14, elapsed / 10)
                env.step(np.zeros(14))
        self.assert_repeats_identical(records)


if __name__ == "__main__":
    unittest.main()
