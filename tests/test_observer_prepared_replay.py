"""Differential replay against the uncached algorithm, including invalid intervals."""

import unittest
from unittest.mock import patch

import numpy as np

from robojudo.config.x2 import x2_locomimic_observer
from robojudo.utils.observers import JointObserver
from robojudo.utils.observers.runtime import _PreparedHistory


class UncachedObserver(JointObserver):
    # Frozen reference implementation from before replay preparation.
    def _reference_history(self, qd, stamps, disturbance=None):
        """Replay accepted commands only over each joint's observed receive interval.

        Receive time is not physical acquisition time. No assumption that Python's
        monotonic clock shares an epoch with the native clock is needed.
        """
        if self.sample_times is None:
            return (qd.copy(), np.ones(self._shape)), np.zeros(self._shape), np.zeros(self._shape, dtype=bool)
        elapsed = stamps - self.sample_times
        fresh = (elapsed > 0) & (elapsed <= self.max_dt)
        position, velocity = self.sample_position.copy(), self.sample_velocity.copy()
        dq, dv = np.zeros(self._shape), np.zeros(self._shape)
        covered = np.zeros(self._shape)
        for i, (start, goal, effort) in enumerate(self.command_history):
            end = self.command_history[i + 1][0] if i + 1 < len(self.command_history) else np.inf
            duration = np.maximum(0, np.minimum(stamps, end) - np.maximum(self.sample_times, start))
            active = fresh & (duration > 0)
            if not active.any():
                continue
            result = self.observer.predict_state(
                position,
                velocity,
                goal,
                effort,
                self.observer.estimated_disturbance_nm if disturbance is None else disturbance,
                np.where(active, duration, 0.02),
                dq,
                dv,
            )
            position, velocity, dq, dv = [
                np.where(active, new, old) for new, old in zip(result, (position, velocity, dq, dv), strict=True)
            ]
            covered += duration
        covered_interval = fresh & np.isclose(covered, elapsed, rtol=0, atol=1e-9)
        if disturbance is not None:
            # Candidate solves must see the actual replay, including an invalid
            # gain. Substituting measured velocity would invent a perfect fit.
            return (velocity, np.where(covered_interval, dv, 1)), np.maximum(elapsed, 0), covered_interval
        fresh = covered_interval & (dv > 0)
        return (np.where(fresh, velocity, qd), np.where(fresh, dv, 1)), np.maximum(elapsed, 0), fresh

    def _prepare_history(self, qd, stamps):
        return _PreparedHistory(lambda disturbance=None: self._reference_history(qd, stamps, disturbance))


def make_observer(cls):
    cfg = x2_locomimic_observer().env
    dof = cfg.dof
    return cls(cfg.observer, dof.joint_names, dof.stiffness, dof.damping, actuator_limits=dof.torque_limits, max_dt=0.1)


class PreparedReplayTests(unittest.TestCase):
    def test_asynchronous_replay_matches_uncached_reference_exactly(self):
        actual, reference = make_observer(JointObserver), make_observer(UncachedObserver)
        rng = np.random.default_rng(872)
        size = actual._feedback_shape[0]
        for step in range(160):
            now = 100 + step * 0.02
            stamps = now - rng.uniform(0, 0.012, size) + np.linspace(-0.035, 0, size)
            if step % 11 == 0:
                # No forward sample for some joints, an expired interval for others.
                stamps[::2] -= 0.2
            goal, q, v = (rng.normal(0, scale, size) for scale in (0.8, 0.3, 1.5))
            if step % 29 == 0:
                actual.reset()
                reference.reset()
            outputs = []
            for ctrl in (actual, reference):
                effort = ctrl.propose(goal, q, v, now, sample_times=stamps)
                # Include acknowledged clipping and two commands per sample interval.
                ctrl.acknowledge(effort * (0.5 if step % 3 else 0), command_time=now)
                if step % 4 == 0:
                    ctrl.pending_target = ctrl.pending_target + 0.01
                    ctrl.acknowledge(ctrl.issued_nm, command_time=now + 0.005)
                outputs.append(effort)
            np.testing.assert_array_equal(*outputs)
            self.assertEqual(actual.get_status(), reference.get_status())

    def test_candidate_predictions_preserve_partial_coverage_and_invalid_gains(self):
        actual, reference = make_observer(JointObserver), make_observer(UncachedObserver)
        n = len(actual.indices)
        qd = np.linspace(-1, 1, n)
        for ctrl in (actual, reference):
            ctrl.sample_times = np.full(n, 100.0)
            ctrl.sample_position = np.linspace(-0.3, 0.3, n)
            ctrl.sample_velocity = qd.copy()
            ctrl.command_history = [
                (100.005, np.full(n, 0.4), np.zeros(n)),
                (100.015, np.full(n, -0.4), np.ones(n)),
            ]
        stamps = np.linspace(100, 100.12, n)
        # Some joints start after the first command and have full coverage.
        actual.sample_times[::2] += 0.005
        reference.sample_times[::2] += 0.005
        predictor = actual._prepare_history(qd, stamps)
        for disturbance in (None, np.zeros(n), actual.observer.limits, -actual.observer.limits):
            expected = reference._reference_history(qd, stamps, disturbance)
            result = predictor(disturbance)
            for left, right in zip((*result[0], *result[1:]), (*expected[0], *expected[1:]), strict=True):
                np.testing.assert_array_equal(left, right)
        # Fully covered candidate replay must expose a nonpositive gain so the
        # nonlinear solver can reject it; nominal replay masks it as not fresh.
        stamps = actual.sample_times + 0.02
        with patch.object(actual.observer, "predict_state", return_value=(qd, qd, qd, -np.ones(n))):
            predictor = actual._prepare_history(qd, stamps)
            (_, gain), _, covered = predictor(np.zeros(n))
            self.assertTrue(covered[::2].all())
            np.testing.assert_array_equal(gain[::2], -1)
            self.assertFalse(predictor()[2].any())
        # Repeated samples cannot update the estimate or issue new correction.
        _, _, fresh = actual._prepare_history(qd, actual.sample_times)()
        self.assertFalse(fresh.any())


if __name__ == "__main__":
    unittest.main()
