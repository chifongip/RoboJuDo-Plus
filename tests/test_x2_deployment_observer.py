"""Finite load steps and limited-command recovery, without a native adapter."""

import unittest

import numpy as np
from test_x2_arm_observer import config, environment, runtime

from robojudo.config.x2.env.x2_real_env_cfg import X2ArmObserverCfg
from robojudo.utils.deployment_disturbance_observer import DeploymentDisturbanceObserver


def observer():
    return DeploymentDisturbanceObserver(
        [50.0],
        [3.0],
        [36.0],
        inertia=0.1553240498,
        friction_limit=0.3,
        friction_viscosity=106.8083,
        bandwidth_rad_s=8,
    )


def affine_predictor(disturbance):
    return disturbance.copy(), np.ones(1)


class DeploymentObserverTests(unittest.TestCase):
    def test_finite_release_and_reversal_invert_friction_transition(self):
        for sign in (-1, 1):
            for actual in (0.0, -4.0 * sign):
                with self.subTest(sign=sign, actual=actual):
                    model = observer()
                    zero = np.zeros(1)
                    model.update(zero, zero, zero, 0.02)
                    model.effort[:] = -4 * sign
                    model.estimated_disturbance_nm[:] = 4 * sign
                    q, v, _, _ = model.predict_state(zero, zero, zero, model.effort, np.array([actual]), 0.02)
                    model.update(zero, q, v, 0.02)
                    self.assertAlmostEqual(model.instantaneous_disturbance_nm[0], actual, delta=2e-5)
                    expected = 4 * sign + -np.expm1(-8 * 0.02) * (actual - 4 * sign)
                    self.assertAlmostEqual(model.estimated_disturbance_nm[0], expected, delta=2e-5)

    def test_release_is_symmetric_and_cannot_grow_opposite_torque(self):
        for sign in (-1, 1):
            model = observer()
            model.release_slew = 20
            model.effort[:] = sign * 0.3
            model.estimated_disturbance_nm[:] = sign * 4  # Opposite correction requested.
            # An affine predictor isolates the slew contract from estimator dynamics.
            first = model.update([0], [0], [sign * 4], 0.02, predictor=affine_predictor)
            np.testing.assert_array_equal(first, 0)
            second = model.update([0], [0], [sign * 4], 0.02, predictor=affine_predictor)
            np.testing.assert_allclose(second, -sign * 0.04)

    def test_internal_and_downstream_limiting_allow_unwinding(self):
        for sign in (-1, 1):
            model = observer()
            model.update([0], [0], [0], 0.02)
            model.effort[:] = sign
            model.estimated_disturbance_nm[:] = -4 * sign
            proposed = model.update([sign], [0], [0], 0.02, predictor=affine_predictor)
            self.assertTrue(model.antiwindup[0])
            self.assertLess(abs(model.estimated_disturbance_nm[0]), 4)
            before_ack = model.estimated_disturbance_nm.copy()
            model.acknowledge_effort(proposed * 0)
            np.testing.assert_array_equal(model.estimated_disturbance_nm, before_ack)
            np.testing.assert_array_equal(model.effort, 0)

    def test_limiting_blocks_outward_adaptation_but_preserves_zero_crossing(self):
        for sign in (-1, 1):
            model = observer()
            model._estimate_before_update[:] = -2 * sign
            model.estimated_disturbance_nm[:] = -3 * sign
            model.effort[:] = sign
            model.acknowledge_effort([0])
            np.testing.assert_array_equal(model.estimated_disturbance_nm, -2 * sign)
            model._estimate_before_update[:] = 2 * sign
            model.estimated_disturbance_nm[:] = -3 * sign
            model.effort[:] = sign
            model.acknowledge_effort([0])
            np.testing.assert_array_equal(model.estimated_disturbance_nm, 0)

    def test_bounds_exact_prediction_and_nonfinite_prediction(self):
        model = observer()
        for measured in (-100, 100):
            inferred = model.infer_disturbance(np.array([measured]), affine_predictor, np.array([True]))
            np.testing.assert_array_equal(inferred, np.sign(measured) * model.limits)
        model.estimated_disturbance_nm[:] = 0.123
        np.testing.assert_array_equal(model.infer_disturbance(np.array([0.123]), affine_predictor, [True]), [0.123])
        with self.assertRaises(ValueError):
            model.infer_disturbance(np.zeros(1), lambda d: (d * np.nan, np.ones(1)), np.array([True]))

    def test_repeated_feedback_freezes_release_and_estimation(self):
        model = observer()
        model.release_slew = 20
        model.effort[:] = 4
        model.estimated_disturbance_nm[:] = -4
        result = model.update([0], [0], [100], 0.02, fresh=[False])
        np.testing.assert_array_equal(result, 4)
        np.testing.assert_array_equal(model.estimated_disturbance_nm, -4)

    def test_release_config_validation(self):
        self.assertIsNone(X2ArmObserverCfg().release_slew_nm_s)
        for invalid in (0, -1, float("nan"), float("inf")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                X2ArmObserverCfg(release_slew_nm_s=invalid)

    def test_faster_release_keeps_eligibility_and_generation_resets(self):
        cfg = config().model_copy(update={"release_slew_nm_s": 16})
        env = environment(cfg)
        env._arm_observer.observer.effort[:] = 0.5
        env._arm_observer.observer.estimated_disturbance_nm[:] = -0.5
        env.set_arm_observer_eligible(False)
        np.testing.assert_array_equal(env._arm_observer.observer.effort, 0)
        env.set_arm_observer_eligible(True)
        env.aimdk.reset_on_step = True
        env.step(np.zeros(16))
        np.testing.assert_array_equal(env._arm_observer.issued_nm, 0)
        np.testing.assert_array_equal(env._arm_observer.observer.estimated_disturbance_nm, 0)
        self.assertIsNone(env._arm_observer.sample_times)

    def test_saturation_discontinuity_keeps_best_bounded_candidate(self):
        model = observer()

        def discontinuous_predictor(disturbance):
            return np.where(disturbance < 0, disturbance - 1, disturbance + 1), np.ones(1)

        inferred = model.infer_disturbance(np.zeros(1), discontinuous_predictor, np.array([True]))
        np.testing.assert_array_equal(inferred, 0)
        np.testing.assert_array_equal(model.inversion_residual_rad_s, 1)

    def test_nonlinear_inversion_replays_segmented_accepted_commands(self):
        wrapper = runtime()
        model = wrapper.observer
        q, v, target = np.zeros(16), np.zeros(16), np.full(16, 0.1)
        stamps = np.full(16, 100.0)
        wrapper.propose(target, q, v, 1, sample_times=stamps)
        wrapper.acknowledge(np.zeros(14), command_time=100)
        actual = np.full(14, 0.7)
        q1, v1, _, _ = model.predict_state(
            q[wrapper.indices], v[wrapper.indices], target[wrapper.indices], np.zeros(14), actual, 0.02
        )
        q[wrapper.indices], v[wrapper.indices] = q1, v1
        stamps[:] = 100.02
        proposal = wrapper.propose(target, q, v, 1.02, sample_times=stamps)
        wrapper.acknowledge(np.array(proposal) / 2, command_time=100.02)
        issued = wrapper.issued_nm.copy()
        new_target = -target
        proposal = wrapper.propose(new_target, q, v, 1.025, sample_times=stamps)
        wrapper.acknowledge(proposal, command_time=100.025)
        q2, v2, _, _ = model.predict_state(q1, v1, target[wrapper.indices], issued, actual, 0.005)
        q3, v3, _, _ = model.predict_state(q2, v2, new_target[wrapper.indices], issued, actual, 0.025)
        q[wrapper.indices], v[wrapper.indices] = q3, v3
        stamps[:] = 100.05
        wrapper.propose(new_target, q, v, 1.06, sample_times=stamps)
        np.testing.assert_allclose(model.instantaneous_disturbance_nm, actual, atol=2e-5)

    def test_near_bound_steady_load_and_abrupt_release(self):
        for sign in (-1, 1):
            model = observer()
            model.release_slew = 16
            q, v, target = np.zeros(1), np.zeros(1), np.zeros(1)
            model.update(target, q, v, 0.02)
            for actual in (7.5 * sign, 0):
                for _ in range(600):
                    q, v, _, _ = model.predict_state(q, v, target, model.effort, np.array([actual]), 0.02)
                    model.update(target, q, v, 0.02)
                    self.assertLessEqual(abs(model.effort[0]), 8)
                self.assertAlmostEqual(model.estimated_disturbance_nm[0], actual, delta=2e-4)
                self.assertLess(abs(q[0]), 0.002)


if __name__ == "__main__":
    unittest.main()
