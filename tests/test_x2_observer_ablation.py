"""Validate experimental factor isolation before interpreting simulation results."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from report_x2_observer_ablation import command_check, paired_effects  # noqa: E402
from x2_observer_ablation import VARIANTS, AblationObserver  # noqa: E402

from robojudo.utils.observers import LinearizedDisturbanceObserver, NonlinearDisturbanceObserver  # noqa: E402


def model(cls, **kwargs):
    return cls(
        [50.0], [3.0], [36.0], inertia=0.15, friction_limit=0.3, friction_viscosity=106.8, bandwidth_rad_s=8, **kwargs
    )


class ObserverAblationTests(unittest.TestCase):
    def test_factorial_matrix_is_complete_with_separate_original_reference(self):
        self.assertEqual(len(VARIANTS), 9)
        combinations = {(v["estimator"], v["unwinding"], v["release"]) for k, v in VARIANTS.items() if k != "original"}
        self.assertEqual(len(combinations), 8)
        self.assertTrue(all(v["zero_before_reversal"] for k, v in VARIANTS.items() if k != "original"))
        self.assertFalse(VARIANTS["original"]["zero_before_reversal"])

    def test_nonlinear_directional_variant_matches_production_exactly(self):
        for rate in (2, 16):
            production = model(NonlinearDisturbanceObserver)
            experimental = model(AblationObserver, estimator="nonlinear", unwinding="directional")
            production.release_slew = experimental.release_slew = rate
            for step in range(40):
                goal, q, velocity = [0.2 * np.sin(step)], [0.01 * np.cos(step)], [0.1 * np.sin(step * 0.3)]
                expected = production.update(goal, q, velocity, 0.02)
                actual = experimental.update(goal, q, velocity, 0.02)
                np.testing.assert_array_equal(actual, expected)
                production.acknowledge_effort(expected * 0.5)
                experimental.acknowledge_effort(actual * 0.5)
                np.testing.assert_array_equal(
                    experimental.estimated_disturbance_nm, production.estimated_disturbance_nm
                )

    def test_linearized_factor_preserves_original_filtered_innovation(self):
        original = model(LinearizedDisturbanceObserver)
        experimental = model(AblationObserver, estimator="linearized", unwinding="frozen")
        # Include residuals whose raw innovation/gain exceeds disturbance bounds:
        # clamping before filtering would accidentally change the estimator.
        for residual in (0.001, 0.02, 0.4, -0.4):
            original.reset()
            experimental.reset()
            prediction = (np.array([0.0]), np.array([0.01]))
            original.update([0], [0], [residual], 0.02, prediction=prediction)
            experimental.update(
                [0], [0], [residual], 0.02, predictor=lambda disturbance: (0.01 * disturbance, np.array([0.01]))
            )
            np.testing.assert_allclose(
                experimental.estimated_disturbance_nm, original.estimated_disturbance_nm, rtol=0, atol=1e-14
            )

    def test_unwinding_factor_only_changes_blocked_estimate_reduction(self):
        for estimator in ("linearized", "nonlinear"):
            frozen = model(AblationObserver, estimator=estimator, unwinding="frozen")
            directional = model(AblationObserver, estimator=estimator, unwinding="directional")
            for observer in (frozen, directional):
                observer._estimate_before_update[:] = -4
                observer.estimated_disturbance_nm[:] = -3
                observer.effort[:] = 2
                observer.acknowledge_effort([1])
            np.testing.assert_array_equal(frozen.estimated_disturbance_nm, [-4])
            np.testing.assert_array_equal(directional.estimated_disturbance_nm, [-3])
            np.testing.assert_array_equal(frozen.effort, directional.effort)

    def test_trace_validation_distinguishes_growth_release_and_reversal(self):
        factors = VARIANTS["linearized_directional_16"]
        growth = command_check(np.array([[0.04], [0.2]]), factors, [8])
        self.assertGreater(growth["slew_excess_nm_s"], 0)
        release = command_check(np.array([[0.04], [0.0]]), factors, [8])
        self.assertLessEqual(release["slew_excess_nm_s"], 0)
        reversal = command_check(np.array([[0.01], [-0.01]]), factors, [8])
        self.assertFalse(reversal["reversal_policy_passed"])
        legacy = command_check(np.array([[0.01], [-0.01]]), VARIANTS["original"], [8])
        self.assertTrue(legacy["reversal_policy_passed"])

    def test_effects_match_seed_and_hold_other_factors_fixed(self):
        rows = [
            dict(study="abrupt", seed=seed, variant=name, **VARIANTS[name], carry_rms_deg=value)
            for seed, name, value in (
                (1, "linearized_directional_16", 0.2),
                (1, "nonlinear_directional_16", 0.3),
                (2, "nonlinear_directional_16", 0.8),
                (1, "nonlinear_frozen_16", 0.4),
            )
        ]
        estimator = [r for r in paired_effects(rows) if r["factor"] == "estimator"]
        self.assertEqual(len(estimator), 1)
        self.assertAlmostEqual(estimator[0]["delta_carry_rms_deg"], 0.1)


if __name__ == "__main__":
    unittest.main()
