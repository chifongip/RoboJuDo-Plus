"""Sampled observer contracts on its model and respects actuator constraints."""

import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np

from robojudo.utils.disturbance_observer import BoundedDisturbanceObserver

if "x2_observer_reference" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "x2_observer_reference", Path(__file__).parents[1] / "scripts/x2_observer_reference.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)


def observer(**kwargs):
    return BoundedDisturbanceObserver(
        [50.0], [3.0], [36.0], inertia=0.1, friction_limit=0.3, friction_viscosity=40.0, **kwargs
    )


class DisturbanceObserverTests(unittest.TestCase):
    def test_endpoint_input_gain(self):
        controller = observer()
        for velocity in (0.0, 0.2, -0.3):
            q, v = np.array([0.1]), np.array([velocity])
            for disturbance in (0.0, 0.15, 1.0, -2.0):
                d = np.array([disturbance])
                args = q, v, q, np.array([0.0])
                _, gain = controller.predict_velocity(*args, d, 0.02)
                plus, _ = controller.predict_velocity(*args, d + 1e-6, 0.02)
                minus, _ = controller.predict_velocity(*args, d - 1e-6, 0.02)
                np.testing.assert_allclose(gain, (plus - minus) / 2e-6, rtol=1e-6)

    def test_predictor_retains_friction_and_known_clipped_actuator(self):
        controller = observer()
        zero = np.zeros(1)
        velocity, gain = controller.predict_velocity(zero, zero, zero, zero, zero, 0.02)
        np.testing.assert_array_equal(velocity, 0)
        self.assertGreater(gain[0], 0)
        # Target far outside the actuator range: initial PD drive is clipped.
        target = np.array([10.0])
        nominal, sensitivity = controller.predict_velocity(zero, zero, target, zero, zero, 0.02)
        plus, _ = controller.predict_velocity(zero, zero, target, zero, np.array([1e-6]), 0.02)
        minus, _ = controller.predict_velocity(zero, zero, target, zero, np.array([-1e-6]), 0.02)
        self.assertGreater(nominal[0], 0)
        np.testing.assert_allclose(sensitivity, (plus - minus) / 2e-6, rtol=1e-6)

    def test_exact_constant_disturbance_contraction_and_removal(self):
        controller = observer(bandwidth_rad_s=5.0)
        q, v, target = np.zeros(1), np.zeros(1), np.zeros(1)
        controller.update(target, q, v, 0.02)
        for actual in (-0.15, 0.0):
            initial = abs(controller.estimated_disturbance_nm[0] - actual)
            for _ in range(100):
                # Plant and predictor use the same four implicit substeps.
                h = 0.005
                for _ in range(4):
                    alpha = controller.inertia / h + controller.kd + controller.kp * h
                    force = controller.inertia / h * v + controller.kp * (target - q) + controller.effort + actual
                    trial = force / (alpha + controller.friction_viscosity)
                    v = np.where(
                        abs(controller.friction_viscosity * trial) > controller.friction_limit,
                        (force - np.sign(force) * controller.friction_limit) / alpha,
                        trial,
                    )
                    q += h * v
                controller.update(target, q, v, 0.02)
            self.assertLess(abs(controller.estimated_disturbance_nm[0] - actual), initial * 0.001)

    def test_bounds_slew_and_antiwindup(self):
        controller = observer()
        previous = controller.effort.copy()
        controller.update([0], [0], [0], 0.02)
        for _ in range(500):
            effort = controller.update([0], [0], [-10], 0.02)
            self.assertLessEqual(abs(effort[0]), 8)
            self.assertLessEqual(abs(effort[0] - previous[0]), 0.04000000001)
            self.assertLessEqual(abs(controller.estimated_disturbance_nm[0]), 8)
            previous = effort
        self.assertTrue(controller.antiwindup[0])

    def test_invalid_feedback_resets(self):
        controller = observer()
        controller.effort[:] = 1
        with self.assertRaises(ValueError):
            controller.update([0], [float("nan")], [0], 0.02)
        self.assertEqual(controller.effort[0], 0)
        self.assertIsNone(controller.previous_position)

    def test_invalid_model(self):
        with self.assertRaises(ValueError):
            observer(bandwidth_rad_s=0)


class TestSelectedObserverReference(unittest.TestCase):
    def test_coefficients_match_three_warmup_validated_reference(self):
        reference = sys.modules["x2_observer_reference"]
        parameters = reference.selected_observer_parameters()
        fingerprint = hashlib.sha256(json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        # Frozen from the actual nine-trial validation, not a newly tuned default.
        self.assertEqual(fingerprint, "d23c873f49e82f004cf185211d5cbbf93c81953b2e53f890c7b8653c978cc975")
        self.assertEqual(len(parameters["inertia"]), 14)
        self.assertEqual(len(parameters["friction_viscosity"]), 14)

    def test_reference_returns_independent_parameter_state(self):
        reference = sys.modules["x2_observer_reference"]
        first = reference.selected_observer_parameters()
        first["inertia"][0] = -1
        self.assertGreater(reference.selected_observer_parameters()["inertia"][0], 0)
        self.assertEqual(reference.selected_observer_parameters()["bandwidth_rad_s"], 8.0)


if __name__ == "__main__":
    unittest.main()
