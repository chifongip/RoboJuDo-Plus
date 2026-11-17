"""Hardware-free regressions for the selected fixed-gain bounded torque integral."""

import unittest

import numpy as np

from robojudo.utils.bounded_integral import BoundedIntegralCorrection


class TestIntegralCorrection(unittest.TestCase):
    def controller(self, torque=10):
        return BoundedIntegralCorrection([50], [3], [torque])

    def test_filter_deadband_retention_and_reset(self):
        c = self.controller()
        self.assertEqual(c.update([0.001], [0], [0], 0.02)[0], 0)
        c.reset()
        value = c.update([0.1], [0], [0], 0.02)[0]
        self.assertGreater(value, 0)
        self.assertLess(c.filtered_error[0], 0.1)
        c.filtered_error[:] = 0
        self.assertEqual(c.update([0.1], [0.1], [0], 0.02)[0], value)
        c.reset()
        np.testing.assert_array_equal(c.effort, 0)
        np.testing.assert_array_equal(c.filtered_error, 0)

    def test_slew_effort_bound_and_no_hidden_windup(self):
        c = self.controller()
        c.kp[:] = 0
        c.ki[:] = 100
        previous = 0
        for _ in range(1000):
            value = c.update([0.1], [0], [0], 0.02)[0]
            self.assertLessEqual(abs(value - previous), 0.0400000001)
            self.assertLessEqual(value, 8)
            previous = value
        self.assertEqual(value, 8)
        c.filtered_error[:] = -0.1
        self.assertLess(c.update([0.1], [0.2], [0], 0.02)[0], 8)

    def test_total_saturation_blocks_outward_and_allows_unwinding(self):
        c = self.controller(torque=1)
        c.effort[:] = 0.5
        c.filtered_error[:] = 1
        self.assertEqual(c.update([1], [0], [0], 0.02)[0], 0.5)
        self.assertTrue(c.antiwindup[0])
        c.filtered_error[:] = -0.1
        self.assertLess(c.update([1], [1.1], [-100], 0.02)[0], 0.5)
        self.assertFalse(c.antiwindup[0])

    def test_invalid_inputs_reset_and_fail(self):
        c = self.controller()
        for target, dt in (([np.nan], 0.02), ([0, 0], 0.02), ([0], 0)):
            c.effort[:] = 1
            with self.assertRaises(ValueError):
                c.update(target, [0], [0], dt)
            self.assertEqual(c.effort[0], 0)

    def test_pick_place_adapts_without_force_input_or_reset(self):
        c = self.controller()
        dt = 0.02
        loaded = []
        for step in range(2500):
            # Quasi-static joint: applied torque balances spring and unknown load.
            # The controller receives position, never this load variable.
            load = 2.0 if 10 <= step * dt < 30 else 0.0
            position = (c.effort[0] - load) / c.kp[0]
            effort = c.update([0], [position], [0], dt)[0]
            if 27 <= step * dt < 30:
                loaded.append(effort)
        self.assertGreater(np.mean(loaded), 1.8)
        self.assertLess(abs(effort), 0.2)

    def test_saturation_and_slew_constraints_are_symmetric(self):
        positive, negative = self.controller(), self.controller()
        for _ in range(200):
            a = positive.update([0.1], [0], [0], 0.02)[0]
            b = negative.update([-0.1], [0], [0], 0.02)[0]
            self.assertAlmostEqual(a, -b)

    def test_motion_does_not_change_adaptation_for_identical_error(self):
        stationary, moving = self.controller(), self.controller()
        for i in range(100):
            a = stationary.update([0], [-0.03], [0], 0.02)
            b = moving.update([i * 0.01], [i * 0.01 - 0.03], [0], 0.02)
            np.testing.assert_allclose(a, b, atol=1e-12)

    def test_fixed_parameters_and_scalar_equilibrium(self):
        for setting in ({"integral_gain_scale": 0}, {"filter_tau": -1}, {"deadband_deg": -1}):
            with self.assertRaises(ValueError):
                BoundedIntegralCorrection([10], [1], [100], **setting)
        controller = BoundedIntegralCorrection([10], [1], [100], filter_tau=0.05, deadband_deg=0)
        q, velocity, dt = 0.0, 0.0, 0.01
        for _ in range(10000):
            effort = controller.update([0], [q], [velocity], dt)[0]
            velocity += (-10 * q - velocity + effort - 2.0) / 0.1 * dt
            q += velocity * dt
        self.assertAlmostEqual(q, 0, places=5)


if __name__ == "__main__":
    unittest.main()
