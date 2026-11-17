import unittest

from robojudo.controller.xr_target_filter import filter_targets


class TargetFilterTests(unittest.TestCase):
    def test_large_step_and_reversal(self):
        previous = {"joint": 0.0}
        for desired in (1.0, -1.0):
            result = filter_targets({"joint": desired}, previous, {"joint": [-1, 1]}, {"joint": 2}, 0.02, 0)
            self.assertLessEqual(abs(result["joint"] - previous["joint"]), 0.04)
            previous = result

    def test_ema_disabled_and_enabled(self):
        args = ({"joint": 1.0}, {"joint": 0.0}, {"joint": [-1, 1]}, {"joint": 100}, 0.02)
        self.assertEqual(filter_targets(*args, 0)["joint"], 1)
        self.assertAlmostEqual(filter_targets(*args, 0.95)["joint"], 0.05)

    def test_incompatible_position_and_velocity_bounds_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "no float32 target"):
            filter_targets({"joint": 1.0}, {"joint": 0.0}, {"joint": [0.5, 1]}, {"joint": 1}, 0.02, 0)

    def test_float32_rounding_stays_inside_nonrepresentable_bounds(self):
        for value in (0.1, -0.1):
            with self.subTest(value=value):
                result = filter_targets(
                    {"joint": value}, {"joint": 0.0}, {"joint": [-0.1, 0.1]}, {"joint": 10.0}, 0.02, 0.0
                )
                self.assertLessEqual(result["joint"], 0.1)
                self.assertGreaterEqual(result["joint"], -0.1)
