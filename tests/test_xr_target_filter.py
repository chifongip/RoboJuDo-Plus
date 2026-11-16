import unittest
from pathlib import Path

from robojudo.controller.xr_target_filter import filter_targets


class TargetFilterTests(unittest.TestCase):
    def test_repository_copies_match(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(
            (root / "robojudo/controller/xr_target_filter.py").read_bytes(),
            (root / "third_party/xrobotoolkit_teleop/xrobotoolkit_teleop/integrations/target_filter.py").read_bytes(),
        )

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
