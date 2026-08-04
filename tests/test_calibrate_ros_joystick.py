import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "calibrate_ros_joystick.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("calibrate_ros_joystick", SCRIPT_PATH)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
calibrate = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(calibrate)


def sample(axes, buttons):
    return SimpleNamespace(axes=axes, buttons=buttons)


class TestCalibrateRosJoystick(unittest.TestCase):
    def test_analyze_samples_reports_all_changed_indices(self):
        observation = calibrate.analyze_samples(
            neutral_axes=[0.0, 0.0, 1.0],
            neutral_buttons=[0, 0, 0],
            samples=[
                sample([0.0, -0.8, 1.0], [0, 1, 0]),
                sample([0.0, -1.0, -1.0], [0, 0, 1]),
            ],
            axis_threshold=0.35,
        )

        self.assertEqual([button["index"] for button in observation["buttons"]], [1, 2])
        self.assertEqual([axis["index"] for axis in observation["axes"]], [1, 2])
        self.assertEqual(observation["axes"][1]["neutral"], 1.0)
        self.assertEqual(observation["axes"][1]["peak_value"], -1.0)

    def test_small_stick_drift_is_ignored(self):
        observation = calibrate.analyze_samples(
            neutral_axes=[0.02],
            neutral_buttons=[0],
            samples=[sample([0.08], [0])],
            axis_threshold=0.35,
        )
        self.assertEqual(observation, {"buttons": [], "axes": []})

    def test_profile_prompts_use_physical_labels(self):
        xbox = dict(calibrate.controls_for_profile("xbox"))
        ps5 = dict(calibrate.controls_for_profile("ps5"))
        self.assertIn("A", xbox["face_south"])
        self.assertIn("Cross", ps5["face_south"])
        self.assertIn("Square", ps5["face_west"])


if __name__ == "__main__":
    unittest.main()
