import sys
import time
import unittest
from pathlib import Path

import numpy as np

PACKAGE_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PACKAGE_SRC))

from robojudo_recorder.protocol import ControlSample  # noqa: E402
from robojudo_recorder.profiles import NamedJointProfile  # noqa: E402


class TestControlSample(unittest.TestCase):
    def make_message(self):
        return {
            "episode_id": 1,
            "task": "pick up the cup",
            "robot_type": "x2",
            "timestamp_ns": time.monotonic_ns(),
            "joint_names": ["left_arm", "right_arm"],
            "joint_positions": [0.1, 0.2],
            "joint_position_commands": [0.3, 0.4],
            "velocity_height_command": [0.5, 0.0, -0.1, 0.62],
        }

    def test_builds_named_action(self):
        sample = ControlSample.from_message(self.make_message(), receive_timestamp_ns=123)
        np.testing.assert_allclose(sample.action, [0.3, 0.4, 0.5, 0.0, -0.1, 0.62])
        self.assertEqual(
            sample.action_names,
            [
                "left_arm.pos",
                "right_arm.pos",
                "base.velocity.x",
                "base.velocity.y",
                "base.yaw_rate",
                "base.height",
            ],
        )

    def test_rejects_invalid_shapes(self):
        message = self.make_message()
        message["velocity_height_command"] = [0.0, 0.0, 0.0]
        with self.assertRaisesRegex(ValueError, "shape"):
            ControlSample.from_message(message, receive_timestamp_ns=123)

    def test_profile_rejects_robot_schema_changes(self):
        sample = ControlSample.from_message(self.make_message(), receive_timestamp_ns=123)
        profile = NamedJointProfile.from_sample(sample)
        changed = self.make_message()
        changed["joint_names"] = ["left_arm", "new_hand"]
        changed_sample = ControlSample.from_message(changed, receive_timestamp_ns=124)
        with self.assertRaisesRegex(ValueError, "profile changed"):
            profile.validate(changed_sample)


if __name__ == "__main__":
    unittest.main()
