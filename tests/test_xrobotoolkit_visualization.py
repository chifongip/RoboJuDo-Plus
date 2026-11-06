"""World-frame composition and persistent marker cleanup without a graphics context."""

import unittest
from types import SimpleNamespace

import numpy as np

from robojudo.environment.utils.mujoco_viz import MujocoVisualizer


class TeleopVisualizationTests(unittest.TestCase):
    def setUp(self):
        self.markers = {}
        viewer = SimpleNamespace(add_marker=lambda **marker: self.markers.update({marker["id"]: marker}))
        self.visualizer = MujocoVisualizer(viewer)

    def test_target_translation_and_rgb_axes_follow_rotated_torso_in_world(self):
        torso = np.array([[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 0.8], [0, 0, 0, 1]], dtype=float)
        target = np.eye(4)
        target[:3, 3] = [0.4, 0.2, 0.1]
        self.visualizer.update_teleop_frames(torso, {"left": target.tolist()})
        np.testing.assert_allclose(self.markers[6004]["pos"], [0.8, 2.4, 0.9])
        for axis in range(3):
            marker = self.markers[6005 + axis]
            np.testing.assert_allclose(marker["mat"][:, 2], torso[:3, axis])
            self.assertEqual(marker["rgba"][axis], 1.0)
        np.testing.assert_array_equal(self.markers[6012]["pos"], [0, 0, 0])

    def test_missing_targets_hide_persistent_markers_without_accumulating(self):
        self.visualizer.update_teleop_frames(np.eye(4), {"left": np.eye(4).tolist()})
        self.assertGreater(self.markers[6004]["rgba"][3], 0)
        self.visualizer.update_teleop_frames(np.eye(4), {})
        for marker_id in range(6004, 6012):
            self.assertEqual(self.markers[marker_id]["rgba"][3], 0)
            self.assertEqual(self.markers[marker_id]["label"], "")
        self.assertEqual(len(self.markers), 16)
