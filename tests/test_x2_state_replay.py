import copy
import unittest

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from robojudo.config.x2.env.x2_env_cfg import (
    X2_ARM_JOINT_NAMES,
    X2_HEAD_JOINT_NAMES,
    X2_LEG_JOINT_NAMES,
    X2_WAIST_JOINT_NAMES,
)
from robojudo.config.x2.env.x2_mujuco_env_cfg import X2MujocoEnvCfg
from robojudo.tools.x2_replay import (
    build_odometry_profile,
    grounded_mujoco_seed,
    reconstruct_environment_frames,
    validate_and_select,
)


def _capture(duration=3.0):
    records = [{"kind": "metadata", "schema_version": 1}]
    topic_joints = {
        "/aima/hal/joint/leg/state": X2_LEG_JOINT_NAMES,
        "/aima/hal/joint/waist/state": X2_WAIST_JOINT_NAMES,
        "/aima/hal/joint/arm/state": X2_ARM_JOINT_NAMES,
        "/aima/hal/joint/head/state": X2_HEAD_JOINT_NAMES,
    }
    for index in range(int(duration * 50) + 1):
        timestamp = index / 50.0
        records.append(
            {
                "kind": "imu",
                "topic": "/aima/hal/imu/torso/state",
                "receipt_time": timestamp,
                "stamp": timestamp,
                "quaternion": [0.0, 0.0, 0.0, 1.0],
                "angular_velocity": [0.0, 0.0, 0.0],
                "linear_acceleration": [0.0, 0.0, 9.81],
            }
        )
        for topic, names in topic_joints.items():
            records.append(
                {
                    "kind": "joint",
                    "topic": topic,
                    "receipt_time": timestamp,
                    "stamp": timestamp,
                    "joints": [{"name": name, "position": 0.0, "velocity": 0.0, "effort": 0.0} for name in names],
                }
            )
        if index % 5 == 0:
            yaw_noise = 0.002 * np.sin(index)
            records.append(
                {
                    "kind": "odometry",
                    "topic": "/laser_odometry",
                    "receipt_time": timestamp,
                    "stamp": timestamp,
                    "frame_id": "map",
                    "child_frame_id": "lidar_chest_front",
                    "position": [0.001 * np.sin(index), 0.0, 1.0],
                    "quaternion": Rotation.from_euler("z", yaw_noise).as_quat().tolist(),
                    "pose_covariance": [0.01] + [0.0] * 35,
                    "linear_velocity": [0.0, 0.0, 0.0],
                    "angular_velocity": [0.0, 0.0, 0.0],
                }
            )
    return records


class TestX2StateReplay(unittest.TestCase):
    def test_selects_causally_synchronized_stable_window(self):
        selection = validate_and_select(_capture())

        self.assertEqual(len(selection.frames), 101)
        self.assertEqual(len(selection.snapshot.joints), 31)
        self.assertLessEqual(max(selection.snapshot.ages.values()), 0.1)

    def test_rejects_wrong_odometry_frame(self):
        records = _capture()
        next(record for record in records if record.get("kind") == "odometry")["child_frame_id"] = "pelvis"

        with self.assertRaisesRegex(ValueError, "Unexpected odometry frames"):
            validate_and_select(records)

    def test_rejects_degenerate_odometry(self):
        records = _capture()
        next(record for record in records if record.get("kind") == "odometry")["pose_covariance"][0] = 1.0

        with self.assertRaisesRegex(ValueError, "degenerate"):
            validate_and_select(records)

    def test_rejects_odometry_timeout_gap(self):
        records = [
            record
            for record in _capture()
            if not (record.get("topic") == "/laser_odometry" and 1.0 <= record.get("receipt_time", -1.0) <= 1.3)
        ]

        with self.assertRaisesRegex(ValueError, "delivery gap"):
            validate_and_select(records)

    def test_profile_is_deterministic_and_detrended(self):
        selection = validate_and_select(_capture())

        first = build_odometry_profile(selection)
        second = build_odometry_profile(copy.deepcopy(selection))

        np.testing.assert_allclose(first.sample_times, second.sample_times)
        np.testing.assert_allclose(first.position_residuals, second.position_residuals)
        np.testing.assert_allclose(first.yaw_residuals, second.yaw_residuals)
        np.testing.assert_allclose(first.position_residuals[0], 0.0)
        self.assertEqual(first.sample_times[0], 0.0)

    def test_reconstruction_produces_complete_policy_state_and_seed(self):
        selection = validate_and_select(_capture())

        frames, seed = reconstruct_environment_frames(selection)

        self.assertEqual(len(frames), 101)
        self.assertEqual(frames[0].dof_pos.shape, (31,))
        self.assertEqual(frames[0].torso_pos.shape, (3,))
        self.assertEqual(seed.joint_position.shape, (31,))
        np.testing.assert_allclose(frames[0].base_pos, np.zeros(3), atol=1e-6)
        np.testing.assert_allclose(seed.root_position, np.zeros(3), atol=1e-6)
        self.assertGreater(np.linalg.norm(seed.odometry_origin_position), 0.5)
        np.testing.assert_allclose(seed.root_quaternion, [0.0, 0.0, 0.0, 1.0])

    def test_grounded_seed_uses_physical_foot_contact_height(self):
        selection = validate_and_select(_capture())
        _, seed = reconstruct_environment_frames(selection)
        cfg = X2MujocoEnvCfg()
        model = mujoco.MjModel.from_xml_path(cfg.xml)
        data = mujoco.MjData(model)

        qpos, _, diagnostics = grounded_mujoco_seed(model, data, seed)

        self.assertGreater(qpos[2], 0.45)
        self.assertAlmostEqual(diagnostics["minimum_foot_clearance"], 0.002, places=6)


if __name__ == "__main__":
    unittest.main()
