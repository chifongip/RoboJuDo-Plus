import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from robojudo.environment.env_cfgs import SimulatedOdometryCfg
from robojudo.environment.utils.odometry import (
    OdometryTracker,
    SimulatedOdometry,
    root_pose_to_sensor,
    sensor_pose_to_root,
)


class TestOdometryTransforms(unittest.TestCase):
    def test_sensor_round_trip_with_rotated_root_and_mounts(self):
        root_position = np.array([1.2, -0.4, 0.9])
        root_quaternion = Rotation.from_euler("xyz", [0.1, -0.2, 2.4]).as_quat()
        root_to_torso_position = np.array([0.05, 0.01, 0.34])
        root_to_torso_quaternion = Rotation.from_euler("xyz", [0.02, 0.1, -0.04]).as_quat()
        torso_to_sensor_position = np.array([0.1, -0.02, 0.18])
        torso_to_sensor_quaternion = Rotation.from_euler("xyz", [0.0, -np.pi / 2, np.pi]).as_quat()

        sensor_position, sensor_quaternion = root_pose_to_sensor(
            root_position,
            root_quaternion,
            root_to_torso_position,
            root_to_torso_quaternion,
            torso_to_sensor_position,
            torso_to_sensor_quaternion,
        )
        recovered_position, recovered_quaternion = sensor_pose_to_root(
            sensor_position,
            sensor_quaternion,
            torso_to_sensor_position,
            torso_to_sensor_quaternion,
            root_to_torso_position,
            root_to_torso_quaternion,
        )

        np.testing.assert_allclose(recovered_position, root_position, atol=1e-10)
        relative = Rotation.from_quat(recovered_quaternion).inv() * Rotation.from_quat(root_quaternion)
        self.assertLess(relative.magnitude(), 1e-10)


class TestOdometryTracker(unittest.TestCase):
    def test_world_displacement_is_converted_by_raw_root_heading(self):
        tracker = OdometryTracker(timeout=0.3, velocity_filter_time_constant=0.0)
        heading = Rotation.from_euler("z", np.pi).as_quat()
        tracker.update([0.0, 0.0, 0.8], heading, sample_time=0.0, receipt_time=0.0)
        tracker.update([-0.1, 0.0, 0.8], heading, sample_time=0.1, receipt_time=0.1)

        estimate = tracker.estimate(0.1)

        np.testing.assert_allclose(estimate.linear_velocity_body, [1.0, 0.0, 0.0], atol=1e-6)

    def test_extrapolation_is_bounded_and_staleness_is_reported(self):
        tracker = OdometryTracker(timeout=0.3, velocity_filter_time_constant=0.0)
        tracker.update([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], 0.0, 0.0)
        tracker.update([0.1, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], 0.1, 0.1)

        fresh = tracker.estimate(0.2)
        stale = tracker.estimate(0.5)

        np.testing.assert_allclose(fresh.position, [0.2, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(stale.position, [0.4, 0.0, 0.0], atol=1e-6)
        self.assertFalse(fresh.stale)
        self.assertTrue(stale.stale)


class TestSimulatedOdometry(unittest.TestCase):
    @staticmethod
    def _identity_converter(position, quaternion):
        return position, quaternion

    def test_ten_hz_sampling_while_called_at_fifty_hz(self):
        source = SimulatedOdometry(
            SimulatedOdometryCfg(enabled=True, update_rate_hz=10.0, velocity_filter_time_constant=0.0)
        )
        for step in range(51):
            timestamp = step * 0.02
            source.update(
                timestamp,
                np.array([timestamp, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0, 1.0]),
                self._identity_converter,
            )

        self.assertEqual(source.generated, 11)
        self.assertEqual(source.delivered, 11)
        np.testing.assert_allclose(source.tracker.estimate(1.0).linear_velocity_body, [1.0, 0.0, 0.0])

    def test_seed_makes_dropout_and_noise_reproducible(self):
        cfg = SimulatedOdometryCfg(
            enabled=True,
            update_rate_hz=10.0,
            dropout_probability=0.25,
            position_noise_std=(0.01, 0.01, 0.01),
            random_seed=7,
        )
        sources = [SimulatedOdometry(cfg), SimulatedOdometry(cfg)]
        for step in range(51):
            timestamp = step * 0.02
            for source in sources:
                source.update(
                    timestamp,
                    np.array([timestamp, 0.0, 0.0]),
                    np.array([0.0, 0.0, 0.0, 1.0]),
                    self._identity_converter,
                )

        self.assertEqual(sources[0].diagnostics(1.0), sources[1].diagnostics(1.0))
        np.testing.assert_allclose(
            sources[0].tracker.estimate(1.0).position,
            sources[1].tracker.estimate(1.0).position,
        )

    def test_degeneracy_window_eventually_marks_estimate_stale(self):
        source = SimulatedOdometry(
            SimulatedOdometryCfg(
                enabled=True,
                update_rate_hz=10.0,
                timeout=0.3,
                degeneracy_windows=[(0.2, 0.7)],
            )
        )
        stale_seen = False
        for step in range(41):
            timestamp = step * 0.02
            estimate = source.update(
                timestamp,
                np.array([timestamp, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0, 1.0]),
                self._identity_converter,
            )
            stale_seen |= estimate is not None and estimate.stale

        self.assertTrue(stale_seen)
        self.assertGreater(source.degenerate, 0)


if __name__ == "__main__":
    unittest.main()
