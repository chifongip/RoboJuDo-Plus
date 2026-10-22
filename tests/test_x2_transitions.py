"""Hardware-free posture trajectory and bounded-controller regressions."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from box import Box

from robojudo.pipeline.upper_body_zmq_pipeline import UpperBodyZmqPipelineMixin

SCRIPTS = Path(__file__).parents[1] / "scripts"
for name in ("x2_integral_sim", "evaluate_x2_transitions"):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
evaluation = sys.modules["evaluate_x2_transitions"]


class TestPostureTransitions(unittest.TestCase):
    def test_schedule_endpoints_and_motion_only_during_carry(self):
        for t, fraction in (
            (0, 0),
            (10, 0),
            (16, 0.8),
            (26, 0.8),
            (30, 1),
            (75, 1),
            (79, 0.8),
            (89, 0.8),
            (95, 0),
            (105, 0),
        ):
            self.assertAlmostEqual(evaluation.schedule(t)[1], fraction)
        for t in np.linspace(0, 105, 10501):
            phase, fraction, load, vx = evaluation.schedule(t)
            self.assertTrue(0 <= fraction <= 1)
            self.assertTrue(0 <= load <= 1)
            self.assertTrue(0 <= vx <= 0.5)
            if phase in evaluation.TRANSITIONS:
                self.assertEqual(vx, 0)
            if vx > 0:
                self.assertEqual(fraction, 1)
                self.assertEqual(load, 1)
        self.assertEqual(evaluation.schedule(50)[3], 0.5)

    def test_quintic_smoothness_and_reference_speed(self):
        for t in (10, 16, 26, 30, 75, 79, 89, 95):
            h = 1e-4
            f = evaluation.schedule(t)[1]
            self.assertLess(abs(evaluation.schedule(t - h)[1] - f), 1e-8)
            self.assertLess(abs(evaluation.schedule(t + h)[1] - f), 1e-8)
        dt = 0.02
        fractions = np.asarray([evaluation.schedule(t)[1] for t in np.arange(0, 105 + dt, dt)])
        delta = np.asarray(list(evaluation.POSTURE.values())) - evaluation.default_posture()
        velocities = np.diff(fractions)[:, None] * delta[None, :] / dt
        self.assertLess(np.max(np.abs(velocities)), 0.35)
        self.assertGreater(np.max(np.abs(velocities)), 0.1)

    def test_postures_preserve_original_defaults_and_limits(self):
        cfg = evaluation.transition_config()
        defaults = evaluation.default_posture()
        self.assertAlmostEqual(defaults[3], -0.87)
        for i, name in enumerate(evaluation.POSTURE):
            j = cfg.joint_default_dof.joint_names.index(name)
            self.assertAlmostEqual(cfg.joint_default_dof.default_pos[j], defaults[i])
            k = cfg.env.dof.joint_names.index(name)
            low, high = cfg.env.dof.position_limits[k]
            self.assertTrue(low <= defaults[i] <= high)
            self.assertTrue(low <= evaluation.POSTURE[name] <= high)
        self.assertEqual(cfg.ctrl, [])

    def test_load_synchronization_and_unloaded_reference(self):
        for t in np.arange(0, 105, 0.02):
            np.testing.assert_array_equal(evaluation.external_force(t, "unloaded"), 0)
            for profile in ("constant", "sway"):
                f = evaluation.external_force(t, profile)
                if t <= 26 or t >= 79:
                    np.testing.assert_array_equal(f, 0)
                self.assertLessEqual(abs(f[0]), 5)
                self.assertTrue(-25 <= f[2] <= 0)
        np.testing.assert_array_equal(evaluation.external_force(50, "constant"), [0, 0, -20])
        self.assertAlmostEqual(evaluation.schedule(28)[2], 0.5)
        self.assertAlmostEqual(evaluation.schedule(77)[2], 0.5)

    def test_deployment_reference_velocity_limiter(self):
        cfg = evaluation.deployment_upper_config()
        cfg.max_joint_velocity_rad_s = 0.05  # Exercise a cap below the requested trajectory speed.
        defaults = evaluation.default_posture()
        env = SimpleNamespace(position_limits=np.asarray([[-4, 4]] * 14))
        pipeline = SimpleNamespace(
            _upper_body_cfg=cfg,
            _upper_body_default=defaults,
            _upper_body_filtered=defaults.copy(),
            _upper_body_indices=np.arange(14),
            _upper_body_enabled=True,
            _upper_body_stream_was_fresh=False,
            env=env,
            dt=0.02,
        )
        data = Box({"UpperBodyZmqCtrl": {"fresh": True, "joint_positions": evaluation.POSTURE}})
        result = UpperBodyZmqPipelineMixin._apply_upper_body_override(pipeline, defaults, data)
        self.assertLessEqual(np.max(np.abs(result - defaults)), 0.05 * 0.02 + 1e-7)

    def test_moving_reference_not_counted_as_sag(self):
        rows = [
            {"elapsed_s": t, "left_torso_z_m": 0.5 + t, "left_torso_x_m": 0, "left_torso_y_m": 0} for t in (1, 2, 3)
        ]
        self.assertEqual(evaluation.hand_difference(rows, rows, 1, 4, "left"), (0, 0))
        displaced = [dict(r, left_torso_z_m=r["left_torso_z_m"] - 0.02) for r in rows]
        mean, peak = evaluation.hand_difference(displaced, rows, 1, 4, "left")
        self.assertAlmostEqual(mean, 0.02)
        self.assertAlmostEqual(peak, 0.02)
        shifted = [dict(r, left_torso_x_m=0.03) for r in displaced]
        self.assertAlmostEqual(evaluation.hand_difference(shifted, rows, 1, 4, "left")[1], np.hypot(0.02, 0.03))
        with self.assertRaises(ValueError):
            evaluation.hand_difference(displaced[:-1], rows, 1, 4, "left")

    def test_reference_drift_is_reported_without_subtraction(self):
        rows = []
        for start in (7, 23, 86, 102):
            for offset in (0.1, 1.1, 2.1):
                t = start + offset
                rows.append({"elapsed_s": t, **{f"{name}_integral_nm": 0.03 * t for name in evaluation.POSTURE}})
        drift = evaluation.baseline_drift(rows)
        self.assertTrue(all(set(value) == {"effort_slope_nm_s"} for value in drift.values()))
        self.assertAlmostEqual(drift["grasp_recovery"]["effort_slope_nm_s"]["left_elbow_joint"], 0.03)

    def test_failure_or_missing_reference_has_no_accuracy_verdict(self):
        row = {"condition": "integral", "profile": "constant", "repetition": 1, "status": "failed"}
        self.assertEqual(evaluation.comparisons([row], Path("/tmp"))[0]["status"], "unavailable")

    def test_integral_harnesses_apply_load_at_hand_center(self):
        for cls, elapsed in ((evaluation.X2TransitionEvaluationEnv, 50),):
            with self.subTest(environment=cls.__name__):
                env = cls(evaluation.transition_config().env)
                try:
                    env.profile = "constant"
                    env.measurement_start = env.data.time - elapsed
                    env.update_load()
                    np.testing.assert_allclose(env.data.xfrc_applied[env.ee_ids, :3], [[0, 0, -20]] * 2)
                    np.testing.assert_array_equal(env.data.xfrc_applied[env.ee_ids, 3:], 0)
                    np.testing.assert_array_equal(env.force_points(), env.data.xipos[env.ee_ids])
                    self.assertTrue(
                        np.all(np.linalg.norm(env.force_points() - env.data.xpos[env.ee_ids], axis=1) > 0.08)
                    )
                finally:
                    env.shutdown()


if __name__ == "__main__":
    unittest.main()
