"""Hardware-free feedback, plant, telemetry and recovery regressions."""

import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from x2_feedback_perturbations import FeedbackChannel, Perturbation, study_cases
from x2_robustness_analysis import recovery
from x2_robustness_sim import PhysicsWriter, load_physics, perturb_plant


class TestPhysicsCoverage(unittest.TestCase):
    def test_both_timesteps_complete_truncated_missing_and_duplicate_samples(self):
        from x2_robustness_analysis import complete_window, physics_timestep

        for dt in (0.005, 0.0025):
            with self.subTest(dt=dt):
                data = {"elapsed_s": np.arange(dt, 2 + dt / 2, dt)}
                self.assertAlmostEqual(physics_timestep(data, {"physics_dt_s": dt}), dt)
                self.assertAlmostEqual(physics_timestep(data, {}), dt)
                self.assertTrue(complete_window(data, 0, 1, dt))
                self.assertTrue(complete_window(data, 1, 2, dt))
                self.assertFalse(complete_window({"elapsed_s": data["elapsed_s"][:-2]}, 1, 2, dt))
                damaged = data["elapsed_s"].copy()
                damaged[len(damaged) // 2 + 4] = damaged[len(damaged) // 2 + 3]
                self.assertFalse(complete_window({"elapsed_s": damaged}, 1, 2, dt))
                self.assertFalse(complete_window({"elapsed_s": np.delete(data["elapsed_s"], -3)}, 1, 2, dt))

    def test_analysis_keeps_refined_windows_and_marks_real_gaps_incomplete(self):
        from unittest.mock import patch

        from x2_integral_sim import POSTURE
        from x2_robustness_analysis import analyze_trial

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "trace.npz").touch()
            for dt in (0.005, 0.0025):
                times = np.arange(dt, 2 + dt / 2, dt)
                data = {"elapsed_s": times}
                for field in ("feedback_age_s", "base_tilt_rad", "base_height_m"):
                    data[field] = np.zeros(len(times))
                for name in POSTURE:
                    for field in (
                        "injected_position_noise_rad",
                        "injected_velocity_noise_rad_s",
                        "saturation_fraction",
                        "position_rad",
                        "target_rad",
                    ):
                        data[f"{name}_{field}"] = np.zeros(len(times))
                summary = dict(
                    physics_trace="trace.npz",
                    physics_dt_s=dt,
                    case="test",
                    condition="observer",
                    profile="unloaded",
                    repetition=1,
                    seed=1,
                    status="completed",
                )
                with (
                    patch("x2_robustness_analysis.load_physics", return_value=data),
                    patch("x2_robustness_analysis.physics_statistics", side_effect=lambda _: {"test": {}}),
                    patch("x2_robustness_analysis.WINDOWS", {"test": (1, 2)}),
                    patch("x2_robustness_analysis.PHASES", [("test", 1, 2)]),
                ):
                    complete = analyze_trial(root, summary)
                    self.assertIn("test", complete["windows"])
                    self.assertTrue(complete["phases"]["test"]["complete"])
                    data["elapsed_s"][-3] = data["elapsed_s"][-4]
                    damaged = analyze_trial(root, summary)
                    self.assertNotIn("test", damaged["windows"])
                    self.assertFalse(damaged["phases"]["test"]["complete"])


class TestStudyPairing(unittest.TestCase):
    def test_full_matrix_preserves_matched_conditions_and_both_payload_directions(self):
        from evaluate_x2_compliant_robustness import jobs_for
        from evaluate_x2_transitions import condition_parameters

        with tempfile.TemporaryDirectory() as directory:
            count = 0
            for case in study_cases():
                groups = defaultdict(list)
                for job in jobs_for(case, Path(directory)):
                    method, profile, seed, output, viewer, parameters, warmup, feedback, rng_seed, payload = job
                    self.assertFalse(viewer)
                    self.assertEqual(parameters, condition_parameters(method))
                    self.assertEqual(feedback, vars(case))
                    self.assertEqual((rng_seed, warmup), (seed, seed + 3))
                    self.assertEqual(
                        payload,
                        {"initial_sign": 1 if output.name == "positive" else -1}
                        if profile == "compliant_payload"
                        else None,
                    )
                    groups[profile, seed, output.name].append(method)
                    count += 1
                self.assertEqual(len(groups), 12)
                for methods in groups.values():
                    self.assertCountEqual(methods, ("pd_only", "integral", "observer"))
            self.assertEqual(count, 792)


class TestFeedback(unittest.TestCase):
    def channel(self, **kwargs):
        return FeedbackChannel(Perturbation("test", **kwargs), 7, np.zeros(14), np.zeros(14), 0)

    def test_identity_and_copy_isolation(self):
        c = self.channel()
        q, v = np.arange(14, dtype=np.float32), -np.arange(14, dtype=np.float32)
        c.acquire(0.005, q, v, 0.005)
        q[:] = 99
        a, b = c.measured()
        np.testing.assert_array_equal(a, np.arange(14))
        np.testing.assert_array_equal(b, -np.arange(14))
        a[:] = 999
        self.assertEqual(c.position[0], 0)

    def test_delayed_packet_never_uses_current_clean_feedback(self):
        c = self.channel(latency_min_s=0.02, latency_max_s=0.02)
        for k in range(1, 7):
            c.acquire(k * 0.005, np.full(14, k), np.full(14, -k), k * 0.005)
            expected = max(0, k - 4)
            self.assertEqual(c.position[0], expected)
            self.assertEqual(c.velocity[0], -expected)

    def test_drop_holds_same_noisy_packet_without_redraw(self):
        c = self.channel(position_std_deg=0.1, dropout_s=0.04)
        c.acquire(11.995, np.ones(14), np.ones(14), 11.995)
        held = c.position.copy()
        for t in np.arange(12, 12.04, 0.005):
            c.acquire(t, np.full(14, 12345), np.full(14, 23456), t)
            np.testing.assert_array_equal(c.position, held)
        c.acquire(12.04, np.full(14, 2), np.full(14, 3), 12.04)
        self.assertLess(np.max(np.abs(c.position - 2)), 0.02)

    def test_seed_reproducibility_and_out_of_order_discard(self):
        config = dict(position_std_deg=0.1, latency_max_s=0.06)
        a, b = self.channel(**config), self.channel(**config)
        stamps = []
        for k in range(1, 301):
            for c in (a, b):
                c.acquire(k * 0.005, np.full(14, k), np.zeros(14), k * 0.005)
            np.testing.assert_array_equal(a.position, b.position)
            stamps.append(a.acquired_time)
        self.assertTrue(np.all(np.diff(stamps) >= 0))
        self.assertGreater(a.out_of_order, 0)

    def test_colored_noise_covariance_and_time_scale(self):
        c = self.channel(colored=True, position_std_deg=0.03, velocity_std_rad_s=0.015)
        samples = []
        for k in range(1, 12001):
            c.acquire(k * 0.005, np.zeros(14), np.zeros(14), k * 0.005)
            samples.append(c.last_noise.copy())
        x = np.asarray(samples)[1000:]
        self.assertAlmostEqual(np.std(x[:, 0]), np.deg2rad(0.03), delta=np.deg2rad(0.004))
        self.assertAlmostEqual(np.std(x[:, 14]), 0.015, delta=0.002)
        self.assertAlmostEqual(np.corrcoef(x[:, 0], x[:, 1])[0, 1], 0.5, delta=0.1)
        self.assertAlmostEqual(np.corrcoef(x[:, 0], x[:, 14])[0, 1], 0.5, delta=0.1)
        self.assertAlmostEqual(np.corrcoef(x[:, 0], x[:, 7])[0, 1], 0, delta=0.12)
        self.assertAlmostEqual(np.corrcoef(x[:-1, 0], x[1:, 0])[0, 1], np.exp(-0.1), delta=0.03)

    def test_matrix_and_invalid_ranges(self):
        cases = study_cases()
        self.assertEqual(len(cases), 22)
        self.assertEqual(len({c.case for c in cases}), 22)
        self.assertEqual(sum(c.shared_pd for c in cases), 2)
        for kwargs in (
            {"position_std_deg": -1},
            {"latency_min_s": 1},
            {"inertia_scale": 0},
            {"velocity_std_rad_s": float("nan")},
        ):
            with self.assertRaises(ValueError):
                Perturbation("invalid", **kwargs)


class TestPlantAndTelemetry(unittest.TestCase):
    def test_plant_changes_only_selected_arrays_and_identity_is_exact(self):
        model = mujoco.MjModel.from_xml_path("assets/robots/x2/x2.xml")
        from x2_integral_sim import POSTURE

        original = {
            f: getattr(model, f).copy()
            for f in ("body_inertia", "body_mass", "body_ipos", "dof_armature", "dof_frictionloss", "dof_damping")
        }
        perturb_plant(model, POSTURE, Perturbation("N0"))
        for f, a in original.items():
            np.testing.assert_array_equal(getattr(model, f), a)
        metadata = perturb_plant(model, POSTURE, Perturbation("test", inertia_scale=1.3, friction_scale=0.5))
        for f in ("body_mass", "body_ipos", "dof_damping", "dof_armature"):
            np.testing.assert_array_equal(getattr(model, f), original[f])
        self.assertEqual(len(metadata["dof_frictionloss"]["indices"]), 14)
        for field, scale in [("body_inertia", 1.3), ("dof_frictionloss", 0.5)]:
            np.testing.assert_allclose(metadata[field]["perturbed"], np.array(metadata[field]["original"]) * scale)

    def test_actual_simulator_uses_stale_channel_and_keeps_four_physics_samples(self):
        from unittest.mock import patch

        from evaluate_x2_transitions import transition_config
        from x2_integral_sim import POSTURE
        from x2_robustness_sim import X2RobustnessEvaluationEnv

        env = X2RobustnessEvaluationEnv(transition_config(False).env)
        self.addCleanup(env.shutdown)
        with tempfile.TemporaryDirectory() as tmp:
            from dataclasses import asdict

            env.configure_robustness(
                asdict(Perturbation("P1", latency_min_s=0.02, latency_max_s=0.02, shared_pd=True)),
                1,
                Path(tmp),
                "integral",
                "unloaded",
                1,
            )
            env.start_feedback(0)
            initial_q, initial_v = env.feedback.measured()
            env.measurement_start = env.data.time
            env.integral_enabled = True
            env.start_telemetry()
            target = env.dof_pos.copy()
            target[env.arm_indices] += 0.001
            with patch.object(env.integral, "update", wraps=env.integral.update) as update:
                env.step(target)
                env.reference_time = 0.02
                env.step(target)
                self.assertEqual(update.call_count, 2)
                for args in update.call_args_list:
                    np.testing.assert_array_equal(args.args[1], initial_q)
                    np.testing.assert_array_equal(args.args[2], initial_v)
                    self.assertEqual(args.args[3], 0.02)
            env.close_telemetry()
            data = load_physics(env.physics_path)
            self.assertEqual(len(data["elapsed_s"]), 8)
            np.testing.assert_allclose(np.diff(data["elapsed_s"]), 0.005, atol=1e-10)
            np.testing.assert_array_equal(data["correction_update_s"], [0] * 4 + [0.02] * 4)
            for i, name in enumerate(POSTURE):
                expected = env.stiffness[env.arm_indices[i]] * (target[env.arm_indices[i]] - initial_q[i])
                expected -= env.damping[env.arm_indices[i]] * initial_v[i]
                np.testing.assert_allclose(data[f"{name}_pd_nm"][:4], expected, atol=1e-12)
            env.command_passive()
            np.testing.assert_array_equal(env.last_applied_torque, 0)
            env.clear_load()

    def test_streamed_chunks_roundtrip_partial_trial(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.npz"
            w = PhysicsWriter(path)
            for i in range(601):
                w.write({"elapsed_s": 0.005 * i, "phase": "walk", "x": float(i)})
            w.close()
            data = load_physics(path)
            np.testing.assert_array_equal(data["x"], np.arange(601))
            self.assertEqual(len(data["phase_id"]), 601)


class TestRecovery(unittest.TestCase):
    def test_warmup_failure_is_retained_without_physics_samples(self):
        from x2_robustness_analysis import analyze_trial

        with tempfile.TemporaryDirectory() as tmp:
            result = analyze_trial(
                Path(tmp),
                {
                    "case": "P2",
                    "condition": "observer",
                    "profile": "sway",
                    "seed": 2,
                    "repetition": 2,
                    "status": "failed",
                    "physics_trace": "missing.npz",
                },
            )
        self.assertEqual(result["case"], "P2")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["phases"], {})

    def traces(self):
        t = np.arange(0.005, 105.001, 0.005)
        u = {"elapsed_s": t}
        # Measured unloaded excursion envelope of approximately 1 mm.
        for a in "xyz":
            u[f"left_torso_{a}_m"] = 0.001 * np.sin(2 * np.pi * t) if a == "z" else np.zeros(len(t))
        return {k: v.copy() for k, v in u.items()}, u

    def test_peak_rebound_residual_and_settling(self):
        d, u = self.traces()
        t = d["elapsed_s"]
        z = np.zeros(len(t))
        z[(t >= 79) & (t < 80)] = 0.01
        z[(t >= 80) & (t < 81)] = -0.004
        d["left_torso_z_m"] += z
        r = recovery(d, u, "left")
        self.assertAlmostEqual(r["peak_m"], 0.01)
        self.assertAlmostEqual(r["vertical_below_reference_peak_m"], 0.004)
        self.assertAlmostEqual(r["settling_time_s"], 2.0, delta=0.006)
        self.assertAlmostEqual(r["vertical_rebound_signed_m"], -0.014)
        self.assertEqual(r["terminal_residual_m"], 0)

    def test_late_escape_and_bias_are_censored(self):
        for start in (79, 88.5, 89.0):
            d, u = self.traces()
            d["left_torso_z_m"][d["elapsed_s"] >= start] += 0.005
            r = recovery(d, u, "left")
            self.assertIsNone(r["settling_time_s"])
            self.assertEqual(r["status"], "not_settled_within_observation")
        with self.assertRaises(ValueError):
            recovery({k: v[:-5000] for k, v in d.items()}, u, "left")

    def test_first_sample_is_not_a_physical_settling_claim(self):
        d, u = self.traces()
        r = recovery(d, u, "left")
        self.assertEqual(r["status"], "first_sample_inside")
        self.assertLess(r["first_sample_offset_s"], 0.00501)


if __name__ == "__main__":
    unittest.main()
