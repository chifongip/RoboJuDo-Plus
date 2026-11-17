"""Differential checks for the optional compiled nonlinear replay/solver."""

import builtins
import importlib.util
import unittest
import weakref
from unittest.mock import patch

import numpy as np

from robojudo.utils.observers import NonlinearDisturbanceObserver

HAS_NUMBA = importlib.util.find_spec("numba") is not None


def missing_backend_dependency(dependency):
    original_import = builtins.__import__

    def import_module(name, *args, **kwargs):
        if name == "accelerated":
            raise ModuleNotFoundError(f"No module named '{dependency}'", name=dependency)
        return original_import(name, *args, **kwargs)

    return patch("builtins.__import__", side_effect=import_module)


def model(backend, n=14):
    return NonlinearDisturbanceObserver(
        np.linspace(10, 100, n),
        np.linspace(1, 5, n),
        np.linspace(4, 36, n),
        inertia=np.linspace(0.05, 0.3, n),
        friction_limit=0.3,
        friction_viscosity=np.linspace(0, 110, n),
        bandwidth_rad_s=8,
        backend=backend,
    )


@unittest.skipUnless(HAS_NUMBA, "Install the observer-fast extra to test compiled kernels")
class AcceleratedObserverTests(unittest.TestCase):
    def test_stale_feedback_skips_prediction_preparation(self):
        for backend in ("numpy", "numba"):
            with self.subTest(backend=backend):
                observer = model(backend, 1)
                observer.update([0], [0], [0], 0.02)
                observer.effort[:] = 0.1
                observer.estimated_disturbance_nm[:] = -0.2
                with (
                    patch.object(observer, "prepare_predictor") as prepare,
                    np.errstate(over="raise", invalid="raise", divide="raise"),
                ):
                    # Coefficient preparation would overflow, but no prediction
                    # is needed when every sample is unchanged.
                    effort = observer.update([0], [0], [0], 1e-320, fresh=[False])
                prepare.assert_not_called()
                np.testing.assert_array_equal(effort, [0.1])
                np.testing.assert_array_equal(observer.estimated_disturbance_nm, [-0.2])

    def test_random_segmented_inversion_matches_numpy_exactly(self):
        rng = np.random.default_rng(828)
        for n in (1, 2, 14):
            reference, compiled = model("numpy", n), model("numba", n)
            for trial in range(40):
                q, v = rng.normal(0, 0.2, n), rng.normal(0, 0.4, n)
                measured = rng.normal(0, 1, n)
                covered = rng.random(n) > 0.2
                fresh = covered & (rng.random(n) > 0.2)
                segments = [
                    (rng.random(n) > 0.2, rng.normal(0, 1, n), rng.uniform(-1, 1, n), rng.uniform(0.001, 0.02, n))
                    for _ in range(trial % 6 + 1)
                ]
                previous = rng.uniform(-1, 1, n)
                reference.estimated_disturbance_nm = previous.copy()
                compiled.estimated_disturbance_nm = previous.copy()

                def predict(disturbance, q=q, v=v, n=n, segments=segments, reference=reference, covered=covered):
                    position, velocity = q.copy(), v.copy()
                    dq, dv = np.zeros(n), np.zeros(n)
                    for active, target, effort, dt in segments:
                        result = reference.predict_state(position, velocity, target, effort, disturbance, dt, dq, dv)
                        position, velocity, dq, dv = [
                            np.where(active, new, old)
                            for new, old in zip(result, (position, velocity, dq, dv), strict=True)
                        ]
                    return velocity, np.where(covered, dv, 1)

                predictor = compiled.prepare_predictor(predict, q, v, segments, covered)
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    try:
                        expected = reference.infer_disturbance(measured, predict, fresh)
                    except ValueError:
                        with self.assertRaises(ValueError):
                            compiled.infer_disturbance(measured, predictor, fresh)
                    else:
                        actual = compiled.infer_disturbance(measured, predictor, fresh)
                        np.testing.assert_array_equal(actual, expected)
                        np.testing.assert_array_equal(
                            compiled.inversion_residual_rad_s, reference.inversion_residual_rad_s
                        )

    def test_closed_loop_load_release_reversal_and_gain_changes(self):
        reference, compiled = model("numpy"), model("numba")
        reference.release_slew = compiled.release_slew = 16
        q, v = np.zeros(14), np.zeros(14)
        for step in range(500):
            if step in (200, 400):
                for observer in (reference, compiled):
                    observer.reset()
                    observer.kp *= 0.8
                    observer.kd *= 1.1
            load = np.full(14, (2 if step < 150 else 0 if step < 300 else -2))
            goal = np.full(14, 0.05 * np.sin(step * 0.1))
            q, v, _, _ = reference.predict_state(q, v, goal, reference.effort, load, 0.02)
            for observer in (reference, compiled):
                effort = observer.update(goal, q, v, 0.02)
                observer.acknowledge_effort(effort * (0.5 if step % 7 == 0 else 1))
            for field in (
                "effort",
                "estimated_disturbance_nm",
                "instantaneous_disturbance_nm",
                "inversion_residual_rad_s",
                "antiwindup",
                "release_limited",
            ):
                np.testing.assert_array_equal(getattr(compiled, field), getattr(reference, field))

    def test_prepared_replay_does_not_leave_reference_cycles(self):
        from robojudo.config.x2 import x2_locomimic_observer
        from robojudo.utils.observers import JointObserver

        cfg = x2_locomimic_observer().env
        cfg.observer.backend = "numba"
        dof = cfg.dof
        ctrl = JointObserver(
            cfg.observer, dof.joint_names, dof.stiffness, dof.damping, actuator_limits=dof.torque_limits, max_dt=0.1
        )
        zeros = np.zeros(len(dof.joint_names))
        stamps = np.full_like(zeros, 100.0)
        effort = ctrl.propose(zeros, zeros, zeros, 1, sample_times=stamps)
        ctrl.acknowledge(effort, command_time=100)
        history = ctrl._prepare_history(zeros[ctrl.indices], stamps[ctrl.indices] + 0.02)
        reference = weakref.ref(history)
        candidate_reference = weakref.ref(history.candidate_predictor)
        del history
        self.assertIsNone(reference())
        self.assertIsNone(candidate_reference())

    def test_x2_headless_physics_matches_numpy_over_load_reversal(self):
        from robojudo.config.x2 import x2_locomimic_observer
        from robojudo.environment import env_registry

        environments = []
        for backend in ("numpy", "numba"):
            cfg = x2_locomimic_observer().env
            cfg.headless, cfg.visualize_extras = True, False
            cfg.observer.backend = backend
            env = env_registry.get(cfg.env_type)(cfg)
            self.addCleanup(env.shutdown)
            environments.append(env)
        reference, compiled = environments
        target = reference.dof_pos.copy()
        arms = reference._observer.indices
        for step in range(120):
            for env in environments:
                env.data.qfrc_applied[-env.num_dofs :][arms] = 2 if step < 40 else 0 if step < 80 else -2
                env.step(target)
            np.testing.assert_array_equal(compiled.data.qpos, reference.data.qpos)
            np.testing.assert_array_equal(compiled.data.qvel, reference.data.qvel)
            np.testing.assert_array_equal(compiled.last_feedforward_effort, reference.last_feedforward_effort)
            self.assertEqual(compiled._observer.get_status(), reference._observer.get_status())

    def test_constructor_warms_only_signature_and_readonly_inputs_do_not_compile(self):
        from robojudo.utils.observers.accelerated import _infer

        compiled = model("numba", 2)
        signatures = list(_infer.signatures)
        self.assertEqual(len(signatures), 1)
        q = np.zeros(4)[::2]
        q.flags.writeable = False
        for _ in range(4):
            compiled.update(q, q, q, 0.02)
        # Also feed strided and read-only arrays directly to the solver boundary.
        predictor = compiled.prepare_predictor(
            lambda d: (q, np.ones(2)),
            q,
            q,
            [(np.ones(2, dtype=bool), q, q, 0.02)],
            np.ones(2, dtype=bool),
        )
        predictor.infer(q, [True, True])
        self.assertEqual(_infer.signatures, signatures)

    def test_custom_dynamics_and_custom_prediction_callbacks_are_preserved(self):
        compiled = model("numba", 1)

        def callback(d):
            return d.copy(), np.ones(1)

        compiled.update([0], [0], [0.25], 0.02, predictor=callback)
        self.assertAlmostEqual(compiled.instantaneous_disturbance_nm[0], 0.25, delta=1e-5)
        with patch.object(compiled, "predict_state"):
            self.assertIs(compiled.prepare_predictor(callback, np.zeros(1), np.zeros(1), [], [True]), callback)

    def test_nonfinite_and_overflowing_candidates_fail_closed(self):
        compiled = model("numba", 1)
        q = np.zeros(1)
        for value in (np.nan, np.inf):
            with self.assertRaises(ValueError):
                compiled.prepare_predictor(
                    lambda d: (d, d), q, q, [(np.ones(1, dtype=bool), np.array([value]), q, 0.02)], [True]
                )
        predictor = compiled.prepare_predictor(
            lambda d: (d, d),
            q,
            q,
            [(np.ones(1, dtype=bool), np.array([1e308]), q, 0.02)],
            [True],
        )
        with self.assertRaises(ValueError):
            predictor.infer(q, [True])
        valid = compiled.prepare_predictor(lambda d: (d, d), q, q, [], [False])
        with self.assertRaises(ValueError):
            valid.infer([np.nan], [False])


class ObserverBackendTests(unittest.TestCase):
    def test_linearized_config_rejects_explicit_numba(self):
        from robojudo.config.x2 import x2_locomimic_observer

        cfg = x2_locomimic_observer().env.observer
        cfg.algorithm, cfg.backend, cfg.release_slew_nm_s = "linearized", "numba", None
        with self.assertRaisesRegex(ValueError, "only the nonlinear"):
            type(cfg).model_validate(cfg.model_dump())

    def test_explicit_numpy_and_invalid_backend(self):
        self.assertEqual(model("numpy", 1).backend, "numpy")
        with self.assertRaises(ValueError):
            model("unknown", 1)

    def test_missing_optional_dependency_falls_back_only_for_auto(self):
        with missing_backend_dependency("numba"):
            self.assertEqual(model("auto", 1).backend, "numpy")
            with self.assertRaisesRegex(ImportError, "observer-fast"):
                model("numba", 1)

    def test_broken_backend_dependency_preserves_original_error(self):
        for backend in ("auto", "numba"):
            with self.subTest(backend=backend), missing_backend_dependency("llvmlite"):
                with self.assertRaises(ModuleNotFoundError) as raised:
                    model(backend, 1)
                self.assertEqual(raised.exception.name, "llvmlite")


if __name__ == "__main__":
    unittest.main()
