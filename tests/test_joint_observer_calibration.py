import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import evaluate_joint_observer as calibration
from scripts.observer_carry_pose import G1_HAND_AXES, X2_HAND_AXES, hand_pose, retarget_g1_carry, standing_state
from scripts.test_upper_body_zmq import PROFILES, validate_profile


class JointObserverCalibrationTests(unittest.TestCase):
    def test_greybox_fit_recovers_model_and_keeps_pulses_out_of_selection(self):
        effort = np.array(
            [calibration.excitation(step * calibration.PROBE_DT, 0.25) for step in range(calibration.PROBE_SAMPLES)]
        )
        parameters = np.array([0.1, 3.0, 50.0, 0.3, 30.0])
        response = calibration.rollout(np.log(parameters), effort)
        result = calibration.fit_response(effort, response)
        self.assertTrue(result["adequate"])
        np.testing.assert_allclose(list(result["parameters"].values()), parameters, rtol=1e-8)
        response[calibration.MULTISINE_SAMPLES :] *= 2
        bad_pulse = calibration.fit_response(effort, response)
        self.assertEqual(bad_pulse["parameters"], result["parameters"])
        self.assertFalse(bad_pulse["adequate"])

    def test_selector_uses_medians_armature_and_actual_pd_damping(self):
        fits = [
            {
                "adequate": True,
                "parameter_at_search_boundary": False,
                "parameters": {"inertia": inertia, "damping": damping, "friction_viscosity": slope},
            }
            for inertia, damping, slope in ((0.01, 5.0, 10.0), (0.08, 3.0, 2.0), (0.06, 10.0, 10.0))
        ]
        self.assertEqual(
            calibration.select_joint(fits, "left_elbow_joint", 0.03, 3.0),
            {"inertia": 0.06, "friction_viscosity": 12.0},
        )
        self.assertEqual(
            calibration.select_joint(fits, "left_wrist_roll_joint", 0.01, 1.5),
            {"inertia": 0.01, "friction_viscosity": 13.5},
        )
        self.assertEqual(
            calibration.select_joint(fits, "left_wrist_roll_joint", 0.01, 1.5, fixed_inertia=0.01020),
            {"inertia": 0.01020, "friction_viscosity": 13.5},
        )
        fits[0]["parameter_at_search_boundary"] = True
        with self.assertRaisesRegex(ValueError, "Standing validation"):
            calibration.select_joint(fits, "left_elbow_joint", 0.03, 3.0)

    def test_g1_wrist_fit_keeps_inertia_fixed_and_seeds_friction(self):
        effort = np.array(
            [calibration.excitation(step * calibration.PROBE_DT, 0.25) for step in range(calibration.PROBE_SAMPLES)]
        )
        parameters = [0.01020, 3.0, 50.0, 0.10, 30.0]
        response = calibration.rollout(np.log(parameters), effort)
        with patch.object(calibration, "least_squares", wraps=calibration.least_squares) as optimize:
            result = calibration.fit_response(
                effort,
                response,
                friction_seed_nm=calibration.G1_FRICTION_SEED_NM,
                fixed_inertia=calibration.G1_WRIST_ROLL_INERTIA_KG_M2,
            )
        optimize.assert_called_once()
        np.testing.assert_array_equal(optimize.call_args.args[1], np.log(parameters[1:]))
        self.assertEqual(result["parameters"]["inertia"], 0.01020)
        self.assertTrue(result["adequate"])
        self.assertFalse(result["parameter_at_search_boundary"])

    def test_setup_uses_calibration_checkpoint_and_registered_arm_order(self):
        for robot, policy, count in (("x2", "policy_fine_tune", 14), ("g1", "policy_23dof_stiff", 10)):
            with self.subTest(robot=robot):
                cfg, _, _, _, names, neutral, carry, force = calibration.calibration_setup(robot)
                self.assertEqual(cfg.loco_policy.policy_name, policy)
                self.assertEqual(force, 20.0)
                self.assertEqual(names, calibration.ROBOTS[robot][0]().env.observer.joint_names)
                self.assertEqual(len(names), count)
                self.assertEqual(neutral.shape, carry.shape)
                self.assertEqual(set(cfg.env.feedforward_effort_limits), set(names))

    def test_retargeted_g1_carry_shifts_hand_load_points_back_and_down(self):
        setups = {}
        points = {}
        for robot, axes in (("g1", G1_HAND_AXES), ("x2", X2_HAND_AXES)):
            cfg, _, _, body, names, _, carry, _ = calibration.calibration_setup(robot)
            targets = dict(zip(names, carry, strict=True))
            setups[robot] = cfg.env, targets
            _, data = standing_state(cfg.env.xml, cfg.env.dof, targets)
            points[robot] = [hand_pose(data, f"{side}_{body}", axes)[0] for side in ("left", "right")]
        np.testing.assert_allclose(points["g1"], np.asarray(points["x2"]) + [-0.10, 0, -0.15], rtol=0, atol=1e-5)

        g1, targets = setups["g1"]
        x2, x2_targets = setups["x2"]
        solution = retarget_g1_carry(g1.xml, g1.dof, list(targets), x2.xml, x2.dof, x2_targets)
        np.testing.assert_allclose(solution, list(targets.values()), rtol=0, atol=1e-6)
        profile = PROFILES["g1-23"]
        validate_profile(profile)
        self.assertEqual(profile.poses["custom_arms"], targets)

    def test_probe_retries_excited_trace_once_at_half_amplitude(self):
        with patch.object(calibration, "acquire_probe", side_effect=[ValueError("clipped"), "half.npz"]) as acquire:
            self.assertEqual(calibration.acquire_trial("g1", "grasp", 1, Path(".")), ("grasp", 1, "half.npz", None))
            self.assertEqual(acquire.call_args.kwargs, {"scale": 0.5})
        with patch.object(calibration, "acquire_probe", side_effect=ValueError("clipped")) as acquire:
            self.assertIn("clipped", calibration.acquire_trial("g1", "grasp", None, Path("."))[-1])
            acquire.assert_called_once()

    def test_trace_reader_requires_correct_robot_and_all_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps({"robot": "x2", "joint_names": ["a"], "traces": {}}))
            with self.assertRaisesRegex(ValueError, "robot or joint order"):
                calibration.read_manifest(root, "g1", ["a"], np.array([0.1]))
            probe = {"carry_force_per_hand_n": 10.0, "carry_target_rad": {"a": 0.1}}
            manifest = {"robot": "g1", "joint_names": ["a"], "traces": {}, "probe": probe}
            for force, angle in ((10.0, 0.1), (20.0, 0.2)):
                with self.subTest(force=force, angle=angle):
                    probe.update(carry_force_per_hand_n=force, carry_target_rad={"a": angle})
                    (root / "manifest.json").write_text(json.dumps(manifest))
                    with self.assertRaisesRegex(ValueError, "carry load and pose"):
                        calibration.read_manifest(root, "g1", ["a"], np.array([0.1]))

            probe.update(carry_force_per_hand_n=20.0, carry_target_rad={"a": 0.1})
            (root / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(calibration.read_manifest(root, "g1", ["a"], np.array([0.1])), manifest)
            probe["control_dt_s"] = 0.01
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "probe protocol"):
                calibration.read_manifest(root, "g1", ["a"], np.array([0.1]))

            trace = root / "incomplete.npz"
            np.savez(trace, q=np.zeros((4, 1)), dq=np.zeros((4, 1)), effort=np.zeros(4))
            with self.assertRaisesRegex(ValueError, "Incomplete trace"):
                calibration.load_probe(trace, 0, "a")

    def test_historical_reader_ignores_clipped_trial(self):
        completed = {
            "operating_point": "default",
            "joint_name": "a",
            "status": "completed",
            "samples": calibration.PROBE_SAMPLES,
            "saturated_physics_fraction": 0,
            "max_excitation_clip_nm": 0,
            "amplitude_scale": 1.0,
            "trace": "valid.csv",
        }
        clipped = {**completed, "amplitude_scale": 0.5, "max_excitation_clip_nm": 0.1, "trace": "clipped.csv"}
        manifest = {
            "posture_rad": {"a": 0.1},
            "dt_s": calibration.PROBE_DT,
            "duration_s": calibration.PROBE_DURATION,
            "frequencies_hz": calibration.FREQUENCIES.tolist(),
            "operating_times": calibration.OPERATING_TIMES,
            "force_per_hand_n": [0, 0, -calibration.CARRY_FORCE_PER_HAND_N],
            "trials": [clipped, completed],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps(manifest))
            replay = calibration.read_manifest(root, "x2", ["a"], np.array([0.1]))
        self.assertEqual(replay["traces"]["default"]["a"], "valid.csv")
