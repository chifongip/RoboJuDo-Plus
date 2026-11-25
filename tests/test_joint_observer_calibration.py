import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from scripts import evaluate_joint_observer as calibration
from scripts.test_upper_body_zmq import PROFILES, validate_profile

G1_HAND_AXES = np.eye(3)
X2_HAND_AXES = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])


def standing_state(xml, dof, arm_targets):
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    targets = dict(zip(dof.joint_names, dof.default_pos, strict=True)) | arm_targets
    for name, value in targets.items():
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_fwdPosition(model, data)
    return model, data


def hand_pose(data, body_name, local_axes):
    pelvis = data.body("pelvis")
    hand = data.body(body_name)
    world_to_pelvis = pelvis.xmat.reshape(3, 3).T
    return (
        world_to_pelvis @ (hand.xipos - pelvis.xpos),
        world_to_pelvis @ hand.xmat.reshape(3, 3) @ local_axes,
    )


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
                fixed_inertia=calibration.G1_23_WRIST_ROLL_INERTIA_KG_M2,
            )
        optimize.assert_called_once()
        np.testing.assert_array_equal(optimize.call_args.args[1], np.log(parameters[1:]))
        self.assertEqual(result["parameters"]["inertia"], 0.01020)
        self.assertTrue(result["adequate"])
        self.assertFalse(result["parameter_at_search_boundary"])

    def test_setup_uses_calibration_checkpoint_and_registered_arm_order(self):
        for robot, policy, count in (
            ("x2", "policy_fine_tune", 14),
            ("g1-23", "policy_23dof_stiff", 10),
            ("g1-29", "policy_29dof_stiff", 14),
        ):
            with self.subTest(robot=robot):
                cfg, _, _, _, names, neutral, carry, force = calibration.calibration_setup(robot)
                self.assertEqual(cfg.loco_policy.policy_name, policy)
                self.assertEqual(force, 20.0)
                self.assertEqual(names, calibration.ROBOTS[robot][0]().env.observer.joint_names)
                self.assertEqual(len(names), count)
                self.assertEqual(neutral.shape, carry.shape)
                self.assertEqual(set(cfg.env.feedforward_effort_limits), set(names))

    def test_29_dof_calibration_uses_wrist_yaw_load_bodies_and_carry_pose(self):
        from robojudo.config.g1.carry_pose import G1_CARRY_POSE
        from robojudo.config.g1.env.g1_env_cfg import G1_29_ARM_JOINT_NAMES
        from robojudo.config.g1.observer_parameters import selected_observer_parameters

        cfg, _, _, body, names, _, carry, _ = calibration.calibration_setup("g1-29")
        self.assertEqual(names, list(G1_29_ARM_JOINT_NAMES))
        self.assertEqual(body, "wrist_yaw_link")
        self.assertEqual(dict(zip(names, carry, strict=True)), G1_CARRY_POSE)
        self.assertEqual(cfg.env.observer.joint_names, names)
        self.assertEqual(cfg.env.dof.num_dofs, 29)
        self.assertEqual(cfg.env.observer.inertia, selected_observer_parameters()["inertia"])
        self.assertEqual(cfg.env.observer.friction_viscosity, selected_observer_parameters()["friction_viscosity"])
        parameters = selected_observer_parameters()
        parameters["inertia"][0] = 0
        self.assertGreater(selected_observer_parameters()["inertia"][0], 0)

    def test_29_dof_manifest_requires_current_small_wrist_excitation(self):
        names = ["left_wrist_yaw_joint"]
        carry = np.array([0.1])
        manifest = {
            "robot": "g1-29",
            "joint_names": names,
            "traces": {},
            "probe": calibration.probe_metadata(names, carry, robot="g1-29"),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            self.assertEqual(calibration.read_manifest(root, "g1-29", names, carry), manifest)
            manifest["probe"].pop("small_wrist_probe_fraction")
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "small-wrist probe amplitude"):
                calibration.read_manifest(root, "g1-29", names, carry)

    def test_29_dof_small_wrist_probe_exceeds_friction(self):
        for side in ("left", "right"):
            for axis in ("pitch", "yaw"):
                name = f"{side}_wrist_{axis}_joint"
                self.assertEqual(calibration.probe_amplitude("g1-29", name, 5.0), 0.125)
        self.assertEqual(calibration.probe_amplitude("g1-29", "left_wrist_roll_joint", 25.0), 0.25)
        self.assertEqual(calibration.probe_amplitude("x2", "left_wrist_yaw_joint", 5.0), 0.05)

    def test_g1_carry_shifts_hand_load_points_back_and_down_from_x2(self):
        points = {}
        for robot, axes in (("g1-23", G1_HAND_AXES), ("x2", X2_HAND_AXES)):
            cfg, _, _, body, names, _, carry, _ = calibration.calibration_setup(robot)
            targets = dict(zip(names, carry, strict=True))
            _, data = standing_state(cfg.env.xml, cfg.env.dof, targets)
            points[robot] = [hand_pose(data, f"{side}_{body}", axes)[0] for side in ("left", "right")]
        np.testing.assert_allclose(points["g1-23"], np.asarray(points["x2"]) + [-0.05, 0, -0.15], rtol=0, atol=1e-5)

        from robojudo.config.g1.carry_pose import G1_23_CARRY_POSE

        profile = PROFILES["g1-23"]
        validate_profile(profile)
        self.assertEqual(profile.poses["custom_arms"], G1_23_CARRY_POSE)

    def test_g1_29_carry_is_5cm_forward_of_23_with_same_hand_orientations(self):
        from robojudo.config.g1.carry_pose import G1_23_CARRY_POSE, G1_CARRY_POSE
        from robojudo.config.g1.env.g1_env_cfg import G1_23DoF, G1_29DoF

        model23, data23 = standing_state("assets/robots/g1/g1_23dof_rev_1_0.xml", G1_23DoF(), G1_23_CARRY_POSE)
        model29, data29 = standing_state("assets/robots/g1/g1_29dof_rev_1_0.xml", G1_29DoF(), G1_CARRY_POSE)
        for side in ("left", "right"):
            with self.subTest(side=side):
                point23, orientation23 = hand_pose(data23, f"{side}_wrist_roll_rubber_hand", G1_HAND_AXES)
                point29, orientation29 = hand_pose(data29, f"{side}_wrist_yaw_link", G1_HAND_AXES)
                np.testing.assert_allclose(point29 - point23, [0.05, 0, 0], rtol=0, atol=1e-5)
                np.testing.assert_allclose(orientation29, orientation23, rtol=0, atol=1e-5)
        for model, data in ((model23, data23), (model29, data29)):
            self.assertFalse(
                any(
                    contact.dist < 0 and model.geom_bodyid[contact.geom1] and model.geom_bodyid[contact.geom2]
                    for contact in data.contact
                )
            )
        profile = PROFILES["g1-29"]
        validate_profile(profile)
        self.assertEqual(profile.poses["custom_arms"], G1_CARRY_POSE)

    def test_probe_retries_excited_trace_once_at_half_amplitude(self):
        with patch.object(calibration, "acquire_probe", side_effect=[ValueError("clipped"), "half.npz"]) as acquire:
            self.assertEqual(calibration.acquire_trial("g1-23", "grasp", 1, Path(".")), ("grasp", 1, "half.npz", None))
            self.assertEqual(acquire.call_args.kwargs, {"scale": 0.5})
        with patch.object(calibration, "acquire_probe", side_effect=ValueError("clipped")) as acquire:
            self.assertIn("clipped", calibration.acquire_trial("g1-23", "grasp", None, Path("."))[-1])
            acquire.assert_called_once()

    def test_trace_reader_requires_correct_robot_and_all_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps({"robot": "x2", "joint_names": ["a"], "traces": {}}))
            with self.assertRaisesRegex(ValueError, "robot or joint order"):
                calibration.read_manifest(root, "g1-23", ["a"], np.array([0.1]))
            probe = {"carry_force_per_hand_n": 10.0, "carry_target_rad": {"a": 0.1}}
            manifest = {"robot": "g1-23", "joint_names": ["a"], "traces": {}, "probe": probe}
            for force, angle in ((10.0, 0.1), (20.0, 0.2)):
                with self.subTest(force=force, angle=angle):
                    probe.update(carry_force_per_hand_n=force, carry_target_rad={"a": angle})
                    (root / "manifest.json").write_text(json.dumps(manifest))
                    with self.assertRaisesRegex(ValueError, "carry load and pose"):
                        calibration.read_manifest(root, "g1-23", ["a"], np.array([0.1]))

            probe.update(carry_force_per_hand_n=20.0, carry_target_rad={"a": 0.1})
            (root / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(calibration.read_manifest(root, "g1-23", ["a"], np.array([0.1])), manifest)
            with self.assertRaisesRegex(ValueError, "robot or joint order"):
                calibration.read_manifest(root, "g1-29", ["a"], np.array([0.1]))
            probe["control_dt_s"] = 0.01
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "probe protocol"):
                calibration.read_manifest(root, "g1-23", ["a"], np.array([0.1]))

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
