import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from box import Box


class FakePolicyEnv:
    def __init__(self, dof_cfg):
        self.dof_cfg = dof_cfg
        self.default_pos = np.asarray(dof_cfg.default_pos, dtype=np.float32)
        self.dof_pos = self.default_pos.copy()
        self.joint_names = list(dof_cfg.joint_names)
        self.reset_count = 0
        self.elastic_band = SimpleNamespace(active=False, rest_length=0.7)

    def reset(self):
        self.reset_count += 1
        self.elastic_band.active = True
        self.elastic_band.rest_length = 0.0

    def update_dof_cfg(self, override_cfg=None):
        self.last_dof_cfg = override_cfg

    def set_control_joint_names(self, joint_names):
        self.control_joint_names = list(joint_names)


class TestX2LocomanipulationLocoMimic(unittest.TestCase):
    def test_mujoco_base_linear_velocity_is_heading_invariant_after_alignment(self):
        from unittest.mock import patch

        import mujoco
        from scipy.spatial.transform import Rotation

        from robojudo.config.x2.x2_cfg import x2_locomimic_beyondmimic
        from robojudo.environment.mujoco_env import MujocoEnv

        class HeadlessViewer:
            def __init__(self, *args, **kwargs):
                self.cam = SimpleNamespace(distance=0, elevation=0, azimuth=0, lookat=np.zeros(3))
                self.is_alive = False

            def close(self):
                pass

        body_velocity = np.asarray([0.4, -0.2, 0.1], dtype=np.float32)
        with patch("robojudo.environment.mujoco_env.mujoco_viewer.MujocoViewer", HeadlessViewer):
            env = MujocoEnv(x2_locomimic_beyondmimic().env)
        self.addCleanup(env.shutdown)

        for heading in (0.0, 90.0, 180.0):
            with self.subTest(heading=heading):
                mujoco.mj_resetDataKeyframe(env.model, env.data, 0)
                initial_quat = Rotation.from_quat(env.data.qpos[3:7][[1, 2, 3, 0]])
                raw_quat = (Rotation.from_euler("z", heading, degrees=True) * initial_quat).as_quat()
                env.data.qpos[3:7] = raw_quat[[3, 0, 1, 2]]
                env.data.qvel[:] = 0.0
                env.data.qvel[:3] = Rotation.from_quat(raw_quat).apply(body_velocity)
                mujoco.mj_forward(env.model, env.data)

                env.base_align.set_base()
                env.update()
                env.reset_alignment()

                np.testing.assert_allclose(env.base_lin_vel, body_velocity, atol=1e-6)

    def test_aligned_debug_arrow_is_drawn_in_raw_mujoco_world(self):
        from scipy.spatial.transform import Rotation

        from robojudo.environment.utils.mujoco_viz import MujocoVisualizer
        from robojudo.utils.rotation import TransformAlignment

        markers = []
        viewer = SimpleNamespace(add_marker=lambda **marker: markers.append(marker))
        alignment = TransformAlignment(
            quat=Rotation.from_euler("z", 180.0, degrees=True).as_quat(),
            pos=np.asarray([2.0, 3.0, 0.0]),
            yaw_only=True,
            xy_only=True,
        )
        visualizer = MujocoVisualizer(viewer, alignment=alignment)

        visualizer.draw_arrow(
            origin=np.asarray([1.0, 0.0, 0.5]),
            root_quat=np.asarray([0.0, 0.0, 0.0, 1.0]),
            vec_local=np.asarray([0.2, 0.0, 0.0]),
            color=[1.0, 0.0, 0.0, 1.0],
            aligned_frame=True,
        )

        np.testing.assert_allclose(markers[0]["pos"], [1.0, 3.0, 0.5], atol=1e-7)
        np.testing.assert_allclose(markers[0]["mat"] @ [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], atol=1e-7)

    def test_beyondmimic_reference_aligns_to_robot_anchor_at_activation(self):
        from scipy.spatial.transform import Rotation

        from robojudo.config.x2.policy.x2_beyondmimic_policy_cfg import X2BeyondMimicPolicyCfg
        from robojudo.policy.x2_beyondmimic_policy import X2BeyondMimicPolicy

        with self.assertRaisesRegex(ValueError, "override_robot_anchor_pos must be False"):
            X2BeyondMimicPolicy(
                X2BeyondMimicPolicyCfg(
                    max_timestep=6747,
                    policy_name="Solo_dance",
                    without_state_estimator=False,
                    override_robot_anchor_pos=True,
                ),
                "cpu",
            )

        policy = X2BeyondMimicPolicy(
            X2BeyondMimicPolicyCfg(
                max_timestep=6747,
                policy_name="Solo_dance",
                without_state_estimator=False,
            ),
            "cpu",
        )
        self.addCleanup(policy.close_progress)

        robot_pos = np.asarray([1.2, -0.7, 0.93], dtype=np.float32)
        robot_quat = Rotation.from_euler("xyz", [0.08, -0.04, np.pi]).as_quat()
        env_data = Box({"torso_pos": robot_pos, "torso_quat": robot_quat})

        policy.reset_alignment(env_data)
        _, _, _, anchor_pos, anchor_quat, _ = policy._get_command(env_data, Box({}))

        np.testing.assert_allclose(anchor_pos, robot_pos, atol=1e-5)
        anchor_yaw = Rotation.from_quat(anchor_quat).as_euler("xyz")[2]
        robot_yaw = Rotation.from_quat(robot_quat).as_euler("xyz")[2]
        yaw_error = np.arctan2(np.sin(anchor_yaw - robot_yaw), np.cos(anchor_yaw - robot_yaw))
        self.assertAlmostEqual(yaw_error, 0.0, places=5)

        observations = []
        actions = []
        for heading in (0.0, np.pi / 2, np.pi):
            policy.reset()
            heading_quat = Rotation.from_euler("xyz", [0.08, -0.04, heading]).as_quat()
            heading_env_data = Box(
                {
                    "dof_pos": policy.default_dof_pos.copy(),
                    "dof_vel": np.zeros(policy.num_dofs, dtype=np.float32),
                    "base_ang_vel": np.asarray([0.01, -0.02, 0.03], dtype=np.float32),
                    "base_lin_vel": np.asarray([0.04, -0.01, 0.0], dtype=np.float32),
                    "torso_pos": robot_pos,
                    "torso_quat": heading_quat,
                }
            )
            policy.reset_alignment(heading_env_data)
            observation, _ = policy.get_observation(heading_env_data, Box({}))
            observations.append(observation)
            actions.append(policy.get_action(observation))

        for observation in observations[1:]:
            np.testing.assert_allclose(observation, observations[0], atol=1e-5)
        for action in actions[1:]:
            np.testing.assert_allclose(action, actions[0], atol=1e-5)

    def test_x2_real_configures_beyondmimic_without_external_odometry(self):
        from robojudo.config.x2.x2_cfg import (
            x2_locomimic_beyondmimic,
            x2_locomimic_beyondmimic_real,
        )

        sim_cfg = x2_locomimic_beyondmimic()
        real_cfg = x2_locomimic_beyondmimic_real()

        self.assertEqual(
            [(policy.policy_name, policy.without_state_estimator) for policy in sim_cfg.mimic_policies],
            [
                ("Walk2_subject1_wose", True),
                ("Walk2_subject1", False),
                ("Solo_dance", False),
                ("Walk1_subject1_wose", True),
            ],
        )
        self.assertEqual(
            [(policy.policy_name, policy.without_state_estimator) for policy in real_cfg.mimic_policies],
            [
                ("Walk2_subject1_wose", True),
                ("Walk2_subject1", False),
                ("Solo_dance", False),
                ("Walk1_subject1_wose", True),
            ],
        )
        self.assertTrue(all(Path(policy.policy_file).is_file() for policy in sim_cfg.mimic_policies))
        self.assertTrue(all(Path(policy.policy_file).is_file() for policy in real_cfg.mimic_policies))
        self.assertEqual(real_cfg.env.odometry_type, "DUMMY")
        self.assertEqual(real_cfg.ctrl[0].ctrl_type, "RosJoystickCtrl")
        self.assertEqual(real_cfg.ctrl[0].profile, "xbox_bluetooth")
        self.assertEqual(real_cfg.ctrl[0].topic, "/joy")

    def test_aimdk_odometry_supplies_aligned_root_pose(self):
        from scipy.spatial.transform import Rotation

        from robojudo.environment.agibot_cpp_env import AgiBotCppEnv
        from robojudo.utils.rotation import TransformAlignment

        raw_heading = Rotation.from_euler("z", 180.0, degrees=True).as_quat()
        body_velocity = np.asarray([0.4, -0.2, 0.1], dtype=np.float32)
        raw_position = np.asarray([1.5, 2.25, 0.87], dtype=np.float32)
        state = SimpleNamespace(
            motor_state=SimpleNamespace(q=np.zeros(31), dq=np.zeros(31)),
            imu_state=SimpleNamespace(
                quaternion=raw_heading,
                gyroscope=[0.01, -0.02, 0.03],
                accelerometer=[0.0, 0.0, 9.81],
            ),
            odometry_state=SimpleNamespace(
                valid=True,
                sequence=1,
                stamp_sec=10,
                stamp_nanosec=0,
                position=raw_position,
                quaternion=raw_heading,
                linear_velocity=body_velocity,
            ),
        )

        env = AgiBotCppEnv.__new__(AgiBotCppEnv)
        env.enabled = False
        env.aimdk = SimpleNamespace(get_robot_state=lambda: state)
        env.cfg_env = SimpleNamespace(
            aimdk=SimpleNamespace(
                state_timeout=0.1,
                odometry_timeout=0.3,
                odometry_velocity_filter_time_constant=0.15,
                shutdown_damping=5.0,
                torso_to_odometry_sensor_position=[0.0, 0.0, 0.0],
                torso_to_odometry_sensor_quaternion=[0.0, 0.0, 0.0, 1.0],
            )
        )
        env._odometry_type = "AIMDK"
        env.born_place_align = True
        env.base_align = TransformAlignment(
            quat=raw_heading,
            pos=np.asarray([1.0, 2.0, 0.0]),
            yaw_only=True,
            xy_only=True,
        )
        env.update_with_fk = True
        env._torso_name = "torso_link"
        env.kinematics = SimpleNamespace(
            forward=lambda **_: {
                "torso_link": {
                    "pos": np.zeros(3),
                    "quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
                }
            }
        )
        env._last_odometry_sequence = None
        env._last_odometry_stamp = None
        env._last_odometry_root_pos = None
        env._last_odometry_root_quat = None
        env._filtered_base_lin_vel = np.zeros(3, dtype=np.float32)
        env._last_odometry_receipt_time = None
        env.fk = lambda: {
            "torso_link": {
                "pos": env._base_pos.copy(),
                "quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
                "ang_vel": np.zeros(3),
            }
        }

        env.update()

        expected_position = env.base_align.align_pos(raw_position)
        np.testing.assert_allclose(env.base_pos, expected_position, atol=1e-6)
        np.testing.assert_allclose(env.torso_pos, expected_position, atol=1e-6)
        np.testing.assert_allclose(env.base_lin_vel, np.zeros(3), atol=1e-6)
        np.testing.assert_allclose(env.base_quat, [0.0, 0.0, 0.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(env.torso_quat, [0.0, 0.0, 0.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(env.fk_info["torso_link"]["pos"], expected_position, atol=1e-6)

    def test_superodom_sensor_pose_converts_through_torso_and_waist_to_pelvis(self):
        from scipy.spatial.transform import Rotation

        from robojudo.environment.agibot_cpp_env import AgiBotCppEnv

        pelvis_pos = np.asarray([1.2, -0.4, 0.72])
        pelvis_rot = Rotation.from_euler("z", 180.0, degrees=True)
        pelvis_to_torso_pos = np.asarray([0.0, 0.0, 0.155])
        pelvis_to_torso_rot = Rotation.from_euler("xyz", [0.08, -0.21, 0.12])
        torso_to_sensor_pos = np.asarray([0.102632855873251, 0.0, 0.181586916322065])
        torso_to_sensor_rot = Rotation.from_euler("x", -np.pi / 2)

        world_to_torso_rot = pelvis_rot * pelvis_to_torso_rot
        world_to_torso_pos = pelvis_pos + pelvis_rot.apply(pelvis_to_torso_pos)
        world_to_sensor_rot = world_to_torso_rot * torso_to_sensor_rot
        world_to_sensor_pos = world_to_torso_pos + world_to_torso_rot.apply(torso_to_sensor_pos)

        env = AgiBotCppEnv.__new__(AgiBotCppEnv)
        env._dof_pos = np.zeros(31)
        env._torso_name = "torso_link"
        env.cfg_env = SimpleNamespace(
            aimdk=SimpleNamespace(
                torso_to_odometry_sensor_position=torso_to_sensor_pos.tolist(),
                torso_to_odometry_sensor_quaternion=torso_to_sensor_rot.as_quat().tolist(),
            )
        )
        env.kinematics = SimpleNamespace(
            forward=lambda **_: {
                "torso_link": {
                    "pos": pelvis_to_torso_pos,
                    "quat": pelvis_to_torso_rot.as_quat(),
                }
            }
        )

        converted_pos, converted_quat = env._odometry_sensor_pose_to_root(
            world_to_sensor_pos,
            world_to_sensor_rot.as_quat(),
        )

        np.testing.assert_allclose(converted_pos, pelvis_pos, atol=1e-7)
        np.testing.assert_allclose(
            Rotation.from_quat(converted_quat).as_matrix(),
            pelvis_rot.as_matrix(),
            atol=1e-7,
        )

    def test_odometry_position_delta_produces_body_velocity_at_180_degree_heading(self):
        from scipy.spatial.transform import Rotation

        from robojudo.environment.agibot_cpp_env import AgiBotCppEnv

        heading = Rotation.from_euler("z", np.pi).as_quat()
        odometry = SimpleNamespace(
            sequence=1,
            stamp_sec=10,
            stamp_nanosec=0,
            position=[0.0, 0.0, 0.0],
            quaternion=heading,
        )
        env = AgiBotCppEnv.__new__(AgiBotCppEnv)
        env.cfg_env = SimpleNamespace(
            aimdk=SimpleNamespace(
                odometry_timeout=0.3,
                odometry_velocity_filter_time_constant=0.0,
            )
        )
        env._last_odometry_sequence = None
        env._last_odometry_stamp = None
        env._last_odometry_root_pos = None
        env._last_odometry_root_quat = None
        env._filtered_base_lin_vel = np.zeros(3, dtype=np.float32)
        env._last_odometry_receipt_time = None
        env._odometry_sensor_pose_to_root = lambda position, quaternion: (position, quaternion)

        env._update_odometry_state(odometry)
        odometry.sequence = 2
        odometry.stamp_nanosec = 100_000_000
        odometry.position = [-0.1, 0.0, 0.0]
        env._update_odometry_state(odometry)

        np.testing.assert_allclose(env._filtered_base_lin_vel, [1.0, 0.0, 0.0], atol=1e-6)

    def test_relative_start_odometry_removes_superodom_sensor_origin_without_changing_delta(self):
        from unittest.mock import patch

        from robojudo.environment.agibot_cpp_env import AgiBotCppEnv

        odometry = SimpleNamespace(
            sequence=1,
            stamp_sec=10,
            stamp_nanosec=0,
            position=[-0.35, 0.12, -0.61],
            quaternion=[0.0, 0.0, 0.0, 1.0],
        )
        env = AgiBotCppEnv.__new__(AgiBotCppEnv)
        env.cfg_env = SimpleNamespace(
            aimdk=SimpleNamespace(
                odometry_timeout=0.3,
                odometry_velocity_filter_time_constant=0.0,
                odometry_position_mode="RELATIVE_START",
            )
        )
        env._last_odometry_sequence = None
        env._last_odometry_stamp = None
        env._last_odometry_root_pos = None
        env._last_odometry_root_quat = None
        env._filtered_base_lin_vel = np.zeros(3, dtype=np.float32)
        env._last_odometry_receipt_time = None
        env._odometry_position_origin = None
        env._odometry_sensor_pose_to_root = lambda position, quaternion: (position, quaternion)

        with patch("robojudo.environment.agibot_cpp_env.time.monotonic", return_value=10.0):
            first_position, _ = env._update_odometry_state(odometry)
            odometry.sequence = 2
            odometry.stamp_nanosec = 100_000_000
            odometry.position = [-0.25, 0.12, -0.61]
            second_position, _ = env._update_odometry_state(odometry)

        np.testing.assert_allclose(first_position, np.zeros(3), atol=1e-6)
        np.testing.assert_allclose(second_position, [0.1, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(env._filtered_base_lin_vel, [1.0, 0.0, 0.0], atol=1e-6)

    def test_pipeline_composes_locomanipulation_loco_mimic_and_four_mode_behavior(self):
        from robojudo.pipeline.locomanipulation_loco_mimic_pipeline import (
            LocomanipulationLocoMimicPipelineMixin,
        )
        from robojudo.pipeline.rl_loco_mimic_pipeline import RlLocoMimicPipeline
        from robojudo.pipeline.x2_locomanipulation_loco_mimic_pipeline import (
            X2LocomanipulationLocoMimicPipeline,
        )
        from robojudo.pipeline.x2_locomanipulation_pipeline import X2FourModePipelineMixin

        self.assertTrue(issubclass(X2LocomanipulationLocoMimicPipeline, LocomanipulationLocoMimicPipelineMixin))
        self.assertTrue(issubclass(X2LocomanipulationLocoMimicPipeline, X2FourModePipelineMixin))
        self.assertTrue(issubclass(X2LocomanipulationLocoMimicPipeline, RlLocoMimicPipeline))

    def test_configs_register_expected_policies_and_controls(self):
        from robojudo.config.x2 import x2_locomimic, x2_locomimic_real
        from robojudo.config.x2.env.x2_env_cfg import X2_ARM_JOINT_NAMES
        from robojudo.config.x2.pipeline.x2_loco_mimic_pipeline_cfg import (
            X2LocomanipulationLocoMimicPipelineCfg,
        )

        sim_cfg = x2_locomimic()
        real_cfg = x2_locomimic_real()

        self.assertIsInstance(sim_cfg, X2LocomanipulationLocoMimicPipelineCfg)
        self.assertEqual(sim_cfg.pipeline_type, "X2LocomanipulationLocoMimicPipeline")
        self.assertEqual(sim_cfg.env.env_type, "MujocoEnv")
        self.assertEqual(sim_cfg.env.sim_dt, 0.005)
        self.assertEqual(sim_cfg.env.sim_decimation, 4)
        self.assertEqual(sim_cfg.loco_policy.policy_type, "X2LocomanipulationPolicy")
        self.assertEqual(
            [policy.policy_type for policy in sim_cfg.mimic_policies],
            ["X2DeployPolicy", "X2DeployPolicy"],
        )
        self.assertIsNot(sim_cfg.mimic_policies[0], sim_cfg.mimic_policies[1])
        for policy in sim_cfg.mimic_policies:
            self.assertIn("x2_rl_deploy", policy.policy_file)
            self.assertEqual(policy.max_timestep, 2820)
        self.assertEqual(len(real_cfg.mimic_policies), 2)
        self.assertTrue(all(policy.max_timestep == 2820 for policy in real_cfg.mimic_policies))
        self.assertEqual(sim_cfg.upper_dof_num, 16)
        self.assertEqual(len(sim_cfg.upper_dof_pos_default), 16)
        self.assertEqual(sim_cfg.upper_dof_override_indices, list(range(-16, -2)))
        self.assertTrue(sim_cfg.realign_on_policy_switch)
        self.assertEqual(sim_cfg.ctrl[-1].joint_names, X2_ARM_JOINT_NAMES)
        self.assertEqual(sim_cfg.ctrl[0].triggers["Back"], "[POLICY_LOCO]")
        self.assertEqual(sim_cfg.ctrl[0].triggers["Start"], "[POLICY_MIMIC]")
        self.assertEqual(sim_cfg.ctrl[0].triggers["RB"], "[POLICY_SWITCH],NEXT")
        self.assertEqual(sim_cfg.ctrl[0].triggers["LB"], "[POLICY_SWITCH],LAST")
        self.assertEqual(sim_cfg.ctrl[0].triggers["L"], "[UPPER_BODY_TOGGLE]")
        self.assertEqual(sim_cfg.ctrl[0].triggers["R"], "[POLICY_RECOVERY]")
        self.assertEqual(sim_cfg.ctrl[1].triggers_extra["r"], "[POLICY_RECOVERY]")
        self.assertEqual(sim_cfg.recovery_policy.policy_type, "AmpRecoveryPolicy")
        self.assertEqual(sim_cfg.recovery_policy.action_dof.num_dofs, 29)
        self.assertTrue(sim_cfg.do_safety_check)
        self.assertEqual(sim_cfg.ctrl[1].triggers_extra["["], "[POLICY_MIMIC]")
        self.assertEqual(real_cfg.env.env_type, "AgiBotCppEnv")
        self.assertTrue(real_cfg.do_safety_check)
        self.assertTrue(real_cfg.realign_on_policy_switch)
        self.assertEqual(real_cfg.ctrl[0].ctrl_type, "RosJoystickCtrl")
        self.assertEqual(real_cfg.ctrl[0].profile, "xbox_bluetooth")
        self.assertEqual(real_cfg.ctrl[0].topic, "/joy")
        self.assertEqual(real_cfg.ctrl[0].triggers["Back"], "[POLICY_LOCO]")
        self.assertEqual(real_cfg.ctrl[0].triggers["RB"], "[POLICY_SWITCH],NEXT")
        self.assertEqual(real_cfg.ctrl[0].triggers["LB"], "[POLICY_SWITCH],LAST")
        self.assertEqual(real_cfg.ctrl[0].triggers["R"], "[POLICY_RECOVERY]")
        self.assertNotIn("LB+RB+Y", real_cfg.ctrl[0].triggers)

    def test_policy_manager_switches_both_directions_and_can_cancel(self):
        from robojudo.config.x2.policy.x2_amp_recovery_policy_cfg import X2AmpRecoveryPolicyCfg
        from robojudo.config.x2.policy.x2_deploy_policy_cfg import X2DeployPolicyCfg
        from robojudo.config.x2.policy.x2_locomanipulation_policy_cfg import (
            X2LocomanipulationEnvDoF,
            X2LocomanipulationPolicyCfg,
        )
        from robojudo.pipeline.rl_loco_mimic_pipeline import PolicyInterpManager

        env = FakePolicyEnv(X2LocomanipulationEnvDoF())
        manager = PolicyInterpManager(
            X2LocomanipulationPolicyCfg(),
            [X2DeployPolicyCfg(), X2DeployPolicyCfg()],
            env,
            cfg_policy_recovery=X2AmpRecoveryPolicyCfg(),
            loco_dof_pos=env.default_pos,
        )
        manager.DURATIONS_LOCO_MIMIC = [0, 2, 1]
        manager.DURATIONS_MIMIC_LOCO = [1, 2, 1]
        env_data = Box(
            {
                "base_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                "base_ang_vel": np.zeros(3, dtype=np.float32),
                "dof_pos": env.default_pos.copy(),
                "dof_vel": np.zeros(31, dtype=np.float32),
            }
        )
        ctrl_data = Box({})

        self.assertEqual(manager.policy_mimic_idx, 0)
        self.assertEqual(manager.policy_mimic_ids, [1, 2])
        self.assertEqual(manager.policy_recovery_id, 3)
        manager.toggle_mimic_policy(1)
        self.assertEqual(manager.policy_mimic_idx, 1)
        manager.toggle_mimic_policy(1)
        self.assertEqual(manager.policy_mimic_idx, 0)
        manager.toggle_mimic_policy(-1)
        self.assertEqual(manager.policy_mimic_idx, 1)

        manager.switch_to_mimic()
        for _ in range(5):
            manager.step(env_data, ctrl_data)
        self.assertEqual(manager.current_policy_id, manager.policy_mimic_ids[1])
        self.assertEqual(manager.interp_state, PolicyInterpManager.InterpState.IDLE)
        manager.toggle_mimic_policy(1)
        self.assertEqual(manager.policy_mimic_idx, 1)
        self.assertEqual(env.reset_count, 0)
        self.assertFalse(env.elastic_band.active)
        self.assertEqual(env.elastic_band.rest_length, 0.7)

        manager.switch_to_loco()
        for _ in range(5):
            manager.step(env_data, ctrl_data)
        self.assertEqual(manager.current_policy_id, manager.policy_loco_id)
        self.assertEqual(manager.interp_state, PolicyInterpManager.InterpState.IDLE)
        self.assertEqual(env.reset_count, 0)
        self.assertFalse(env.elastic_band.active)
        self.assertEqual(env.elastic_band.rest_length, 0.7)

        manager.toggle_mimic_policy(-1)
        self.assertEqual(manager.policy_mimic_idx, 0)
        manager.switch_to_mimic()
        manager.step(env_data, ctrl_data)
        manager.toggle_mimic_policy(1)
        self.assertEqual(manager.policy_mimic_idx, 0)
        manager.reset_to_loco(refresh_env=False)
        self.assertEqual(manager.current_policy_id, manager.policy_loco_id)
        self.assertEqual(manager.interp_state, PolicyInterpManager.InterpState.IDLE)
        self.assertFalse(manager.timer.has_pending())
        np.testing.assert_array_equal(manager.override_dof_pos, manager.loco_dof_pos)

        self.assertTrue(manager.activate_recovery())
        self.assertEqual(manager.current_policy_id, manager.policy_recovery_id)
        self.assertIs(env.last_dof_cfg, manager.policy.cfg_action_dof)
        self.assertEqual(env.reset_count, 0)
        completed = []
        self.assertTrue(manager.switch_to_loco(callback_end=lambda: completed.append(True)))
        for _ in range(5):
            manager.step(env_data, ctrl_data)
        self.assertEqual(manager.current_policy_id, manager.policy_loco_id)
        self.assertEqual(completed, [True])

    def test_zmq_is_available_only_for_idle_loco_and_disables_on_mimic(self):
        from robojudo.pipeline.rl_loco_mimic_pipeline import PolicyInterpManager
        from robojudo.pipeline.x2_locomanipulation_loco_mimic_pipeline import (
            X2LocomanipulationLocoMimicPipeline,
        )

        pipeline = X2LocomanipulationLocoMimicPipeline.__new__(X2LocomanipulationLocoMimicPipeline)
        manager = SimpleNamespace(
            current_policy_id=0,
            policy_loco_id=0,
            interp_state=PolicyInterpManager.InterpState.IDLE,
            switch_to_mimic=lambda: (setattr(manager, "mimic_requested", True), True)[1],
        )
        pipeline.policy_manager = manager
        pipeline.policy_locomotion_mimic_flag = 0
        pipeline._upper_body_cfg = object()
        pipeline._upper_body_enabled = True
        pipeline._upper_body_stream_was_fresh = True

        self.assertTrue(pipeline._upper_body_control_available())
        manager.interp_state = PolicyInterpManager.InterpState.IN_PROGRESS
        self.assertFalse(pipeline._upper_body_control_available())
        manager.interp_state = PolicyInterpManager.InterpState.IDLE

        pipeline._process_policy_commands(["[POLICY_MIMIC]"], {})
        self.assertFalse(pipeline._upper_body_enabled)
        self.assertTrue(manager.mimic_requested)
        self.assertEqual(pipeline.policy_locomotion_mimic_flag, 1)

    def test_zmq_filter_syncs_to_loco_target_after_mimic(self):
        from robojudo.pipeline.rl_loco_mimic_pipeline import PolicyInterpManager
        from robojudo.pipeline.x2_locomanipulation_loco_mimic_pipeline import (
            X2LocomanipulationLocoMimicPipeline,
        )

        pipeline = X2LocomanipulationLocoMimicPipeline.__new__(X2LocomanipulationLocoMimicPipeline)
        manager = SimpleNamespace(
            current_policy_id=1,
            policy_loco_id=0,
            interp_state=PolicyInterpManager.InterpState.IN_PROGRESS,
        )
        pipeline.policy_manager = manager
        pipeline._upper_body_cfg = SimpleNamespace(ema_alpha=0.9, joint_names=["left_arm", "right_arm"])
        pipeline._upper_body_indices = np.asarray([1, 3], dtype=np.int32)
        pipeline._upper_body_default = np.asarray([0.2, -0.2], dtype=np.float32)
        pipeline._upper_body_filtered = np.asarray([1.5, -1.5], dtype=np.float32)
        pipeline._upper_body_enabled = False
        pipeline._upper_body_stream_was_fresh = True
        pipeline._upper_body_override_was_available = True
        pipeline.env = SimpleNamespace(position_limits=np.asarray([[-2.0, 2.0]] * 4, dtype=np.float32))
        loco_target = np.asarray([0.0, 0.2, 0.0, -0.2], dtype=np.float32)
        pipeline._get_policy_step = lambda env_data, ctrl_data: (loco_target.copy(), {})
        ctrl_data = Box(
            {
                "UpperBodyZmqCtrl": {
                    "fresh": True,
                    "joint_positions": {"left_arm": 1.0, "right_arm": -1.0},
                }
            }
        )

        mimic_target, _ = pipeline._step_rl_policy(Box({}), ctrl_data, dry_run=True)
        np.testing.assert_array_equal(mimic_target, loco_target)
        np.testing.assert_array_equal(pipeline._upper_body_filtered, [1.5, -1.5])
        self.assertFalse(pipeline._upper_body_override_was_available)

        manager.current_policy_id = manager.policy_loco_id
        manager.interp_state = PolicyInterpManager.InterpState.IDLE
        resumed_target, _ = pipeline._step_rl_policy(Box({}), ctrl_data, dry_run=True)

        np.testing.assert_array_equal(resumed_target, loco_target)
        np.testing.assert_array_equal(pipeline._upper_body_filtered, pipeline._upper_body_default)
        self.assertFalse(pipeline._upper_body_enabled)
        self.assertFalse(pipeline._upper_body_stream_was_fresh)
        self.assertTrue(pipeline._upper_body_override_was_available)

    def test_disabling_zmq_during_loco_keeps_smooth_return_to_default(self):
        from robojudo.pipeline.rl_loco_mimic_pipeline import PolicyInterpManager
        from robojudo.pipeline.x2_locomanipulation_loco_mimic_pipeline import (
            X2LocomanipulationLocoMimicPipeline,
        )

        pipeline = X2LocomanipulationLocoMimicPipeline.__new__(X2LocomanipulationLocoMimicPipeline)
        pipeline.policy_manager = SimpleNamespace(
            current_policy_id=0,
            policy_loco_id=0,
            interp_state=PolicyInterpManager.InterpState.IDLE,
        )
        pipeline._upper_body_cfg = SimpleNamespace(ema_alpha=0.5, joint_names=["left_arm"])
        pipeline._upper_body_indices = np.asarray([1], dtype=np.int32)
        pipeline._upper_body_default = np.asarray([0.0], dtype=np.float32)
        pipeline._upper_body_filtered = np.asarray([1.0], dtype=np.float32)
        pipeline._upper_body_enabled = False
        pipeline._upper_body_stream_was_fresh = False
        pipeline._upper_body_override_was_available = True
        pipeline.env = SimpleNamespace(position_limits=np.asarray([[-2.0, 2.0]] * 2, dtype=np.float32))
        loco_target = np.asarray([0.0, 0.0], dtype=np.float32)
        pipeline._get_policy_step = lambda env_data, ctrl_data: (loco_target.copy(), {})

        pd_target, _ = pipeline._step_rl_policy(Box({}), Box({}), dry_run=True)

        self.assertEqual(pd_target[1], 0.5)
        self.assertEqual(pipeline._upper_body_filtered[0], 0.5)
        self.assertTrue(pipeline._upper_body_override_was_available)

    def test_leaving_rl_resets_loco_and_cancels_policy_state(self):
        from robojudo.pipeline.x2_locomanipulation_loco_mimic_pipeline import (
            X2LocomanipulationLocoMimicPipeline,
        )

        pipeline = X2LocomanipulationLocoMimicPipeline.__new__(X2LocomanipulationLocoMimicPipeline)
        progress = SimpleNamespace(close_calls=0)
        manager = SimpleNamespace(
            reset_to_loco_calls=[],
            policy=SimpleNamespace(close_progress=lambda: setattr(progress, "close_calls", progress.close_calls + 1)),
        )
        manager.reset_to_loco = lambda refresh_env: manager.reset_to_loco_calls.append(refresh_env)
        pipeline.policy_manager = manager
        pipeline.policy_locomotion_mimic_flag = 1
        pipeline._upper_body_cfg = object()
        pipeline._upper_body_enabled = True
        pipeline._upper_body_stream_was_fresh = True

        pipeline._on_leave_rl()

        self.assertFalse(pipeline._upper_body_enabled)
        self.assertEqual(progress.close_calls, 1)
        self.assertEqual(manager.reset_to_loco_calls, [True])
        self.assertEqual(pipeline.policy_locomotion_mimic_flag, 0)
        self.assertFalse(pipeline._upper_body_override_was_available)

    def test_policy_time_and_switch_timer_advance_only_in_rl(self):
        from robojudo.pipeline.x2_locomanipulation_loco_mimic_pipeline import (
            X2LocomanipulationLocoMimicPipeline,
        )

        pipeline = X2LocomanipulationLocoMimicPipeline.__new__(X2LocomanipulationLocoMimicPipeline)
        calls = SimpleNamespace(policy_post=0, manager_step=0, ctrl_post=0)
        pipeline.timestep = 0
        pipeline.cfg = SimpleNamespace(debug=SimpleNamespace(log_obs=False))
        pipeline.visualizer = None
        pipeline.policy_manager = SimpleNamespace(
            policy=SimpleNamespace(
                post_step_callback=lambda commands: setattr(calls, "policy_post", calls.policy_post + 1)
            ),
            warmup_policy_indices=set(),
            step=lambda env_data, ctrl_data: setattr(calls, "manager_step", calls.manager_step + 1),
        )
        pipeline.ctrl_manager = SimpleNamespace(
            post_step_callback=lambda ctrl_data: setattr(calls, "ctrl_post", calls.ctrl_post + 1)
        )
        env_data = Box({})
        ctrl_data = Box({"COMMANDS": []})

        pipeline._post_mode_step(env_data, ctrl_data, {}, np.zeros(31), rl_active=False)
        self.assertEqual((calls.policy_post, calls.manager_step, calls.ctrl_post), (0, 0, 1))

        pipeline.policy_manager.warmup_policy_indices = {0, 1}
        pipeline._post_mode_step(env_data, ctrl_data, {}, np.zeros(31), rl_active=False)
        self.assertEqual((calls.policy_post, calls.manager_step, calls.ctrl_post), (0, 1, 2))

        pipeline.policy_manager.warmup_policy_indices.clear()
        pipeline._post_mode_step(env_data, ctrl_data, {}, np.zeros(31), rl_active=True)
        self.assertEqual((calls.policy_post, calls.manager_step, calls.ctrl_post), (1, 2, 3))

    def test_motion_done_callback_starts_return_to_loco(self):
        from robojudo.pipeline.x2_locomanipulation_loco_mimic_pipeline import (
            X2LocomanipulationLocoMimicPipeline,
        )

        pipeline = X2LocomanipulationLocoMimicPipeline.__new__(X2LocomanipulationLocoMimicPipeline)
        manager = SimpleNamespace(return_requested=False)
        manager.switch_to_loco = lambda: (setattr(manager, "return_requested", True), True)[1]
        pipeline.policy_manager = manager
        pipeline.policy_locomotion_mimic_flag = 1

        commands = []
        pipeline._process_policy_commands(commands, {"CALLBACK": ["[MOTION_DONE]"]})

        self.assertEqual(commands, ["[POLICY_LOCO]"])
        self.assertTrue(manager.return_requested)
        self.assertEqual(pipeline.policy_locomotion_mimic_flag, 0)


if __name__ == "__main__":
    unittest.main()
