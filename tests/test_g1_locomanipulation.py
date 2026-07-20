import hashlib
import unittest
from types import SimpleNamespace

import mujoco
import numpy as np
from box import Box


class TestG1Locomanipulation(unittest.TestCase):
    def test_configs_pair_models_with_recorded_presets(self):
        from robojudo.config.g1.g1_cfg import (
            g1_23_locomanipulation_default,
            g1_23_locomanipulation_stiff,
            g1_29_locomanipulation_stiff,
        )

        cases = [
            (g1_23_locomanipulation_default(), 23, 13, 360, "default", "policy_23dof_default"),
            (g1_23_locomanipulation_stiff(), 23, 13, 360, "stiff", "policy_23dof_stiff"),
            (g1_29_locomanipulation_stiff(), 29, 15, 430, "stiff", "policy_29dof_stiff"),
        ]
        for cfg, obs_dofs, action_dofs, num_obs, preset, policy_name in cases:
            with self.subTest(policy_name=policy_name):
                self.assertEqual(cfg.pipeline_type, "G1LocomanipulationPipeline")
                self.assertEqual(cfg.env.sim_dt, 0.005)
                self.assertEqual(cfg.env.sim_decimation, 4)
                self.assertEqual(cfg.policy.policy_type, "G1LocomanipulationPolicy")
                self.assertEqual(cfg.policy.policy_name, policy_name)
                self.assertEqual(cfg.policy.pd_gain_preset, preset)
                self.assertEqual(cfg.policy.obs_dof.num_dofs, obs_dofs)
                self.assertEqual(cfg.policy.action_dof.num_dofs, action_dofs)
                self.assertEqual(cfg.policy.num_obs, num_obs)
                self.assertEqual(cfg.env.dof.joint_names, cfg.policy.obs_dof.joint_names)
                self.assertEqual(cfg.env.dof.stiffness, cfg.policy.obs_dof.stiffness)
                self.assertEqual(cfg.ctrl[0].triggers["Start"], "[UPPER_BODY_TOGGLE]")
                self.assertEqual(cfg.ctrl[1].triggers["t"], "[UPPER_BODY_TOGGLE]")
                self.assertEqual(
                    {key: cfg.ctrl[1].triggers[key] for key in ("7", "8", "9")},
                    {
                        "7": "[ELASTIC_BAND_LOWER]",
                        "8": "[ELASTIC_BAND_LIFT]",
                        "9": "[ELASTIC_BAND_TOGGLE]",
                    },
                )
                self.assertEqual(cfg.env.elastic_band.body_name, "torso_link")
                self.assertEqual(cfg.env.elastic_band.anchor_point, (0.0, 0.0, 3.0))
                self.assertEqual(cfg.env.elastic_band.stiffness, 200.0)
                self.assertEqual(cfg.env.elastic_band.damping, 100.0)
                self.assertEqual(cfg.env.elastic_band.rest_length, 0.0)
                self.assertTrue(cfg.env.elastic_band.active)
                self.assertEqual(
                    {key: cfg.ctrl[0].triggers[key] for key in ("A", "B", "Y", "X")},
                    {
                        "A": "[PASSIVE_DEFAULT]",
                        "B": "[DAMPING_DEFAULT]",
                        "Y": "[JOINT_DEFAULT]",
                        "X": "[RL_DEFAULT]",
                    },
                )
                self.assertEqual(cfg.joint_default_duration, 1.5)
                self.assertEqual(cfg.default_damping, 5.0)
                self.assertEqual(cfg.joint_default_dof.joint_names, cfg.env.dof.joint_names)
                self.assertEqual(cfg.joint_default_dof.default_pos, cfg.policy.obs_dof.default_pos)
                self.assertEqual(cfg.joint_default_dof.stiffness, cfg.policy.obs_dof.stiffness)
                self.assertEqual(cfg.joint_default_dof.damping, cfg.policy.obs_dof.damping)
                self.assertEqual(round(cfg.joint_default_duration * cfg.policy.freq), 75)
                self.assertEqual(
                    cfg.ctrl[-1].joint_names,
                    cfg.policy.obs_dof.joint_names[action_dofs:],
                )

    def test_real_configs_use_native_unitree_layouts_and_safety_controls(self):
        from robojudo.config.g1.g1_cfg import (
            g1_23_locomanipulation_default_real,
            g1_23_locomanipulation_stiff_real,
            g1_29_locomanipulation_stiff_real,
        )

        from robojudo.config.g1.env.g1_env_cfg import G1_23_DOF_INDICES

        motor_mapping_23 = [*range(13), *range(15, 20), *range(22, 27)]
        self.assertEqual(G1_23_DOF_INDICES, motor_mapping_23)
        cases = [
            (g1_23_locomanipulation_default_real(), 23, 29, motor_mapping_23),
            (g1_23_locomanipulation_stiff_real(), 23, 29, motor_mapping_23),
            (g1_29_locomanipulation_stiff_real(), 29, None, None),
        ]
        for cfg, num_dofs, motor_dof_count, joint2motor_idx in cases:
            with self.subTest(config=type(cfg).__name__):
                self.assertEqual(cfg.env.env_type, "UnitreeCppEnv")
                self.assertEqual(cfg.env.dof.num_dofs, num_dofs)
                self.assertEqual(cfg.env.motor_dof_count, motor_dof_count)
                self.assertEqual(cfg.env.joint2motor_idx, joint2motor_idx)
                self.assertEqual(cfg.env.forward_kinematic.kinematic_joint_names, cfg.env.dof.joint_names)
                self.assertEqual(cfg.env.unitree.command_timeout, 0.1)
                self.assertEqual(cfg.env.unitree.state_timeout, 0.1)
                self.assertEqual(cfg.env.unitree.shutdown_damping, 5.0)
                self.assertTrue(cfg.do_safety_check)
                self.assertEqual([ctrl.ctrl_type for ctrl in cfg.ctrl], ["UnitreeCtrl", "UpperBodyZmqCtrl"])
                self.assertEqual(cfg.ctrl[0].combination_init_buttons, ["L1", "R1"])
                self.assertEqual(
                    cfg.ctrl[0].triggers,
                    {
                        "A": "[PASSIVE_DEFAULT]",
                        "B": "[DAMPING_DEFAULT]",
                        "Y": "[JOINT_DEFAULT]",
                        "X": "[RL_DEFAULT]",
                        "Start": "[UPPER_BODY_TOGGLE]",
                        "L1+R1+A": "[SHUTDOWN]",
                    },
                )

    def test_unitree_cpp_mode_adapter_validates_and_clamps_commands(self):
        from robojudo.config.g1.g1_cfg import g1_23_locomanipulation_stiff_real
        from robojudo.environment.unitree_cpp_env import UnitreeCppEnv

        class FakeUnitree:
            def __init__(self):
                self.calls = []

            def set_passive(self):
                self.calls.append(("passive",))

            def set_damping(self, damping):
                self.calls.append(("damping", damping))

            def arm_position_control(self):
                self.calls.append(("arm",))

            def step(self, target):
                self.calls.append(("step", target))

        cfg = g1_23_locomanipulation_stiff_real().env
        env = UnitreeCppEnv.__new__(UnitreeCppEnv)
        env.cfg_env = cfg
        env.enabled = True
        env.unitree = FakeUnitree()
        env.num_dofs = cfg.dof.num_dofs
        env._motor_dof_count = cfg.motor_dof_count
        env._dof_idx = np.asarray(cfg.joint2motor_idx, dtype=np.int32)
        env.joint_names = cfg.dof.joint_names
        env.position_limits = np.asarray(cfg.dof.position_limits)
        env._control_joint_names = set(cfg.dof.joint_names)
        env._last_clamp_log_time = 0.0
        env.hand_retarget = None

        env.set_control_joint_names(env.joint_names)
        env.command_passive()
        env.command_damping(6.0)
        env.arm_position_control()
        target = np.asarray(cfg.dof.default_pos, dtype=np.float64)
        target[0] = env.position_limits[0, 1] + 1.0
        env.step(target)

        self.assertEqual(env.unitree.calls[:3], [("passive",), ("damping", 6.0), ("arm",)])
        self.assertEqual(env.unitree.calls[-1][0], "step")
        motor_target = np.asarray(env.unitree.calls[-1][1])
        self.assertEqual(motor_target.shape, (29,))
        self.assertEqual(motor_target[0], env.position_limits[0, 1])
        np.testing.assert_array_equal(motor_target[[13, 14, 20, 21, 27, 28]], 0.0)
        with self.assertRaisesRegex(ValueError, "complete environment joint layout"):
            env.set_control_joint_names(env.joint_names[:-1])
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            env.step(np.full(env.num_dofs, np.nan))
        self.assertEqual(env.unitree.calls[-1], ("damping", 5.0))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            env.command_damping(-1.0)

    def test_unitree_cpp_23_dof_transport_expands_gains_and_selects_feedback(self):
        from robojudo.config.g1.g1_cfg import g1_23_locomanipulation_stiff_real
        from robojudo.environment.unitree_cpp_env import UnitreeCppEnv

        class FakeUnitree:
            def __init__(self):
                self.gains = None

            def set_gains(self, stiffness, damping):
                self.gains = np.asarray(stiffness), np.asarray(damping)

        cfg = g1_23_locomanipulation_stiff_real().env
        env = UnitreeCppEnv.__new__(UnitreeCppEnv)
        env.cfg_env = cfg
        env.enabled = True
        env.unitree = FakeUnitree()
        env.num_dofs = 23
        env._motor_dof_count = 29
        env._dof_idx = np.asarray(cfg.joint2motor_idx, dtype=np.int32)

        stiffness = np.arange(1, 24, dtype=np.float64)
        damping = stiffness / 10.0
        env.set_gains(stiffness, damping)

        motor_stiffness, motor_damping = env.unitree.gains
        np.testing.assert_array_equal(motor_stiffness[env._dof_idx], stiffness)
        np.testing.assert_array_equal(motor_damping[env._dof_idx], damping)
        np.testing.assert_array_equal(motor_stiffness[[13, 14, 20, 21, 27, 28]], 0.0)
        np.testing.assert_array_equal(motor_damping[[13, 14, 20, 21, 27, 28]], 0.0)

        motor_feedback = np.arange(29, dtype=np.float32)
        np.testing.assert_array_equal(
            env._motor_to_logical(motor_feedback, dtype=np.float32),
            motor_feedback[env._dof_idx],
        )

    def test_unitree_transport_mapping_validation_rejects_duplicate_and_out_of_range_slots(self):
        from robojudo.config.g1.env.g1_real_env_cfg import G1_23RealEnvCfg

        duplicate_mapping = [*range(13), *range(15, 20), *range(22, 26), 25]
        with self.assertRaisesRegex(ValueError, "entries must be unique"):
            G1_23RealEnvCfg(joint2motor_idx=duplicate_mapping)

        out_of_range_mapping = [*range(13), *range(15, 20), *range(22, 26), 29]
        with self.assertRaisesRegex(ValueError, "within motor_dof_count"):
            G1_23RealEnvCfg(joint2motor_idx=out_of_range_mapping)

    def test_unitree_cpp_stale_state_enters_damping(self):
        from robojudo.config.g1.g1_cfg import g1_29_locomanipulation_stiff_real
        from robojudo.environment.unitree_cpp_env import UnitreeCppEnv

        class FakeUnitree:
            def __init__(self):
                self.damping = []

            def state_is_fresh(self, timeout):
                self.timeout = timeout
                return False

            def set_damping(self, damping):
                self.damping.append(damping)

        cfg = g1_29_locomanipulation_stiff_real().env
        env = UnitreeCppEnv.__new__(UnitreeCppEnv)
        env.cfg_env = cfg
        env.enabled = True
        env.unitree = FakeUnitree()

        with self.assertRaisesRegex(RuntimeError, "low state became stale"):
            env.update()
        self.assertEqual(env.unitree.timeout, 0.1)
        self.assertEqual(env.unitree.damping, [5.0])

    def test_elastic_band_attaches_to_both_g1_models_and_resets(self):
        from robojudo.config.g1.g1_cfg import (
            g1_23_locomanipulation_default,
            g1_29_locomanipulation_stiff,
        )
        from robojudo.environment.utils.elastic_band import ElasticBand

        for cfg in (g1_23_locomanipulation_default(), g1_29_locomanipulation_stiff()):
            with self.subTest(num_dofs=cfg.env.dof.num_dofs):
                model = mujoco.MjModel.from_xml_path(cfg.env.xml)
                data = mujoco.MjData(model)
                if model.nkey:
                    mujoco.mj_resetDataKeyframe(model, data, 0)
                else:
                    mujoco.mj_resetData(model, data)
                mujoco.mj_forward(model, data)
                band = ElasticBand(cfg.env.elastic_band, model, data)

                force = band.apply()
                self.assertTrue(np.isfinite(force).all())
                self.assertGreater(np.linalg.norm(force), 0.0)
                self.assertEqual(band.lower(), 0.1)
                self.assertEqual(band.lift(), 0.0)
                self.assertFalse(band.toggle())

                band.reset()
                self.assertTrue(band.active)
                self.assertEqual(band.rest_length, 0.0)

    def test_g1_pipeline_routes_elastic_band_commands(self):
        from robojudo.pipeline.g1_locomanipulation_pipeline import G1LocomanipulationPipeline

        class FakeEnv:
            def __init__(self):
                self.calls = []

            def lower_elastic_band(self):
                self.calls.append("lower")

            def lift_elastic_band(self):
                self.calls.append("lift")

            def toggle_elastic_band(self):
                self.calls.append("toggle")

        pipeline = G1LocomanipulationPipeline.__new__(G1LocomanipulationPipeline)
        pipeline.env = FakeEnv()
        pipeline._process_commands(
            ["[ELASTIC_BAND_LOWER]", "[ELASTIC_BAND_LIFT]", "[ELASTIC_BAND_TOGGLE]"]
        )
        self.assertEqual(pipeline.env.calls, ["lower", "lift", "toggle"])

    def test_model_assets_match_supplied_exports(self):
        from robojudo.config import ASSETS_DIR

        expected = {
            "policy_23dof_default.onnx": "a1763bf76b0d52ce07ab5e0d02da1618bea6531398a1bc895fe854ebea331598",
            "policy_23dof_stiff.onnx": "1ae49337e134e74d78a10020769751a2fd4d7a138d002a63c04c7978381c7d08",
            "policy_29dof_stiff.onnx": "3cd13ffd6443123a6e81ebe1439f02143b7bbe20cdbdb24880f27595dcde5994",
        }
        for filename, digest in expected.items():
            with self.subTest(filename=filename):
                model = ASSETS_DIR / "models/g1/locomanipulation" / filename
                self.assertEqual(hashlib.sha256(model.read_bytes()).hexdigest(), digest)

    def test_all_onnx_contracts_and_inference_are_finite(self):
        from robojudo.config.g1.policy.g1_locomanipulation_policy_cfg import (
            G1Locomanipulation23PolicyCfg,
            G1Locomanipulation29PolicyCfg,
        )
        from robojudo.policy.g1_locomanipulation_policy import G1LocomanipulationPolicy

        configs = [
            G1Locomanipulation23PolicyCfg(
                policy_name="policy_23dof_default",
                pd_gain_preset="default",
            ),
            G1Locomanipulation23PolicyCfg(),
            G1Locomanipulation29PolicyCfg(),
        ]
        for cfg in configs:
            with self.subTest(policy_name=cfg.policy_name):
                policy = G1LocomanipulationPolicy(cfg, "cpu")
                action = policy.get_action(np.zeros(cfg.num_obs, dtype=np.float32))
                self.assertEqual(action.shape, (cfg.action_dof.num_dofs,))
                self.assertTrue(np.isfinite(action).all())

    def test_model_rejects_wrong_pd_and_action_scale_preset(self):
        from robojudo.config.g1.policy.g1_locomanipulation_policy_cfg import (
            G1Locomanipulation23PolicyCfg,
        )
        from robojudo.policy.g1_locomanipulation_policy import G1LocomanipulationPolicy

        cfg = G1Locomanipulation23PolicyCfg(
            policy_name="policy_23dof_default",
            pd_gain_preset="stiff",
        )
        with self.assertRaisesRegex(ValueError, "metadata does not match"):
            G1LocomanipulationPolicy(cfg, "cpu")

    def test_observation_uses_five_samples_per_term(self):
        from robojudo.config.g1.policy.g1_locomanipulation_policy_cfg import (
            G1Locomanipulation23PolicyCfg,
            G1Locomanipulation29PolicyCfg,
        )
        from robojudo.policy.g1_locomanipulation_policy import G1LocomanipulationPolicy

        for cfg in (G1Locomanipulation23PolicyCfg(), G1Locomanipulation29PolicyCfg()):
            with self.subTest(policy_name=cfg.policy_name):
                policy = G1LocomanipulationPolicy(cfg, "cpu")
                env_data = Box(
                    {
                        "base_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                        "base_ang_vel": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
                        "dof_pos": np.asarray(cfg.obs_dof.default_pos, dtype=np.float32),
                        "dof_vel": np.zeros(cfg.obs_dof.num_dofs, dtype=np.float32),
                    }
                )
                obs, _ = policy.get_observation(env_data, Box({}))
                self.assertEqual(obs.shape, (cfg.num_obs,))
                np.testing.assert_array_equal(obs[:15], np.tile(env_data.base_ang_vel, 5))

    def test_upper_body_override_changes_only_non_action_joints(self):
        from robojudo.config.g1.g1_cfg import g1_23_locomanipulation_stiff
        from robojudo.pipeline.g1_locomanipulation_pipeline import G1LocomanipulationPipeline

        cfg = g1_23_locomanipulation_stiff()
        pipeline = G1LocomanipulationPipeline.__new__(G1LocomanipulationPipeline)
        pipeline._upper_body_cfg = cfg.ctrl[-1].model_copy(update={"ema_alpha": 0.0})
        pipeline._upper_body_enabled = True
        pipeline._upper_body_stream_was_fresh = False
        pipeline._upper_body_indices = np.asarray(
            [cfg.env.dof.joint_names.index(name) for name in pipeline._upper_body_cfg.joint_names]
        )
        pipeline._upper_body_default = np.asarray(cfg.env.dof.default_pos)[pipeline._upper_body_indices]
        pipeline._upper_body_filtered = pipeline._upper_body_default.copy()
        pipeline.env = SimpleNamespace(position_limits=np.asarray(cfg.env.dof.position_limits))
        target = np.asarray(cfg.env.dof.default_pos, dtype=np.float32)
        left_shoulder = cfg.env.dof.joint_names.index("left_shoulder_pitch_joint")

        result = pipeline._apply_pd_target_override(
            target,
            Box(
                {
                    "UpperBodyZmqCtrl": {
                        "fresh": True,
                        "joint_positions": {"left_shoulder_pitch_joint": 0.8},
                    }
                }
            ),
        )

        self.assertEqual(result[left_shoulder], 0.8)
        np.testing.assert_array_equal(result[:13], target[:13])

    def test_four_mode_sequence_and_upper_body_synchronization(self):
        from robojudo.pipeline.g1_locomanipulation_pipeline import G1ControlMode, G1LocomanipulationPipeline

        class FakeEnv:
            def __init__(self):
                self.joint_names = ["joint_a", "joint_b"]
                self.dof_pos = np.asarray([0.0, 0.4], dtype=np.float32)
                self.elastic_band = SimpleNamespace(active=False, rest_length=0.7)
                self.control_joint_names = None
                self.stiffness = None
                self.damping = None
                self.targets = []
                self.position_armed = False

            def set_control_joint_names(self, names):
                self.control_joint_names = list(names)

            def set_gains(self, stiffness, damping):
                self.stiffness = np.asarray(stiffness)
                self.damping = np.asarray(damping)

            def arm_position_control(self):
                self.position_armed = True

            def step(self, target):
                self.targets.append(np.asarray(target))

        class FakePolicy:
            def __init__(self):
                self.reset_count = 0

            def reset(self):
                self.reset_count += 1

        pipeline = G1LocomanipulationPipeline.__new__(G1LocomanipulationPipeline)
        pipeline._mode_robot_name = "G1"
        pipeline.mode = G1ControlMode.PASSIVE_DEFAULT
        pipeline.env = FakeEnv()
        pipeline.policy = FakePolicy()
        pipeline._joint_default_start = None
        pipeline._joint_default_step = 0
        pipeline._joint_default_steps = 3
        pipeline._joint_default_complete = False
        pipeline._joint_default_target = np.asarray([0.6, -0.3], dtype=np.float32)
        pipeline._joint_default_stiffness = np.asarray([40.0, 50.0], dtype=np.float32)
        pipeline._joint_default_damping = np.asarray([4.0, 5.0], dtype=np.float32)
        pipeline._rl_stiffness = np.asarray([100.0, 120.0], dtype=np.float32)
        pipeline._rl_damping = np.asarray([2.5, 3.0], dtype=np.float32)
        pipeline._upper_body_cfg = object()
        pipeline._upper_body_enabled = False
        pipeline._upper_body_stream_was_fresh = True
        pipeline._upper_body_indices = np.asarray([1], dtype=np.int32)
        pipeline._upper_body_filtered = np.asarray([-1.0], dtype=np.float32)

        self.assertFalse(pipeline._enter_mode(G1ControlMode.RL_DEFAULT))
        self.assertTrue(pipeline._enter_mode(G1ControlMode.JOINT_DEFAULT))
        self.assertEqual(pipeline.env.control_joint_names, pipeline.env.joint_names)
        self.assertTrue(pipeline.env.position_armed)
        for _ in range(3):
            pipeline._step_joint_default()
        np.testing.assert_allclose(pipeline.env.targets[-1], pipeline._joint_default_target)
        self.assertTrue(pipeline._joint_default_complete)

        self.assertTrue(pipeline._enter_mode(G1ControlMode.RL_DEFAULT))
        self.assertEqual(pipeline.policy.reset_count, 1)
        np.testing.assert_array_equal(pipeline.env.stiffness, pipeline._rl_stiffness)
        np.testing.assert_array_equal(pipeline.env.damping, pipeline._rl_damping)
        np.testing.assert_array_equal(pipeline._upper_body_filtered, np.asarray([0.4], dtype=np.float32))
        self.assertFalse(pipeline._upper_body_stream_was_fresh)

        pipeline._upper_body_enabled = True
        pipeline._enter_mode(G1ControlMode.DAMPING_DEFAULT)
        self.assertFalse(pipeline._upper_body_enabled)
        self.assertFalse(pipeline._joint_default_complete)
        self.assertFalse(pipeline.env.elastic_band.active)
        self.assertEqual(pipeline.env.elastic_band.rest_length, 0.7)

    def test_upper_body_toggle_requires_g1_rl_mode(self):
        from robojudo.config.g1.g1_cfg import g1_23_locomanipulation_stiff
        from robojudo.pipeline.g1_locomanipulation_pipeline import G1ControlMode, G1LocomanipulationPipeline

        pipeline = G1LocomanipulationPipeline.__new__(G1LocomanipulationPipeline)
        pipeline._upper_body_cfg = g1_23_locomanipulation_stiff().ctrl[-1]
        pipeline._upper_body_enabled = False
        pipeline._upper_body_stream_was_fresh = False
        pipeline.mode = G1ControlMode.JOINT_DEFAULT
        pipeline._toggle_upper_body()
        self.assertFalse(pipeline._upper_body_enabled)

        pipeline.mode = G1ControlMode.RL_DEFAULT
        pipeline._toggle_upper_body()
        self.assertTrue(pipeline._upper_body_enabled)
        pipeline._toggle_upper_body()
        self.assertFalse(pipeline._upper_body_enabled)

    def test_reborn_and_shutdown_return_to_safe_modes(self):
        from robojudo.pipeline.g1_locomanipulation_pipeline import G1ControlMode, G1LocomanipulationPipeline

        class FakeEnv:
            def __init__(self):
                self.reborn_count = 0
                self.damping_commands = []
                self.shutdown_count = 0

            def reborn(self):
                self.reborn_count += 1

            def command_damping(self, damping):
                self.damping_commands.append(damping)

            def shutdown(self):
                self.shutdown_count += 1

        class FakePolicy:
            def __init__(self):
                self.reset_count = 0

            def reset(self):
                self.reset_count += 1

        pipeline = G1LocomanipulationPipeline.__new__(G1LocomanipulationPipeline)
        pipeline._mode_robot_name = "G1"
        pipeline.mode = G1ControlMode.RL_DEFAULT
        pipeline.env = FakeEnv()
        pipeline.policy = FakePolicy()
        pipeline._default_damping = 5.0
        pipeline._joint_default_complete = True
        pipeline._joint_default_start = np.asarray([0.0], dtype=np.float32)
        pipeline._upper_body_cfg = object()
        pipeline._upper_body_enabled = True
        pipeline._upper_body_stream_was_fresh = True
        pipeline._shutdown_requested = False
        pipeline.should_exit = False

        pipeline._process_commands(["[SIM_REBORN]"])
        self.assertEqual(pipeline.env.reborn_count, 1)
        self.assertEqual(pipeline.policy.reset_count, 1)
        self.assertEqual(pipeline.mode, G1ControlMode.PASSIVE_DEFAULT)
        self.assertFalse(pipeline._upper_body_enabled)
        self.assertFalse(pipeline._joint_default_complete)

        pipeline._process_commands(["[SHUTDOWN]"])
        self.assertEqual(pipeline.mode, G1ControlMode.DAMPING_DEFAULT)
        self.assertEqual(pipeline.env.damping_commands, [5.0])
        self.assertEqual(pipeline.env.shutdown_count, 1)
        self.assertTrue(pipeline.should_exit)

    def test_x2_and_g1_use_shared_policy_runtime(self):
        from robojudo.policy.g1_locomanipulation_policy import G1LocomanipulationPolicy
        from robojudo.policy.locomanipulation_policy import LocomanipulationPolicyBase
        from robojudo.policy.x2_locomanipulation_policy import X2LocomanipulationPolicy

        self.assertTrue(issubclass(G1LocomanipulationPolicy, LocomanipulationPolicyBase))
        self.assertTrue(issubclass(X2LocomanipulationPolicy, LocomanipulationPolicyBase))

    def test_x2_and_g1_use_shared_control_modes(self):
        from robojudo.pipeline.four_mode_pipeline import ControlMode, FourModePipelineMixin
        from robojudo.pipeline.g1_locomanipulation_pipeline import G1ControlMode, G1ModePipelineMixin
        from robojudo.pipeline.x2_deploy_pipeline import X2ControlMode, X2ModePipelineMixin

        self.assertIs(G1ControlMode, ControlMode)
        self.assertIs(X2ControlMode, ControlMode)
        self.assertTrue(issubclass(G1ModePipelineMixin, FourModePipelineMixin))
        self.assertTrue(issubclass(X2ModePipelineMixin, FourModePipelineMixin))


if __name__ == "__main__":
    unittest.main()
