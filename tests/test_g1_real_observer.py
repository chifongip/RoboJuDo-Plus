import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from pydantic import ValidationError

from robojudo.config.config_manager import ConfigManager
from robojudo.config.g1.env.g1_env_cfg import G1_29_ARM_JOINT_NAMES
from robojudo.config.g1.env.g1_real_env_cfg import G1ObserverRealEnvCfg
from robojudo.config.observer_cfg import JointObserverCfg
from robojudo.environment import unitree_cpp_env
from robojudo.pipeline.four_mode_pipeline import ControlMode, FourModePipelineMixin

OBSERVED_SLOTS = [*range(15, 20), *range(22, 27)]


class FakeObserverController:
    def __init__(self):
        self.generation = 0
        self.commands = []
        self.generation_changed = False

    def get_safety_status(self):
        return SimpleNamespace(reset_generation=self.generation, latched=False)

    def step(self, positions, efforts=None, generation=None):
        self.commands.append((positions, efforts, generation))
        return SimpleNamespace(
            command_time_sec=10.005,
            reset_generation=None if generation is None else generation + self.generation_changed,
            arm_efforts=efforts,
        )

    def clear_arm_effort(self):
        self.generation += 1

    def set_damping(self, value):
        self.commands.append(("damping", value))

    def set_passive(self):
        self.commands.append(("passive",))

    def arm_position_control(self):
        self.commands.append(("arm",))

    def state_is_fresh(self, timeout):
        return True


class G1RealObserverTests(unittest.TestCase):
    def observer_preset(self):
        return ConfigManager("g1_23_locomanipulation_locomimic_real_observer").get_cfg()

    def test_deployment_preset_is_off_and_inherits_the_stiff_baseline(self):
        baseline = ConfigManager("g1_23_locomanipulation_locomimic_real").get_cfg()
        cfg = self.observer_preset()
        self.assertEqual(cfg.model_dump(exclude={"env"}), baseline.model_dump(exclude={"env"}))
        self.assertEqual(cfg.env.model_dump(exclude={"observer"}), baseline.env.model_dump())
        self.assertEqual(cfg.env.observer.mode, "OFF")
        limits = dict(zip(baseline.env.dof.joint_names, baseline.env.dof.torque_limits, strict=True))
        self.assertEqual(cfg.env.observer.actuator_limits_nm, [limits[name] for name in cfg.env.observer.joint_names])

    def test_active_uses_shared_observer_limits(self):
        cfg = self.observer_preset().env
        cfg.observer.mode = "ACTIVE"
        type(cfg).model_validate(cfg.model_dump())
        cfg.observer.actuator_limits_nm = None
        with self.assertRaisesRegex(ValidationError, "explicit actuator limits"):
            type(cfg).model_validate(cfg.model_dump())

    def make_environment(self, mode, cfg=None):
        if cfg is None:
            cfg = self.observer_preset().env
        cfg.observer.mode = mode
        env = unitree_cpp_env.UnitreeCppEnv.__new__(unitree_cpp_env.UnitreeCppEnv)
        env.cfg_env = cfg
        env.enabled = True
        env.unitree = FakeObserverController()
        env.robot_state = SimpleNamespace(received_time_sec=10.0, age_sec=0.0)
        env.num_dofs = cfg.dof.num_dofs
        env.joint_names = cfg.dof.joint_names
        env.torque_limits = np.asarray(cfg.dof.torque_limits)
        env.stiffness = np.asarray(cfg.dof.stiffness)
        env.damping = np.asarray(cfg.dof.damping)
        env.position_limits = np.asarray(cfg.dof.position_limits)
        env._dof_pos = np.asarray(cfg.dof.default_pos)
        env._dof_vel = np.zeros(env.num_dofs)
        env._motor_dof_count = cfg.motor_dof_count or env.num_dofs
        env._dof_idx = None if cfg.joint2motor_idx is None else np.asarray(cfg.joint2motor_idx)
        env._observer_eligible = False
        env._observer_generation = None
        env._observer_snapshot_started = time.monotonic()
        env._last_clamp_log_time = 0.0
        env._control_joint_names = set(env.joint_names)
        native_cfg = {}
        with (
            patch.object(unitree_cpp_env, "UnitreeController", FakeObserverController),
            patch.object(unitree_cpp_env, "RobotState", SimpleNamespace(received_time_sec=0, age_sec=0)),
            patch.object(
                unitree_cpp_env.unitree_cpp, "UnitreeConfig", SimpleNamespace(arm_motor_indices=[]), create=True
            ),
        ):
            env._configure_observer(native_cfg)
        env.set_observer_eligible(True)
        return env, native_cfg

    def test_shadow_maps_final_target_and_keeps_cached_sample_time(self):
        env, native_cfg = self.make_environment("SHADOW")
        self.assertFalse(native_cfg["arm_effort_enabled"])
        target = env._dof_pos.copy()
        target[13] = 0.123456789
        env.step(target)
        before = env._observer.sample_times.copy()
        env.step(target)
        commands = env.unitree.commands
        self.assertEqual(len(commands), 2)
        target_native, efforts, _ = commands[0]
        self.assertEqual(target_native[15], float(np.float32(target[13])))
        np.testing.assert_array_equal(efforts, 0)
        np.testing.assert_array_equal(env._observer.sample_times, before)
        status = env.get_observer_status()
        self.assertAlmostEqual(status["feedback_age_at_command_sec"], 0.005)
        np.testing.assert_array_equal(status["issued_nm"], 0)

    def test_active_config_limits_only_arm_slots_and_new_generation_clears_history(self):
        env, native_cfg = self.make_environment("ACTIVE")
        self.assertEqual(env._dof_idx[env._observer.indices].tolist(), OBSERVED_SLOTS)
        self.assertEqual(native_cfg["arm_motor_indices"], OBSERVED_SLOTS)
        self.assertTrue(native_cfg["arm_effort_enabled"])
        self.assertEqual(native_cfg["arm_effort_limits"], env.cfg_env.observer.resolve_correction_limits())
        env.step(env._dof_pos)
        self.assertTrue(env._observer.command_history)
        env.unitree.generation_changed = True
        env.step(env._dof_pos)
        self.assertFalse(env._observer.command_history)

    def test_full_g1_arm_commands_follow_observer_joint_order(self):
        baseline = ConfigManager("g1_29_locomanipulation_locomimic_real").get_cfg().env
        names = list(reversed(G1_29_ARM_JOINT_NAMES))
        limit_by_joint = dict(zip(baseline.dof.joint_names, baseline.dof.torque_limits, strict=True))
        observer = JointObserverCfg(
            mode="ACTIVE",
            joint_names=names,
            algorithm="nonlinear",
            inertia=0.1,
            slew_nm_s=2.0,
            actuator_limits_nm=[limit_by_joint[name] for name in names],
        )
        cfg = G1ObserverRealEnvCfg(**baseline.model_dump(), observer=observer)
        env, native_cfg = self.make_environment("ACTIVE", cfg)
        self.assertEqual(native_cfg["arm_motor_indices"], list(range(28, 14, -1)))
        self.assertEqual(native_cfg["arm_total_torque_limits"], observer.actuator_limits_nm)
        env.step(env._dof_pos)
        self.assertEqual(len(env.unitree.commands[-1][0]), 29)
        self.assertEqual(len(env.unitree.commands[-1][1]), 14)
        self.assertEqual(env.get_observer_status()["joint_names"], names)

        invalid = cfg.model_dump()
        invalid["observer"]["joint_names"][0] = "left_knee_joint"
        with self.assertRaisesRegex(ValidationError, "must match the arm joints"):
            G1ObserverRealEnvCfg.model_validate(invalid)

    def test_stale_snapshot_falls_back_to_position_only(self):
        env, _ = self.make_environment("SHADOW")
        env.robot_state.age_sec = 0.2
        env.step(env._dof_pos)
        self.assertIsNone(env.unitree.commands[-1][1])
        self.assertFalse(env._observer.command_history)

    def make_control_step(self):
        pipeline = FourModePipelineMixin.__new__(FourModePipelineMixin)
        pipeline.cfg = SimpleNamespace(debug=SimpleNamespace(log_obs=True))
        pipeline.env = Mock(dof_pos=np.zeros(23))
        pipeline.ctrl_manager = Mock()
        pipeline.ctrl_manager.get_ctrl_data.return_value = {"COMMANDS": []}
        pipeline._process_commands = Mock()
        pipeline._safety_check_before_command = Mock()
        pipeline._shutdown_requested = False
        pipeline.mode = ControlMode.RL_DEFAULT
        pipeline.timestep = 1
        return pipeline

    def test_failed_mode_transition_enters_damping(self):
        pipeline = self.make_control_step()
        pipeline._process_commands.side_effect = RuntimeError("rearm requires fresh feedback")
        pipeline._force_damping = Mock()
        pipeline._post_mode_step = Mock()
        pipeline.step()
        pipeline._force_damping.assert_called_once_with("rearm requires fresh feedback")

    def test_debug_logs_receipt_before_policy_transition_resets_history(self):
        pipeline = self.make_control_step()
        pipeline.debug_logger = Mock()
        pipeline._step_rl_policy = Mock(return_value=(np.zeros(23), {}))
        pipeline.env.get_observer_status.return_value = {"issued_nm": [0.2]}

        def post_step(env_data, ctrl_data, extras, pd_target, rl_active):
            pipeline.env.get_observer_status.return_value = {"issued_nm": [0.0]}
            pipeline._log_control_step(env_data, ctrl_data, extras, pd_target)

        pipeline._post_mode_step = post_step
        pipeline.step()
        self.assertEqual(pipeline.debug_logger.log.call_args.kwargs["extras"]["observer"]["issued_nm"], [0.2])
