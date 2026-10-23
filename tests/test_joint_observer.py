"""Generic runtime timing and environment-specific configuration requirements."""

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from robojudo.config.observer_cfg import JointObserverCfg
from robojudo.config.x2 import x2_locomimic_real, x2_locomimic_real_observer
from robojudo.environment.agibot_cpp_env import AgiBotCppEnv
from robojudo.utils.observers import JointObserver


def config(algorithm="nonlinear"):
    return JointObserverCfg(
        algorithm=algorithm,
        mode="ACTIVE",
        joint_names=["elbow", "shoulder"],
        inertia=[0.1, 0.2],
        correction_limits_nm=0.5,
        slew_nm_s=2,
    )


class JointObserverTests(unittest.TestCase):
    def test_arbitrary_names_and_timestamped_interval_replay_for_both_algorithms(self):
        for algorithm in ("linearized", "nonlinear"):
            with self.subTest(algorithm=algorithm):
                ctrl = JointObserver(
                    config(algorithm),
                    ["shoulder", "other", "elbow"],
                    [10] * 3,
                    [1] * 3,
                    actuator_limits=[4] * 3,
                    max_dt=0.1,
                )
                q, v, target = np.zeros(3), np.zeros(3), np.zeros(3)
                ctrl.propose(target, q, v, 1, sample_times=np.full(3, 100))
                ctrl.acknowledge(np.zeros(2), command_time=100)
                actual = np.array([0.3, -0.2])
                next_q, next_v, _, _ = ctrl.observer.predict_state(
                    q[ctrl.indices], v[ctrl.indices], target[ctrl.indices], np.zeros(2), actual, 0.02
                )
                q[ctrl.indices], v[ctrl.indices] = next_q, next_v
                effort = ctrl.propose(target, q, v, 1.03, sample_times=np.full(3, 100.02))
                ctrl.acknowledge(effort, command_time=100.021)
                np.testing.assert_allclose(
                    ctrl.observer.estimated_disturbance_nm,
                    -np.expm1(-config().bandwidth_rad_s * 0.02) * actual,
                    atol=1e-6,
                )
                status = ctrl.get_status()
                self.assertEqual(status["joint_names"], ["elbow", "shoulder"])
                self.assertEqual(status["algorithm"], algorithm)
                status["issued_nm"][0] = 100
                self.assertLessEqual(abs(ctrl.get_status()["issued_nm"][0]), 0.5)
                # Each joint owns its own receive interval.
                previous = ctrl.observer.estimated_disturbance_nm.copy()
                stamps = np.full(3, 100.02)
                stamps[ctrl.indices[0]] = 100.04
                v[ctrl.indices[0]] += 0.02
                ctrl.propose(target, q, v, 1.05, sample_times=stamps)
                self.assertEqual(ctrl.observer.estimated_disturbance_nm[1], previous[1])
                self.assertNotEqual(ctrl.observer.estimated_disturbance_nm[0], previous[0])

    def test_simulation_fallback_cannot_authorize_hardware_active(self):
        cfg = config()
        JointObserver(cfg, ["elbow", "shoulder"], [10] * 2, [1] * 2, actuator_limits=[4] * 2, max_dt=0.1)
        with self.assertRaisesRegex(ValueError, "explicit actuator limits"):
            cfg.validate_hardware()
        cfg.actuator_limits_nm = [4, 4]
        cfg.validate_hardware()
        cfg.actuator_limits_nm = [float("nan"), 4]
        with self.assertRaises(ValueError):
            JointObserver(cfg, ["elbow", "shoulder"], [10] * 2, [1] * 2, max_dt=0.1)

    def test_hardware_boundary_rejects_missing_limits_and_non_arm_joints(self):
        env = AgiBotCppEnv.__new__(AgiBotCppEnv)
        env.cfg_env = x2_locomimic_real_observer().env
        cfg = env.cfg_env.observer
        cfg.mode, cfg.actuator_limits_nm = "ACTIVE", None
        with self.assertRaisesRegex(ValueError, "explicit actuator limits"):
            env._build_observer()
        cfg.actuator_limits_nm = [10] * 14
        cfg.joint_names[0] = "left_hip_pitch_joint"
        with self.assertRaisesRegex(ValueError, "configured arm joints"):
            env._build_observer()

    def test_clean_configuration_break_and_explicit_algorithm(self):
        base = x2_locomimic_real().env
        self.assertIsNone(base.observer)
        with self.assertRaisesRegex(ValueError, "arm_observer was removed"):
            type(base).model_validate({**base.model_dump(), "arm_observer": {"mode": "OFF"}})
        data = config().model_dump()
        data.pop("algorithm")
        with self.assertRaises(ValueError):
            JointObserverCfg.model_validate(data)
        with self.assertRaisesRegex(ValueError, "symmetric"):
            JointObserverCfg.model_validate({**config("linearized").model_dump(), "release_slew_nm_s": 16})


class HardwareObserverConstructorTests(unittest.TestCase):
    """Exercise the real environment constructor with the native module replaced."""

    def environment_config(self, algorithm, mode):
        cfg = x2_locomimic_real_observer().env
        cfg.forward_kinematic, cfg.update_with_fk = None, False
        observer = cfg.observer
        observer.algorithm, observer.mode = algorithm, mode
        if algorithm == "linearized":
            observer.release_slew_nm_s = None
        # Distinct limits and a nontrivial permutation expose incorrect SDK order.
        observer.correction_limits_nm = [0.1 + 0.01 * i for i in range(len(observer.joint_names))]
        permutation = np.random.default_rng(31).permutation(len(observer.joint_names))
        for field in ("joint_names", "inertia", "friction_viscosity", "actuator_limits_nm", "correction_limits_nm"):
            values = getattr(observer, field)
            setattr(observer, field, [values[i] for i in permutation])
        return cfg

    def fake_sdk(self, cfg):
        controller = Mock(
            spec=[
                "get_state_freshness_report",
                "get_safety_status",
                "get_robot_state",
                "arm_position_control",
                "clear_arm_effort",
                "self_check",
                "shutdown",
                "step",
            ]
        )
        controller.get_state_freshness_report.return_value = SimpleNamespace(required_streams_fresh=True)
        controller.get_safety_status.return_value = SimpleNamespace(state="ACTIVE", reset_generation=0)
        controller.get_robot_state.return_value = SimpleNamespace(joint_received_time_sec=[100.0] * cfg.dof.num_dofs)
        controller.self_check.return_value = True
        module = ModuleType("aimdk_cpp")
        module.AimdkController = Mock(return_value=controller)
        return module, controller

    def test_full_constructor_sets_native_limits_and_mode_for_both_algorithms(self):
        for algorithm in ("linearized", "nonlinear"):
            for mode in ("OFF", "SHADOW", "ACTIVE"):
                with self.subTest(algorithm=algorithm, mode=mode):
                    cfg = self.environment_config(algorithm, mode)
                    sdk, controller = self.fake_sdk(cfg)
                    with patch.dict(sys.modules, {"aimdk_cpp": sdk}):
                        env = AgiBotCppEnv(cfg)
                        self.addCleanup(env.shutdown)
                    native_cfg = sdk.AimdkController.call_args.args[0]
                    self.assertIs(env.aimdk, controller)
                    controller.self_check.assert_called_once_with(cfg.aimdk.startup_state_timeout)
                    self.assertEqual(native_cfg["arm_joint_names"], cfg.arm_joint_names)
                    if mode == "OFF":
                        self.assertIsNone(env._observer)
                        self.assertFalse(native_cfg.get("arm_effort_enabled", False))
                    else:
                        self.assertEqual(env._observer.algorithm, algorithm)
                        self.assertEqual(env._observer.mode, mode)
                        self.assertEqual(native_cfg["arm_effort_enabled"], mode == "ACTIVE")
                        named_limits = dict(
                            zip(cfg.observer.joint_names, cfg.observer.correction_limits_nm, strict=True)
                        )
                        np.testing.assert_array_equal(
                            native_cfg["arm_effort_limits"], [named_limits[name] for name in cfg.arm_joint_names]
                        )

    def test_full_constructor_rejects_invalid_active_config_before_native_creation(self):
        for invalid in ("missing_limits", "non_arm_joint"):
            with self.subTest(invalid=invalid):
                cfg = self.environment_config("nonlinear", "ACTIVE")
                if invalid == "missing_limits":
                    cfg.observer.actuator_limits_nm = None
                else:
                    cfg.observer.joint_names[0] = cfg.leg_joint_names[0]
                sdk, _ = self.fake_sdk(cfg)
                with patch.dict(sys.modules, {"aimdk_cpp": sdk}), self.assertRaises(ValueError):
                    AgiBotCppEnv(cfg)
                sdk.AimdkController.assert_not_called()

    def test_native_command_and_clipped_receipt_preserve_named_joint_order(self):
        for algorithm in ("linearized", "nonlinear"):
            with self.subTest(algorithm=algorithm):
                cfg = self.environment_config(algorithm, "ACTIVE")
                sdk, controller = self.fake_sdk(cfg)
                with patch.dict(sys.modules, {"aimdk_cpp": sdk}):
                    env = AgiBotCppEnv(cfg)
                    self.addCleanup(env.shutdown)
                env.set_observer_eligible(True)
                env._observer_feedback_time = 1.0
                env._observer_sample_times = np.full(env.num_dofs, 100.0)
                target = env.dof_pos
                observer = env._observer
                proposed = observer.observer.limits * np.where(np.arange(len(observer.joint_names)) % 2, -0.5, 0.5)
                named_effort = dict(zip(observer.joint_names, proposed, strict=True))
                native_effort = np.array([named_effort[name] for name in cfg.arm_joint_names])
                controller.step.return_value = SimpleNamespace(
                    reset_generation=0, command_time_sec=100.0, arm_efforts=(native_effort / 2).tolist()
                )
                # Stub estimator output only; exercise the actual environment
                # step, SDK ordering and runtime receipt reconciliation.
                observer.pending_target = target[observer.indices].copy()
                observer.proposed_nm, observer.limited_nm = proposed.copy(), proposed.copy()
                observer.observer.effort = proposed.copy()
                with (
                    patch.object(observer, "propose", return_value=proposed),
                    patch("robojudo.environment.agibot_cpp_env.time.monotonic", return_value=1.0),
                ):
                    env.step(target)
                args = controller.step.call_args.args
                np.testing.assert_array_equal(args[0], target)
                np.testing.assert_array_equal(args[1], native_effort)
                self.assertEqual(args[2], 0)
                np.testing.assert_array_equal(observer.issued_nm, proposed / 2)
                np.testing.assert_array_equal(observer.observer.effort, proposed / 2)
                np.testing.assert_array_equal(observer.command_history[-1][2], proposed / 2)


if __name__ == "__main__":
    unittest.main()
