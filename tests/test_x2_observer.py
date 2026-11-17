"""Deployment observer tests using fake commands only; no ROS or robot connection."""

import importlib.util
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from robojudo.config.observer_cfg import JointObserverCfg
from robojudo.config.x2.observer_parameters import JOINT_NAMES, selected_observer_parameters
from robojudo.environment.agibot_cpp_env import AgiBotCppEnv
from robojudo.pipeline.four_mode_pipeline import ControlMode, FourModePipelineMixin
from robojudo.utils.observers import JointObserver


def config(mode="ACTIVE"):
    # Synthetic test limits, not deployment recommendations.
    parameters = selected_observer_parameters()
    parameters.pop("method")
    return JointObserverCfg(
        algorithm="nonlinear",
        joint_names=list(JOINT_NAMES),
        **parameters,
        mode=mode,
        actuator_limits_nm=[10.0] * 14,
        correction_limit_nm=1.0,
        slew_nm_s=2.0,
    )


def runtime(cfg=None):
    names = ["leg", "waist", *reversed(JOINT_NAMES)]
    cfg = JointObserverCfg.model_validate((cfg or config()).model_dump())
    cfg.validate_hardware()
    return JointObserver(cfg, names, [20] * 16, [2] * 16, actuator_limits=[10] * 16, max_dt=0.1)


class FakeAdapter:
    def __init__(self):
        self.generation = 0
        self.state = "ACTIVE"
        self.fresh = True
        self.commands = []
        self.effort = np.zeros(14)
        self.reset_on_step = False

    def clear_arm_effort(self):
        self.generation += 1
        self.effort[:] = 0

    def get_safety_status(self):
        return SimpleNamespace(reset_generation=self.generation, state=self.state, latched=self.state == "DAMPING")

    def get_state_freshness_report(self, *args):
        return SimpleNamespace(required_streams_fresh=self.fresh)

    def step(self, positions, efforts=None, expected=None):
        if self.reset_on_step:
            self.clear_arm_effort()
        self.effort = np.asarray(efforts if efforts is not None and expected == self.generation else np.zeros(14))
        self.commands.append((np.asarray(positions), self.effort.copy()))
        return SimpleNamespace(
            arm_efforts=self.effort.tolist(), reset_generation=self.generation, command_time_sec=time.monotonic()
        )

    def set_passive(self):
        self.clear_arm_effort()

    def set_damping(self, value):
        self.clear_arm_effort()

    def arm_position_control(self):
        self.clear_arm_effort()

    def set_control_joint_names(self, names):
        self.clear_arm_effort()

    def set_gains(self, kp, kd):
        self.clear_arm_effort()

    def shutdown(self):
        self.clear_arm_effort()


def environment(cfg=None):
    env = AgiBotCppEnv.__new__(AgiBotCppEnv)
    env.enabled = True
    env.aimdk = FakeAdapter()
    env._observer = runtime(cfg)
    env._configure_observer_order(list(reversed(JOINT_NAMES)))
    env._observer_eligible = True
    env._observer_reset_generation = 0
    env._observer_feedback_time = time.monotonic()
    env._observer_sample_times = np.full(16, env._observer_feedback_time)
    env.joint_names = ["leg", "waist", *reversed(JOINT_NAMES)]
    env.num_dofs = 16
    env.stiffness, env.damping = np.full(16, 20.0), np.full(16, 2.0)
    env._dof_pos, env._dof_vel = np.zeros(16), np.zeros(16)
    env.position_limits = np.tile([-1.0, 1.0], (16, 1))
    env._control_joint_names = set(env.joint_names)
    env._last_clamp_log_time = 0.0
    env.cfg_env = SimpleNamespace(aimdk=SimpleNamespace(state_timeout=0.1, odometry_timeout=0.1, shutdown_damping=5.0))
    return env


class ArmObserverTests(unittest.TestCase):
    def test_repeated_native_sample_freezes_adaptation_despite_new_commands(self):
        env = environment()
        env._observer_feedback_time = 1.0
        env._observer_sample_times[:] = 50.0  # Native and Python clock epochs differ.
        target = np.full(16, 0.1)
        for now in (1.0, 1.02, 1.04, 1.06, 1.08):
            with patch("robojudo.environment.agibot_cpp_env.time.monotonic", return_value=now):
                env.step(target)
            np.testing.assert_array_equal(env._observer.issued_nm, 0)
            np.testing.assert_array_equal(env._observer.observer.estimated_disturbance_nm, 0)
        with patch("robojudo.environment.agibot_cpp_env.time.monotonic", return_value=1.11):
            env.step(target)
        self.assertIsNone(env._observer.sample_times)
        self.assertFalse(env._observer.command_history)

    def test_receive_intervals_replay_accepted_targets_across_inference_jitter(self):
        controller = runtime()
        q, v = np.zeros(16), np.zeros(16)
        target = np.full(16, 0.1)
        stamps = np.full(16, 100.0)
        controller.propose(target, q, v, 1, sample_times=stamps)
        controller.acknowledge(np.zeros(14), command_time=100.005)
        # First interval has no command coverage before 100.005; initialize only.
        stamps[:] = 100.02
        controller.propose(target, q, v, 1.03, sample_times=stamps)
        controller.acknowledge(np.zeros(14), command_time=100.021)
        new_target = -target
        controller.propose(new_target, q, v, 1.04, sample_times=stamps)
        controller.acknowledge(np.zeros(14), command_time=100.025)
        model = controller.observer
        zeros = np.zeros(14)
        q1, v1, _, _ = model.predict_state(zeros, zeros, target[controller.indices], zeros, zeros, 0.005)
        q2, v2, _, _ = model.predict_state(q1, v1, new_target[controller.indices], zeros, zeros, 0.025)
        q[controller.indices], v[controller.indices] = q2, v2
        stamps[:] = 100.05  # 30 ms feedback interval, but 40 ms command interval.
        controller.propose(new_target, q, v, 1.08, sample_times=stamps)
        np.testing.assert_allclose(model.velocity_innovation, 0, atol=1e-11)
        np.testing.assert_allclose(model.estimated_disturbance_nm, 0, atol=1e-10)
        # A partial arm refresh must not adapt other joints from repeated samples.
        controller.acknowledge(np.zeros(14), command_time=100.055)
        previous = model.estimated_disturbance_nm.copy()
        stamps[controller.indices[0]] = 100.07
        v[controller.indices[0]] -= 0.1
        controller.propose(new_target, q, v, 1.10, sample_times=stamps)
        np.testing.assert_array_equal(model.estimated_disturbance_nm[1:], previous[1:])
        self.assertNotEqual(model.estimated_disturbance_nm[0], previous[0])

    def test_configuration_defaults_and_explicit_active_limits(self):
        self.assertEqual(config("OFF").mode, "OFF")
        self.assertEqual(config("SHADOW").mode, "SHADOW")
        for kwargs in (
            {"actuator_limits_nm": None},
            {"actuator_limits_nm": [1]},
            {"actuator_limits_nm": [float("nan")] * 14},
            {"slew_nm_s": float("inf")},
            {"correction_limit_fraction": 1.1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                candidate = JointObserverCfg.model_validate({**config().model_dump(), **kwargs})
                candidate.validate_hardware()

    def test_mutated_configuration_is_revalidated_before_hardware_fallback(self):
        cfg = config("SHADOW").model_copy(update={"actuator_limits_nm": None})
        cfg.mode = "ACTIVE"
        with self.assertRaises(ValueError):
            runtime(cfg)
        for field, invalid in (("correction_limit_nm", float("nan")), ("slew_nm_s", -2)):
            cfg = config()
            setattr(cfg, field, invalid)
            with self.subTest(field=field), self.assertRaises(ValueError):
                runtime(cfg)

    def test_runtime_mode_cannot_be_changed_through_source_configuration(self):
        cfg = config("SHADOW")
        controller = runtime(cfg)
        cfg.mode = controller.cfg.mode = "ACTIVE"
        self.assertEqual(controller.mode, "SHADOW")
        np.testing.assert_array_equal(controller.propose(np.zeros(16), np.zeros(16), np.zeros(16), 1), 0)
        with self.assertRaises(ValueError):
            controller.acknowledge(np.ones(14) * 0.1)
        np.testing.assert_array_equal(controller.issued_nm, 0)

    def test_invalid_runtime_vectors_time_and_receipts_reset_all_state(self):
        zeros = np.zeros(16)
        for target, position, velocity, now in (
            (np.zeros((16, 1)), zeros, zeros, 1.02),
            (zeros, np.zeros(2), zeros, 1.02),
            (zeros, zeros, np.full(16, np.nan), 1.02),
            (zeros, zeros, zeros, float("nan")),
            (zeros, zeros, zeros, float("inf")),
            (np.full(16, 1e308), zeros, zeros, 1.02),
        ):
            controller = runtime()
            controller.propose(zeros, zeros, zeros, 1)
            with self.subTest(now=now), self.assertRaises((ValueError, FloatingPointError)):
                controller.propose(target, position, velocity, now)
            self.assertIsNone(controller.last_time)
            self.assertIsNone(controller.observer.previous_position)
            np.testing.assert_array_equal(controller.issued_nm, 0)
        for receipt in ([], np.zeros((14, 1)), np.full(14, np.nan), np.ones(14) * 0.5):
            controller = runtime()
            controller.propose(zeros, zeros, zeros, 1)
            with self.subTest(receipt=receipt), self.assertRaises(ValueError):
                controller.acknowledge(receipt)
            self.assertIsNone(controller.last_time)
            np.testing.assert_array_equal(controller.issued_nm, 0)

    def test_cached_snapshot_age_cannot_be_hidden_by_newer_backend_feedback(self):
        env = environment()
        env.born_place_align = env.update_with_fk = False
        env._odometry_type = "NONE"
        env.aimdk.get_robot_state = lambda: SimpleNamespace(
            joint_age_sec=[0.09] * 16,
            joint_received_time_sec=[0.91] * 16,
            motor_state=SimpleNamespace(q=[0.0] * 16, dq=[-0.1] * 16),
            imu_state=SimpleNamespace(quaternion=[0, 0, 0, 1], gyroscope=[0, 0, 0], accelerometer=[0, 0, 0]),
        )
        with patch("robojudo.environment.agibot_cpp_env.time.monotonic", return_value=1):
            env.update()
        self.assertAlmostEqual(env._observer_feedback_time, 0.91)
        # New packets have reached the backend, but this Python snapshot is now stale.
        env.aimdk.fresh = True
        with (
            patch("robojudo.environment.agibot_cpp_env.time.monotonic", return_value=1.03),
            patch.object(env._observer, "propose") as propose,
        ):
            env.step(np.zeros(16))
        propose.assert_not_called()
        np.testing.assert_array_equal(env.aimdk.effort, 0)
        self.assertIsNone(env._observer.observer.previous_position)

    def test_malformed_target_damps_instead_of_leaving_pending_effort(self):
        for target in (np.zeros((16, 1)), [0] * 15, None, ["bad"] * 16):
            env = environment()
            env.aimdk.effort[:] = 0.5
            with self.subTest(target=target), self.assertRaises(ValueError):
                env.step(target)
            np.testing.assert_array_equal(env.aimdk.effort, 0)
            self.assertFalse(env.aimdk.commands)

    def test_joint_mapping_and_actual_elapsed_slew(self):
        controller = runtime()
        q, qd = np.zeros(16), np.zeros(16)
        controller.propose(q, q, qd, 1.0)
        for now in (1.013, 1.044, 1.061):
            previous = controller.issued_nm.copy()
            previous_time = controller.last_time
            qd[2:] -= np.linspace(0.01, 0.14, 14)
            effort = controller.propose(q, q, qd, now)
            controller.acknowledge(effort)
            np.testing.assert_allclose(effort, controller.issued_nm)
            self.assertTrue(np.all(abs(controller.issued_nm - previous) <= 2 * (now - previous_time) + 1e-12))
        self.assertGreater(np.max(abs(controller.issued_nm)), 0)
        self.assertTrue(np.all(abs(controller.issued_nm) <= 1))
        # A missed deadline restarts history with no accumulated effort.
        np.testing.assert_array_equal(controller.propose(q, q, qd, 2), 0)

    def test_shadow_estimates_with_zero_issued_correction(self):
        controller = runtime(config("SHADOW"))
        q, qd = np.zeros(16), np.zeros(16)
        for i in range(8):
            qd[2:] = -0.01 * i
            effort = controller.propose(q, q, qd, 1 + i * 0.02)
            np.testing.assert_array_equal(effort, 0)
            controller.acknowledge(effort)
            np.testing.assert_array_equal(controller.observer.effort, 0)
        self.assertGreater(np.max(abs(controller.proposed_nm)), 0)
        self.assertGreater(np.max(abs(controller.observer.estimated_disturbance_nm)), 0)

    def test_headroom_and_downstream_reconciliation(self):
        controller = runtime()
        q = np.zeros(16)
        controller.propose(q, q, q, 1)
        controller.observer.effort[:] = 0.5
        controller.observer.estimated_disturbance_nm[:] = -0.5
        target = np.ones(16)  # Existing PD already exceeds the synthetic actuator limit.
        np.testing.assert_array_equal(controller.propose(target, q, q, 1.02), 0)
        controller.acknowledge(np.zeros(14))
        np.testing.assert_array_equal(controller.observer.effort, 0)
        self.assertTrue(controller.observer.antiwindup.all())
        self.assertTrue(np.all(abs(controller.observer.estimated_disturbance_nm) <= 1))

    def test_final_clipped_targets_and_unchanged_non_arm_positions(self):
        env = environment()
        target = np.linspace(-2, 2, 16)
        with patch.object(env._observer, "propose", wraps=env._observer.propose) as propose:
            env.step(target)
        np.testing.assert_array_equal(propose.call_args.args[0], np.clip(target, -1, 1))
        np.testing.assert_array_equal(env.aimdk.commands[-1][0], np.clip(target, -1, 1))
        self.assertEqual(env.aimdk.commands[-1][1].shape, (14,))

    def test_stale_feedback_and_transient_watchdog_generation_reset(self):
        env = environment()
        for cause in ("stale", "generation", "hold", "old_sample", "race"):
            with self.subTest(cause=cause):
                env.aimdk.fresh = cause != "stale"
                env.aimdk.state = "HOLD" if cause == "hold" else "ACTIVE"
                env._observer_feedback_time = time.monotonic() - (1 if cause == "old_sample" else 0)
                env.aimdk.reset_on_step = cause == "race"
                env._observer.observer.effort[:] = 0.5
                if cause == "generation":
                    env.aimdk.generation += 1
                env.step(np.zeros(16))
                np.testing.assert_array_equal(env.aimdk.effort, 0)
                np.testing.assert_array_equal(env._observer.observer.effort, 0)

    def test_lifecycle_clear_and_gain_refresh(self):
        for method, args in (
            ("command_passive", ()),
            ("command_damping", ()),
            ("arm_position_control", ()),
            ("set_control_joint_names", (["leg"],)),
            ("shutdown", ()),
            ("set_gains", ([30] * 16, [3] * 16)),
            ("set_observer_eligible", (False,)),
        ):
            with self.subTest(method=method):
                env = environment()
                env._observer.observer.effort[:] = 0.5
                env.aimdk.effort[:] = 0.5
                getattr(env, method)(*args)
                self.assertIsNone(env._observer.observer.previous_position)
                np.testing.assert_array_equal(env.aimdk.effort, 0)
                np.testing.assert_array_equal(env._observer.observer.effort, 0)
                if method == "set_gains":
                    np.testing.assert_array_equal(env._observer.observer.kp, 30)
                    np.testing.assert_array_equal(env._observer.observer.kd, 3)

    def test_environment_disable_immediately_clears_pending_correction(self):
        env = environment()
        env.aimdk.effort[:] = 0.5
        env._observer.observer.effort[:] = 0.5
        env.enabled = False
        np.testing.assert_array_equal(env.aimdk.effort, 0)
        self.assertIsNone(env._observer.observer.previous_position)

    def test_control_joint_iterators_are_not_consumed_before_native_submission(self):
        env = environment()
        env.aimdk.set_control_joint_names = Mock()
        env.set_control_joint_names(name for name in ("leg", "waist"))
        env.aimdk.set_control_joint_names.assert_called_once_with(["leg", "waist"])
        self.assertEqual(env._control_joint_names, {"leg", "waist"})
        env.aimdk.set_control_joint_names.side_effect = ValueError("unknown joint")
        with self.assertRaises(ValueError):
            env.set_control_joint_names(["unknown"])
        self.assertEqual(env._control_joint_names, {"leg", "waist"})

    def test_disabled_uses_original_position_only_call(self):
        env = environment()
        env._observer = None
        env.aimdk = Mock()
        target = np.linspace(-0.5, 0.5, 16)
        env.step(target)
        env.aimdk.step.assert_called_once_with(target.tolist())

    def test_active_transmission_then_disable_clears_pending_effort(self):
        env = environment()
        with patch("robojudo.environment.agibot_cpp_env.time.monotonic", return_value=1.0):
            env._observer_feedback_time = 1.0
            env._observer_sample_times[:] = 1.0
            env.step(np.zeros(16))
        env._dof_vel[2:] = -0.1
        with patch("robojudo.environment.agibot_cpp_env.time.monotonic", return_value=1.02):
            env._observer_feedback_time = 1.0
            env._observer_sample_times[:] = 1.02
            env.step(np.zeros(16))
        self.assertGreater(np.max(abs(env.aimdk.effort)), 0)
        np.testing.assert_array_equal(env.aimdk.effort[::-1], env._observer.observer.effort)
        env.set_observer_eligible(False)
        np.testing.assert_array_equal(env.aimdk.effort, 0)
        self.assertIsNone(env._observer.observer.previous_position)

    def test_shadow_status_is_named_and_does_not_expose_mutable_state(self):
        env = environment(config("SHADOW"))
        env.step(np.zeros(16))
        status = env.get_observer_status()
        self.assertEqual(status["mode"], "SHADOW")
        self.assertEqual(status["joint_names"], list(JOINT_NAMES))
        self.assertEqual(status["model_qualification"], "simulation_only")
        np.testing.assert_array_equal(status["issued_nm"], 0)
        status["issued_nm"][0] = 99
        np.testing.assert_array_equal(env._observer.issued_nm, 0)
        status["estimated_disturbance_nm"][0] = 99
        np.testing.assert_array_equal(env._observer.observer.estimated_disturbance_nm, 0)

    def test_observer_logs_once_per_second_after_submission(self):
        for mode in ("SHADOW", "ACTIVE"):
            with self.subTest(mode=mode):
                env = environment(config(mode))
                with patch("robojudo.environment.agibot_cpp_env.logger") as log:
                    log.isEnabledFor.return_value = True
                    for now in (10.0, 10.02, 11.01):
                        env._observer_feedback_time = now
                        env._step_with_observer(np.zeros(16), now)
                    self.assertEqual(log.info.call_count, 2)
                    args = log.info.call_args.args
                    self.assertEqual(args[1], mode)
                    message = args[0] % args[1:]
                    self.assertNotIn("\n", message)
                    self.assertLess(len(message), 300)
                    self.assertIn("bound=0 limited=0", message)
                    for values, key in zip(args[2:4], ("estimated_disturbance_nm", "issued_nm"), strict=True):
                        np.testing.assert_allclose(
                            np.fromstring(values, sep=","), env.get_observer_status()[key], atol=0.05
                        )
                    # Repeated feedback preserves the bounded estimate while
                    # the new PD target removes all compensation headroom.
                    env._observer.observer.estimated_disturbance_nm[:] = -1
                    env._observer.observer.effort[:] = 0.5
                    env._observer.last_time = 12.0
                    env._observer_feedback_time = 12.02
                    env._step_with_observer(np.ones(16), 12.02)
                    self.assertEqual(log.info.call_count, 3)
                    args = log.info.call_args.args
                    self.assertIn("bound=14 limited=14", args[0] % args[1:])
                    np.testing.assert_array_equal(env._observer.issued_nm, 0)
                self.assertEqual(len(env.aimdk.commands), 4)
                np.testing.assert_array_equal(env.aimdk.effort, 0)

    def test_observer_logging_skips_ineligible_and_stale_feedback(self):
        for cfg, eligible, fresh in (
            (config(), False, True),
            (config(), True, False),
            (config("SHADOW"), False, True),
            (config("SHADOW"), True, False),
        ):
            with self.subTest(mode=cfg.mode, eligible=eligible, fresh=fresh):
                env = environment(cfg)
                env._observer_eligible = eligible
                env.aimdk.fresh = fresh
                with patch("robojudo.environment.agibot_cpp_env.logger") as log:
                    env.step(np.zeros(16))
                    log.info.assert_not_called()

    def test_invalid_feedback_and_rejected_submission_clear(self):
        env = environment()
        env.step(np.zeros(16))
        env._dof_vel[:] = np.nan
        with self.assertRaises(ValueError):
            env.step(np.zeros(16))
        self.assertIsNone(env._observer.observer.previous_position)
        env._dof_vel[:] = 0
        env.aimdk.step = Mock(side_effect=RuntimeError("rejected"))
        with self.assertRaises(RuntimeError):
            env.step(np.zeros(16))
        self.assertIsNone(env._observer.observer.previous_position)
        np.testing.assert_array_equal(env.aimdk.effort, 0)


class ObserverTeardownTests(unittest.TestCase):
    def test_pipeline_errors_shutdown_x2_before_other_cleanup(self):
        spec = importlib.util.spec_from_file_location(
            "review_run_pipeline", Path(__file__).parents[1] / "scripts" / "run_pipeline.py"
        )
        run_pipeline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_pipeline)

        args = SimpleNamespace(
            config="test",
            record=False,
            record_endpoint=None,
            record_task=None,
            gr00t_task=None,
            gr00t_command_endpoint=None,
        )
        cfg = SimpleNamespace(
            env=SimpleNamespace(is_sim=False, env_type="AgiBotCppEnv"),
            pipeline_type="X2LocomanipulationPipeline",
            run_fullspeed=True,
        )
        for failing_stage in ("prepare", "step", "recording"):
            with self.subTest(failing_stage=failing_stage):
                events = []
                pipeline = SimpleNamespace(
                    env=SimpleNamespace(shutdown=lambda events=events: events.append("shutdown")),
                    prepare=Mock(),
                    step=Mock(side_effect=RuntimeError("step failed")),
                    close_recording=lambda events=events: events.append("recording"),
                    ctrl_manager=SimpleNamespace(close=lambda events=events: events.append("controllers")),
                )
                if failing_stage == "prepare":
                    pipeline.prepare.side_effect = RuntimeError("prepare failed")
                if failing_stage == "recording":

                    def recording_error(events=events):
                        events.append("recording")
                        raise RuntimeError("recording failed")

                    pipeline.close_recording = recording_error
                with (
                    patch.object(run_pipeline, "parse_args", return_value=args),
                    patch.object(run_pipeline, "ConfigManager", return_value=SimpleNamespace(get_cfg=lambda: cfg)),
                    patch.object(run_pipeline.robojudo.pipeline, "X2LocomanipulationPipeline", return_value=pipeline),
                    self.assertRaises(RuntimeError),
                ):
                    run_pipeline.main()
                self.assertEqual(events, ["shutdown", "recording", "controllers"])


class ObserverPolicyGateTests(unittest.TestCase):
    def test_mode_entry_clears_observer_before_any_position_command(self):
        pipeline = FourModePipelineMixin.__new__(FourModePipelineMixin)
        pipeline.env = environment()
        pipeline._on_leave_rl = Mock()
        pipeline._on_enter_rl = Mock()
        pipeline._set_upper_body_enabled = Mock()
        pipeline._rl_stiffness = pipeline._joint_default_stiffness = [20] * 16
        pipeline._rl_damping = pipeline._joint_default_damping = [2] * 16
        for mode in ControlMode:
            pipeline.mode = ControlMode.RL_DEFAULT
            pipeline.env._observer_eligible = True
            pipeline.env._observer.observer.effort[:] = 0.5
            pipeline.env.aimdk.effort[:] = 0.5
            pipeline._enter_mode(mode, force=True)
            self.assertFalse(pipeline.env._observer_eligible)
            self.assertIsNone(pipeline.env._observer.observer.previous_position)
            np.testing.assert_array_equal(pipeline.env.aimdk.effort, 0)

    def test_mimic_request_clears_effort_in_the_same_control_cycle(self):
        from robojudo.pipeline.rl_loco_mimic_pipeline import PolicyInterpManager
        from robojudo.pipeline.x2_locomanipulation_loco_mimic_pipeline import X2LocomanipulationLocoMimicPipeline
        from robojudo.policy.x2_locomanipulation_policy import X2LocomanipulationPolicy

        pipeline = X2LocomanipulationLocoMimicPipeline.__new__(X2LocomanipulationLocoMimicPipeline)
        pipeline.env = environment()
        pipeline.mode = ControlMode.RL_DEFAULT
        pipeline._set_upper_body_enabled = Mock()
        inner = X2LocomanipulationPolicy.__new__(X2LocomanipulationPolicy)
        pipeline._inner_policy = lambda: inner
        manager = SimpleNamespace(
            current_policy_id=0,
            policy_loco_id=0,
            interp_state=PolicyInterpManager.InterpState.IDLE,
            InterpState=PolicyInterpManager.InterpState,
        )
        pipeline.policy_manager = manager
        pipeline._sync_observer_eligibility()
        pipeline.env.aimdk.effort[:] = 0.5
        pipeline.env._observer.observer.effort[:] = 0.5

        def switch():
            manager.interp_state = PolicyInterpManager.InterpState.START
            return True

        manager.switch_to_mimic = switch
        pipeline._process_policy_commands(["[POLICY_MIMIC]"], {})
        self.assertFalse(pipeline.env._observer_eligible)
        np.testing.assert_array_equal(pipeline.env.aimdk.effort, 0)
        self.assertIsNone(pipeline.env._observer.observer.previous_position)

    def test_only_idle_locomanipulation_family_in_rl_default(self):
        from robojudo.pipeline.rl_loco_mimic_pipeline import PolicyInterpManager
        from robojudo.policy.x2_gr00t_locomanipulation_policy import X2Gr00tLocomanipulationPolicy
        from robojudo.policy.x2_locomanipulation_policy import X2LocomanipulationPolicy

        pipeline = FourModePipelineMixin.__new__(FourModePipelineMixin)
        pipeline.env = environment()
        for policy_class in (X2LocomanipulationPolicy, X2Gr00tLocomanipulationPolicy, object):
            inner = policy_class.__new__(policy_class)
            pipeline._inner_policy = lambda policy=inner: policy
            for mode in ControlMode:
                pipeline.mode = mode
                for state in PolicyInterpManager.InterpState:
                    pipeline.policy_manager = SimpleNamespace(
                        current_policy_id=0,
                        policy_loco_id=0,
                        interp_state=state,
                        InterpState=PolicyInterpManager.InterpState,
                    )
                    pipeline._send_policy_target(np.zeros(16))
                    expected = policy_class is not object and mode == ControlMode.RL_DEFAULT
                    expected = expected and state == PolicyInterpManager.InterpState.IDLE
                    self.assertEqual(pipeline.env._observer_eligible, expected)
            pipeline.mode = ControlMode.RL_DEFAULT
            pipeline.policy_manager.current_policy_id = 1
            pipeline._send_policy_target(np.zeros(16))
            self.assertFalse(pipeline.env._observer_eligible)


if __name__ == "__main__":
    unittest.main()
