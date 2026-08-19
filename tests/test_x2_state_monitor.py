import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

MONITOR_PATH = Path(__file__).resolve().parents[1] / "scripts" / "monitor_x2_state.py"
MONITOR_SPEC = importlib.util.spec_from_file_location("monitor_x2_state", MONITOR_PATH)
assert MONITOR_SPEC is not None and MONITOR_SPEC.loader is not None
monitor = importlib.util.module_from_spec(MONITOR_SPEC)
MONITOR_SPEC.loader.exec_module(monitor)


def freshness_report(**overrides):
    telemetry = SimpleNamespace(
        topic="/aima/hal/joint/leg/state",
        received_count=123,
        last_receive_age_sec=0.015,
        receive_rate_hz=99.5,
        last_inter_arrival_sec=0.010,
        max_inter_arrival_sec=0.040,
        sequence_gap_count=2,
        sequence_nonmonotonic_count=1,
        last_sequence=456,
        last_header_stamp_sec=100,
        last_header_stamp_nanosec=200,
        last_measurement_stamp_sec=99,
        last_measurement_stamp_nanosec=900,
        last_joint_names=["left_knee_joint"],
    )
    values = {
        "required_streams_fresh": False,
        "reasons": ["imu_stale", "joints_missing", "joints_stale", "odometry_missing"],
        "imu_received": True,
        "imu_age_sec": 0.125,
        "missing_joint_names": ["head_yaw_joint"],
        "stale_joint_names": ["left_knee_joint"],
        "joint_age_sec": {"left_knee_joint": 0.142},
        "odometry_required": True,
        "odometry_received": False,
        "odometry_valid": False,
        "odometry_degenerate": False,
        "odometry_age_sec": None,
        "last_odometry_rejection_reason": "frame_mismatch",
        "last_odometry_rejection_age_sec": 0.02,
        "stream_telemetry": {"leg": telemetry},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TestX2StateMonitor(unittest.TestCase):
    def test_real_config_builds_a_passive_controller(self):
        controller_cfg, cfg_env = monitor.build_controller_config("x2_real")

        self.assertFalse(controller_cfg["act"])
        self.assertTrue(controller_cfg["node_name"].startswith("robojudo_x2_state_monitor_"))
        self.assertFalse(controller_cfg["enable_odometry"])
        self.assertEqual(controller_cfg["joint_names"], cfg_env.dof.joint_names)

    def test_sim_config_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not a real X2"):
            monitor.build_controller_config("x2")

    def test_report_serialization_preserves_timeout_evidence(self):
        serialized = monitor.report_to_dict(freshness_report())

        self.assertEqual(serialized["imu"]["age_sec"], 0.125)
        self.assertEqual(serialized["joints"]["missing"], ["head_yaw_joint"])
        self.assertEqual(serialized["joints"]["age_sec"]["left_knee_joint"], 0.142)
        self.assertEqual(serialized["odometry"]["last_rejection_reason"], "frame_mismatch")
        self.assertEqual(serialized["stream_telemetry"]["leg"]["receive_rate_hz"], 99.5)
        self.assertEqual(serialized["stream_telemetry"]["leg"]["sequence_gap_count"], 2)
        self.assertEqual(
            serialized["stream_telemetry"]["leg"]["measurement_stamp"], {"sec": 99, "nanosec": 900}
        )

    def test_telemetry_summary_includes_rate_gap_and_sequence_diagnostics(self):
        summary = monitor.format_stream_telemetry(freshness_report())

        self.assertIn("leg: 99.5 Hz", summary)
        self.assertIn("age=0.015s", summary)
        self.assertIn("max_gap=0.040s", summary)
        self.assertIn("sequence gaps=2", summary)
        self.assertIn("nonmonotonic=1", summary)

    def test_age_only_changes_do_not_create_state_transitions(self):
        first = freshness_report(imu_age_sec=0.125)
        second = freshness_report(imu_age_sec=0.225)

        self.assertEqual(monitor.report_fingerprint(first), monitor.report_fingerprint(second))

    def test_error_detail_names_each_failed_stream(self):
        detail = monitor.format_state_freshness_report(freshness_report())

        self.assertIn("IMU stale (0.125s old)", detail)
        self.assertIn("joints never received: head_yaw_joint", detail)
        self.assertIn("stale joints: left_knee_joint (0.142s)", detail)
        self.assertIn("odometry never accepted", detail)
        self.assertIn("last odometry rejection: frame_mismatch (0.020s ago)", detail)

    def test_environment_hard_timeout_reports_cause_without_a_second_damping_command(self):
        from robojudo.environment.agibot_cpp_env import AgiBotCppEnv

        report = freshness_report(
            reasons=["imu_stale"],
            missing_joint_names=[],
            stale_joint_names=[],
            odometry_required=False,
            last_odometry_rejection_reason="",
            last_odometry_rejection_age_sec=None,
        )
        backend = SimpleNamespace(
            damping=None,
            get_state_freshness_report=lambda *timeouts: report,
            set_damping=lambda damping: setattr(backend, "damping", damping),
        )
        env = AgiBotCppEnv.__new__(AgiBotCppEnv)
        env.enabled = True
        env.aimdk = backend
        env.cfg_env = SimpleNamespace(
            aimdk=SimpleNamespace(
                state_timeout=0.1,
                state_damping_timeout=0.1,
                odometry_damping_timeout=0.1,
                shutdown_damping=5.0,
            )
        )

        with self.assertRaisesRegex(RuntimeError, r"IMU stale \(0.125s old\).+damping is latched"):
            env.update()

        self.assertIsNone(backend.damping)

    def test_environment_surfaces_a_latched_command_watchdog(self):
        from robojudo.environment.agibot_cpp_env import AgiBotCppEnv

        fresh = freshness_report(
            required_streams_fresh=True,
            reasons=[],
            missing_joint_names=[],
            stale_joint_names=[],
            odometry_required=False,
            last_odometry_rejection_reason="",
            last_odometry_rejection_age_sec=None,
        )
        backend = SimpleNamespace(
            get_state_freshness_report=lambda *timeouts: fresh,
            get_safety_status=lambda: SimpleNamespace(state="DAMPING", fault="COMMAND_TIMEOUT", latched=True),
        )
        env = AgiBotCppEnv.__new__(AgiBotCppEnv)
        env.enabled = True
        env.aimdk = backend
        env.cfg_env = SimpleNamespace(
            aimdk=SimpleNamespace(state_timeout=0.1, state_damping_timeout=0.5, odometry_damping_timeout=0.5)
        )

        with self.assertRaisesRegex(RuntimeError, r"damping is latched \(COMMAND_TIMEOUT\).+re-arm"):
            env.update()

    def test_environment_allows_backend_hold_to_publish_without_forcing_damping(self):
        from robojudo.environment.agibot_cpp_env import AgiBotCppEnv

        fresh = freshness_report(
            required_streams_fresh=True,
            reasons=[],
            missing_joint_names=[],
            stale_joint_names=[],
            odometry_required=False,
            last_odometry_rejection_reason="",
            last_odometry_rejection_age_sec=None,
        )
        state = SimpleNamespace(
            motor_state=SimpleNamespace(q=[0.0], dq=[0.0]),
            imu_state=SimpleNamespace(
                quaternion=[0.0, 0.0, 0.0, 1.0],
                gyroscope=[0.0, 0.0, 0.0],
                accelerometer=[0.0, 0.0, 0.0],
            ),
        )
        backend = SimpleNamespace(
            get_state_freshness_report=lambda *timeouts: fresh,
            get_safety_status=lambda: SimpleNamespace(state="HOLD", fault="COMMAND_TIMEOUT", latched=False),
            get_robot_state=lambda: state,
        )
        env = AgiBotCppEnv.__new__(AgiBotCppEnv)
        env.enabled = True
        env.aimdk = backend
        env.cfg_env = SimpleNamespace(
            aimdk=SimpleNamespace(state_timeout=0.1, state_damping_timeout=0.5, odometry_damping_timeout=0.5)
        )
        env.born_place_align = False
        env._odometry_type = "NONE"
        env.update_with_fk = False

        env.update()

        self.assertEqual(env._last_safety_state, "HOLD")
        self.assertEqual(env._dof_pos.tolist(), [0.0])

    def test_startup_recheck_accepts_state_that_recovered_after_timeout(self):
        from robojudo.environment.agibot_cpp_env import AgiBotCppEnv

        backend = SimpleNamespace(
            self_check=lambda: False,
            get_state_freshness_report=lambda timeout: freshness_report(
                required_streams_fresh=True,
                reasons=[],
                missing_joint_names=[],
                stale_joint_names=[],
                odometry_required=False,
                last_odometry_rejection_reason="",
                last_odometry_rejection_age_sec=None,
            ),
        )
        env = AgiBotCppEnv.__new__(AgiBotCppEnv)
        env.aimdk = backend
        env._odometry_type = "NONE"
        env.cfg_env = SimpleNamespace(aimdk=SimpleNamespace(state_timeout=0.1))

        env.self_check()

    def test_existing_output_requires_an_explicit_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "state.jsonl"
            output_path.write_text("original\n", encoding="utf-8")

            with self.assertRaisesRegex(SystemExit, "--append or --overwrite"):
                monitor.open_output_file(output_path)

            self.assertEqual(output_path.read_text(encoding="utf-8"), "original\n")

    def test_output_append_and_overwrite_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "state.jsonl"
            output_path.write_text("old\n", encoding="utf-8")

            with monitor.open_output_file(output_path, append=True) as output:
                output.write("new\n")
            self.assertEqual(output_path.read_text(encoding="utf-8"), "old\nnew\n")

            with monitor.open_output_file(output_path, overwrite=True) as output:
                output.write("replacement\n")
            self.assertEqual(output_path.read_text(encoding="utf-8"), "replacement\n")


if __name__ == "__main__":
    unittest.main()
