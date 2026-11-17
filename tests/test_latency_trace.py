"""Diagnostic tracing stays bounded and cannot manufacture a latency improvement."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robojudo.utils.latency_trace import LatencyTrace, command_stamps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_xr_latency_trace import analyze  # noqa: E402


class LatencyTraceTests(unittest.TestCase):
    def test_package_copies_match(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(
            (root / "robojudo/utils/latency_trace.py").read_bytes(),
            (root / "third_party/xrobotoolkit_teleop/xrobotoolkit_teleop/common/latency_trace.py").read_bytes(),
        )

    def test_disabled_trace_has_no_clock_or_export_io(self):
        with patch("robojudo.utils.latency_trace.clock_domain") as clock, patch.object(Path, "write_text") as write:
            trace = LatencyTrace()
            trace.record("ignored")
            trace.export()
        clock.assert_not_called()
        write.assert_not_called()
        self.assertEqual(trace.records, [])

    def test_delayed_window_overflow_and_export_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory, patch("time.monotonic", return_value=10.0) as clock:
            path = Path(directory) / "trace.json"
            trace = LatencyTrace(path, duration_s=1.0, delay_s=1.0)
            trace.MAX_RECORDS = 2
            trace.record("too_early")
            clock.return_value = 11.1
            trace.record("first")
            trace.record("second")
            trace.record("overflow")
            self.assertFalse(path.exists())
            self.assertEqual(len(trace.records), 2)
            self.assertTrue(trace.overflow)
            self.assertFalse(trace.snapshot()["window_complete"])
            clock.return_value = 12.1
            trace.record("too_late")
            trace.export()
            saved = json.loads(path.read_text())
            self.assertTrue(saved["window_complete"])
            self.assertTrue(saved["overflow"])

    def test_stamp_whitelist_excludes_unbounded_and_invalid_values(self):
        self.assertEqual(
            command_stamps(
                {"received": 10.0, "extra": [0] * 100, "acknowledged": float("nan"), "left_write_completed": True}
            ),
            {"received": 10.0},
        )


class TraceAnalysisTests(unittest.TestCase):
    def fixture(self):
        domain = dict(clock="monotonic", boot_id="boot", time_namespace="time:[1]")
        base = dict(
            version=1, clock_domain=domain, started_at=0.0, duration_s=30.0, window_complete=True, overflow=False
        )
        key = dict(session_id="robot", generation=1, bridge_id="bridge", frame_id=1, state_id=11)
        robot = dict(
            base,
            records=[
                dict(
                    key,
                    event="robot_acknowledged",
                    at=10.008,
                    stamps={
                        "received": 10.002,
                        "left_write_completed": 10.006,
                        "right_write_completed": 10.007,
                        "arm_submission_completed": 10.005,
                    },
                ),
                dict(key, event="feedback_send", at=10.009, accepted=True),
            ],
        )
        bridge = dict(
            base,
            records=[
                dict(key, event="command_published", at=10.001, sampled_at=10.0),
                dict(key, event="feedback_received", at=10.010, valid=True),
                dict(key, event="feedback_received", at=10.020, state_id=12, valid=True),
                dict(key, event="completion_observed", at=10.020, metric_at=10.020, state_id=12),
            ],
        )
        return robot, bridge

    def test_separates_write_latency_from_skipped_early_snapshot(self):
        result = analyze(*self.fixture())
        self.assertTrue(result["eligible_for_comparison"])
        self.assertAlmostEqual(result["metrics"]["sample_to_write_completion_ms"]["mean"], 7)
        self.assertAlmostEqual(result["metrics"]["first_receive_to_observation_ms"]["mean"], 10)
        self.assertEqual(result["evidence"]["earlier_valid_snapshot_skipped"], 1)
        self.assertEqual(result["evidence"]["first_ack_absent"], 0)

    def test_absent_first_ack_is_not_misreported_as_bridge_selection(self):
        robot, bridge = self.fixture()
        bridge["records"].pop(1)
        result = analyze(robot, bridge)
        self.assertEqual(result["evidence"]["first_ack_absent"], 1)
        self.assertEqual(result["evidence"]["earlier_valid_snapshot_skipped"], 0)

    def test_invalid_first_ack_is_distinguished_from_packet_absence(self):
        robot, bridge = self.fixture()
        bridge["records"][1]["valid"] = False
        result = analyze(robot, bridge)
        self.assertEqual(result["evidence"]["first_ack_received_invalid"], 1)
        self.assertEqual(result["evidence"]["first_ack_absent"], 0)

    def test_incomplete_overflowed_or_cross_host_runs_are_not_comparable(self):
        for field, value in (("window_complete", False), ("overflow", True), ("clock_domain", None)):
            robot, bridge = self.fixture()
            bridge[field] = value
            result = analyze(robot, bridge)
            self.assertFalse(result["eligible_for_comparison"])
            if field == "clock_domain":
                self.assertEqual(result["metrics"]["sample_to_write_completion_ms"], {"n": 0})

    def test_missing_write_stamps_cannot_substitute_acknowledgement_time(self):
        robot, bridge = self.fixture()
        del robot["records"][0]["stamps"]["left_write_completed"]
        result = analyze(robot, bridge)
        self.assertFalse(result["eligible_for_comparison"])
        self.assertEqual(result["evidence"]["missing_write_stamps"], 1)
        self.assertEqual(result["metrics"]["sample_to_write_completion_ms"], {"n": 0})

    def test_earlier_observation_does_not_manufacture_faster_write_completion(self):
        robot, bridge = self.fixture()
        before = analyze(robot, bridge)
        bridge["records"][-1].update(at=10.011, metric_at=10.010, state_id=11)
        after = analyze(robot, bridge)
        self.assertEqual(
            before["metrics"]["sample_to_write_completion_ms"], after["metrics"]["sample_to_write_completion_ms"]
        )
        self.assertGreater(
            before["metrics"]["first_receive_to_observation_ms"]["p95"],
            after["metrics"]["first_receive_to_observation_ms"]["p95"],
        )

    def test_older_trace_cannot_claim_actual_diagnostic_processing_time(self):
        robot, bridge = self.fixture()
        del bridge["records"][-1]["metric_at"]
        result = analyze(robot, bridge)
        self.assertEqual(result["metrics"]["first_receive_to_observation_ms"], {"n": 0})
        self.assertAlmostEqual(result["metrics"]["first_receive_to_metric_timestamp_ms"]["mean"], 10)

    def test_frame_identity_mismatch_cannot_join_unrelated_commands(self):
        robot, bridge = self.fixture()
        bridge["records"][0]["frame_id"] = 2
        result = analyze(robot, bridge)
        self.assertFalse(result["eligible_for_comparison"])
        self.assertEqual(result["evidence"]["matched_frames"], 0)

    def test_negative_cross_process_interval_invalidates_result(self):
        robot, bridge = self.fixture()
        bridge = copy.deepcopy(bridge)
        bridge["records"][0]["sampled_at"] = 10.0075
        result = analyze(robot, bridge)
        self.assertFalse(result["eligible_for_comparison"])


class FeedbackDecisionTests(unittest.TestCase):
    def rows(self, gain=0.0, throughput=50.0, resets=0):
        rows = []
        for observe in (False, True):
            rows.append(
                {
                    "workload": "holding",
                    "collision_checking": False,
                    "trial": 1,
                    "observe_all_completions": observe,
                    "completed_hz": throughput if observe else 50.0,
                    "command_counts": {"rejected": 0},
                    "correlation": {
                        "eligible_for_comparison": True,
                        "evidence": {"first_ack_absent": 0, "generation_resets": resets if observe else 0},
                        "metrics": {"sample_to_write_completion_ms": {"p95": 20.0 - gain if observe else 20.0}},
                    },
                }
            )
        return rows

    def test_reporting_only_improvement_does_not_recommend_adoption(self):
        from benchmark_xr_executor import feedback_decision

        decision = feedback_decision(self.rows())
        self.assertFalse(decision["adoption_recommended"])
        self.assertFalse(decision["buffer_experiment_justified"])

    def test_response_gain_requires_throughput_and_generation_guards(self):
        from benchmark_xr_executor import feedback_decision

        self.assertTrue(feedback_decision(self.rows(gain=6.0))["adoption_recommended"])
        self.assertFalse(feedback_decision(self.rows(gain=6.0, throughput=40.0))["adoption_recommended"])
        self.assertFalse(feedback_decision(self.rows(gain=6.0, resets=1))["adoption_recommended"])

    def test_incomplete_pair_is_not_evidence_for_adoption(self):
        from benchmark_xr_executor import feedback_decision

        self.assertFalse(feedback_decision(self.rows(gain=6.0)[:1])["adoption_recommended"])
