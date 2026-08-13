import importlib.util
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "stress_x2_state_subscribers.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("stress_x2_state_subscribers", SCRIPT_PATH)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
stress = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(stress)


class TestX2StateSubscriberStress(unittest.TestCase):
    def test_defaults_match_latest_sample_state_subscribers(self):
        args = stress.parse_args([])

        self.assertEqual(args.processes, 4)
        self.assertEqual(args.copies_per_topic, 1)
        self.assertEqual(args.qos_depth, 1)
        self.assertEqual(args.gap_threshold, 0.1)

    def test_metrics_distinguish_delivery_gap_from_header_progress(self):
        metrics = stress.StreamMetrics()

        self.assertIsNone(metrics.observe(1_000_000_000, 10_000_000_000, 0.1))
        event = metrics.observe(1_200_000_000, 10_002_000_000, 0.1)

        self.assertEqual(event, {"receive_gap_sec": 0.2, "header_gap_sec": 0.002})
        self.assertEqual(metrics.receive_gap_count, 1)
        self.assertEqual(metrics.header_gap_count, 0)
        self.assertAlmostEqual(metrics.summary()["receive_rate_hz"], 5.0)

    def test_metrics_capture_source_and_receive_gap_together(self):
        metrics = stress.StreamMetrics()
        metrics.observe(1_000_000_000, 10_000_000_000, 0.1)

        event = metrics.observe(1_250_000_000, 10_240_000_000, 0.1)

        self.assertEqual(event, {"receive_gap_sec": 0.25, "header_gap_sec": 0.24})
        self.assertEqual(metrics.receive_gap_count, 1)
        self.assertEqual(metrics.header_gap_count, 1)

    def test_each_stream_uses_an_explicit_callback_group(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("MutuallyExclusiveCallbackGroup", source)
        self.assertIn("callback_groups[stream]", source)
        self.assertIn("callback_group=callback_groups[stream]", source)


if __name__ == "__main__":
    unittest.main()
