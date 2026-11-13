"""Read-only receive probe metrics and SDK process watchdog."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from probe_omnihand_receive import read_hand, run_probe  # noqa: E402


class ReceiveProbeTests(unittest.TestCase):
    def test_read_latency_failures_and_measurement_age_include_pacing(self):
        clock = [0.0]

        class Hand:
            calls = 0

            def get_all_joint_angles(self):
                self.calls += 1
                clock[0] += 0.03
                return [float("nan") if self.calls == 2 else 0.0] * 19

        def sleep(seconds):
            clock[0] += seconds

        with (
            patch("probe_omnihand_receive.time.monotonic", side_effect=lambda: clock[0]),
            patch("probe_omnihand_receive.time.sleep", side_effect=sleep),
        ):
            result = read_hand(Hand(), "left", 0.11, 25, 50)
        self.assertEqual(result["attempts"], 3)
        self.assertEqual(result["valid"], 2)
        self.assertEqual(result["failures"], 1)
        self.assertAlmostEqual(result["valid_gap_ms"]["max"], 80)
        self.assertAlmostEqual(result["previous_measurement_age_ms"]["max"], 110)
        np.testing.assert_allclose([row["read_ms"] for row in result["rows"]], [30, 30, 30])

    def test_spawned_probe_never_writes_and_terminates_stuck_reads(self):
        sdk = """
import time
from types import SimpleNamespace
HandType = SimpleNamespace(LEFT="left", RIGHT="right")
class Hand:
    def __init__(self, side): self.side, self.timeout = side, 200
    def set_frame_recv_timeout(self, value): self.timeout = value
    def get_vendor_info(self):
        return SimpleNamespace(product_seq_num=self.side, product_model="O12")
    def get_device_info(self): return SimpleNamespace(hand_device_id=1)
    def get_all_joint_angles(self):
        if self.timeout == 10: time.sleep(60)
        return [0.] * 19
    def init(self): raise AssertionError("must not initialize motors")
    def set_all_active_joint_angles(self, *args): raise AssertionError("must not write")
class OmniHandPro2025:
    @staticmethod
    def create_hand_by_hcan(**kwargs): return Hand(kwargs["hand_type"])
"""
        config = {
            "left_hcan_serial": "adapter-left",
            "right_hcan_serial": "adapter-right",
            "left_hand_serial": "left",
            "right_hand_serial": "right",
        }
        with tempfile.TemporaryDirectory() as directory, patch.object(sys, "path", [directory, *sys.path]):
            Path(directory, "omnihand.py").write_text(sdk)
            result = run_probe(config, 25, 0.03, 100, 100, 10, 0.2)
            self.assertEqual(result["status"], "ok", result)
            self.assertTrue(all(hand["valid"] > 0 and not hand["failures"] for hand in result["hands"]))
            stuck = run_probe(config, 10, 0.03, 100, 100, 10, 0.2)
            self.assertEqual(stuck["status"], "error")
            self.assertIn("reads did not complete", stuck["error"])


if __name__ == "__main__":
    unittest.main()
