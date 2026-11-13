"""Measurement recovery and native-worker supervision without robot hardware."""

import multiprocessing as mp
import queue
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from robojudo.controller.ctrl_cfgs import OmniHandCfg
from robojudo.controller.omnihand_runtime import OMNIHAND_JOINT_NAMES, OmniHandRuntime, _hand_loop, _worker


class MeasurementRecoveryTests(unittest.TestCase):
    def test_reconnect_configuration_rejects_invalid_budgets(self):
        for field in ("reconnect_timeout_s", "reconnect_interval_s"):
            for value in (0, -1, float("inf"), float("nan")):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    OmniHandCfg(**{field: value})
        with self.assertRaisesRegex(ValueError, "at least"):
            OmniHandCfg(reconnect_timeout_s=0.1)

    def test_rejected_reads_keep_timestamp_and_recover(self):
        for invalid in (
            [],
            [0.0] * 18,
            [float("nan")] * 19,
            [float("inf")] * 19,
            ["bad"] * 19,
            OSError("receive timeout"),
        ):
            with self.subTest(invalid=str(invalid)[:30]):
                stop, commands = threading.Event(), queue.Queue()
                hand, fail = Mock(), Mock()
                hand.get_all_joint_angles.side_effect = [[0.0] * 19, invalid, [0.1] * 19]
                gate = mp.get_context("spawn").Array("q", [0, 0])
                states = []
                clock = [10.0]

                def wait(_, clock=clock, states=states, stop=stop):
                    clock[0] += 0.01
                    if len(states) == 3:
                        stop.set()

                with (
                    patch("time.monotonic", side_effect=lambda clock=clock: clock[0]),
                    patch.object(stop, "wait", side_effect=wait),
                ):
                    _hand_loop(
                        OmniHandCfg(),
                        "right",
                        hand,
                        commands,
                        stop,
                        gate,
                        lambda side, state, states=states: states.append(state),
                        fail,
                    )
                fail.assert_not_called()
                self.assertEqual(hand.get_all_joint_angles.call_count, 3)
                self.assertEqual(states[1]["positions"], states[0]["positions"])
                self.assertEqual(states[1]["sampled_at"], states[0]["sampled_at"])
                self.assertEqual(states[1]["consecutive_failures"], 1)
                self.assertGreater(states[2]["sampled_at"], states[1]["sampled_at"])
                self.assertEqual(states[2]["consecutive_failures"], 0)
                self.assertIsNone(states[2]["last_read_error"])

    def test_stale_other_hand_prevents_write_at_dispatch(self):
        stop = threading.Event()
        gate = mp.get_context("spawn").Array("q", [1, 1])
        measurements = [10.0, 9.0]
        commands = queue.Queue()
        commands.put((1, 11.0, (1, 1), [0.0] * 12, 10.0))
        hand = Mock()
        hand.get_all_joint_angles.return_value = [0.0] * 19
        with patch("time.monotonic", return_value=10.0), patch.object(stop, "wait", side_effect=lambda _: stop.set()):
            _hand_loop(OmniHandCfg(), "left", hand, commands, stop, gate, Mock(), Mock(), measurements)
        hand.set_all_active_joint_angles.assert_not_called()

    def test_worker_watchdog_revokes_gate_while_both_reads_block(self):
        stop, release = threading.Event(), threading.Event()
        gate = mp.get_context("spawn").Array("q", [1, 1, 0])
        measurements = mp.get_context("spawn").Array("d", [1.0, 1.0], lock=False)
        hand = Mock()
        hand.get_all_joint_angles.side_effect = lambda: (release.wait(2), [0.0] * 19)[1]
        with patch("robojudo.controller.omnihand_runtime.create_hand", return_value=hand):
            worker = threading.Thread(
                target=_worker,
                args=(OmniHandCfg(), queue.Queue(), queue.Queue(), stop, gate, queue.Queue(), measurements),
            )
            worker.start()
            try:
                deadline = time.monotonic() + 1
                while gate[1] and time.monotonic() < deadline:
                    time.sleep(0.005)
                self.assertEqual(gate[1], 0)
                self.assertGreater(gate[0], 1)
            finally:
                stop.set()
                release.set()
                worker.join(2)
            self.assertFalse(worker.is_alive())


# The control file injects failures into a real spawned SDK owner. The write
# journal detects stale-command replay and accidental writes during reconnect.
SDK = """
import os
import time
from pathlib import Path
from types import SimpleNamespace
HandType = SimpleNamespace(LEFT="left", RIGHT="right")
root = Path(__file__).parent
class Hand:
    def __init__(self, side): self.side, self.reads = side, 0
    def set_frame_recv_timeout(self, timeout): pass
    def get_vendor_info(self):
        serial = self.side
        if (root / "mode").read_text() == "identity": serial = "wrong"
        return SimpleNamespace(product_model="O12", product_seq_num=serial)
    def get_device_info(self): return SimpleNamespace(hand_device_id=1)
    def get_all_joint_angles(self):
        self.reads += 1
        mode = (root / "mode").read_text()
        if self.side == "right":
            if mode == "startup" and self.reads == 1: return []
            if mode == "blocked": time.sleep(60)
            if mode == "invalid": return []
            if mode == "exception": raise OSError("receive failed")
        return [0.] * 19
    def set_all_active_joint_angles(self, angles):
        if (root / "mode").read_text() == "write": raise OSError("write failed")
        with (root / "writes").open("a") as output:
            output.write(f"{os.getpid()} {self.side} {angles[5]}\\n")
    def init(self): raise AssertionError("verified reconnect must not initialize motors")
class OmniHandPro2025:
    @staticmethod
    def create_hand_by_hcan(**kwargs):
        if (root / "mode").read_text() == "unavailable": raise OSError("USB unavailable")
        return Hand(kwargs["hand_type"])
"""


class SupervisedRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.directory = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(self.directory)
        (self.root / "omnihand.py").write_text(SDK)
        self.mode("ok")
        self.enterContext(patch.object(sys, "path", [self.directory, *sys.path]))

    def mode(self, value):
        # Atomic replacement prevents the SDK seeing a partially written mode.
        temporary = self.root / "next_mode"
        temporary.write_text(value)
        temporary.replace(self.root / "mode")

    def runtime(self, **kwargs):
        cfg = OmniHandCfg(
            left_hcan_serial="adapter-left",
            right_hcan_serial="adapter-right",
            left_hand_serial="left",
            right_hand_serial="right",
            joint_state_timeout_s=0.08,
            reconnect_timeout_s=0.2,
            reconnect_interval_s=0.05,
            startup_timeout_s=4.0,
            **kwargs,
        )
        result = OmniHandRuntime(cfg)
        self.addCleanup(result.close)
        return result

    def until(self, runtime, predicate, timeout=8):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            data = runtime.get_data()
            if predicate(data):
                return data
            time.sleep(0.005)
        self.fail(f"recovery deadline exceeded: {data['connection_status']}")

    def test_stale_stream_recovers_same_worker_and_preserves_requested_takeover(self):
        runtime = self.runtime()
        runtime.set_takeover_enabled(True)
        initial = runtime.get_data()
        self.mode("invalid")
        stale = self.until(runtime, lambda data: not data["joint_state_fresh"])
        self.assertFalse(stale["enabled"])
        self.assertTrue(runtime.takeover_requested)
        self.assertGreater(stale["stream_generation"], initial["stream_generation"])
        runtime.set_joint_commands([0.0] * 12, [0.0] * 12, 123, 55)
        self.assertFalse((self.root / "writes").exists())
        self.mode("ok")
        recovered = self.until(runtime, lambda data: data["joint_state_fresh"] and data["enabled"])
        self.assertEqual(recovered["connection_generation"], initial["connection_generation"])
        self.assertIsNone(recovered["applied"])
        runtime.set_joint_commands([0.0] * 12, [0.0] * 12, 124, 56)
        self.until(runtime, lambda data: data["applied_frame_id"] == 56)

    def test_blocked_read_reconnects_without_replaying_commands(self):
        runtime = self.runtime()
        runtime.set_takeover_enabled(True)
        old_process = runtime._process
        self.mode("blocked")
        self.until(runtime, lambda data: data["connection_status"] == "reconnecting")
        started = time.monotonic()
        for _ in range(100):
            self.assertFalse(runtime.get_data()["joint_state_fresh"])
        self.assertLess(time.monotonic() - started, 0.2)
        self.mode("ok")
        data = self.until(runtime, lambda data: data["connection_generation"] > 1 and data["enabled"])
        self.assertFalse(old_process.is_alive())
        self.assertIsNone(data["applied"])
        self.assertFalse((self.root / "writes").exists())

    def test_reconnect_preserves_last_issued_targets_without_replaying_them(self):
        runtime = self.runtime()
        runtime.set_takeover_enabled(True)
        targets = dict.fromkeys(OMNIHAND_JOINT_NAMES, 0.0)
        targets["L_index_mcp_joint"] = targets["R_index_mcp_joint"] = 0.2
        runtime.submit(targets, (1, 1), time.monotonic() + 1)
        self.until(runtime, lambda data: data["applied"] == (1, 1))
        written = (self.root / "writes").read_text()
        self.mode("invalid")
        self.until(runtime, lambda data: data["connection_status"] == "reconnecting")
        self.mode("ok")
        data = self.until(runtime, lambda data: data["connection_generation"] > 1 and data["enabled"])
        self.assertEqual(data["targets"], targets)
        self.assertIsNone(data["applied"])
        self.assertEqual((self.root / "writes").read_text(), written)

    def test_worker_outage_is_not_lost_when_feedback_recovers_before_parent_poll(self):
        runtime = self.runtime()
        runtime.set_takeover_enabled(True)
        generation = runtime.get_data()["stream_generation"]
        with runtime._lock:
            self.mode("invalid")
            time.sleep(0.12)
            self.assertEqual(runtime._gate[1], 0)
            self.mode("ok")
            time.sleep(0.04)
        # Allow the supervisor to consume the recovered sample before get_data
        # can observe and revoke the old cached measurement itself.
        time.sleep(0.04)
        data = self.until(runtime, lambda data: data["enabled"] and data["joint_state_fresh"])
        self.assertGreater(data["stream_generation"], generation)

    def test_reconnect_identity_mismatch_is_latched(self):
        runtime = self.runtime()
        self.mode("invalid")
        self.until(runtime, lambda data: data["connection_status"] == "reconnecting")
        self.mode("identity")
        with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
            self.until(runtime, lambda _: False)
        runtime.set_takeover_enabled(True)
        self.assertEqual(runtime._gate[1], 0)

    def test_write_exception_faults_instead_of_resuming(self):
        runtime = self.runtime()
        runtime.set_takeover_enabled(True)
        self.mode("write")
        runtime.submit(dict.fromkeys(OMNIHAND_JOINT_NAMES, 0.0), (1, 1), time.monotonic() + 1)
        with self.assertRaisesRegex(RuntimeError, "write failed"):
            self.until(runtime, lambda _: False)
        self.assertEqual(runtime._connection_generation, 1)

    def test_explicit_disable_during_outage_survives_reconnection(self):
        runtime = self.runtime()
        runtime.set_takeover_enabled(True)
        self.mode("invalid")
        self.until(runtime, lambda data: data["connection_status"] == "reconnecting")
        runtime.set_takeover_enabled(False)
        self.mode("ok")
        data = self.until(runtime, lambda data: data["connection_generation"] > 1 and data["joint_state_fresh"])
        self.assertFalse(data["enabled"])
        self.assertFalse(runtime.takeover_requested)

    def test_unexpected_worker_exit_revokes_pending_work_and_reconnects(self):
        runtime = self.runtime()
        runtime.set_takeover_enabled(True)
        previous = runtime._process
        previous.kill()
        previous.join(1)
        data = self.until(runtime, lambda data: data["connection_generation"] > 1 and data["enabled"])
        self.assertIsNone(data["applied"])
        self.assertIsNone(runtime._pending)
        self.assertFalse((self.root / "writes").exists())

    def test_shutdown_during_reconnect_reaps_worker(self):
        runtime = self.runtime()
        self.mode("blocked")
        self.until(runtime, lambda data: data["connection_status"] == "reconnecting")
        runtime.close()
        self.assertFalse(runtime._supervisor.is_alive())
        self.assertIsNone(runtime._process)
        runtime.close()

    def test_startup_tolerates_invalid_first_sample(self):
        self.mode("startup")
        runtime = self.runtime()
        self.assertTrue(runtime.get_data()["joint_state_fresh"])
        self.assertEqual(runtime._connection_generation, 1)

    def test_failed_reconnect_attempts_keep_retrying_until_adapter_returns(self):
        runtime = self.runtime()
        runtime.set_takeover_enabled(True)
        self.mode("invalid")
        self.until(runtime, lambda data: data["connection_status"] == "reconnecting")
        self.mode("unavailable")
        self.until(runtime, lambda data: data["connection_generation"] >= 3)
        self.mode("ok")
        data = self.until(runtime, lambda data: data["enabled"] and data["joint_state_fresh"])
        self.assertGreaterEqual(data["connection_generation"], 3)
        self.assertFalse((self.root / "writes").exists())

    def test_disabled_reconnect_latches_persistent_read_outage(self):
        runtime = self.runtime(auto_reconnect=False)
        self.mode("exception")
        with self.assertRaisesRegex(RuntimeError, "communication deadline"):
            self.until(runtime, lambda _: False)
        self.assertEqual(runtime._connection_generation, 1)

    def test_index_selected_adapters_do_not_reconnect(self):
        sdk = SDK.replace(
            'def init(self): raise AssertionError("verified reconnect must not initialize motors")',
            "def init(self): return True",
        ).replace("class OmniHandPro2025:", "class OmniHandPro2025:\n    kDefaultHandDeviceId = 1")
        (self.root / "omnihand.py").write_text(sdk)
        runtime = OmniHandRuntime(
            OmniHandCfg(joint_state_timeout_s=0.08, reconnect_timeout_s=0.2, startup_timeout_s=4.0)
        )
        self.addCleanup(runtime.close)
        self.mode("invalid")
        with self.assertRaisesRegex(RuntimeError, "communication deadline"):
            self.until(runtime, lambda _: False)
        self.assertEqual(runtime._connection_generation, 1)


if __name__ == "__main__":
    unittest.main()
