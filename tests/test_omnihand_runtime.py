"""Shared OmniHand runtime contracts for dex-teleop and checked XR callers."""

import multiprocessing as mp
import queue
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from robojudo.controller.ctrl_cfgs import OmniHandCfg
from robojudo.controller.omnihand_runtime import (
    OMNIHAND_JOINT_NAMES,
    OMNIHAND_LEFT_LIMITS,
    OMNIHAND_RIGHT_LIMITS,
    OmniHandRuntime,
    _CommandMailbox,
    create_hand,
)


class TestOmniHandConfig(unittest.TestCase):
    def test_timeouts_and_rates_must_be_finite(self):
        for field in ("command_timeout_s", "joint_state_fps", "joint_state_timeout_s", "startup_timeout_s"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                OmniHandCfg(**{field: float("inf")})

    def test_hcan_exposes_adapter_indices_without_channel_configuration(self):
        cfg = OmniHandCfg(transport="hcan", left_adapter_index=1, right_adapter_index=0)
        self.assertEqual(cfg.transport, "hcan")
        self.assertFalse(hasattr(cfg, "left_channel_id"))

    def test_socketcan_interfaces_must_not_be_empty(self):
        with self.assertRaisesRegex(ValueError, "interfaces"):
            OmniHandCfg(transport="socketcan", left_interface=" ")


class TestOmniHandRuntime(unittest.TestCase):
    def test_command_mailbox_retains_latest_frame_during_producer_burst(self):
        self.runtime.set_takeover_enabled(True)
        for frame_id in range(100):
            self.runtime.set_joint_commands(np.zeros(12), np.zeros(12), 123, frame_id)
        message = self.runtime._commands.get_nowait()
        self.assertEqual(message[0], self.runtime._gate[0])
        self.assertEqual(message[2][1], 99)
        self.assertEqual(len(message[3]), 24)
        with self.assertRaises(queue.Empty):
            self.runtime._commands.get_nowait()

    def setUp(self):
        self.clock = self.enterContext(patch("time.monotonic", return_value=10.0))
        runtime = OmniHandRuntime.__new__(OmniHandRuntime)
        runtime.cfg = OmniHandCfg()
        runtime.read_only = False
        runtime._gate = mp.get_context("spawn").Array("q", [0, 0, 0])
        runtime._lock = threading.RLock()
        runtime._shutdown = threading.Event()
        runtime._requested = False
        runtime._healthy = True
        runtime._attempt_ready = True
        runtime._seen_outages = 0
        runtime._measurements = [10.0, 10.0]
        runtime._stream_generation = 1
        runtime._connection_generation = 1
        runtime._status = "streaming"
        runtime._last_error = None
        runtime._initial_targets = None
        runtime._pending = None
        runtime._sequence = 0
        runtime._commands = _CommandMailbox(mp.get_context("spawn"))
        runtime._stop = mp.get_context("spawn").Event()
        runtime._closed = False
        runtime._error = None
        runtime._states = {
            side: {
                "positions": [0.0] * 19,
                "sampled_at": 10.0,
                "targets": None,
                "applied": None,
                "commanded_at": None,
                "generation": None,
            }
            for side in ("left", "right")
        }
        self.runtime = runtime
        self.positions = dict.fromkeys(OMNIHAND_JOINT_NAMES, 0.0)

    def test_disabled_dex_frames_do_not_enqueue_hardware_work(self):
        self.runtime.set_joint_commands(np.zeros(12), np.zeros(12), 123, 1)
        with self.assertRaises(queue.Empty):
            self.runtime._commands.get_nowait()
        self.assertFalse(self.runtime.get_data()["fresh"])
        self.assertTrue(self.runtime.get_data()["joint_state_fresh"])

    def test_dex_frames_keep_newest_clipped_targets_and_source_identity(self):
        self.runtime.set_takeover_enabled(True)
        clipped = self.runtime.set_joint_commands(np.full(12, 100.0), np.full(12, -100.0), 10, 1)
        np.testing.assert_allclose(clipped[:12], OMNIHAND_LEFT_LIMITS[:, 1])
        np.testing.assert_allclose(clipped[12:], OMNIHAND_RIGHT_LIMITS[:, 0])
        self.runtime.set_joint_commands(np.zeros(12), np.zeros(12), 20, 2)
        frame = self.runtime._commands.get_nowait()
        self.assertEqual(frame[2][1:], (2, 20))
        self.assertEqual(frame[3], [0.0] * 24)

    def test_checked_frames_reject_invalid_targets_and_multiple_pending_commands(self):
        self.runtime.set_takeover_enabled(True)
        bad = {**self.positions, "L_index_mcp_joint": 100.0}
        with self.assertRaisesRegex(ValueError, "limits"):
            self.runtime.submit(bad, (1, 1), 10.1)
        self.runtime.submit(self.positions, (1, 2), 10.1)
        with self.assertRaisesRegex(RuntimeError, "pending"):
            self.runtime.submit(self.positions, (1, 3), 10.1)

    def test_expired_or_disabled_checked_frames_are_rejected(self):
        with self.assertRaises(RuntimeError):
            self.runtime.submit(self.positions, (1, 1), 10.1)
        self.runtime.set_takeover_enabled(True)
        with self.assertRaises(RuntimeError):
            self.runtime.submit(self.positions, (1, 1), 9.9)

    def test_disable_revokes_pending_generation_and_applied_identity(self):
        self.runtime.set_takeover_enabled(True)
        self.runtime.submit(self.positions, (1, 1), 10.1)
        generation = self.runtime._gate[0]
        for state in self.runtime._states.values():
            state.update(applied=(1, 7, 123), commanded_at=10.0, generation=generation, targets=[0.0] * 12)
        self.assertEqual(self.runtime.get_data()["applied_frame_id"], 7)
        self.runtime.reset()
        self.runtime.set_takeover_enabled(True)
        result = self.runtime.get_data()
        self.assertIsNone(self.runtime._pending)
        self.assertIsNone(result["applied_frame_id"])
        self.assertFalse(result["left_fresh"])
        self.assertFalse(result["fresh"])

    def test_command_leases_and_sample_ages_remain_independent(self):
        self.runtime.set_takeover_enabled(True)
        for state in self.runtime._states.values():
            state.update(applied=(1, 7, 123), commanded_at=10.0, generation=self.runtime._gate[0], targets=[0.0] * 12)
        result = self.runtime.get_data()
        self.assertTrue(result["fresh"])
        self.assertEqual(result["applied_source_timestamp_ns"], 123)
        self.clock.return_value = 10.3
        for state in self.runtime._states.values():
            state["sampled_at"] = 10.3
        self.runtime._measurements[:] = [10.3, 10.3]
        result = self.runtime.get_data()
        self.assertFalse(result["fresh"])
        self.assertTrue(result["joint_state_fresh"])

    def test_snapshot_expiry_revokes_generation_between_supervisor_ticks(self):
        runtime = self.runtime
        runtime._healthy = True
        runtime._stream_generation = 1
        runtime._status = "streaming"
        runtime.set_takeover_enabled(True)
        generation = runtime._gate[0]
        self.clock.return_value = 10.3
        result = runtime.get_data()
        self.assertFalse(result["joint_state_fresh"])
        self.assertFalse(result["enabled"])
        self.assertEqual(result["stream_generation"], 2)
        self.assertGreater(runtime._gate[0], generation)
        self.assertTrue(runtime.takeover_requested)

    def test_invalid_dex_arrays_are_rejected(self):
        for left in ([0.0] * 11, [float("nan")] * 12):
            with self.assertRaisesRegex(ValueError, "12 finite"):
                self.runtime.set_joint_commands(left, np.zeros(12))

    def test_expiry_at_submission_is_recoverable_and_preserves_takeover_request(self):
        runtime = self.runtime
        runtime.set_takeover_enabled(True)
        self.clock.return_value = 10.3
        with self.assertRaisesRegex(ValueError, "measurements are stale"):
            runtime.submit(self.positions, (1, 1), 10.5)
        self.assertEqual(runtime._gate[1], 0)
        self.assertTrue(runtime.takeover_requested)
        self.assertIsNone(runtime._pending)

    def test_xr_can_require_serials_without_removing_index_support(self):
        with self.assertRaisesRegex(ValueError, "serial"):
            OmniHandRuntime(OmniHandCfg(), require_serials=True)

    def test_latched_fault_cannot_reenable_commands(self):
        runtime = self.runtime
        runtime.set_takeover_enabled(True)
        runtime._error = "right: read measurements: SDK failure"
        runtime._set_health(False)
        runtime._stop.set()
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "right: read measurements: SDK failure"):
                OmniHandRuntime._poll(runtime)
            self.assertEqual(runtime._gate[1], 0)
            runtime.set_takeover_enabled(True)
            self.assertEqual(runtime._gate[1], 0)

    def test_cleanup_handles_process_start_failure(self):
        runtime = self.runtime
        process = runtime._process = Mock(pid=None)
        runtime._state_queue = Mock()
        runtime._reap_worker()
        process.join.assert_not_called()
        process.terminate.assert_not_called()
        runtime._state_queue.close.assert_called_once()
        self.assertIsNone(runtime._process)


class TestOmniHandTransports(unittest.TestCase):
    def test_existing_transport_factory_arguments_remain_supported(self):
        hand = Mock()
        hand.init.return_value = True
        vendor = SimpleNamespace(
            kDefaultHandDeviceId=1,
            create_hand_by_hcan=Mock(return_value=hand),
            create_hand_by_zlgcan=Mock(return_value=hand),
            create_hand_socketcan=Mock(return_value=hand),
        )
        module = SimpleNamespace(HandType=SimpleNamespace(LEFT=1, RIGHT=2), OmniHandPro2025=vendor)
        with patch.dict(sys.modules, omnihand=module):
            cfg = OmniHandCfg(left_adapter_index=4, right_adapter_index=5)
            create_hand(cfg, "left")
            create_hand(cfg, "right")
            self.assertEqual(
                [call.kwargs["canfd_device_id"] for call in vendor.create_hand_by_hcan.call_args_list], [4, 5]
            )
            self.assertEqual(
                [call.kwargs["canfd_channel_id"] for call in vendor.create_hand_by_hcan.call_args_list], [0, 0]
            )
            create_hand(OmniHandCfg(transport="zlgcan"), "right")
            self.assertEqual(vendor.create_hand_by_zlgcan.call_args.kwargs["canfd_channel_id"], 1)
            create_hand(OmniHandCfg(transport="socketcan", right_interface="can8"), "right")
            self.assertEqual(vendor.create_hand_socketcan.call_args.kwargs["can_interface"], "can8")
        self.assertEqual(hand.init.call_count, 4)
