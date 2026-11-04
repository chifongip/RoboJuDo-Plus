import queue
import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from robojudo.controller.ctrl_cfgs import OmniHandCfg
from robojudo.controller.omnihand_runtime import (
    OMNIHAND_LEFT_LIMITS,
    OMNIHAND_RIGHT_LIMITS,
    OmniHandRuntime,
)


class ObservedCommandQueue(queue.Queue):
    """Signal when the worker has finished handling its first dequeued frame."""

    def __init__(self):
        super().__init__(maxsize=1)
        self.processed = threading.Event()
        self._dequeued = False

    def get(self, *args, **kwargs):
        if self._dequeued:
            self.processed.set()
        result = super().get(*args, **kwargs)
        self._dequeued = True
        return result

    def get_nowait(self):
        # Producer-side queue clearing is not a completed worker iteration.
        return super().get(block=False)


class TestOmniHandConfig(unittest.TestCase):
    def test_hcan_exposes_adapter_indices_without_channel_configuration(self):
        cfg = OmniHandCfg(transport="hcan", left_adapter_index=1, right_adapter_index=0)
        self.assertEqual(cfg.transport, "hcan")
        self.assertFalse(hasattr(cfg, "left_channel_id"))

    def test_socketcan_interfaces_must_not_be_empty(self):
        with self.assertRaisesRegex(ValueError, "interfaces"):
            OmniHandCfg(transport="socketcan", left_interface=" ")


class TestOmniHandRuntime(unittest.TestCase):
    def test_disable_during_left_write_cancels_right_and_frame_publication(self):
        entered = threading.Event()
        release = threading.Event()
        sent = []

        class FakeHand:
            def __init__(self, side):
                self.side = side

            def set_all_active_joint_angles(self, positions):
                if self.side == "left":
                    entered.set()
                    if not release.wait(2):
                        raise TimeoutError("test did not release left write")
                sent.append(self.side)

            def get_all_active_joint_angles(self):
                return np.zeros(12)

        runtime = OmniHandRuntime(OmniHandCfg(command_timeout_s=10), hand_factory=lambda _, side: FakeHand(side))
        try:
            commands = ObservedCommandQueue()
            runtime._command_queue = commands
            runtime.set_takeover_enabled(True)
            runtime.set_joint_commands(np.zeros(12), np.zeros(12), 123, 8)
            self.assertTrue(entered.wait(1))
            runtime.set_takeover_enabled(False)
            # A quick re-enable must not revive a frame from the previous takeover.
            runtime.set_takeover_enabled(True)
            release.set()
            self.assertTrue(commands.processed.wait(1))
            self.assertEqual(sent, ["left"])
            data = runtime.get_data()
            self.assertIsNone(data["applied_frame_id"])
            self.assertIsNone(data["applied_source_timestamp_ns"])
            self.assertFalse(data["left_fresh"])
            self.assertFalse(data["right_fresh"])
        finally:
            release.set()
            runtime.close()

    def test_expired_frame_is_dropped_before_either_write(self):
        class FakeHand:
            def get_all_active_joint_angles(self):
                return np.zeros(12)

            def set_all_active_joint_angles(self, positions):
                sent.append(positions)

        sent = []
        # Hold the worker in startup feedback while a frame waits in its queue.
        entered = threading.Event()
        release = threading.Event()
        clock = SimpleNamespace(monotonic=lambda: now[0], time_ns=time.time_ns)
        now = [10.0]
        runtime = OmniHandRuntime.__new__(OmniHandRuntime)
        original_update = OmniHandRuntime._update_measured_joint_positions

        def blocked_update(inst, hands, timestamp):
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test did not release feedback")
            original_update(inst, hands, timestamp)

        # Run construction separately because startup waits for the first feedback.
        errors = []

        def start():
            try:
                runtime.__init__(OmniHandCfg(command_timeout_s=0.03), hand_factory=lambda *_: FakeHand())
            except Exception as exc:
                errors.append(exc)

        with (
            patch("robojudo.controller.omnihand_runtime.time", clock),
            patch.object(OmniHandRuntime, "_update_measured_joint_positions", blocked_update),
        ):
            constructor = threading.Thread(target=start)
            constructor.start()
            try:
                self.assertTrue(entered.wait(1))
                commands = ObservedCommandQueue()
                runtime._command_queue = commands
                runtime.set_takeover_enabled(True)
                runtime.set_joint_commands(np.zeros(12), np.zeros(12), 123, 9)
                now[0] += 0.1
                release.set()
                constructor.join(1)
                self.assertFalse(constructor.is_alive())
                self.assertEqual(errors, [])
                self.assertTrue(commands.processed.wait(1))
                self.assertEqual(sent, [])
                self.assertIsNone(runtime.get_data()["applied_frame_id"])
            finally:
                release.set()
                constructor.join(3)
                runtime.close()

    def test_frame_expiring_during_left_write_cancels_right(self):
        now = [10.0]
        sent = []
        clock = SimpleNamespace(monotonic=lambda: now[0], time_ns=time.time_ns)

        class FakeHand:
            def __init__(self, side):
                self.side = side

            def get_all_active_joint_angles(self):
                return np.zeros(12)

            def set_all_active_joint_angles(self, positions):
                sent.append(self.side)
                now[0] += 0.1

        with patch("robojudo.controller.omnihand_runtime.time", clock):
            runtime = OmniHandRuntime(OmniHandCfg(command_timeout_s=0.03), hand_factory=lambda _, side: FakeHand(side))
            try:
                commands = ObservedCommandQueue()
                runtime._command_queue = commands
                runtime.set_takeover_enabled(True)
                runtime.set_joint_commands(np.zeros(12), np.zeros(12), 123, 9)
                self.assertTrue(commands.processed.wait(1))
                self.assertEqual(sent, ["left"])
                data = runtime.get_data()
                self.assertIsNone(data["applied_frame_id"])
                self.assertFalse(data["fresh"])
            finally:
                runtime.close()

    def test_runtime_applies_controller_queued_dual_hand_frame(self):
        class FakeHand:
            def __init__(self):
                self.positions = np.zeros(12)

            def set_all_active_joint_angles(self, positions):
                self.positions = np.asarray(positions)

            def get_all_active_joint_angles(self):
                return self.positions

        hands = {"left": FakeHand(), "right": FakeHand()}
        runtime = OmniHandRuntime(
            OmniHandCfg(joint_state_fps=100.0),
            hand_factory=lambda _, side: hands[side],
        )
        try:
            runtime.set_takeover_enabled(True)
            runtime.set_joint_commands(np.full(12, 0.1), np.full(12, 0.2), 123, 9)
            deadline = time.monotonic() + 1.0
            data = runtime.get_data()
            while data["applied_frame_id"] != 9 and time.monotonic() < deadline:
                time.sleep(0.005)
                data = runtime.get_data()

            self.assertEqual(data["applied_frame_id"], 9)
            self.assertEqual(data["applied_source_timestamp_ns"], 123)
            np.testing.assert_allclose(
                hands["left"].positions,
                np.clip(0.1, OMNIHAND_LEFT_LIMITS[:, 0], OMNIHAND_LEFT_LIMITS[:, 1]),
            )
            np.testing.assert_allclose(
                hands["right"].positions,
                np.clip(0.2, OMNIHAND_RIGHT_LIMITS[:, 0], OMNIHAND_RIGHT_LIMITS[:, 1]),
            )
        finally:
            runtime.close()

    def test_joint_command_queue_keeps_only_newest_clipped_atomic_frame(self):
        runtime = OmniHandRuntime.__new__(OmniHandRuntime)
        runtime._command_queue = queue.Queue(maxsize=1)
        runtime._lock = threading.Lock()
        runtime._takeover_generation = 0

        first = runtime.set_joint_commands(np.full(12, 100.0), np.full(12, -100.0), 10, 1)
        second = runtime.set_joint_commands(np.zeros(12), np.zeros(12), 20, 2)

        np.testing.assert_allclose(first[:12], OMNIHAND_LEFT_LIMITS[:, 1])
        np.testing.assert_allclose(first[12:], OMNIHAND_RIGHT_LIMITS[:, 0])
        command = runtime._command_queue.get_nowait()
        self.assertEqual(command.frame_id, 2)
        self.assertEqual(command.source_timestamp_ns, 20)
        np.testing.assert_array_equal(second, np.zeros(24))

    def test_hcan_factory_always_uses_channel_zero(self):
        calls = []

        class FakeHand:
            def init(self):
                return True

        class FakeOmniHandPro2025:
            kDefaultHandDeviceId = 7

            @staticmethod
            def create_hand_by_hcan(**kwargs):
                calls.append(kwargs)
                return FakeHand()

        module = SimpleNamespace(
            HandType=SimpleNamespace(LEFT=1, RIGHT=2),
            OmniHandPro2025=FakeOmniHandPro2025,
        )
        cfg = OmniHandCfg(transport="hcan", left_adapter_index=4, right_adapter_index=5)
        with patch.dict(sys.modules, {"omnihand": module}):
            OmniHandRuntime._create_hand(cfg, "left")
            OmniHandRuntime._create_hand(cfg, "right")

        self.assertEqual([call["canfd_device_id"] for call in calls], [4, 5])
        self.assertEqual([call["canfd_channel_id"] for call in calls], [0, 0])

    def test_data_requires_fresh_commands_and_joint_states(self):
        runtime = OmniHandRuntime.__new__(OmniHandRuntime)
        runtime.cfg = OmniHandCfg()
        runtime._error = None
        runtime._lock = __import__("threading").Lock()
        runtime._enabled = True
        now = time.monotonic()
        runtime._last_command_at = {"left": now, "right": now}
        runtime._last_joint_state_at = {"left": now, "right": now}
        runtime._applied_commands = {
            "left": np.zeros(12, dtype=np.float32),
            "right": np.ones(12, dtype=np.float32),
        }
        runtime._measured_joint_positions = {
            "left": np.full(12, 2.0, dtype=np.float32),
            "right": np.full(12, 3.0, dtype=np.float32),
        }
        runtime._applied_source_timestamp_ns = 123
        runtime._applied_frame_id = 7

        data = runtime.get_data()
        self.assertTrue(data["fresh"])
        self.assertEqual(len(data["joint_names"]), 24)
        np.testing.assert_array_equal(data["joint_position_commands"][:12], np.zeros(12))
        np.testing.assert_array_equal(data["joint_positions"][12:], np.full(12, 3.0))
        self.assertEqual(data["applied_frame_id"], 7)

        runtime._last_command_at["right"] = now - 1.0
        self.assertFalse(runtime.get_data()["fresh"])


if __name__ == "__main__":
    unittest.main()
