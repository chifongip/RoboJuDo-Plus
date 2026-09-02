import queue
import sys
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


class TestOmniHandConfig(unittest.TestCase):
    def test_hcan_exposes_adapter_indices_without_channel_configuration(self):
        cfg = OmniHandCfg(transport="hcan", left_adapter_index=1, right_adapter_index=0)
        self.assertEqual(cfg.transport, "hcan")
        self.assertFalse(hasattr(cfg, "left_channel_id"))

    def test_socketcan_interfaces_must_not_be_empty(self):
        with self.assertRaisesRegex(ValueError, "interfaces"):
            OmniHandCfg(transport="socketcan", left_interface=" ")


class TestOmniHandRuntime(unittest.TestCase):
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
