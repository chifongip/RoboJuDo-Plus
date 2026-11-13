"""Fresh body snapshots and bounded hand IPC without touching robot hardware."""

import multiprocessing as mp
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from robojudo.controller.omnihand_runtime import _LatestSnapshot
from robojudo.environment.agibot_cpp_env import AgiBotCppEnv
from robojudo.environment.base_env import Environment


def hold_snapshot_lock(slot, ready):
    slot._lock.acquire()
    ready.set()
    # Parent deliberately terminates this writer while it owns the lock.
    mp.get_context("spawn").Event().wait(60)


class FeedbackSnapshotTests(unittest.TestCase):
    def test_native_feedback_bypasses_policy_cache_and_keeps_paired_timestamps(self):
        env = AgiBotCppEnv.__new__(AgiBotCppEnv)
        env.joint_names, env.num_dofs = ["arm"], 1
        env._dof_pos = np.array([0.1])
        env._joint_sample_times = np.array([1.0])
        env.aimdk = Mock()
        env.aimdk.get_robot_state.return_value = SimpleNamespace(
            motor_state=SimpleNamespace(q=[0.2]),
            joint_received_time_sec=[2.0],
        )
        env.update = Mock(side_effect=AssertionError("policy/observer update must not run"))
        result = env.get_joint_feedback()
        self.assertAlmostEqual(result["positions"]["arm"], 0.2)
        self.assertEqual(result["sample_times"], {"arm": 2.0})
        np.testing.assert_array_equal(env._dof_pos, [0.1])
        np.testing.assert_array_equal(env._joint_sample_times, [1.0])
        env.update.assert_not_called()
        env.aimdk.get_robot_state.return_value.joint_received_time_sec = []
        with self.assertRaisesRegex(ValueError, "timestamps"):
            env.get_joint_feedback()
        del env.aimdk.get_robot_state.return_value.joint_received_time_sec
        with self.assertRaisesRegex(ValueError, "timestamps"):
            env.get_joint_feedback()

    def test_base_snapshot_is_a_copy_with_original_timestamps(self):
        env = SimpleNamespace(joint_names=["arm"], dof_pos=np.array([0.1]), _joint_sample_times=[1.0])
        result = Environment.get_joint_feedback(env)
        env.dof_pos[0] = 0.2
        env._joint_sample_times[0] = 2.0
        self.assertEqual(result, {"positions": {"arm": 0.1}, "sample_times": {"arm": 1.0}})

    def test_slot_keeps_latest_sequence_without_consuming_it(self):
        slot = _LatestSnapshot(mp.get_context("spawn"))
        self.assertIsNone(slot.read_after(0))
        for index in range(100):
            self.assertTrue(slot.put_nowait({"sampled_at": index}))
        sequence, stamp, snapshot = slot.read_after(0)
        self.assertEqual(sequence, 100)
        self.assertEqual(snapshot, {"sampled_at": 99})
        self.assertEqual(slot.read_after(0), (sequence, stamp, snapshot))
        self.assertIsNone(slot.read_after(sequence))
        with self.assertRaisesRegex(ValueError, "capacity"):
            slot.put_nowait(bytes(16_385))
        self.assertEqual(slot.read_after(0)[0], sequence)

    def test_slot_contention_and_killed_writer_do_not_block_reader_or_replacement(self):
        context = mp.get_context("spawn")
        slot = _LatestSnapshot(context)
        ready = context.Event()
        writer = context.Process(target=hold_snapshot_lock, args=(slot, ready))
        writer.start()
        try:
            self.assertTrue(ready.wait(10))
            self.assertIsNone(slot.read_after(0))
            self.assertFalse(slot.put_nowait({"sampled_at": 1}))
            writer.terminate()
            writer.join(5)
            self.assertFalse(writer.is_alive())
            self.assertIsNone(slot.read_after(0))
            replacement = _LatestSnapshot(context)
            replacement.put_nowait({"sampled_at": 2})
            self.assertEqual(replacement.read_after(0)[2]["sampled_at"], 2)
        finally:
            if writer.is_alive():
                writer.kill()
                writer.join(5)
            writer.close()


if __name__ == "__main__":
    unittest.main()
