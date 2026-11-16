"""Direct hints cannot replace authoritative, matching dual-hand completion."""

import multiprocessing as mp
import threading
import unittest
from unittest.mock import Mock, patch

from robojudo.controller.omnihand_runtime import _LatestSnapshot, _WorkerSnapshotPublisher


class WorkerCompletionPublisherTests(unittest.TestCase):
    def setUp(self):
        self.clock = self.enterContext(patch("time.monotonic", return_value=10.0))
        context = mp.get_context("spawn")
        self.slot = _LatestSnapshot(context)
        self.gate = context.Array("q", [3, 1, 0])
        self.stop = threading.Event()
        self.socket = Mock()
        self.publisher = _WorkerSnapshotPublisher(self.slot, self.gate, self.stop, self.socket)

    @staticmethod
    def state(frame=1, generation=3, **kwargs):
        return {"applied": (2, frame), "generation": generation, "positions": [0.0] * 19, "sampled_at": 10.0, **kwargs}

    def complete(self, frame=1):
        for side in ("left", "right"):
            self.publisher.publish(side, self.state(frame))

    def test_hint_follows_matching_published_snapshot_and_is_sent_once(self):
        observed = []
        self.socket.send.side_effect = lambda _: observed.append(self.slot.read_after(0))
        self.publisher.publish("left", self.state())
        self.socket.send.assert_not_called()
        self.publisher.publish("right", self.state(frame=2))
        self.socket.send.assert_not_called()
        self.publisher.publish("right", self.state())
        self.socket.send.assert_called_once()
        _, _, snapshot = observed[0]
        self.assertEqual(snapshot["left"]["applied"], snapshot["right"]["applied"])
        self.assertEqual(snapshot["left"]["completion_ready_at"], 10.0)
        self.clock.return_value = 10.02
        self.publisher.publish("left", self.state(sampled_at=10.02))
        self.assertEqual(self.slot.read_after(0)[2]["left"]["completion_ready_at"], 10.0)
        self.socket.send.assert_called_once()

    def test_busy_slot_retries_newest_snapshot_without_another_read(self):
        self.slot._lock.acquire()
        try:
            self.complete()
            self.clock.return_value = 10.01
            self.publisher.publish("left", self.state(sampled_at=10.01))
            self.publisher.retry()
            self.socket.send.assert_not_called()
        finally:
            self.slot._lock.release()
        self.publisher.retry()
        self.socket.send.assert_called_once()
        _, _, snapshot = self.slot.read_after(0)
        self.assertEqual(snapshot["left"]["sampled_at"], 10.01)
        self.assertEqual(snapshot["left"]["completion_ready_at"], 10.0)
        self.publisher.retry()
        self.socket.send.assert_called_once()

    def test_revocation_cancels_pending_hint(self):
        self.slot._lock.acquire()
        try:
            self.complete()
        finally:
            self.slot._lock.release()
        with self.gate.get_lock():
            self.gate[0] += 1
            self.gate[1] = 0
        self.publisher.retry()
        self.socket.send.assert_not_called()
        self.assertIsNone(self.slot.read_after(0))
        self.complete()
        self.socket.send.assert_not_called()

    def test_replacement_never_retries_old_frame(self):
        self.slot._lock.acquire()
        try:
            self.complete()
            self.publisher.publish("left", self.state(frame=2))
        finally:
            self.slot._lock.release()
        self.publisher.retry()
        self.socket.send.assert_not_called()
        self.publisher.publish("right", self.state(frame=2))
        self.socket.send.assert_called_once()
        snapshot = self.slot.read_after(0)[2]
        self.assertEqual(snapshot["left"]["applied"], (2, 2))
        self.assertEqual(snapshot["right"]["applied"], (2, 2))

    def test_stop_cancels_pending_hint(self):
        self.slot._lock.acquire()
        try:
            self.complete()
        finally:
            self.slot._lock.release()
        self.stop.set()
        self.publisher.retry()
        self.socket.send.assert_not_called()

    def test_full_notification_socket_keeps_authoritative_completion(self):
        self.socket.send.side_effect = BlockingIOError()
        self.complete()
        self.assertIsNotNone(self.slot.read_after(0))
        self.publisher.retry()
        self.socket.send.assert_called_once()

    def test_notification_failure_is_bounded_and_does_not_fail_publication(self):
        self.socket.send.side_effect = OSError("closed endpoint")
        with self.assertLogs("robojudo.controller.omnihand_runtime", level="WARNING") as logs:
            self.complete()
            self.complete(frame=2)
        self.assertEqual(len(logs.output), 1)
        self.assertEqual(self.slot.read_after(0)[2]["left"]["applied"], (2, 2))

    def test_default_path_does_not_retry_or_require_a_notification_socket(self):
        publisher = _WorkerSnapshotPublisher(self.slot, self.gate, self.stop)
        self.slot._lock.acquire()
        try:
            for side in ("left", "right"):
                publisher.publish(side, self.state())
        finally:
            self.slot._lock.release()
        publisher.retry()
        self.assertIsNone(self.slot.read_after(0))


if __name__ == "__main__":
    unittest.main()
