"""Control deadlines remain authoritative while completions are serviced."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

# A ROS installation also provides a package named scripts.
spec = importlib.util.spec_from_file_location(
    "latency_run_pipeline", Path(__file__).parents[1] / "scripts/run_pipeline.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
wait_for_pipeline = runner.wait_for_pipeline


class PipelineIdleWaitTests(unittest.TestCase):
    def test_completion_does_not_advance_control_tick(self):
        clock = [10.0]
        marker = object()
        pipeline = SimpleNamespace(
            get_idle_notifications=lambda: (marker,), service_idle_notifications=Mock(), step=Mock(), should_exit=False
        )
        waits = []

        def ready(readers, writers, errors, timeout):
            waits.append(timeout)
            if len(waits) == 1:
                clock[0] = 10.006
                return [marker], [], []
            clock[0] = 10.02
            return [], [], []

        with (
            patch.object(runner.time, "monotonic", side_effect=lambda: clock[0]),
            patch.object(runner.select, "select", side_effect=ready),
        ):
            wait_for_pipeline(pipeline, 10.02)
        pipeline.service_idle_notifications.assert_called_once()
        pipeline.step.assert_not_called()
        self.assertAlmostEqual(waits[0], 0.02)
        self.assertAlmostEqual(waits[1], 0.014)

    def test_pipeline_without_notifications_uses_remaining_sleep(self):
        with (
            patch.object(runner.time, "monotonic", return_value=10),
            patch.object(runner.time, "sleep") as sleep,
        ):
            wait_for_pipeline(SimpleNamespace(), 10.02)
        sleep.assert_called_once()
        self.assertAlmostEqual(sleep.call_args.args[0], 0.02)

    def test_overrun_never_sleeps_or_services_more_work(self):
        pipeline = SimpleNamespace(get_idle_notifications=lambda: (), service_idle_notifications=Mock())
        with (
            patch.object(runner.time, "monotonic", return_value=10.03),
            patch.object(runner.time, "sleep") as sleep,
        ):
            wait_for_pipeline(pipeline, 10.02)
        sleep.assert_not_called()
        pipeline.service_idle_notifications.assert_not_called()


if __name__ == "__main__":
    unittest.main()
