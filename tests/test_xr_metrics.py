"""Robot rates count completed pipeline ticks, not asynchronous feedback servicing."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from robojudo.pipeline.upper_body_zmq_pipeline import UpperBodyZmqPipelineMixin


class XrMetricsTests(unittest.TestCase):
    def setUp(self):
        self.clock = self.enterContext(patch("time.monotonic", return_value=10.0))
        self.counts = {"received": 0, "completed": 0}
        self.receiver = SimpleNamespace(feedback=SimpleNamespace(command_counts=self.counts))
        self.pipeline = UpperBodyZmqPipelineMixin.__new__(UpperBodyZmqPipelineMixin)
        self.pipeline._feedback_receiver = lambda: self.receiver

    def test_measured_rates_and_reset(self):
        self.pipeline.report_xr_metrics()
        with self.assertNoLogs("robojudo.pipeline.upper_body_zmq_pipeline", level="INFO"):
            for tick in range(1, 50):
                self.clock.return_value = 10 + tick * 0.02
                self.counts.update(received=tick * 2, completed=tick // 2)
                self.pipeline.report_xr_metrics()
        self.clock.return_value = 11.0
        self.counts.update(received=100, completed=25)
        with self.assertLogs("robojudo.pipeline.upper_body_zmq_pipeline", level="INFO") as logs:
            self.pipeline.report_xr_metrics()
        self.assertIn("receive=100.0 Hz | control=50.0 Hz | complete=25.0 Hz", logs.output[0])
        self.clock.return_value = 12.0
        with self.assertLogs("robojudo.pipeline.upper_body_zmq_pipeline", level="INFO") as logs:
            self.pipeline.report_xr_metrics()
        self.assertIn("receive=0.0 Hz | control=1.0 Hz | complete=0.0 Hz", logs.output[0])
        self.counts.update(received=0, completed=0)
        with self.assertNoLogs("robojudo.pipeline.upper_body_zmq_pipeline", level="INFO"):
            self.pipeline.report_xr_metrics()

    def test_idle_feedback_does_not_count_as_a_control_tick(self):
        self.pipeline.report_xr_metrics()
        initial = self.pipeline._xr_metrics
        self.receiver._hand_runtime = SimpleNamespace(notification_socket=object(), drain_notifications=Mock())
        self.pipeline._upper_body_feedback_body = {}
        self.pipeline._publish_upper_body_feedback = Mock()
        for _ in range(20):
            self.pipeline.service_idle_notifications()
        self.assertEqual(self.pipeline._xr_metrics, initial)
        self.assertEqual(self.pipeline._publish_upper_body_feedback.call_count, 20)

    def test_no_feedback_produces_no_xr_log(self):
        self.pipeline._feedback_receiver = lambda: None
        with self.assertNoLogs("robojudo.pipeline.upper_body_zmq_pipeline", level="INFO"):
            self.pipeline.report_xr_metrics()

    def test_launcher_reports_once_after_each_successful_step(self):
        spec = importlib.util.spec_from_file_location(
            "metrics_run_pipeline", Path(__file__).parents[1] / "scripts/run_pipeline.py"
        )
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        events = []
        pipeline = SimpleNamespace(should_exit=False, env=SimpleNamespace(), ctrl_manager=SimpleNamespace(close=Mock()))

        def step():
            events.append("step")
            pipeline.should_exit = True

        pipeline.step = step
        pipeline.report_xr_metrics = lambda: events.append("report")
        cfg = SimpleNamespace(
            env=SimpleNamespace(is_sim=True, env_type="test"),
            run_fullspeed=True,
            pipeline_type="X2LocomanipulationPipeline",
        )
        args = SimpleNamespace(
            config="test",
            record=False,
            record_endpoint=None,
            record_task=None,
            gr00t_task=None,
            gr00t_command_endpoint=None,
        )
        with (
            patch.object(runner, "parse_args", return_value=args),
            patch.object(runner, "ConfigManager", return_value=SimpleNamespace(get_cfg=lambda: cfg)),
            patch.object(runner.robojudo.pipeline, cfg.pipeline_type, return_value=pipeline),
        ):
            runner.main()
        self.assertEqual(events, ["step", "report"])
