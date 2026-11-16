"""Policy-boundary completion checks reuse the existing combined-frame barrier."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from test_upper_body_feedback import MemoryContext

from robojudo.controller.ctrl_cfgs import OmniHandCfg, UpperBodyOmniHandZmqCtrlCfg, UpperBodyZmqCtrlCfg
from robojudo.controller.omnihand_runtime import ALL_HAND_NAMES, OMNIHAND_JOINT_NAMES
from robojudo.controller.upper_body_omnihand_zmq_ctrl import UpperBodyOmniHandZmqCtrl
from robojudo.pipeline.x2_locomanipulation_loco_mimic_pipeline import X2LocomanipulationLocoMimicPipeline


class UpperBodyCompletionCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.clock = self.enterContext(patch("time.monotonic", return_value=10.0))
        self.ready_at = float("inf")
        self.identity = None
        self.fresh, self.available, self.intent = True, True, "TRACK"
        self.inference_s, self.arm_step_s = 0.006, 0.0
        self.cfg = UpperBodyOmniHandZmqCtrlCfg(
            joint_names=["arm"],
            ema_alpha=0,
            xr_target_filter=True,
            hand_command_mode="direct",
            feedback_endpoint="tcp://127.0.0.1:8571",
            omnihand=OmniHandCfg(direct_completion_notifications=True),
        )
        self.runtime = Mock()
        self.runtime.get_data.side_effect = self.snapshot
        self.env = SimpleNamespace(
            cfg_env=SimpleNamespace(is_sim=False),
            joint_names=["arm"],
            position_limits=np.array([[-1.0, 1.0]]),
            step=Mock(side_effect=self.arm_step),
            get_joint_feedback=lambda: {"positions": {"arm": 0.01}, "sample_times": {"arm": 10.0}},
        )
        with (
            patch("robojudo.controller.upper_body_zmq_ctrl.zmq.Context.instance", return_value=MemoryContext()),
            patch.object(UpperBodyOmniHandZmqCtrl, "_create_hand_runtime", return_value=self.runtime),
        ):
            self.receiver = UpperBodyOmniHandZmqCtrl(self.cfg, self.env)
        self.addCleanup(self.receiver.close)
        self.feedback = self.receiver.feedback
        p = X2LocomanipulationLocoMimicPipeline.__new__(X2LocomanipulationLocoMimicPipeline)
        p.env, p.dt = self.env, 0.02
        p._upper_body_cfg = self.cfg
        p._upper_body_indices = np.array([0])
        p._upper_body_filtered = p._upper_body_default = np.zeros(1)
        p._upper_body_enabled = p._upper_body_override_was_available = True
        p._upper_body_pending = None
        p._upper_body_command_submitted = p._recording_active = False
        p._upper_body_control_available = lambda: self.available
        p._feedback_intent = lambda: self.intent
        p._feedback_receiver = lambda: self.receiver
        p.mode = SimpleNamespace(value="RL_DEFAULT")
        p._get_policy_step = self.inference
        p._send_policy_target = p._send_upper_body_target
        self.pipeline = p
        p._publish_upper_body_feedback(None, np.zeros(1))
        self.receiver.get_data()  # Drain the initial control-intent reset.
        state = self.feedback.socket.frames[-1]
        positions = dict(state["last_commanded_positions"], arm=float(np.float32(0.01)))
        proposal = dict(
            schema_version=2,
            type="upper_body_proposal",
            session_id=state["session_id"],
            generation=state["generation"],
            state_id=state["state_id"],
            command_id=state["command_id"],
            bridge_id="checkpoint-test",
            frame_id=1,
            intent="TRACK",
            status="tracking",
            positions=positions,
            desired_positions=positions.copy(),
            filter_contract="robojudo_filter_v1",
            hand_command_mode="direct",
        )
        self.receiver._socket.messages.append(proposal)
        target = p._apply_feedback_proposal(np.zeros(1), {self.cfg.ctrl_type: self.receiver.get_data()}, self.receiver)
        p._send_upper_body_target(target)
        self.identity = self.receiver.pending[1]
        p._publish_upper_body_feedback(None, target)
        self.proposal = self.receiver.pending[0]
        self.runtime.submit.reset_mock()
        self.env.step.reset_mock()
        p._upper_body_command_submitted = False
        self.clock.return_value = 10.02
        self.arm_step_s = 0.003

    def snapshot(self):
        return {
            "positions": dict.fromkeys(ALL_HAND_NAMES, 0.0),
            "sample_times": dict.fromkeys(ALL_HAND_NAMES, 10.0),
            "targets": dict.fromkeys(OMNIHAND_JOINT_NAMES, 0.0),
            "joint_state_fresh": self.fresh,
            "applied": self.identity if self.clock.return_value >= self.ready_at else None,
        }

    def inference(self, *_):
        self.clock.return_value += self.inference_s
        return np.array([0.01], dtype=np.float32), {}

    def arm_step(self, *_):
        self.clock.return_value += self.arm_step_s

    def checkpoint(self, **kwargs):
        return self.pipeline._service_upper_body_completion_checkpoint("before_policy", **kwargs)

    def cycle(self):
        self.clock.return_value += 0.004
        target, _ = self.pipeline._step_rl_policy(None, {}, dry_run=False)
        self.pipeline._publish_upper_body_feedback(None, target)
        if self.receiver.pending is not None:
            self.clock.return_value = max(self.clock.return_value, self.ready_at + 0.0001)
            self.pipeline.service_idle_notifications()
        self.assertEqual(self.feedback.command_counts["completed"], 1)
        self.env.step.assert_called_once()
        self.runtime.submit.assert_not_called()
        np.testing.assert_allclose(target, [0.01])
        return 1000 * (self.feedback._command_timestamps["acknowledged"] - self.ready_at)

    def test_before_inference_acknowledges_once_without_hardware_or_extra_publication(self):
        self.ready_at = 10.019
        count = len(self.feedback.socket.frames)
        self.assertTrue(self.checkpoint())
        frame = self.feedback.socket.frames[-1]
        self.assertEqual(frame["applied_frame_id"], 1)
        self.assertEqual(frame["last_commanded_positions"], self.proposal["positions"])
        self.assertEqual(frame["feedback_diagnostics"]["completion_observation_source"], "before_policy")
        self.assertEqual(len(self.feedback.socket.frames), count + 1)
        self.assertFalse(self.checkpoint())
        self.assertEqual(len(self.feedback.socket.frames), count + 1)
        self.runtime.submit.assert_not_called()
        self.env.step.assert_not_called()

    def test_incomplete_frame_has_no_publication_or_acknowledgement(self):
        count = len(self.feedback.socket.frames)
        self.assertFalse(self.checkpoint())
        self.assertEqual(len(self.feedback.socket.frames), count)
        self.assertEqual(self.feedback.command_id, 0)

    def test_after_inference_checkpoint_and_policy_timing(self):
        self.ready_at = 10.025
        self.assertAlmostEqual(self.cycle(), 5)
        self.assertEqual(self.feedback._completion_observation_source, "after_policy")
        self.assertIn("checkpoint_before", self.pipeline._upper_body_timing_ms)
        self.assertIn("checkpoint_after", self.pipeline._upper_body_timing_ms)

    def test_policy_timing_excludes_both_checkpoint_costs(self):
        perf = [0.0]
        order = []

        def checkpoint(source, **kwargs):
            order.append(source)
            perf[0] += 0.001
            return False

        def inference(*args):
            order.append("policy")
            perf[0] += 0.003
            return self.inference(*args)

        with (
            patch("time.perf_counter", side_effect=lambda: perf[0]),
            patch.object(self.pipeline, "_service_upper_body_completion_checkpoint", side_effect=checkpoint),
            patch.object(self.pipeline, "_get_policy_step", side_effect=inference),
        ):
            self.pipeline._step_rl_policy(None, {}, dry_run=False)
        timing = self.pipeline._upper_body_timing_ms
        self.assertAlmostEqual(timing["policy"], 3)
        self.assertAlmostEqual(timing["checkpoint_before"], 1)
        self.assertAlmostEqual(timing["checkpoint_after"], 1)
        self.assertEqual(order, ["before_policy", "policy", "after_policy"])

    def test_recording_keeps_existing_post_step_completion_path(self):
        self.pipeline._recording_active = True
        self.ready_at = 10.021
        self.assertAlmostEqual(self.cycle(), 12)
        self.assertEqual(self.feedback._completion_observation_source, "periodic")
        self.assertTrue(self.pipeline._upper_body_stream_was_fresh)
        self.assertNotIn("checkpoint_before", self.pipeline._upper_body_timing_ms)

    def test_guard_cases_do_not_acknowledge(self):
        cases = (
            "disabled",
            "other_controller",
            "dry_run",
            "recording",
            "partial",
            "stale",
            "obsolete",
            "session",
            "intent",
            "arm_unfinished",
            "unavailable",
            "selected_proposal",
            "simulation",
        )
        for case in cases:
            with self.subTest(case=case):
                sample = type(self)()
                sample.setUp()
                try:
                    sample.ready_at = 10.019
                    p, r = sample.pipeline, sample.receiver
                    if case == "disabled":
                        sample.cfg.omnihand.direct_completion_notifications = False
                    if case == "other_controller":
                        p._upper_body_cfg = UpperBodyZmqCtrlCfg(joint_names=["arm"])
                    if case == "recording":
                        p._recording_active = True
                    if case == "partial":
                        sample.ready_at = float("inf")
                    if case == "stale":
                        sample.fresh = False
                    if case == "obsolete":
                        sample.feedback.generation += 1
                    if case == "session":
                        sample.proposal["session_id"] = "obsolete"
                    if case == "intent":
                        sample.intent = "HOLD"
                    if case == "arm_unfinished":
                        del sample.proposal["_command_timestamps"]["arm_submission_completed"]
                    if case == "unavailable":
                        sample.available = False
                    if case == "selected_proposal":
                        p._upper_body_pending = {"frame_id": 2}
                    if case == "simulation":
                        r._hand_runtime = None
                    count = len(sample.feedback.socket.frames)
                    self.assertFalse(sample.checkpoint(dry_run=case == "dry_run"))
                    self.assertEqual(sample.feedback.command_id, 0)
                    self.assertEqual(len(sample.feedback.socket.frames), count)
                    sample.env.step.assert_not_called()
                    sample.runtime.submit.assert_not_called()
                finally:
                    sample.doCleanups()

    def test_generation_change_during_snapshot_read_cannot_acknowledge(self):
        self.ready_at = 10.019

        def changed():
            self.feedback.generation += 1
            self.receiver.pending = None
            return self.snapshot()

        self.runtime.get_data.side_effect = changed
        self.assertFalse(self.checkpoint())
        self.assertEqual(self.feedback.command_id, 0)

    def test_snapshot_failure_uses_existing_invalidation(self):
        self.runtime.get_data.side_effect = RuntimeError("worker failed")
        self.assertFalse(self.checkpoint())
        self.assertIsNone(self.receiver.pending)
        self.assertEqual(self.feedback.command_id, 0)
        self.env.step.assert_not_called()

    def test_old_baseline_is_rejected_after_early_completion(self):
        self.ready_at = 10.019
        self.assertTrue(self.checkpoint())
        self.receiver._socket.messages.append(dict(self.proposal, frame_id=2))
        stream = self.receiver.get_data()
        target = self.pipeline._apply_feedback_proposal(np.zeros(1), {self.cfg.ctrl_type: stream}, self.receiver)
        self.assertIsNone(self.pipeline._upper_body_pending)
        np.testing.assert_allclose(target, [0.01])
        self.runtime.submit.assert_not_called()

    def test_completion_phase_sweep_and_long_inference_limit(self):
        current, checkpoints = [], []
        for enabled, result in ((False, current), (True, checkpoints)):
            for phase in np.linspace(0, 20, 81):
                sample = type(self)()
                sample.setUp()
                try:
                    sample.ready_at = 10.02 + phase / 1000
                    if enabled:
                        result.append(sample.cycle())
                    else:
                        with patch.object(
                            sample.pipeline, "_service_upper_body_completion_checkpoint", return_value=None
                        ):
                            result.append(sample.cycle())
                finally:
                    sample.doCleanups()
        self.assertAlmostEqual(np.percentile(current, 95), 12)
        self.assertAlmostEqual(np.percentile(checkpoints, 95), 4.75)
        self.ready_at, self.inference_s = 10.025, 0.015
        self.assertAlmostEqual(self.cycle(), 14)


if __name__ == "__main__":
    unittest.main()
