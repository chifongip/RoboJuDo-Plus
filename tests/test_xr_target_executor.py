"""Authoritative combined transitions without hardware or an IK solver."""

import copy
import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from test_upper_body_feedback import MemoryContext

from robojudo.config.x2 import x2_xrobotoolkit_omnihand_real_direct_executor
from robojudo.controller.upper_body_feedback import UpperBodyFeedback
from robojudo.pipeline.upper_body_zmq_pipeline import UpperBodyZmqPipelineMixin


@unittest.skipUnless(importlib.util.find_spec("xrobotoolkit_teleop"), "install XR executor support")
class ExecutorTests(unittest.TestCase):
    def setUp(self):
        self.clock = self.enterContext(patch("time.monotonic", return_value=10.0))
        self.cfg = next(
            c for c in x2_xrobotoolkit_omnihand_real_direct_executor().ctrl if c.ctrl_type == "UpperBodyOmniHandZmqCtrl"
        )
        self.feedback = UpperBodyFeedback(self.cfg, MemoryContext())
        self.addCleanup(self.feedback.close)
        self.executor = self.feedback.executor
        self.profile = self.executor.profile
        self.baseline = {n: float(np.float32(-0.87 if "elbow" in n else 0.0)) for n in self.cfg.joint_names}
        self.baseline.update(dict.fromkeys(self.profile.hands.active, 0.0))
        self.measured = dict.fromkeys(self.profile.observed_joints, 0.0)
        self.measured.update(self.baseline)
        self.publish()

    def publish(self, intent="TRACK"):
        self.feedback.publish(
            positions=self.measured,
            sample_times=dict.fromkeys(self.measured, 10.0),
            targets=self.baseline,
            defaults=self.baseline,
            limits={**self.profile.spec.limits, **self.profile.hands.limits},
            intent=intent,
            mode="RL_DEFAULT",
            control_dt=0.02,
        )
        return self.feedback.socket.frames[-1]

    def desired(self, frame_id=1):
        state = self.feedback.socket.frames[-1]
        return dict(
            schema_version=3,
            type="upper_body_desired",
            bridge_id="bridge",
            frame_id=frame_id,
            **{k: state[k] for k in ("session_id", "generation", "state_id", "command_id", "intent")},
            status="homing" if state["intent"] == "HOME" else "tracking",
            hand_command_mode="direct",
            executor=dict(self.executor.contract),
            active_joint_names=list(self.feedback.command_names),
            velocity_limits=dict(arm=1.0, **{"return": 0.5}, hand=2.0),
            positions=dict(self.baseline),
        )

    def prepare(self, desired):
        return self.executor.prepare(self.feedback, desired, self.measured)

    def test_waiting_desired_survives_completion_and_uses_local_baseline(self):
        first, waiting = self.desired(), self.desired(2)
        joint = self.cfg.joint_names[0]
        first["positions"][joint] += 0.3
        waiting["positions"][joint] += 0.3
        self.feedback.validate(first, first["positions"], receiving=True)
        issued = self.prepare(first)
        self.feedback.validate(waiting, waiting["positions"], receiving=True)
        self.feedback.acknowledge(issued)
        self.assertEqual(waiting["command_id"], 0)
        self.assertEqual(self.feedback.command_id, 1)
        next_issued = self.prepare(waiting)
        self.assertGreater(next_issued["positions"][joint], issued["positions"][joint])
        self.assertLessEqual(next_issued["positions"][joint] - issued["positions"][joint], 0.02)
        with self.assertRaisesRegex(ValueError, "already consumed"):
            self.prepare(waiting)

    def test_executor_can_filter_locally_without_bridge_prediction(self):
        # EMA is safe to configure for locally validated schema 3 transitions.
        validated = type(self.cfg).model_validate({**self.cfg.model_dump(), "ema_alpha": 0.5})
        self.feedback.cfg = validated
        desired = self.desired()
        joint = self.cfg.joint_names[0]
        desired["positions"][joint] += 0.01
        issued = self.prepare(desired)
        self.assertAlmostEqual(issued["positions"][joint] - self.baseline[joint], 0.005, places=6)

    def test_inactive_joints_hold_local_baseline_after_delayed_feedback(self):
        first = self.desired()
        joint = self.cfg.joint_names[0]
        first["positions"][joint] += 0.3
        issued = self.prepare(first)
        self.feedback.acknowledge(issued)
        held = self.desired(2)
        held["active_joint_names"].remove(joint)
        held["positions"][joint] = self.baseline[joint]
        self.assertEqual(self.prepare(held)["positions"][joint], issued["positions"][joint])

    def test_direct_fingers_and_home_hold(self):
        desired = self.desired()
        desired["positions"]["L_index_mcp_joint"] = 0.8
        issued = self.prepare(desired)
        self.assertEqual(issued["positions"]["L_index_mcp_joint"], float(np.float32(0.8)))
        self.feedback.acknowledge(issued)
        self.baseline = dict(issued["positions"])
        self.publish("HOME")
        homing = self.prepare(self.desired(2))
        # Intent reset rebases from the supplied current coherent targets.
        self.assertEqual(
            homing["positions"]["L_index_mcp_joint"], self.feedback._completed_positions["L_index_mcp_joint"]
        )

    def test_deadline_generation_policy_replay_and_target_mutation(self):
        desired = self.desired()
        self.feedback.validate(desired, desired["positions"], receiving=True)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.feedback.validate(desired, desired["positions"], receiving=True)
        bad = copy.deepcopy(desired)
        bad["executor"]["collision_checking"] = True
        with self.assertRaisesRegex(ValueError, "collision policy"):
            self.prepare(bad)
        issued = self.prepare(desired)
        issued["positions"][self.cfg.joint_names[0]] += 0.001
        with self.assertRaisesRegex(ValueError, "target changed"):
            self.feedback.validate(issued, issued["positions"])
        self.clock.return_value = 10.11
        with self.assertRaisesRegex(ValueError, "expired"):
            self.feedback.validate(issued, issued["positions"])
        self.feedback.reset()
        with self.assertRaisesRegex(ValueError, "obsolete"):
            self.prepare(desired)

    def test_collision_failure_or_expiry_does_not_advance_baseline(self):
        baseline = dict(self.feedback._completed_positions)
        with patch.object(self.executor.model, "validate_transition", side_effect=RuntimeError("collision")):
            with self.assertRaisesRegex(ValueError, "collision"):
                self.prepare(self.desired())
        self.assertEqual(self.feedback._completed_positions, baseline)
        self.assertEqual(self.feedback.command_id, 0)

        def expire(*args):
            self.clock.return_value = 10.11

        with patch.object(self.executor.model, "validate_transition", side_effect=expire):
            with self.assertRaisesRegex(ValueError, "expired"):
                self.prepare(self.desired(2))
        self.assertEqual(self.feedback._completed_positions, baseline)

    def test_pipeline_collision_rejection_holds_without_revoking_clutch(self):
        pipeline = UpperBodyZmqPipelineMixin.__new__(UpperBodyZmqPipelineMixin)
        receiver = SimpleNamespace(
            feedback=self.feedback,
            pending=None,
            synchronize_intent=Mock(),
            synchronize_generation=Mock(),
            _clear_positions=Mock(),
            _log_invalid_message=Mock(),
            invalidate=Mock(),
            hand_snapshot=lambda: {"positions": self.measured, "fresh": True},
        )
        pipeline._upper_body_enabled = True
        pipeline._upper_body_control_available = lambda: True
        pipeline._feedback_intent = lambda: "TRACK"
        pipeline._upper_body_cfg = self.cfg
        pipeline._upper_body_indices = np.arange(14)
        pipeline._upper_body_filtered = np.array([self.baseline[n] for n in self.cfg.joint_names])
        before = pipeline._upper_body_filtered.copy()
        generation = self.feedback.generation
        with patch.object(self.executor.model, "validate_transition", side_effect=RuntimeError("collision")):
            target = pipeline._apply_feedback_proposal(
                np.zeros(14),
                {
                    self.cfg.ctrl_type: {
                        "fresh": True,
                        "proposal": self.desired(),
                        "joint_positions": self.baseline,
                    }
                },
                receiver,
            )
        np.testing.assert_array_equal(target, before)
        self.assertIsNone(pipeline._upper_body_pending)
        self.assertEqual(self.feedback.generation, generation)
        receiver.invalidate.assert_not_called()
        receiver._clear_positions.assert_called_once()

    def test_enabled_collision_model_initialization_failure_is_fatal(self):
        from robojudo.controller.xr_target_executor import XRTargetExecutor

        self.cfg.executor_collision_checking = True
        with (
            patch(
                "xrobotoolkit_teleop.ik.pinocchio_model.PinocchioModel", side_effect=RuntimeError("native query failed")
            ),
            self.assertRaisesRegex(RuntimeError, "native query failed"),
        ):
            XRTargetExecutor(self.cfg)

    def test_pipeline_does_not_prepare_while_combined_frame_in_flight(self):
        pipeline = UpperBodyZmqPipelineMixin.__new__(UpperBodyZmqPipelineMixin)
        receiver = SimpleNamespace(feedback=self.feedback, pending=object(), synchronize_intent=Mock())
        pipeline._upper_body_enabled = True
        pipeline._upper_body_control_available = lambda: True
        pipeline._feedback_intent = lambda: "TRACK"
        pipeline._upper_body_cfg = self.cfg
        pipeline._upper_body_indices = np.arange(14)
        pipeline._upper_body_filtered = np.array([self.baseline[n] for n in self.cfg.joint_names])
        with patch.object(self.executor, "prepare") as prepare:
            pipeline._apply_feedback_proposal(
                np.zeros(14),
                {
                    self.cfg.ctrl_type: {
                        "fresh": True,
                        "proposal": self.desired(),
                        "joint_positions": self.baseline,
                    }
                },
                receiver,
            )
        prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
