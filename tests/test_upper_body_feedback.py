"""Protocol and pipeline lifecycle checks without sockets or robot hardware."""

import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import zmq
from box import Box

from robojudo.config.x2 import x2_locomanipulation, x2_xrobotoolkit, x2_xrobotoolkit_real
from robojudo.controller.ctrl_cfgs import UpperBodyZmqCtrlCfg
from robojudo.controller.upper_body_zmq_ctrl import UpperBodyZmqCtrl
from robojudo.pipeline.four_mode_pipeline import ControlMode
from robojudo.pipeline.x2_locomanipulation_pipeline import X2LocomanipulationPipeline


class MemorySocket:
    def __init__(self):
        self.messages = []
        self.frames = []

    def setsockopt(self, *args):
        pass

    def getsockopt(self, *args):
        return False

    def connect(self, endpoint):
        pass

    def bind(self, endpoint):
        pass

    def close(self, **kwargs):
        pass

    def send_json(self, frame, **kwargs):
        self.frames.append(json.loads(json.dumps(frame, allow_nan=False)))

    def recv(self, **kwargs):
        if not self.messages:
            raise zmq.Again()
        return json.dumps(self.messages.pop(0)).encode()


class MemoryContext:
    def socket(self, kind):
        return MemorySocket()


class UpperBodyFeedbackTests(unittest.TestCase):
    def test_publication_uses_new_joint_snapshot_without_changing_policy_cache(self):
        cached = self.env.dof_pos.copy()
        positions = dict(zip(self.env.joint_names, cached.tolist(), strict=True))
        positions[self.env.joint_names[0]] += 0.01
        self.env.get_joint_feedback = lambda: {
            "positions": positions.copy(),
            "sample_times": dict.fromkeys(self.env.joint_names, 9.999),
        }
        frame = self.publish()
        self.assertEqual(frame["positions"], positions)
        self.assertAlmostEqual(frame["joint_age_s"][self.env.joint_names[0]], 0.001)
        np.testing.assert_array_equal(self.env.dof_pos, cached)
        self.assertIn("body_snapshot", frame["pipeline_timing_ms"])
        self.assertAlmostEqual(frame["feedback_diagnostics"]["source_age_ms"]["body"], 1)

    def test_publication_diagnostics_label_previous_send_and_keep_original_deadline(self):
        first = self.publish()
        self.clock.return_value = 10.02
        second = self.publish()
        diagnostics = second["feedback_diagnostics"]
        self.assertIn(diagnostics["oldest_joint"], self.env.joint_names)
        self.assertEqual(diagnostics["previous_publication"]["state_id"], first["state_id"])
        self.assertGreaterEqual(diagnostics["previous_publication"]["encode_send_ms"], 0)
        self.assertAlmostEqual(self.feedback._states[second["state_id"]][0], 10.1)

    def test_young_retained_measurements_do_not_override_connection_readiness(self):
        frame = self.publish()
        self.feedback.publish(
            positions=frame["positions"],
            sample_times=dict.fromkeys(frame["positions"], 10.0),
            targets=frame["last_commanded_positions"],
            defaults=frame["default_positions"],
            limits=frame["position_limits"],
            intent=frame["intent"],
            mode=frame["mode"],
            control_dt=frame["control_dt"],
            measurements_available=np.bool_(False),
        )
        self.assertFalse(self.feedback.socket.frames[-1]["valid"])

    def setUp(self):
        self.clock = self.enterContext(patch("time.monotonic", return_value=10.0))
        self.cfg = x2_locomanipulation()
        self.cfg.ctrl = x2_xrobotoolkit().ctrl
        self.arm_cfg = next(cfg for cfg in self.cfg.ctrl if isinstance(cfg, UpperBodyZmqCtrlCfg))
        with patch("robojudo.controller.upper_body_zmq_ctrl.zmq.Context.instance", return_value=MemoryContext()):
            self.receiver = UpperBodyZmqCtrl(self.arm_cfg)
        self.addCleanup(self.receiver.close)
        self.feedback = self.receiver.feedback
        dof = self.cfg.env.dof
        self.env = SimpleNamespace(
            joint_names=dof.joint_names,
            dof_pos=np.asarray(dof.default_pos, dtype=np.float32),
            default_pos=np.asarray(dof.default_pos, dtype=np.float32),
            position_limits=np.asarray(dof.position_limits),
        )
        self.sent = []
        self.env.step = lambda target, hand=None: self.sent.append(target.copy())
        pipeline = X2LocomanipulationPipeline.__new__(X2LocomanipulationPipeline)
        pipeline.cfg, pipeline.env, pipeline.dt = self.cfg, self.env, 0.02
        pipeline.mode = ControlMode.RL_DEFAULT
        pipeline.ctrl_manager = SimpleNamespace(controllers={"UpperBodyZmqCtrl": {"inst": self.receiver}})
        pipeline._upper_body_cfg = self.arm_cfg
        pipeline._upper_body_enabled = False
        pipeline._upper_body_generation = None
        pipeline._upper_body_pending = None
        pipeline._upper_body_command_submitted = False
        pipeline._configure_upper_body_override()
        self.pipeline = pipeline
        self.env_data = Box(dof_pos=self.env.dof_pos, joint_sample_times=np.full(len(dof.joint_names), 10.0))

    def publish(self):
        target = self.sent[-1] if self.sent else self.env.default_pos
        self.pipeline._publish_upper_body_feedback(self.env_data, target)
        return self.feedback.socket.frames[-1]

    def proposal(self, state, delta=0.01, frame_id=1, bridge_id="test-bridge"):
        targets = state["last_commanded_positions"].copy()
        targets["left_elbow_joint"] += delta
        return {
            "schema_version": 1,
            "type": "upper_body_proposal",
            "session_id": state["session_id"],
            "generation": state["generation"],
            "bridge_id": bridge_id,
            "frame_id": frame_id,
            "state_id": state["state_id"],
            "command_id": state["command_id"],
            "intent": state["intent"],
            "status": "homing" if state["intent"] == "HOME" else "tracking",
            "positions": targets,
        }

    def receive(self, proposal):
        self.receiver.get_data()  # Drain packets preceding a reset before publishing.
        self.receiver._socket.messages.append(proposal)
        return {"UpperBodyZmqCtrl": self.receiver.get_data()}

    def test_presets_reuse_controller_and_pipeline_without_timeout_policy_switch(self):
        for cfg in (x2_xrobotoolkit(), x2_xrobotoolkit_real()):
            upper = next(item for item in cfg.ctrl if isinstance(item, UpperBodyZmqCtrlCfg))
            self.assertEqual(cfg.pipeline_type, "X2LocomanipulationLocoMimicPipeline")
            self.assertEqual(upper.ctrl_type, "UpperBodyZmqCtrl")
            self.assertEqual(upper.max_joint_velocity_rad_s, 8.0)
            self.assertEqual(upper.return_joint_velocity_rad_s, 1.0)
            self.assertEqual(upper.ema_alpha, 0.0)
        legacy = UpperBodyZmqCtrlCfg(joint_names=self.arm_cfg.joint_names)
        self.assertIsNone(legacy.feedback_endpoint)
        self.assertNotIn("timeout_behavior", UpperBodyZmqCtrlCfg.model_fields)
        self.assertEqual(legacy.max_joint_velocity_rad_s, 1.0)
        with self.assertRaises(ValueError):
            UpperBodyZmqCtrlCfg(joint_names=legacy.joint_names, feedback_endpoint="tcp://127.0.0.1:8571")

    def test_pending_home_can_apply_while_tracking_is_disabled(self):
        state = self.publish()
        self.assertEqual(state["intent"], "HOME")
        proposal = self.proposal(state)
        target = self.pipeline._apply_upper_body_override(self.env.default_pos, self.receive(proposal))
        expected = np.asarray(list(proposal["positions"].values()), dtype=np.float32)
        np.testing.assert_array_equal(target[self.pipeline._upper_body_indices], expected)
        self.assertFalse(self.pipeline._upper_body_stream_was_fresh)
        self.pipeline._send_policy_target(target)
        self.publish()
        self.assertEqual(self.feedback.command_id, 1)
        self.assertEqual(len(self.sent), 1)
        with self.assertRaises(ValueError):
            self.feedback.validate(proposal, proposal["positions"])

    def test_offline_bridge_holds_and_preserves_home_request(self):
        self.publish()
        previous = self.pipeline._upper_body_filtered.copy()
        previous[3] += 0.2
        self.pipeline._upper_body_filtered = previous.copy()
        for _ in range(3):
            target = self.pipeline._apply_upper_body_override(self.env.default_pos, {})
            np.testing.assert_array_equal(target[self.pipeline._upper_body_indices], previous)
            self.assertEqual(self.publish()["intent"], "HOME")

    def test_tracking_outage_holds_and_mode_change_revokes_proposals(self):
        self.pipeline._set_upper_body_enabled(True)
        state = self.publish()
        proposal = self.proposal(state, delta=0.1)
        target = self.pipeline._apply_upper_body_override(self.env.default_pos, self.receive(proposal))
        self.assertTrue(self.pipeline._upper_body_stream_was_fresh)
        self.pipeline._send_policy_target(target)
        self.publish()
        self.clock.return_value = 10.3
        held = self.pipeline._apply_upper_body_override(self.env.default_pos, {})
        np.testing.assert_array_equal(held, target)
        self.pipeline.mode = ControlMode.DAMPING_DEFAULT
        self.pipeline._reset_upper_body_feedback()
        self.assertEqual(self.publish()["intent"], "HOLD")
        with self.assertRaises(ValueError):
            self.feedback.validate(proposal, proposal["positions"])

    def test_expired_and_cached_measurements_are_rejected(self):
        state = self.publish()
        proposal = self.proposal(state)
        self.clock.return_value = 10.101
        with self.assertRaisesRegex(ValueError, "expired"):
            self.feedback.validate(proposal, proposal["positions"])
        cached = self.publish()
        self.assertGreater(cached["state_id"], state["state_id"])
        self.assertFalse(cached["valid"])
        with self.assertRaises(ValueError):
            self.feedback.validate(self.proposal(cached), proposal["positions"])

    def test_delayed_command_is_rechecked_at_application(self):
        state = self.publish()
        data = self.receive(self.proposal(state))
        self.clock.return_value = 10.101
        previous = self.pipeline._upper_body_filtered.copy()
        target = self.pipeline._apply_upper_body_override(self.env.default_pos, data)
        np.testing.assert_array_equal(target[self.pipeline._upper_body_indices], previous)

    def test_tracking_command_outage_revokes_generation_even_with_fresh_robot_state(self):
        self.pipeline._set_upper_body_enabled(True)
        state = self.publish()
        self.receive(self.proposal(state))
        self.clock.return_value = 10.2
        self.receiver.get_data()
        self.clock.return_value = 10.3
        self.receiver.get_data()
        self.env_data.joint_sample_times[:] = 10.3
        recovered = self.publish()
        self.assertTrue(recovered["valid"])
        self.assertGreater(recovered["generation"], state["generation"])
        self.assertEqual(recovered["reset_reason"], "tracking command timeout")

    def test_return_is_also_bounded_by_overall_arm_speed_override(self):
        self.feedback.cfg.max_joint_velocity_rad_s = 0.25
        state = self.publish()
        too_fast = self.proposal(state, delta=0.01)
        with self.assertRaisesRegex(ValueError, "velocity"):
            self.feedback.validate(too_fast, too_fast["positions"])
        allowed = self.proposal(state, delta=0.004)
        self.feedback.validate(allowed, allowed["positions"])

    def test_duplicate_partial_invalid_and_over_speed_commands_are_rejected(self):
        state = self.publish()
        valid = self.proposal(state)
        self.feedback.validate(valid, valid["positions"], receiving=True)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.feedback.validate(valid, valid["positions"], receiving=True)
        for mutation in ("partial", "speed", "session", "boolean", "limits"):
            bad = copy.deepcopy(valid)
            bad["frame_id"] = 2
            if mutation == "partial":
                bad["positions"].pop("right_elbow_joint")
            elif mutation == "speed":
                bad["positions"]["left_elbow_joint"] += 0.03
            elif mutation == "session":
                bad["session_id"] = "expired"
            elif mutation == "boolean":
                bad["generation"] = True
            else:
                bad["positions"]["left_elbow_joint"] = 100
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.feedback.validate(bad, bad["positions"], receiving=True)

    def test_bridge_restart_rotates_generation_and_requires_new_feedback(self):
        state = self.publish()
        valid = self.proposal(state)
        self.feedback.validate(valid, valid["positions"], receiving=True)
        restart = self.proposal(state, bridge_id="restarted")
        with self.assertRaisesRegex(ValueError, "restarted"):
            self.feedback.validate(restart, restart["positions"], receiving=True)
        next_state = self.publish()
        self.assertGreater(next_state["generation"], state["generation"])
        fresh = self.proposal(next_state, bridge_id="restarted")
        self.feedback.validate(fresh, fresh["positions"], receiving=True)

    def test_nonfinite_measurements_publish_invalid_json_safe_state(self):
        self.env_data.dof_pos = self.env.dof_pos.copy()
        self.env_data.dof_pos[0] = np.nan
        state = self.publish()
        self.assertFalse(state["valid"])
        self.assertIsNone(state["positions"][self.env.joint_names[0]])

    def test_policy_joint_ownership_prevents_even_held_arm_override(self):
        self.pipeline._upper_body_control_available = lambda: False
        policy_target = self.env.default_pos.copy()
        policy_target[self.pipeline._upper_body_indices] += 0.05
        target = self.pipeline._apply_upper_body_override(policy_target, {})
        np.testing.assert_array_equal(target, policy_target)
        self.assertEqual(self.publish()["intent"], "HOLD")

    def test_visualization_messages_cannot_refresh_commands_and_expire(self):
        state = self.publish()
        message = self.proposal(state)
        message.update(type="upper_body_visualization", positions=None, target_poses={"left": np.eye(4).tolist()})
        data = self.receive(message)["UpperBodyZmqCtrl"]
        self.assertEqual(set(data["target_poses"]), {"left"})
        self.assertFalse(data["fresh"])
        self.assertIsNone(self.feedback.last_proposal_at)
        self.assertEqual(self.feedback.command_id, 0)
        self.clock.return_value = 10.101
        self.assertEqual(self.receiver.get_data()["target_poses"], {})

    def test_visualization_rejects_invalid_transforms_without_rejecting_valid_commands(self):
        state = self.publish()
        message = self.proposal(state)
        message["target_poses"] = {"left": np.zeros((4, 4)).tolist()}
        data = self.receive(message)["UpperBodyZmqCtrl"]
        self.assertTrue(data["fresh"])
        self.assertEqual(data["target_poses"], {})

    def test_expired_visualization_is_debug_only_and_does_not_hide_command_rejections(self):
        state = self.publish()
        message = self.proposal(state)
        message.update(type="upper_body_visualization", positions=None, target_poses={"left": np.eye(4).tolist()})
        self.clock.return_value = 10.101
        with self.assertLogs("robojudo.controller.upper_body_zmq_ctrl", level="DEBUG") as logs:
            self.receive(message)
        self.assertTrue(all(record.levelname == "DEBUG" for record in logs.records))
        self.assertEqual(self.receiver._last_invalid_log_at, float("-inf"))
        with self.assertLogs("robojudo.controller.upper_body_zmq_ctrl", level="WARNING") as logs:
            self.receive(self.proposal(state))
        self.assertIn("Rejected upper-body ZMQ message", logs.output[0])
        self.assertIn("expired", logs.output[0])

    def test_ack_keeps_fresh_visualization_but_rejects_old_command_baseline(self):
        state = self.publish()
        proposal = self.proposal(state)
        self.feedback.validate(proposal, proposal["positions"], receiving=True)
        self.feedback.acknowledge()
        visual = dict(proposal, target_poses={"left": np.eye(4).tolist()})
        poses, _ = self.feedback.visualization(visual)
        self.assertIn("left", poses)
        with self.assertRaisesRegex(ValueError, "superseded"):
            self.feedback.validate(proposal, proposal["positions"])

    def test_older_visualization_cannot_replace_newer_target(self):
        old = self.publish()
        self.clock.return_value = 10.02
        self.env_data.joint_sample_times[:] = 10.02
        new = self.publish()
        for state, offset in ((new, 0.3), (old, 0.1)):
            target = np.eye(4)
            target[0, 3] = offset
            message = self.proposal(state)
            message.update(type="upper_body_visualization", positions=None, target_poses={"left": target.tolist()})
            self.receive(message)
        self.assertEqual(self.receiver.get_data()["target_poses"]["left"][0][3], 0.3)

    def test_proposal_arriving_during_inference_is_applied_in_the_same_cycle(self):
        self.pipeline._set_upper_body_enabled(True)
        state = self.publish()
        proposal = self.proposal(state)
        self.receiver.get_data()  # Drain the enable-session reset before inference.

        def infer(obs):
            self.clock.return_value = 10.06
            self.receiver._socket.messages.append(proposal)
            return self.env.default_pos.copy()

        self.pipeline.policy = SimpleNamespace(
            get_observation=lambda env_data, ctrl_data: (None, {}), get_pd_target=infer
        )
        target, _ = self.pipeline._step_rl_policy(self.env_data, {"UpperBodyZmqCtrl": {"fresh": False}}, dry_run=False)
        index = self.env.joint_names.index("left_elbow_joint")
        self.assertAlmostEqual(target[index], proposal["positions"]["left_elbow_joint"], places=6)
        self.assertEqual(len(self.sent), 1)
        self.assertIsNotNone(self.pipeline._upper_body_pending)

    def test_post_step_feedback_uses_latest_state_and_its_real_sample_times(self):
        self.clock.return_value = 10.08
        latest = Box(dof_pos=self.env.dof_pos.copy(), joint_sample_times=np.full(len(self.env.joint_names), 10.08))
        latest.dof_pos[0] += 0.1
        self.env.get_data = lambda: latest
        self.pipeline._upper_body_command_submitted = True
        state = self.publish()
        self.assertAlmostEqual(state["positions"][self.env.joint_names[0]], float(latest.dof_pos[0]))
        self.assertTrue(all(age == 0 for age in state["joint_age_s"].values()))
        latest.joint_sample_times[:] = 9.0
        self.assertFalse(self.publish()["valid"])


if __name__ == "__main__":
    unittest.main()
