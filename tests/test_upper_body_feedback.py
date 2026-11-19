"""Protocol and pipeline lifecycle checks without sockets or robot hardware."""

import copy
import importlib.util
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

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


class TransportExecutor:
    """Stand-in numerical executor for transport/barrier tests with a one-joint model.

    Real model, filtering and collision validation are covered in test_xr_target_executor.
    """

    contract = {}

    def __init__(self, cfg):
        from robojudo.controller.omnihand_runtime import OMNIHAND_JOINT_NAMES

        self.hand_names = list(OMNIHAND_JOINT_NAMES) if cfg.xr_profile == "x2_omnihand" else []

    def reset(self):
        pass

    def close(self):
        pass

    def validate(self, feedback, message, positions):
        pass

    def prepare(self, feedback, message, measured):
        feedback.validate(message, message["positions"])
        return message


@unittest.skipUnless(importlib.util.find_spec("xrobotoolkit_teleop"), "install XR executor support")
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

    def test_runtimes_without_notifications_keep_periodic_fallback(self):
        self.receiver._hand_runtime = SimpleNamespace()
        self.assertEqual(self.pipeline.get_idle_notifications(), ())
        self.pipeline.service_idle_notifications()
        self.assertEqual(self.feedback.socket.frames, [])

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

    def desired(self, state, delta=0.01, frame_id=1, bridge_id="test-bridge"):
        targets = state["last_commanded_positions"].copy()
        targets["left_elbow_joint"] += delta
        return {
            "schema_version": 3,
            "type": "upper_body_desired",
            "session_id": state["session_id"],
            "generation": state["generation"],
            "bridge_id": bridge_id,
            "frame_id": frame_id,
            "state_id": state["state_id"],
            "command_id": state["command_id"],
            "intent": state["intent"],
            "status": "homing" if state["intent"] == "HOME" else "tracking",
            "positions": {n: float(np.float32(v)) for n, v in targets.items()},
            "desired_positions": targets.copy(),
            "executor": dict(self.feedback.executor.contract),
            "active_joint_names": list(targets),
            "velocity_limits": {"arm": 8.0, "return": 1.0},
        }

    def receive(self, desired):
        self.receiver.get_data()  # Drain packets preceding a reset before publishing.
        self.receiver._socket.messages.append(desired)
        return {"UpperBodyZmqCtrl": self.receiver.get_data()}

    def test_arm_recording_uses_completed_issued_targets_once(self):
        self.pipeline._set_upper_body_enabled(True)
        state = self.publish()
        desired = self.desired(state, delta=0.05)
        target = self.pipeline._apply_upper_body_override(self.env.default_pos, self.receive(desired))
        self.pipeline._recorder_client = Mock()
        self.pipeline._recording_active = True
        self.pipeline._recording_paused = False
        self.pipeline._send_upper_body_target(target)
        self.publish()
        completed = self.feedback._completed_positions.copy()
        newer = target.copy()
        newer[self.pipeline._upper_body_indices] += 0.1
        extras = {"locomotion_command": np.zeros(4)}
        self.pipeline._record_upper_body_sample(self.env_data, extras, newer, rl_active=True)
        recorder = self.pipeline._recorder_client
        recorder.submit.assert_called_once()
        sample = recorder.submit.call_args.kwargs
        self.assertEqual(sample["joint_names"], self.arm_cfg.joint_names)
        np.testing.assert_allclose(sample["joint_position_commands"], [completed[n] for n in self.arm_cfg.joint_names])
        self.pipeline._record_upper_body_sample(self.env_data, extras, newer, rl_active=True)
        recorder.submit.assert_called_once()

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
        UpperBodyZmqCtrlCfg(joint_names=legacy.joint_names, feedback_endpoint="tcp://127.0.0.1:8571")

    def test_pending_home_can_apply_while_tracking_is_disabled(self):
        state = self.publish()
        self.assertEqual(state["intent"], "HOME")
        desired = self.desired(state)
        target = self.pipeline._apply_upper_body_override(self.env.default_pos, self.receive(desired))
        expected = np.asarray(list(desired["positions"].values()), dtype=np.float32)
        np.testing.assert_array_equal(target[self.pipeline._upper_body_indices], expected)
        self.assertFalse(self.pipeline._upper_body_stream_was_fresh)
        self.pipeline._send_policy_target(target)
        self.publish()
        self.assertEqual(self.feedback.command_id, 1)
        self.assertEqual(len(self.sent), 1)
        with self.assertRaises(ValueError):
            self.feedback.validate(desired, desired["positions"])

    def test_offline_bridge_holds_and_preserves_home_request(self):
        self.publish()
        previous = self.pipeline._upper_body_filtered.copy()
        previous[3] += 0.2
        self.pipeline._upper_body_filtered = previous.copy()
        for _ in range(3):
            target = self.pipeline._apply_upper_body_override(self.env.default_pos, {})
            np.testing.assert_array_equal(target[self.pipeline._upper_body_indices], previous)
            self.assertEqual(self.publish()["intent"], "HOME")

    def test_tracking_outage_holds_and_mode_change_revokes_desired_targets(self):
        self.pipeline._set_upper_body_enabled(True)
        state = self.publish()
        desired = self.desired(state, delta=0.1)
        target = self.pipeline._apply_upper_body_override(self.env.default_pos, self.receive(desired))
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
            self.feedback.validate(desired, desired["positions"])

    def test_expired_and_cached_measurements_are_rejected(self):
        state = self.publish()
        desired = self.desired(state)
        self.clock.return_value = 10.101
        with self.assertRaisesRegex(ValueError, "expired"):
            self.feedback.validate(desired, desired["positions"])
        cached = self.publish()
        self.assertGreater(cached["state_id"], state["state_id"])
        self.assertFalse(cached["valid"])
        with self.assertRaises(ValueError):
            self.feedback.validate(self.desired(cached), desired["positions"])

    def test_delayed_command_is_rechecked_at_application(self):
        state = self.publish()
        data = self.receive(self.desired(state))
        self.clock.return_value = 10.101
        previous = self.pipeline._upper_body_filtered.copy()
        target = self.pipeline._apply_upper_body_override(self.env.default_pos, data)
        np.testing.assert_array_equal(target[self.pipeline._upper_body_indices], previous)

    def test_tracking_command_outage_revokes_generation_even_with_fresh_robot_state(self):
        self.pipeline._set_upper_body_enabled(True)
        state = self.publish()
        self.receive(self.desired(state))
        self.clock.return_value = 10.2
        self.receiver.get_data()
        self.clock.return_value = 10.3
        self.receiver.get_data()
        self.env_data.joint_sample_times[:] = 10.3
        recovered = self.publish()
        self.assertTrue(recovered["valid"])
        self.assertGreater(recovered["generation"], state["generation"])
        self.assertEqual(recovered["reset_reason"], "tracking command timeout")

    def test_stalled_poll_revokes_the_lease_as_a_recoverable_timeout(self):
        self.pipeline._set_upper_body_enabled(True)
        state = self.publish()
        self.assertTrue(self.receive(self.desired(state))["UpperBodyZmqCtrl"]["fresh"])
        self.clock.return_value = 10.0 + self.arm_cfg.timeout_s + 0.01
        self.assertFalse(self.receiver.get_data()["fresh"])
        self.assertEqual(self.feedback.generation, state["generation"] + 1)
        self.assertEqual(self.feedback.reset_reason, "tracking command timeout")

    def test_rejected_combined_waist_transition_holds_both_commands(self):
        waist = "waist_yaw_joint"
        index = self.env.joint_names.index(waist)
        pipeline = self.pipeline
        pipeline._xr_checked_waist_yaw = 0.1
        pipeline._upper_body_previous = pipeline._upper_body_filtered.copy()
        pipeline._upper_body_filtered = pipeline._upper_body_previous + 0.01
        self.feedback.latest_measurements = dict(zip(self.env.joint_names, self.env.dof_pos.tolist(), strict=True))
        self.feedback._completed_positions = dict(
            zip(self.arm_cfg.joint_names, pipeline._upper_body_previous, strict=True)
        )
        self.feedback._states[1] = (10.1, {"generation": self.feedback.generation})
        desired = {"positions": dict(zip(self.arm_cfg.joint_names, pipeline._upper_body_filtered, strict=True))}
        pipeline._upper_body_pending = desired
        target = self.env.default_pos.copy()
        target[index] = 0.2
        target[pipeline._upper_body_indices] = pipeline._upper_body_filtered
        receiver = SimpleNamespace(
            feedback=self.feedback,
            pending=None,
            hand_snapshot=lambda: {"fresh": True, "positions": {}},
            _log_invalid_message=Mock(),
        )
        with patch.object(
            self.feedback.executor.model, "validate_transition", side_effect=RuntimeError("collision")
        ) as check:
            self.assertFalse(pipeline._check_xr_waist_transition(target, receiver, desired))
        self.assertEqual(check.call_args.kwargs, {"independent_joints": (waist,)})
        self.assertAlmostEqual(check.call_args.args[2][waist], 0.2)
        self.assertAlmostEqual(float(target[index]), 0.1)
        np.testing.assert_array_equal(target[pipeline._upper_body_indices], pipeline._upper_body_previous)
        self.assertIsNone(pipeline._upper_body_pending)

    def test_return_is_bounded_by_robot_executor(self):
        self.feedback.cfg.max_joint_velocity_rad_s = 0.25
        state = self.publish()
        desired = self.desired(state, delta=0.1)
        issued = self.feedback.executor.prepare(self.feedback, desired, state["positions"])
        self.assertLessEqual(
            abs(issued["positions"]["left_elbow_joint"] - state["last_commanded_positions"]["left_elbow_joint"]),
            0.005001,
        )

    def test_duplicate_partial_invalid_and_over_speed_commands_are_rejected(self):
        state = self.publish()
        valid = self.desired(state)
        self.feedback.validate(valid, valid["positions"], receiving=True)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.feedback.validate(valid, valid["positions"], receiving=True)
        for mutation in ("partial", "session", "boolean", "limits"):
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
        valid = self.desired(state)
        self.feedback.validate(valid, valid["positions"], receiving=True)
        restart = self.desired(state, bridge_id="restarted")
        with self.assertRaisesRegex(ValueError, "restarted"):
            self.feedback.validate(restart, restart["positions"], receiving=True)
        next_state = self.publish()
        self.assertGreater(next_state["generation"], state["generation"])
        fresh = self.desired(next_state, bridge_id="restarted")
        self.feedback.validate(fresh, fresh["positions"], receiving=True)

    def test_completion_identity_and_targets_are_cleared_on_generation_reset(self):
        state = self.publish()
        desired = self.desired(state)
        self.feedback.acknowledge(desired)
        completed = self.publish()
        self.assertEqual(completed["applied_bridge_id"], desired["bridge_id"])
        self.assertEqual(completed["applied_frame_id"], desired["frame_id"])
        self.assertEqual(completed["last_commanded_positions"], desired["positions"])
        self.feedback.reset()
        reset = self.publish()
        self.assertIsNone(reset["applied_bridge_id"])
        self.assertIsNone(reset["applied_frame_id"])
        self.assertGreater(reset["generation"], completed["generation"])

    def test_invalid_initial_targets_recover_without_a_generation_reset(self):
        for invalid in (np.nan, np.inf, -np.inf):
            with self.subTest(invalid=invalid):
                self.feedback.reset()
                self.pipeline._upper_body_filtered[0] = invalid
                first = self.publish()
                self.assertFalse(first["valid"])
                self.assertIsNone(self.feedback._completed_positions)
                self.assertIsNone(first["last_commanded_positions"][self.arm_cfg.joint_names[0]])

                self.pipeline._upper_body_filtered[0] = self.pipeline._upper_body_default[0]
                recovered = self.publish()
                self.assertTrue(recovered["valid"])
                self.assertEqual(recovered["generation"], first["generation"])
                self.assertEqual(self.feedback._completed_positions, recovered["last_commanded_positions"])
                self.assertTrue(all(np.isfinite(value) for value in recovered["last_commanded_positions"].values()))

    def test_completed_baseline_cannot_hide_nonfinite_current_targets(self):
        self.publish()
        self.pipeline._upper_body_filtered[0] = np.nan
        state = self.publish()
        self.assertFalse(state["valid"])
        self.assertTrue(all(np.isfinite(value) for value in state["last_commanded_positions"].values()))

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
        message = self.desired(state)
        message.update(type="upper_body_visualization", positions=None, target_poses={"left": np.eye(4).tolist()})
        data = self.receive(message)["UpperBodyZmqCtrl"]
        self.assertEqual(set(data["target_poses"]), {"left"})
        self.assertFalse(data["fresh"])
        self.assertIsNone(self.feedback.last_desired_at)
        self.assertEqual(self.feedback.command_id, 0)
        self.clock.return_value = 10.101
        self.assertEqual(self.receiver.get_data()["target_poses"], {})

    def test_visualization_rejects_invalid_transforms_without_rejecting_valid_commands(self):
        state = self.publish()
        message = self.desired(state)
        message["target_poses"] = {"left": np.zeros((4, 4)).tolist()}
        data = self.receive(message)["UpperBodyZmqCtrl"]
        self.assertTrue(data["fresh"])
        self.assertEqual(data["target_poses"], {})

    def test_expired_visualization_is_debug_only_and_does_not_hide_command_rejections(self):
        state = self.publish()
        message = self.desired(state)
        message.update(type="upper_body_visualization", positions=None, target_poses={"left": np.eye(4).tolist()})
        self.clock.return_value = 10.101
        with self.assertLogs("robojudo.controller.upper_body_zmq_ctrl", level="DEBUG") as logs:
            self.receive(message)
        self.assertTrue(all(record.levelname == "DEBUG" for record in logs.records))
        self.assertEqual(self.receiver._last_invalid_log_at, float("-inf"))
        with self.assertLogs("robojudo.controller.upper_body_zmq_ctrl", level="WARNING") as logs:
            self.receive(self.desired(state))
        self.assertIn("Rejected upper-body ZMQ message", logs.output[0])
        self.assertIn("expired", logs.output[0])

    def test_completion_timestamps_follow_acknowledged_desired_and_reset(self):
        state = self.publish()
        desired = self.desired(state)
        desired["_command_timestamps"] = {"received": 9.98, "acknowledged": 9.99}
        self.feedback.acknowledge(desired)
        frame = self.publish()
        stamps = frame["feedback_diagnostics"]["command_timestamps"]
        self.assertEqual(
            stamps, {"received": 9.98, "acknowledged": 9.99, "feedback_built": 10.0, "first_feedback_built": 10.0}
        )
        self.assertEqual(frame["applied_frame_id"], desired["frame_id"])
        self.feedback.reset()
        self.assertNotIn("command_timestamps", self.publish()["feedback_diagnostics"])

    def test_first_ack_state_id_tracks_accepted_send_attempt_not_delivery(self):
        state = self.publish()
        desired = self.desired(state)
        self.feedback.acknowledge(desired)
        first = self.publish()
        next_state = self.publish()
        self.assertEqual(first["feedback_diagnostics"]["first_ack_state_id"], first["state_id"])
        self.assertEqual(next_state["feedback_diagnostics"]["first_ack_state_id"], first["state_id"])
        # A successful memory PUB send is not evidence of subscriber delivery.
        self.feedback.reset()
        self.assertIsNone(self.feedback._first_ack_state_id)

    def test_silent_pub_drop_preserves_first_attempt_marker_in_later_snapshot(self):
        state = self.publish()
        self.feedback.acknowledge(self.desired(state))
        with patch.object(self.feedback.socket, "send_json", return_value=None):
            self.publish()
        missing_state_id = self.feedback.state_id
        later = self.publish()
        self.assertEqual(later["feedback_diagnostics"]["first_ack_state_id"], missing_state_id)
        self.assertNotIn(missing_state_id, [f["state_id"] for f in self.feedback.socket.frames])

    def test_first_feedback_timestamp_retries_failed_send_and_stays_fixed(self):
        desired = self.desired(self.publish())
        desired["_command_timestamps"] = {"received": 9.98, "acknowledged": 9.99}
        self.feedback.acknowledge(desired)
        with patch.object(self.feedback.socket, "send_json", side_effect=zmq.Again()):
            self.publish()
        self.assertIsNone(self.feedback._first_feedback_built)
        self.clock.return_value = 10.02
        first = self.publish()
        self.clock.return_value = 10.04
        later = self.publish()
        self.assertEqual(first["feedback_diagnostics"]["command_timestamps"]["first_feedback_built"], 10.02)
        stamps = later["feedback_diagnostics"]["command_timestamps"]
        self.assertEqual(stamps["first_feedback_built"], 10.02)
        self.assertEqual(stamps["feedback_built"], 10.04)
        self.feedback.acknowledge(dict(desired, frame_id=desired["frame_id"] + 1))
        self.assertIsNone(self.feedback._first_feedback_built)
        self.clock.return_value = 10.06
        self.assertEqual(self.publish()["feedback_diagnostics"]["command_timestamps"]["first_feedback_built"], 10.06)
        self.feedback.reset()
        self.assertIsNone(self.feedback._first_feedback_built)

    def test_ack_keeps_fresh_visualization_and_desired_lease(self):
        state = self.publish()
        desired = self.desired(state)
        self.feedback.validate(desired, desired["positions"], receiving=True)
        self.feedback.acknowledge()
        visual = dict(desired, target_poses={"left": np.eye(4).tolist()})
        poses, _ = self.feedback.visualization(visual)
        self.assertIn("left", poses)
        self.feedback.validate(desired, desired["positions"])

    def test_older_visualization_cannot_replace_newer_target(self):
        old = self.publish()
        self.clock.return_value = 10.02
        self.env_data.joint_sample_times[:] = 10.02
        new = self.publish()
        for state, offset in ((new, 0.3), (old, 0.1)):
            target = np.eye(4)
            target[0, 3] = offset
            message = self.desired(state)
            message.update(type="upper_body_visualization", positions=None, target_poses={"left": target.tolist()})
            self.receive(message)
        self.assertEqual(self.receiver.get_data()["target_poses"]["left"][0][3], 0.3)

    def test_desired_arriving_during_inference_is_applied_in_the_same_cycle(self):
        self.pipeline._set_upper_body_enabled(True)
        state = self.publish()
        desired = self.desired(state)
        self.receiver.get_data()  # Drain the enable-session reset before inference.

        def infer(obs):
            self.clock.return_value = 10.06
            self.receiver._socket.messages.append(desired)
            return self.env.default_pos.copy()

        self.pipeline.policy = SimpleNamespace(
            get_observation=lambda env_data, ctrl_data: (None, {}), get_pd_target=infer
        )
        target, _ = self.pipeline._step_rl_policy(self.env_data, {"UpperBodyZmqCtrl": {"fresh": False}}, dry_run=False)
        index = self.env.joint_names.index("left_elbow_joint")
        self.assertAlmostEqual(target[index], desired["positions"]["left_elbow_joint"], places=6)
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
