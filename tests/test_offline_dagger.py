import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import zmq

from robojudo.controller.casia_hand_runtime import CASIA_JOINT_NAMES
from robojudo.controller.ctrl_cfgs import Gr00tZmqCtrlCfg
from robojudo.controller.gr00t_zmq_ctrl import Gr00tZmqCtrl
from robojudo.pipeline.gr00t_locomanipulation_pipeline import Gr00tLocomanipulationPipelineMixin
from robojudo.policy.gr00t_locomanipulation_policy import Gr00tLocomanipulationPolicyMixin


class _Recorder:
    def __init__(self):
        self.samples = []

    def submit(self, **sample):
        self.samples.append(sample)


class _EmptySocket:
    def recv_json(self, flags=0):
        raise zmq.Again()


class _MessageSocket:
    def __init__(self, *messages):
        self.messages = list(messages)

    def recv_json(self, flags=0):
        del flags
        if not self.messages:
            raise zmq.Again()
        return self.messages.pop(0)


class _ManualLocomotionBase:
    def _get_commands(self, ctrl_data):
        self.manual_ctrl_data = ctrl_data
        return np.asarray([0.35, -0.2, 0.15, 0.73, 0.4], dtype=np.float32)


class _DaggerLocomotionPolicy(Gr00tLocomanipulationPolicyMixin, _ManualLocomotionBase):
    pass


class TestOfflineDaggerController(unittest.TestCase):
    """Protect the offline DAgger session and policy/expert arbitration contract."""

    def make_controller(self):
        controller = Gr00tZmqCtrl.__new__(Gr00tZmqCtrl)
        controller.cfg_ctrl = Gr00tZmqCtrlCfg(
            joint_names=["left_arm", "right_arm"],
            offline_dagger_enabled=True,
        )
        controller._joint_names = ("left_arm", "right_arm")
        controller._joint_name_set = set(controller._joint_names)
        controller._observation_stream_id = "robot-run"
        controller._observation_snapshot_lock = threading.Lock()
        controller._takeover_enabled = True
        controller._control_session = 3
        controller._expert_intervention = True
        controller._intervention_session = 1
        controller._socket = _EmptySocket()
        controller._expert_socket = _EmptySocket()
        controller._latest_positions = {"left_arm": 0.1, "right_arm": -0.1}
        controller._latest_policy_hands = None
        controller._latest_locomotion_command = np.asarray([0.2, 0.0, 0.0, 0.7])
        controller._latest_sequence = 5
        controller._latest_command_stream_id = "robot-run"
        controller._latest_command_session = 3
        controller._last_received_at = 10.0
        controller._latest_expert_positions = {"left_arm": 0.4, "right_arm": -0.5}
        controller._latest_expert_hands = None
        controller._latest_expert_frame_id = 12
        controller._latest_expert_session = 1
        controller._expert_last_received_at = 10.0
        controller._expert_action_received_at = 10.0
        controller._expert_stream_ready = True
        controller._last_invalid_log_at = float("-inf")
        controller._hand_runtime = None
        controller._observation_error = None
        controller._observation_ready = None
        controller._published_observations = 0
        controller._dropped_observations = 0
        controller._camera_encoder_drops = {}
        return controller

    @staticmethod
    def expert_message(*, frame_id=12, session=1, active=True, valid=True):
        return {
            "schema_version": 1,
            "type": "synchronized_teleop_frame",
            "frame_id": frame_id,
            "arm": {
                "valid": True,
                "joint_names": ["left_arm", "right_arm"],
                "qpos": [0.4, -0.5],
            },
            "dagger": {
                "feedback_fresh": True,
                "intervention_active": active,
                "expert_valid": valid,
                "stream_id": "robot-run",
                "intervention_session": session,
            },
        }

    def test_only_current_intervention_session_is_executable(self):
        controller = self.make_controller()
        ready, positions, _, frame_id, session = controller._decode_expert_message(
            self.expert_message(), 1
        )
        self.assertTrue(ready)
        self.assertEqual(positions, {"left_arm": 0.4, "right_arm": -0.5})
        self.assertEqual((frame_id, session), (12, 1))

        ready, positions, _, _, _ = controller._decode_expert_message(
            self.expert_message(session=0), 1
        )
        self.assertTrue(ready)
        self.assertIsNone(positions)

    def test_select_level_creates_one_session_per_rising_edge(self):
        controller = self.make_controller()
        controller._expert_intervention = False
        controller._intervention_session = 0

        held = {"UnitreeCtrl": {"fresh": True, "pressed_buttons": ["Select"]}}
        controller._update_expert_intervention(held)
        controller._update_expert_intervention(held)
        self.assertTrue(controller._expert_intervention)
        self.assertEqual(controller._intervention_session, 1)

        controller._update_expert_intervention(
            {"UnitreeCtrl": {"fresh": True, "pressed_buttons": []}}
        )
        controller._update_expert_intervention(held)
        self.assertEqual(controller._intervention_session, 2)

        # The recorder pause chord must not create a DAgger intervention.
        controller._update_expert_intervention(
            {"UnitreeCtrl": {"fresh": True, "pressed_buttons": ["L1", "R1", "Select"]}}
        )
        self.assertFalse(controller._expert_intervention)

    def test_expert_arm_hand_candidate_does_not_require_fresh_policy_chunk(self):
        controller = self.make_controller()
        with patch("robojudo.controller.gr00t_zmq_ctrl.time.monotonic", return_value=10.0):
            data = controller.get_data()
        self.assertTrue(data["fresh"])
        self.assertTrue(data["policy_fresh"])
        self.assertTrue(data["expert_applied"])
        self.assertEqual(data["action_source"], "expert")
        self.assertEqual(data["joint_positions"], {"left_arm": 0.4, "right_arm": -0.5})

        controller._last_received_at = 9.0
        with patch("robojudo.controller.gr00t_zmq_ctrl.time.monotonic", return_value=10.0):
            data = controller.get_data()
        self.assertTrue(data["fresh"])
        self.assertFalse(data["policy_fresh"])
        self.assertTrue(data["expert_applied"])

    def test_new_invalid_expert_frame_revokes_previous_candidate_immediately(self):
        controller = self.make_controller()
        controller._expert_socket = _MessageSocket(
            self.expert_message(frame_id=13, valid=False)
        )

        controller._receive_expert_available(10.01)

        self.assertEqual(controller._latest_expert_frame_id, 13)
        self.assertEqual(controller._latest_expert_session, 1)
        self.assertEqual(controller._latest_expert_positions, {})
        self.assertIsNone(controller._expert_action_received_at)
        with patch("robojudo.controller.gr00t_zmq_ctrl.time.monotonic", return_value=10.01):
            data = controller.get_data()
        self.assertFalse(data["expert_applied"])
        self.assertEqual(data["action_source"], "policy")
        self.assertEqual(data["expert_frame_id"], 13)
        self.assertEqual(data["joint_positions"], {"left_arm": 0.1, "right_arm": -0.1})

    def test_recorded_hand_action_matches_current_arbitration_before_worker_applies_it(self):
        class DelayedHandRuntime:
            def __init__(self):
                self.snapshot = {
                    "joint_names": list(CASIA_JOINT_NAMES),
                    "joint_positions": np.zeros(20),
                    "joint_position_commands": np.zeros(20),
                    "enabled": True,
                    "joint_state_fresh": True,
                    "fresh": False,
                }
                self.accepted = None
                self.command_count = 0

            def get_data(self):
                return self.snapshot

            def set_joint_commands(self, left, right, timestamp_ns, frame_id):
                # Model asynchronous execution and SDK clipping: feedback still
                # reports the previous command after the new one is enqueued.
                self.accepted = np.clip(np.concatenate((left, right)), -0.05, 0.05)
                self.command_count += 1
                return self.accepted.copy()

        controller = self.make_controller()
        controller._hand_runtime = DelayedHandRuntime()
        controller._latest_expert_hands = (np.ones(10), np.ones(10))
        controller._latest_policy_hands = (-np.ones(10), -np.ones(10))
        controller._hand_command_filtered = None
        controller._last_hand_command_at = None
        pipeline = Gr00tLocomanipulationPipelineMixin()
        pipeline._upper_body_cfg = SimpleNamespace(
            offline_dagger_enabled=True, joint_names=list(controller._joint_names)
        )
        pipeline._upper_body_indices = np.asarray([0, 1])
        pipeline._upper_body_enabled = True
        pipeline._upper_body_stream_was_fresh = True
        pipeline._recording_active = True
        pipeline._recording_paused = False
        pipeline._upper_body_control_available = lambda: True
        pipeline._recorder_client = _Recorder()

        for expert, enabled, feedback_fresh in (
            (True, True, True),
            (False, True, True),
            (True, False, True),
            (True, True, False),
        ):
            with self.subTest(expert=expert, enabled=enabled, feedback_fresh=feedback_fresh):
                controller._expert_intervention = expert
                controller._hand_command_filtered = None
                controller._hand_runtime.snapshot.update(enabled=enabled, joint_state_fresh=feedback_fresh)
                with patch("robojudo.controller.gr00t_zmq_ctrl.time.monotonic", return_value=10.01):
                    stream = controller.get_data()
                self.assertEqual(stream["casia_hand"]["fresh"], enabled and feedback_fresh)
                pipeline._gr00t_record_stream = stream
                arms = np.asarray(list(stream["joint_positions"].values()))
                sample_count = len(pipeline._recorder_client.samples)
                pipeline._record_upper_body_sample(
                    SimpleNamespace(dof_pos=np.zeros(2)),
                    {"locomotion_command": stream["locomotion_command"]},
                    arms,
                    rl_active=True,
                )
                if not enabled or not feedback_fresh:
                    self.assertEqual(len(pipeline._recorder_client.samples), sample_count)
                    continue
                self.assertEqual(len(pipeline._recorder_client.samples), sample_count + 1)
                sample = pipeline._recorder_client.samples[-1]
                self.assertEqual(sample["dagger"]["expert_applied"], expert)
                np.testing.assert_allclose(sample["joint_position_commands"][:2], arms)
                np.testing.assert_allclose(sample["joint_position_commands"][2:], 0.05 if expert else -0.05)
                np.testing.assert_array_equal(sample["joint_positions"][2:], np.zeros(20))
                np.testing.assert_array_equal(
                    controller._hand_runtime.snapshot["joint_position_commands"], np.zeros(20)
                )

        command_count = controller._hand_runtime.command_count
        with patch("robojudo.controller.gr00t_zmq_ctrl.time.monotonic", return_value=11.0):
            stream = controller.get_data()
        self.assertFalse(stream["fresh"])
        self.assertEqual(controller._hand_runtime.command_count, command_count)

        # Ordinary GR00T preserves the runtime snapshot and its freshness; only
        # offline DAgger substitutes the current arbitrated recording action.
        controller.cfg_ctrl.offline_dagger_enabled = False
        with patch("robojudo.controller.gr00t_zmq_ctrl.time.monotonic", return_value=10.01):
            stream = controller.get_data()
        self.assertIs(stream["casia_hand"], controller._hand_runtime.snapshot)
        self.assertEqual(controller._hand_runtime.command_count, command_count)

    def test_restarted_expert_frame_sequence_is_accepted_after_timeout(self):
        controller = self.make_controller()
        controller._expert_socket = _MessageSocket(self.expert_message(frame_id=0))

        # A rollback while the accepted stream is fresh is still rejected.
        controller._receive_expert_available(10.1)
        self.assertEqual(controller._latest_expert_frame_id, 12)
        self.assertEqual(controller._expert_last_received_at, 10.0)

        # Rejected rollback frames do not refresh liveness, so the same
        # restarted publisher is accepted once the previous stream times out.
        controller._expert_socket = _MessageSocket(self.expert_message(frame_id=0))
        controller._receive_expert_available(10.3)
        self.assertEqual(controller._latest_expert_frame_id, 0)
        self.assertEqual(controller._latest_expert_session, 1)
        self.assertEqual(controller._expert_last_received_at, 10.3)
        self.assertEqual(controller._expert_action_received_at, 10.3)

    def test_expert_applied_uses_all_four_manual_locomotion_dimensions(self):
        policy = _DaggerLocomotionPolicy()
        policy._gr00t_takeover_was_enabled = False
        policy._command_defaults = np.asarray([0.0, 0.0, 0.0, 0.75, 0.0], dtype=np.float32)
        policy._target_waist_yaw = 0.0
        policy.cmd = policy._command_defaults.copy()
        stream = {
            "takeover_enabled": True,
            "expert_applied": True,
            "locomotion_command": np.asarray([9.0, 9.0, 9.0, 9.0]),
        }
        ctrl_data = {"Gr00tZmqCtrl": stream, "UnitreeCtrl": {"fresh": True}}

        commands = policy._get_commands(ctrl_data)

        np.testing.assert_allclose(commands, [0.35, -0.2, 0.15, 0.73, 0.0])
        np.testing.assert_allclose(stream["locomotion_command"], [0.35, -0.2, 0.15, 0.73])
        self.assertEqual(stream["locomotion_source"], "expert")
        self.assertIs(policy.manual_ctrl_data, ctrl_data)

    def test_g1_offline_dagger_config_is_isolated_and_uses_distinct_record_port(self):
        from robojudo.config.g1.g1_vla_cfg import (
            g1_23_gr00t_locomanipulation_stiff_real,
            g1_23_gr00t_offline_dagger_stiff_real,
        )

        normal = g1_23_gr00t_locomanipulation_stiff_real()
        dagger = g1_23_gr00t_offline_dagger_stiff_real()
        self.assertFalse(normal.ctrl[-1].offline_dagger_enabled)
        self.assertEqual(normal.ctrl[-1].endpoint, "tcp://192.168.123.222:8559")
        self.assertEqual(normal.ctrl[-1].expert_endpoint, "tcp://192.168.123.222:8560")
        self.assertTrue(dagger.ctrl[-1].offline_dagger_enabled)
        self.assertEqual(dagger.ctrl[-1].endpoint, "tcp://192.168.123.222:8559")
        self.assertEqual(dagger.ctrl[-1].expert_endpoint, "tcp://192.168.123.222:8560")
        self.assertEqual(dagger.record.endpoint, "tcp://*:8562")
        self.assertTrue(dagger.record.enabled)
        self.assertEqual([camera.type for camera in normal.ctrl[-1].cameras], ["realsense"] * 3)
        self.assertEqual([camera.type for camera in dagger.ctrl[-1].cameras], ["zmq"] * 3)
        self.assertEqual(
            [camera.options["endpoint"] for camera in dagger.ctrl[-1].cameras],
            [f"tcp://127.0.0.1:{port}" for port in (8571, 8572, 8573)],
        )

    def test_recording_keeps_full_rollout_with_expert_source_label(self):
        pipeline = Gr00tLocomanipulationPipelineMixin.__new__(
            Gr00tLocomanipulationPipelineMixin
        )
        pipeline._upper_body_cfg = SimpleNamespace(
            offline_dagger_enabled=True,
            joint_names=["left_arm", "right_arm"],
        )
        pipeline._upper_body_indices = np.asarray([1, 2])
        pipeline._upper_body_enabled = True
        pipeline._upper_body_stream_was_fresh = True
        pipeline._recording_active = True
        pipeline._recording_paused = False
        pipeline._upper_body_control_available = lambda: True
        pipeline._recorder_client = _Recorder()
        pipeline._gr00t_record_stream = {
            "expert_intervention": True,
            "expert_applied": True,
            "intervention_session": 7,
            "action_source": "expert",
            "expert_frame_id": 41,
            "casia_hand": {
                "fresh": True,
                "joint_names": ["left_thumb", "right_thumb"],
                "joint_positions": np.asarray([0.2, 0.3]),
                "joint_position_commands": np.asarray([0.4, 0.5]),
            },
        }
        env_data = SimpleNamespace(dof_pos=np.asarray([0.0, 0.1, -0.1]))

        pipeline._record_upper_body_sample(
            env_data,
            {"locomotion_command": np.asarray([0.2, 0.0, 0.1, 0.75])},
            np.asarray([0.0, 0.4, -0.5]),
            rl_active=True,
        )

        sample = pipeline._recorder_client.samples[0]
        self.assertEqual(sample["dagger"]["intervention_session"], 7)
        self.assertEqual(sample["dagger"]["action_source"], "expert")
        self.assertEqual(sample["dagger"]["expert_frame_id"], 41)
        self.assertEqual(len(sample["joint_names"]), 4)


if __name__ == "__main__":
    unittest.main()
