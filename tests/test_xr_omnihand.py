"""Serial identity and combined XR application checks without hardware."""

import multiprocessing as mp
import queue
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import zmq
from box import Box
from test_upper_body_feedback import MemoryContext

from robojudo.controller.ctrl_cfgs import OmniHandCfg, UpperBodyOmniHandZmqCtrlCfg
from robojudo.controller.omnihand_runtime import (
    ALL_HAND_NAMES,
    OMNIHAND_JOINT_NAMES,
    OmniHandRuntime,
    _hand_loop,
    _worker,
    connect_verified_hand,
    measured_command_baseline,
)
from robojudo.controller.upper_body_feedback import UpperBodyFeedback


def serial_config():
    return OmniHandCfg(
        left_hcan_serial="adapter-left",
        right_hcan_serial="adapter-right",
        left_hand_serial="LEFT-HAND-EXACT",
        right_hand_serial="RIGHT-HAND-EXACT",
    )


class HcanIdentityTests(unittest.TestCase):
    def test_shared_spawned_worker_verifies_applies_and_reaps(self):
        fake_sdk = """
from types import SimpleNamespace
import multiprocessing as mp
import weakref
assert mp.parent_process() is not None, 'SDK imported in parent'
_opened = {}
HandType = SimpleNamespace(LEFT=1, RIGHT=2)
class Hand:
    def __init__(self, side):
        self.side = side
        self.angles = [0.] * 12
    def set_frame_recv_timeout(self, timeout): pass
    def get_vendor_info(self):
        return SimpleNamespace(product_model='O12', product_seq_num=(
            'LEFT-HAND-EXACT' if self.side == 1 else 'RIGHT-HAND-EXACT'))
    def get_device_info(self): return SimpleNamespace(hand_device_id=1)
    def get_all_joint_angles(self): return self.angles + [0.] * 7
    def set_all_active_joint_angles(self, angles): self.angles = angles
class OmniHandPro2025:
    @staticmethod
    def create_hand_by_hcan(**kwargs):
        side = kwargs['hand_type']
        if side == 1:
            assert not _opened, 'left must open first'
        else:
            assert _opened[1]() is not None, 'left must remain alive in same process'
        hand = Hand(side)
        _opened[side] = weakref.ref(hand)
        return hand
"""
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "omnihand.py").write_text(fake_sdk)
            with patch.object(sys, "path", [directory, *sys.path]):
                runtime = OmniHandRuntime(serial_config())
                process = runtime._process
                try:
                    runtime.set_takeover_enabled(True)
                    positions = dict.fromkeys(OMNIHAND_JOINT_NAMES, 0.0)
                    positions["L_index_mcp_joint"] = float(np.float32(0.1))
                    identity = (1, 7)
                    runtime.submit(positions, identity, time.monotonic() + 1)
                    deadline = time.monotonic() + 1
                    while time.monotonic() < deadline:
                        state = runtime.get_data()
                        if state["applied"] == identity:
                            break
                        time.sleep(0.01)
                    self.assertEqual(state["applied"], identity)
                    self.assertEqual(state["targets"]["L_index_mcp_joint"], positions["L_index_mcp_joint"])
                    self.assertTrue(state["fresh"])
                    runtime.acknowledge()
                    for frame_id in range(50):
                        runtime.set_joint_commands(np.zeros(12), np.zeros(12), 123, frame_id)
                    deadline = time.monotonic() + 1
                    while time.monotonic() < deadline:
                        state = runtime.get_data()
                        if state["applied_frame_id"] == 49:
                            break
                        time.sleep(0.01)
                    self.assertEqual(state["applied_frame_id"], 49)
                    self.assertEqual(state["applied_source_timestamp_ns"], 123)
                finally:
                    runtime.close()
                self.assertFalse(process.is_alive())
                runtime.close()
                # Read-only startup verifies both hands but never writes to either.
                Path(directory, "omnihand.py").write_text(
                    fake_sdk.replace("self.angles = angles", "raise AssertionError('read-only worker wrote motion')")
                )
                runtime = OmniHandRuntime(serial_config(), read_only=True, require_serials=True)
                try:
                    runtime.set_takeover_enabled(True)
                    self.assertFalse(runtime.get_data()["enabled"])
                    self.assertTrue(runtime.get_data()["joint_state_fresh"])
                    with self.assertRaisesRegex(RuntimeError, "unauthorized"):
                        runtime.submit(positions, (2, 8), time.monotonic() + 1)
                finally:
                    runtime.close()
                self.assertFalse(runtime._process.is_alive())

                # Right identity failure must release the process holding the left hand.
                Path(directory, "omnihand.py").write_text(fake_sdk.replace("RIGHT-HAND-EXACT", "WRONG-RIGHT-HAND"))
                failed = OmniHandRuntime.__new__(OmniHandRuntime)
                with self.assertRaises(RuntimeError):
                    failed.__init__(serial_config())
                self.assertFalse(failed._process.is_alive())

                # Native open may block indefinitely; startup and cleanup remain bounded.
                Path(directory, "omnihand.py").write_text(
                    fake_sdk.replace(
                        "side = kwargs['hand_type']",
                        "side = kwargs['hand_type']; __import__('time').sleep(60) if side == 2 else None",
                    )
                )
                blocked = OmniHandRuntime.__new__(OmniHandRuntime)
                cfg = serial_config()
                cfg.startup_timeout_s = 2.0
                with self.assertRaises(TimeoutError):
                    blocked.__init__(cfg)
                self.assertFalse(blocked._process.is_alive())

                # A native read can also hang after successful startup. The parent
                # must become stale and close must still reap the SDK process.
                Path(directory, "omnihand.py").write_text(
                    fake_sdk.replace(
                        "def get_all_joint_angles(self): return self.angles + [0.] * 7",
                        """def get_all_joint_angles(self):
        from pathlib import Path
        if Path(__file__).with_name('block-read').exists():
            __import__('time').sleep(60)
        return self.angles + [0.] * 7""",
                    )
                )
                runtime = OmniHandRuntime(serial_config(), read_only=True)
                try:
                    Path(directory, "block-read").touch()
                    time.sleep(0.4)
                    self.assertFalse(runtime.get_data()["joint_state_fresh"])
                finally:
                    started = time.monotonic()
                    runtime.close()
                self.assertLess(time.monotonic() - started, 6)
                self.assertFalse(runtime._process.is_alive())

    def test_measured_baseline_is_bounded_and_float32(self):
        measured = dict.fromkeys(OMNIHAND_JOINT_NAMES, 0.0)
        measured["L_thumb_mcp_joint"] = 1e-20
        measured["L_index_mcp_joint"] = 0.123456789
        result = measured_command_baseline(measured)
        self.assertEqual(result["L_thumb_mcp_joint"], 0.0)
        self.assertTrue(all(float(np.float32(value)) == value for value in result.values()))
        measured["L_thumb_mcp_joint"] = 0.01
        with self.assertRaises(ValueError):
            measured_command_baseline(measured)

    def connect(self, actual="LEFT-HAND-EXACT", device_id=1):
        hand = Mock()
        hand.get_vendor_info.return_value = SimpleNamespace(product_model="O12", product_seq_num=actual)
        hand.get_device_info.return_value = SimpleNamespace(hand_device_id=device_id)
        factory = Mock(return_value=hand)
        module = SimpleNamespace(
            HandType=SimpleNamespace(LEFT=1, RIGHT=2), OmniHandPro2025=SimpleNamespace(create_hand_by_hcan=factory)
        )
        with patch.dict(sys.modules, omnihand=module):
            result = connect_verified_hand(serial_config(), "left")
        return result, factory

    def test_serial_selection_verifies_without_motion(self):
        hand, factory = self.connect("LEFT-HAND-EXACT\0\0")
        factory.assert_called_once_with(
            hand_type=1, hand_device_id=1, hcan_serial_number="adapter-left", canfd_channel_id=0
        )
        hand.set_frame_recv_timeout.assert_called_once_with(200)
        hand.set_all_active_joint_angles.assert_not_called()
        hand.init.assert_not_called()

    def test_longer_serial_and_wrong_device_are_rejected(self):
        for actual, device_id in (("LEFT-HAND-EXACT-extra", 1), ("RIGHT-HAND-EXACT", 1), ("LEFT-HAND-EXACT", 2)):
            with self.subTest(actual=actual, device_id=device_id), self.assertRaisesRegex(RuntimeError, "identity"):
                self.connect(actual, device_id)

    def test_unreadable_identity_is_not_reported_as_serial_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, "adapter-left.*identity unreadable"):
            self.connect("")

    def test_adapter_open_and_identity_read_errors_have_context(self):
        cfg = serial_config()
        factory = Mock(return_value=None)
        module = SimpleNamespace(
            HandType=SimpleNamespace(LEFT=1, RIGHT=2),
            OmniHandPro2025=SimpleNamespace(create_hand_by_hcan=factory),
        )
        with patch.dict(sys.modules, omnihand=module):
            with self.assertRaisesRegex(RuntimeError, "left.*adapter-left.*could not be opened"):
                connect_verified_hand(cfg, "left")
            factory.side_effect = OSError("USB unavailable")
            with self.assertRaisesRegex(RuntimeError, "left.*adapter-left.*open failed.*USB unavailable"):
                connect_verified_hand(cfg, "left")
            factory.side_effect = None
            hand = Mock()
            hand.get_vendor_info.side_effect = TimeoutError("no response")
            factory.return_value = hand
            with self.assertRaisesRegex(RuntimeError, "adapter-left.*identity read failed.*no response"):
                connect_verified_hand(cfg, "left")

    def test_invalid_serial_configuration_is_rejected(self):
        for values in (
            {"left_hcan_serial": "one"},
            {**serial_config().model_dump(), "right_hcan_serial": "adapter-left"},
            {**serial_config().model_dump(), "transport": "socketcan"},
            {**serial_config().model_dump(), "transport": "zlgcan"},
        ):
            with self.assertRaises(ValueError):
                OmniHandCfg(**values)

    def test_workers_drop_expired_or_revoked_commands(self):
        commands, states, stop = queue.Queue(1), queue.Queue(1), threading.Event()
        errors = queue.Queue(1)
        gate = mp.get_context("spawn").Array("q", [3, 1])
        hand = Mock()
        hand.get_all_joint_angles.return_value = [0.0] * 19
        commands.put((2, time.monotonic() + 1, (2, 1), [0.0] * 24, time.monotonic()))
        with patch("robojudo.controller.omnihand_runtime.connect_verified_hand", return_value=hand):
            thread = threading.Thread(target=_worker, args=(serial_config(), commands, states, stop, gate, errors))
            thread.start()
            try:
                state = states.get(timeout=1)["left"]
                self.assertEqual(len(state["positions"]), 19)
                hand.set_all_active_joint_angles.assert_not_called()
                commands.put((3, time.monotonic() - 1, (3, 2), [0.0] * 24, time.monotonic()))
                time.sleep(0.04)
                hand.set_all_active_joint_angles.assert_not_called()
                commands.put((3, time.monotonic() + 1, (3, 3), [0.0] * 24, time.monotonic()))
                deadline = time.monotonic() + 1
                while time.monotonic() < deadline:
                    state = states.get(timeout=1)["left"]
                    if state["applied"] == (3, 3):
                        break
                self.assertEqual(state["applied"], (3, 3))
            finally:
                stop.set()
                thread.join(timeout=1)
                self.assertFalse(thread.is_alive())

    def run_hand_cycle(self, hand, side="left", gate=None):
        """Run one fake-handle cycle with a deterministic clock."""
        states, errors, stop = queue.Queue(), queue.Queue(), threading.Event()
        gate = gate if gate is not None else mp.get_context("spawn").Array("q", [3, 1])
        commands = queue.Queue(1)
        commands.put((3, 10.1, (3, 1), [0.1] * 12, 10.0))

        def fail(side, stage, exc):
            with gate.get_lock():
                gate[0] += 1
                gate[1] = 0
            errors.put(f"{side}: {stage}: {exc}")
            stop.set()

        with patch.object(stop, "wait", side_effect=lambda _: stop.set()):
            _hand_loop(serial_config(), side, hand, commands, stop, gate, lambda _, state: states.put(state), fail)
        return states, errors, gate

    def test_in_flight_expiry_preserves_target_without_acknowledgement(self):
        hand = Mock()
        hand.get_all_joint_angles.return_value = [0.0] * 19
        with patch("time.monotonic", return_value=10.0) as clock:
            hand.set_all_active_joint_angles.side_effect = lambda _: setattr(clock, "return_value", 10.2)
            states, errors, _ = self.run_hand_cycle(hand)
        state = states.get_nowait()
        self.assertTrue(errors.empty())
        self.assertEqual(state["targets"], [0.1] * 12)
        self.assertIsNone(state["applied"])
        self.assertIsNone(state["generation"])

    def test_in_flight_revocation_does_not_acknowledge_write(self):
        hand = Mock()
        hand.get_all_joint_angles.return_value = [0.0] * 19
        gate = mp.get_context("spawn").Array("q", [3, 1])

        def revoke(_):
            with gate.get_lock():
                gate[0] += 1
                gate[1] = 0

        hand.set_all_active_joint_angles.side_effect = revoke
        with patch("time.monotonic", return_value=10.0):
            states, _, _ = self.run_hand_cycle(hand, gate=gate)
        state = states.get_nowait()
        self.assertEqual(state["targets"], [0.1] * 12)
        self.assertIsNone(state["applied"])

    def test_samples_keep_read_start_time(self):
        hand = Mock()
        with patch("time.monotonic", return_value=10.0) as clock:

            def read():
                clock.return_value = 10.05
                return [0.0] * 19

            hand.get_all_joint_angles.side_effect = read
            states, errors, _ = self.run_hand_cycle(hand)
        self.assertTrue(errors.empty())
        self.assertEqual(states.get_nowait()["sampled_at"], 10.0)

    def test_invalid_measurements_and_write_errors_revoke_gate(self):
        for failure in ("shape", "empty", "nan", "inf", "write"):
            with self.subTest(failure=failure), patch("time.monotonic", return_value=10.0):
                hand = Mock()
                hand.get_all_joint_angles.return_value = [0.0] * 19
                if failure == "write":
                    hand.set_all_active_joint_angles.side_effect = OSError("write failed")
                else:
                    hand.get_all_joint_angles.return_value = {
                        "shape": [0.0] * 18,
                        "empty": [],
                        "nan": [float("nan")] * 19,
                        "inf": [0.0] * 18 + [float("inf")],
                    }[failure]
                states, errors, gate = self.run_hand_cycle(hand, "right")
                self.assertTrue(states.empty())
                self.assertEqual(gate[1], 0)
                self.assertGreater(gate[0], 3)
                error = errors.get_nowait()
                self.assertIn("right:", error)
                if failure != "write":
                    angles = np.asarray(hand.get_all_joint_angles.return_value)
                    self.assertIn(f"received shape={angles.shape}", error)
                    self.assertIn(f"nonfinite_indices={np.flatnonzero(~np.isfinite(angles)).tolist()}", error)


class SharedWorkerTimingTests(unittest.TestCase):
    def start_worker(self, left, right):
        self.commands, self.states, self.errors = queue.Queue(1), queue.Queue(1), queue.Queue(1)
        self.stop = threading.Event()
        self.gate = mp.get_context("spawn").Array("q", [3, 1])
        self.enterContext(patch("robojudo.controller.omnihand_runtime.create_hand", side_effect=[left, right]))
        self.thread = threading.Thread(
            target=_worker,
            args=(serial_config(), self.commands, self.states, self.stop, self.gate, self.errors),
        )
        self.thread.start()
        self.addCleanup(self.stop_worker)

    def stop_worker(self):
        self.stop.set()
        self.thread.join(timeout=2)
        self.assertFalse(self.thread.is_alive())
        self.assertTrue(self.errors.empty(), list(self.errors.queue))

    def test_two_80ms_reads_remain_fresh_at_default_threshold(self):
        def read():
            time.sleep(0.08)
            return [0.0] * 19

        hands = [Mock(), Mock()]
        for hand in hands:
            hand.get_all_joint_angles.side_effect = read
        self.start_worker(*hands)
        snapshot = self.states.get(timeout=1)
        deadline = time.monotonic() + 0.8
        while time.monotonic() < deadline:
            try:
                snapshot = self.states.get_nowait()
            except queue.Empty:
                pass
            now = time.monotonic()
            self.assertTrue(all(0 <= now - state["sampled_at"] < 0.25 for state in snapshot.values()))
            time.sleep(0.005)

    def test_acknowledgement_precedes_next_reads_and_retains_sample_times(self):
        release_second, release_third = threading.Event(), threading.Event()
        started_second = [threading.Event(), threading.Event()]
        started_third = [threading.Event(), threading.Event()]
        hands = [Mock(), Mock()]

        def reader(index):
            calls = 0

            def read():
                nonlocal calls
                calls += 1
                if calls == 2:
                    started_second[index].set()
                    if not release_second.wait(timeout=2):
                        raise TimeoutError("test second read blocked")
                elif calls >= 3:
                    started_third[index].set()
                    if not release_third.wait(timeout=2):
                        raise TimeoutError("test third read blocked")
                return [0.0] * 19

            return read

        for index, hand in enumerate(hands):
            hand.get_all_joint_angles.side_effect = reader(index)
        self.start_worker(*hands)
        # Release barriers before cleanup tries to join the worker.
        self.addCleanup(release_third.set)
        self.addCleanup(release_second.set)
        for event in started_second:
            self.assertTrue(event.wait(timeout=1))
        submitted_at = time.monotonic()
        self.commands.put((3, submitted_at + 0.1, (3, 7), [0.1] * 24, submitted_at))
        # Let the dispatcher transfer the pair before the reads finish.
        deadline = time.monotonic() + 0.05
        while not self.commands.empty() and time.monotonic() < deadline:
            time.sleep(0.001)
        self.assertTrue(self.commands.empty())
        for hand in hands:
            hand.set_all_active_joint_angles.assert_not_called()
        release_second.set()
        for event in started_third:
            self.assertTrue(event.wait(timeout=1))
        snapshot = self.states.get(timeout=1)
        self.assertLess(time.monotonic(), submitted_at + 0.1)
        self.assertTrue(all(state["applied"] == (3, 7) for state in snapshot.values()))
        self.assertTrue(all(state["targets"] == [0.1] * 12 for state in snapshot.values()))
        self.assertTrue(all(state["sampled_at"] < submitted_at for state in snapshot.values()))
        for hand in hands:
            hand.set_all_active_joint_angles.assert_called_once_with([0.1] * 12)


class CombinedFeedbackTests(unittest.TestCase):
    def test_legacy_mode_reset_discards_queued_synchronized_frames(self):
        from test_upper_body_omnihand_zmq_ctrl import TestUpperBodyOmniHandZmqCtrl

        from robojudo.config.x2.env.x2_env_cfg import X2_ARM_JOINT_NAMES
        from robojudo.controller.upper_body_omnihand_zmq_ctrl import UpperBodyOmniHandZmqCtrl

        runtime = Mock()
        runtime.get_data.return_value = {"enabled": True, "joint_state_fresh": True}
        cfg = UpperBodyOmniHandZmqCtrlCfg(joint_names=X2_ARM_JOINT_NAMES)
        with (
            patch("robojudo.controller.upper_body_zmq_ctrl.zmq.Context.instance", return_value=MemoryContext()),
            patch.object(UpperBodyOmniHandZmqCtrl, "_create_hand_runtime", return_value=runtime),
        ):
            controller = UpperBodyOmniHandZmqCtrl(cfg)
        self.addCleanup(controller.close)
        controller._socket.recv_json = Mock(side_effect=zmq.Again())
        controller._socket.messages.append(TestUpperBodyOmniHandZmqCtrl.synchronized_frame())
        controller.reset()
        self.assertFalse(controller.get_data()["fresh"])
        self.assertEqual(controller._socket.messages, [])
        runtime.set_joint_commands.assert_not_called()

    def setUp(self):
        self.clock = self.enterContext(patch("time.monotonic", return_value=10.0))
        self.cfg = UpperBodyOmniHandZmqCtrlCfg(
            joint_names=["arm"], ema_alpha=0, feedback_endpoint="tcp://127.0.0.1:8571"
        )
        self.feedback = UpperBodyFeedback(self.cfg, MemoryContext())
        self.addCleanup(self.feedback.close)
        self.names = ["arm", *OMNIHAND_JOINT_NAMES]
        self.targets = dict.fromkeys(self.names, 0.0)

    def proposal(self, intent="TRACK", status="tracking"):
        measured = dict.fromkeys(["arm", *ALL_HAND_NAMES], 0.0)
        self.feedback.publish(
            positions=measured,
            sample_times=dict.fromkeys(measured, 10.0),
            targets=self.targets,
            defaults=self.targets,
            limits=dict.fromkeys(self.names, [-1.0, 1.0]),
            intent=intent,
            mode="RL_DEFAULT",
            control_dt=0.02,
        )
        state = self.feedback.socket.frames[-1]
        return {
            "schema_version": 2,
            "type": "upper_body_proposal",
            "session_id": state["session_id"],
            "generation": state["generation"],
            "state_id": state["state_id"],
            "command_id": state["command_id"],
            "bridge_id": "test",
            "frame_id": 1,
            "intent": intent,
            "status": status,
            "positions": self.targets.copy(),
        }

    def test_complete_fingers_required_and_speed_checked(self):
        proposal = self.proposal()
        self.feedback.validate(proposal, proposal["positions"])
        proposal["positions"]["L_index_mcp_joint"] = 0.1
        with self.assertRaisesRegex(ValueError, "velocity"):
            self.feedback.validate(proposal, proposal["positions"])
        del proposal["positions"]["L_index_mcp_joint"]
        with self.assertRaisesRegex(ValueError, "all configured"):
            self.feedback.validate(proposal, proposal["positions"])

    def test_invalid_numeric_encoding_does_not_refresh_command_lease(self):
        proposal = self.proposal()
        proposal["positions"]["L_index_mcp_joint"] = 0.01
        with self.assertRaisesRegex(ValueError, "float32"):
            self.feedback.validate(proposal, proposal["positions"], receiving=True)
        self.assertIsNone(self.feedback.last_proposal_at)
        self.assertEqual(self.feedback._frame_id, -1)
        self.assertIsNone(self.feedback.bridge_id)

    def test_home_cannot_open_fingers(self):
        proposal = self.proposal("HOME", "homing")
        proposal["positions"]["L_index_mcp_joint"] = 0.01
        with self.assertRaisesRegex(ValueError, "velocity"):
            self.feedback.validate(proposal, proposal["positions"])

    def test_schema_and_measurement_deadlines(self):
        proposal = self.proposal()
        proposal["schema_version"] = 1
        with self.assertRaises(ValueError):
            self.feedback.validate(proposal, proposal["positions"])

        proposal["schema_version"] = 2
        self.clock.return_value = 10.11
        with self.assertRaisesRegex(ValueError, "expired"):
            self.feedback.validate(proposal, proposal["positions"])

    def test_rejected_expiry_does_not_revoke_engagement_before_submission(self):
        from robojudo.pipeline.x2_locomanipulation_loco_mimic_pipeline import X2LocomanipulationLocoMimicPipeline

        proposal = self.proposal()
        generation = self.feedback.generation
        self.clock.return_value = 10.11
        receiver = SimpleNamespace(
            feedback=self.feedback,
            synchronize_intent=Mock(return_value=False),
            synchronize_generation=Mock(),
            _clear_positions=Mock(),
            _log_invalid_message=Mock(),
        )
        pipeline = X2LocomanipulationLocoMimicPipeline.__new__(X2LocomanipulationLocoMimicPipeline)
        pipeline._upper_body_cfg = self.cfg
        pipeline._upper_body_indices = np.array([0])
        pipeline._upper_body_filtered = np.zeros(1)
        pipeline._upper_body_control_available = lambda: True
        pipeline._feedback_intent = lambda: "TRACK"
        pipeline._apply_feedback_proposal(
            np.zeros(1), {self.cfg.ctrl_type: {"fresh": True, "proposal": proposal}}, receiver
        )
        self.assertEqual(self.feedback.generation, generation)
        receiver._clear_positions.assert_called_once()
        self.assertIsNone(pipeline._upper_body_pending)

    def test_safety_hold_after_body_submission_cancels_hand_dispatch(self):
        from robojudo.controller.upper_body_omnihand_zmq_ctrl import UpperBodyOmniHandZmqCtrl
        from robojudo.pipeline.x2_locomanipulation_loco_mimic_pipeline import X2LocomanipulationLocoMimicPipeline

        data = Box(dof_pos=np.zeros(1), joint_sample_times=np.array([10.0]))
        env = SimpleNamespace(
            cfg_env=SimpleNamespace(is_sim=False),
            get_data=lambda: data,
            joint_names=["arm"],
            position_limits=np.array([[-1.0, 1.0]]),
        )
        runtime = Mock()
        runtime.get_data.return_value = {
            "positions": dict.fromkeys(ALL_HAND_NAMES, 0.0),
            "sample_times": dict.fromkeys(ALL_HAND_NAMES, 10.0),
            "targets": dict.fromkeys(OMNIHAND_JOINT_NAMES, 0.0),
            "joint_state_fresh": True,
        }
        with (
            patch("robojudo.controller.upper_body_zmq_ctrl.zmq.Context.instance", return_value=MemoryContext()),
            patch.object(UpperBodyOmniHandZmqCtrl, "_create_hand_runtime", return_value=runtime),
        ):
            receiver = UpperBodyOmniHandZmqCtrl(self.cfg, env)
        self.addCleanup(receiver.close)
        self.feedback = receiver.feedback
        proposal = self.proposal()
        receiver.synchronize_generation()
        runtime.reset.reset_mock()
        pipeline = X2LocomanipulationLocoMimicPipeline.__new__(X2LocomanipulationLocoMimicPipeline)
        pipeline.env, pipeline.dt = env, 0.02
        pipeline.mode = SimpleNamespace(value="RL_DEFAULT")
        pipeline._upper_body_cfg = self.cfg
        pipeline._upper_body_indices = np.array([0])
        pipeline._upper_body_filtered = pipeline._upper_body_default = np.zeros(1)
        pipeline._upper_body_command_submitted = True
        pipeline._upper_body_pending = proposal
        pipeline._upper_body_enabled = True
        pipeline._feedback_receiver = lambda: receiver
        pipeline._feedback_intent = lambda: "HOLD"
        pipeline._publish_upper_body_feedback(data, np.zeros(1))
        runtime.submit.assert_not_called()
        runtime.reset.assert_called_once()
        self.assertEqual(self.feedback.command_id, 0)
        self.assertEqual(self.feedback.socket.frames[-1]["intent"], "HOLD")
        self.assertIsNone(pipeline._upper_body_pending)


class CombinedApplicationTests(unittest.TestCase):
    def test_timely_application_observed_after_deadline_is_acknowledged(self):
        receiver = self.receiver()
        receiver.pending = ({"status": "tracking"}, (2, 10), 10.1)
        receiver._hand_runtime.get_data.return_value = {"joint_state_fresh": True, "applied": (2, 10)}
        with patch("time.monotonic", return_value=10.11), patch.object(receiver, "invalidate") as invalidate:
            self.assertTrue(receiver.finish_pending())
        invalidate.assert_not_called()
        receiver.feedback.acknowledge.assert_called_once()

    def test_unapplied_command_still_expires(self):
        receiver = self.receiver()
        receiver.pending = ({"status": "tracking"}, (2, 10), 10.1)
        receiver._hand_runtime.get_data.return_value = {"joint_state_fresh": True, "applied": None}
        with patch("time.monotonic", return_value=10.11), patch.object(receiver, "invalidate") as invalidate:
            self.assertFalse(receiver.finish_pending())
        invalidate.assert_called_once()
        receiver.feedback.acknowledge.assert_not_called()

    def test_hand_admission_precedes_arm_submission_and_failure_holds_arms(self):
        from robojudo.pipeline.upper_body_zmq_pipeline import UpperBodyZmqPipelineMixin

        for failure in (None, ValueError("expired")):
            with self.subTest(failure=failure):
                pipeline = UpperBodyZmqPipelineMixin.__new__(UpperBodyZmqPipelineMixin)
                receiver = Mock()
                receiver.feedback.schema_version = 2
                receiver.synchronize_intent.return_value = False
                receiver.hand_snapshot.return_value = {"fresh": True}
                order = []

                def submit(proposal, order=order, failure=failure):
                    order.append("hands")
                    if failure:
                        raise failure

                receiver.submit_hands.side_effect = submit
                pipeline._feedback_receiver = lambda receiver=receiver: receiver
                pipeline._feedback_intent = lambda: "TRACK"
                pipeline._upper_body_pending = {"positions": {"arm": 0.2}}
                pipeline._upper_body_indices = np.array([0])
                pipeline._upper_body_previous = np.array([0.1], dtype=np.float32)
                pipeline.env = SimpleNamespace(step=lambda *args, order=order: order.append("arms"))
                target = np.array([0.2], dtype=np.float32)
                pipeline._send_upper_body_target(target)
                self.assertEqual(order, ["hands", "arms"])
                self.assertAlmostEqual(target[0], 0.1 if failure else 0.2)
                self.assertEqual(receiver.invalidate.called, failure is not None)

    def test_feedback_runtime_preserves_configured_read_timeout(self):
        from robojudo.controller.upper_body_omnihand_zmq_ctrl import UpperBodyOmniHandZmqCtrl

        cfg = UpperBodyOmniHandZmqCtrlCfg(joint_names=["arm"], feedback_endpoint="tcp://127.0.0.1:8571", ema_alpha=0)
        receiver = UpperBodyOmniHandZmqCtrl.__new__(UpperBodyOmniHandZmqCtrl)
        for timeout_ms in (200, 50):
            with self.subTest(timeout_ms=timeout_ms):
                cfg.omnihand.frame_recv_timeout_ms = timeout_ms
                with patch("robojudo.controller.upper_body_omnihand_zmq_ctrl.OmniHandRuntime") as runtime:
                    receiver._create_hand_runtime(cfg)
                self.assertEqual(runtime.call_args.args[0].frame_recv_timeout_ms, timeout_ms)
                self.assertEqual(cfg.omnihand.frame_recv_timeout_ms, timeout_ms)

    def test_arm_submission_failure_revokes_pending_hand_writes(self):
        from robojudo.pipeline.upper_body_zmq_pipeline import UpperBodyZmqPipelineMixin

        pipeline = UpperBodyZmqPipelineMixin.__new__(UpperBodyZmqPipelineMixin)
        receiver = Mock()
        receiver.feedback.schema_version = 2
        receiver.synchronize_intent.return_value = False
        receiver.hand_snapshot.return_value = {"fresh": True}
        pipeline._feedback_receiver = lambda: receiver
        pipeline._feedback_intent = lambda: "TRACK"
        pipeline._upper_body_pending = {"positions": {"arm": 0.2}}
        pipeline.env = SimpleNamespace(step=Mock(side_effect=RuntimeError("body failed")))
        with self.assertRaisesRegex(RuntimeError, "body failed"):
            pipeline._send_upper_body_target(np.array([0.2], dtype=np.float32))
        receiver.invalidate.assert_called_once_with("arm submission failed")
        self.assertIsNone(pipeline._upper_body_pending)

    def test_recoverable_application_failure_uses_bridge_rebase_reason(self):
        receiver = self.receiver()
        with patch.object(receiver, "reset"), patch.object(receiver, "_log_invalid_message"):
            receiver.invalidate("application deadline expired", recoverable=True)
        self.assertEqual(receiver.feedback.reset_reason, "tracking command timeout")

    def test_recording_requires_acknowledgement_and_contains_active_fingers(self):
        from robojudo.pipeline.x2_locomanipulation_loco_mimic_pipeline import X2LocomanipulationLocoMimicPipeline

        pipeline = X2LocomanipulationLocoMimicPipeline.__new__(X2LocomanipulationLocoMimicPipeline)
        pipeline._recorder_client = Mock()
        pipeline._recording_active = True
        pipeline._recording_paused = False
        pipeline._upper_body_enabled = True
        pipeline._upper_body_control_available = lambda: True
        pipeline._upper_body_stream_was_fresh = False
        pipeline._upper_body_cfg = SimpleNamespace(joint_names=["arm"])
        pipeline._upper_body_indices = np.array([0])
        snapshot = {
            "fresh": True,
            "positions": dict.fromkeys(ALL_HAND_NAMES, 0.1),
            "targets": dict.fromkeys(OMNIHAND_JOINT_NAMES, 0.2),
        }
        receiver = SimpleNamespace(
            feedback=SimpleNamespace(schema_version=2),
            record_hand_data=lambda: {
                "fresh": snapshot["fresh"],
                "joint_names": list(OMNIHAND_JOINT_NAMES),
                "joint_positions": [snapshot["positions"][n] for n in OMNIHAND_JOINT_NAMES],
                "joint_position_commands": [snapshot["targets"][n] for n in OMNIHAND_JOINT_NAMES],
            },
        )
        pipeline._feedback_receiver = lambda: receiver
        data = SimpleNamespace(dof_pos=np.array([-0.4]))
        extras = {"locomotion_command": [0.0, 0.0, 0.0, 0.7]}
        pipeline._record_upper_body_sample(data, extras, np.array([-0.3]), rl_active=True)
        pipeline._recorder_client.submit.assert_not_called()
        pipeline._upper_body_stream_was_fresh = True
        pipeline._record_upper_body_sample(data, extras, np.array([-0.3]), rl_active=True)
        result = pipeline._recorder_client.submit.call_args.kwargs
        self.assertEqual(result["joint_names"], ["arm", *OMNIHAND_JOINT_NAMES])
        np.testing.assert_allclose(result["joint_positions"], [-0.4, *([0.1] * 24)])
        np.testing.assert_allclose(result["joint_position_commands"], [-0.3, *([0.2] * 24)])
        pipeline._recorder_client.submit.reset_mock()
        receiver.record_hand_data = Mock(side_effect=RuntimeError("hand worker failed"))
        receiver.invalidate = Mock()
        pipeline._record_upper_body_sample(data, extras, np.array([-0.3]), rl_active=True)
        pipeline._recorder_client.submit.assert_not_called()
        receiver.invalidate.assert_called_once_with("hand worker failed")
        self.assertFalse(pipeline._upper_body_stream_was_fresh)

    def receiver(self):
        from robojudo.controller.upper_body_omnihand_zmq_ctrl import UpperBodyOmniHandZmqCtrl

        receiver = UpperBodyOmniHandZmqCtrl.__new__(UpperBodyOmniHandZmqCtrl)
        receiver.feedback = Mock(generation=2)
        receiver._hand_generation = 2
        receiver._hand_runtime = Mock()
        receiver.pending = ({"status": "tracking"}, (2, 10), time.monotonic() + 1)
        return receiver

    def test_both_hand_writes_are_required_for_acknowledgement(self):
        receiver = self.receiver()
        receiver._hand_runtime.get_data.return_value = {"joint_state_fresh": True, "applied": (2, 9)}
        self.assertFalse(receiver.finish_pending())
        receiver.feedback.acknowledge.assert_not_called()
        receiver._hand_runtime.get_data.return_value["applied"] = (2, 10)
        self.assertTrue(receiver.finish_pending())
        receiver.feedback.acknowledge.assert_called_once()
        receiver._hand_runtime.acknowledge.assert_called_once()
        self.assertIsNone(receiver.pending)

    def test_generation_change_revokes_worker_gate(self):
        receiver = self.receiver()
        receiver.feedback.generation = 3
        self.assertFalse(receiver.finish_pending())
        receiver._hand_runtime.reset.assert_called_once()
        receiver.feedback.acknowledge.assert_not_called()

    def test_stale_measurement_cancels_partial_application(self):
        receiver = self.receiver()
        receiver._hand_runtime.get_data.return_value = {"joint_state_fresh": False, "applied": (2, 10)}
        with patch.object(receiver, "invalidate") as invalidate:
            self.assertFalse(receiver.finish_pending())
        invalidate.assert_called_once()
        receiver.feedback.acknowledge.assert_not_called()


class ArticulatedSimulationTests(unittest.TestCase):
    def test_body_addresses_remain_correct_with_interleaved_fingers(self):
        from robojudo.config.x2 import x2_xrobotoolkit_omnihand
        from robojudo.environment.x2_omnihand_mujoco_env import X2OmniHandMujocoEnv

        cfg = x2_xrobotoolkit_omnihand().env
        cfg.headless = True
        cfg.forward_kinematic = None
        cfg.update_with_fk = False
        env = X2OmniHandMujocoEnv(cfg)
        self.addCleanup(env.shutdown)
        env.data.qpos[env.model.joint("right_elbow_joint").qposadr[0]] = -0.4
        env.data.qpos[env.model.joint("L_index_mcp_joint").qposadr[0]] = 0.2
        env.update(simple=True)
        self.assertAlmostEqual(env.dof_pos[env.joint_names.index("right_elbow_joint")], -0.4)
        self.assertEqual(len(env.hand_snapshot()["positions"]), 38)
        targets = dict.fromkeys(OMNIHAND_JOINT_NAMES, 0.0)
        targets["L_index_mcp_joint"] = 0.3
        env.submit_hand_targets(targets)
        env._apply_auxiliary_targets()
        np.testing.assert_allclose(env.data.ctrl[env._hand_actuators], list(targets.values()))
        env.step(env.dof_pos.copy())
        np.testing.assert_allclose(env.data.ctrl[env._hand_actuators], list(targets.values()))
        env.reborn()
        self.assertEqual(env.hand_snapshot()["targets"]["L_index_mcp_joint"], 0.0)
