"""Opt-in two-interpreter MuJoCo/ONNX -> ZMQ -> XR bridge integration check.

Run with XROBOTOOLKIT_TEST_PYTHON pointing at the isolated XR environment's
Python executable. The child uses synthetic XR input and never initializes SDK
or robot hardware. Local TCP sockets are required.
"""

import json
import os
import socket
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import zmq

BRIDGE_PROCESS = r"""
import json
import select
import logging
import os
import sys
import time
import numpy as np
from xrobotoolkit_teleop.integrations.robojudo import RoboJuDoBridge
from xrobotoolkit_teleop.common.teleop_settings import TeleopSettings
from xrobotoolkit_teleop.common.waist_tracking import WaistSample

class SyntheticXR:
    def __init__(self):
        self.grip = 0.0
        self.right_grip = 0.0
        self.trigger = 0.0
        self.right_trigger = 0.0
        self.z = 0.0
        self.freeze = False
        self.freeze_body = False
        self.body_timestamp = 1
        self.height = 0.0
        self.twist = 0.0
        self.heading = 0.0
        self.home = False
        self.ik_delay = 0.0
        self.timestamp = 1
        self.next_update = 0.0
    def get_timestamp_ns(self):
        if not self.freeze and time.monotonic() >= self.next_update:
            self.timestamp += 20_000_000
            self.next_update = time.monotonic() + 0.05
        return self.timestamp
    def get_pose_by_name(self, name):
        return np.array([0., 0., self.z if name == 'left_controller' else 0., 0., 0., 0., 1.])
    def get_key_value_by_name(self, name):
        return {'left_grip': self.grip, 'right_grip': self.right_grip,
                'left_trigger': self.trigger, 'right_trigger': self.right_trigger}.get(name, 0.0)
    def get_button_state_by_name(self, name):
        return self.home
    def get_waist_sample(self, *, pelvis_ik=False, include_triggers=True):
        stamp = self.get_timestamp_ns()
        if not self.freeze_body:
            self.body_timestamp = stamp
        yaw = self.heading
        pelvis = np.array([0., self.height, 0., 0., np.sin(yaw / 2), 0., np.cos(yaw / 2)])
        def pair(angle):
            axis = np.array([-np.cos(angle), 0., np.sin(angle)])
            return np.array([axis, -axis])
        controllers = {}
        for side in ("left", "right"):
            pose = self.get_pose_by_name(f"{side}_controller")
            delta = np.array([np.sin(yaw) * pose[2], self.height, np.cos(yaw) * pose[2]])
            controllers[side] = np.r_[delta, pelvis[3:]]
        triggers = None
        if include_triggers:
            triggers = {s: self.get_key_value_by_name(f"{s}_trigger") for s in controllers}
            if None in triggers.values():
                raise ValueError("synthetic trigger readings unavailable")
        return WaistSample(
            pelvis, controllers, self.body_timestamp, stamp, time.monotonic(), pair(yaw),
            shoulder_positions=pair(yaw + self.twist),
            grips={s: self.get_key_value_by_name(f"{s}_grip") for s in controllers},
            triggers=triggers,
        )
    def close(self):
        pass

def delayed_solve(delay, *args):
    from xrobotoolkit_teleop.integrations.pelvis_ik import solve_arms

    time.sleep(delay)
    return solve_arms(*args)


def main():
    diagnostics = os.environ.get("XROBOTOOLKIT_TEST_DIAGNOSTICS")
    if diagnostics:
        logging.basicConfig(level=logging.DEBUG if diagnostics == "debug" else logging.INFO)
    xr = SyntheticXR()
    settings = TeleopSettings(collision_checking=sys.argv[5] == "True")
    options = dict(settings=settings, xr_client=xr, ik_backend=sys.argv[3], robot=sys.argv[4])
    if sys.argv[6]:
        from xrobotoolkit_teleop.integrations.pelvis_ik import PelvisIKBridge

        bridge = PelvisIKBridge(sys.argv[1], sys.argv[2], posture_endpoint=sys.argv[6], **options)
        submit = bridge.retargeter.pool.submit
        bridge.retargeter.pool.submit = lambda _solve, *args: submit(delayed_solve, xr.ik_delay, *args)
    else:
        bridge = RoboJuDoBridge(sys.argv[1], sys.argv[2], **options)
    try:
        deadline = time.monotonic()
        status = None
        feedback_generation = None
        while True:
            time.sleep(max(0, deadline - time.monotonic()))
            if select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline()
                if not line:
                    break
                for name, value in json.loads(line).items():
                    setattr(xr, name, value)
            bridge.poll()
            current = bridge.status, bridge.hold_reason
            if diagnostics and current != status:
                logging.info("Bridge status %s (%s)", *current)
                status = current
            feedback = bridge._latest
            if diagnostics == "debug" and feedback and feedback["generation"] != feedback_generation:
                feedback_generation = feedback["generation"]
                logging.debug("Feedback generation %s (%s)", feedback_generation, feedback["reset_reason"])
            deadline += 0.02
            now = time.monotonic()
            if deadline < now:
                deadline += (int((now - deadline) / 0.02) + 1) * 0.02
    finally:
        bridge.close()

if __name__ == "__main__":
    main()
"""


class XrobotoolkitFixture:
    with_omnihand = False
    finger_mode = "direct"
    collision_checking = True
    pelvis_ik = False

    def setUp(self):
        from robojudo.config.x2 import (
            x2_xrobotoolkit,
            x2_xrobotoolkit_omnihand,
            x2_xrobotoolkit_omnihand_pelvis_ik,
            x2_xrobotoolkit_pelvis_ik,
        )
        from robojudo.controller.ctrl_cfgs import UpperBodyZmqCtrlCfg
        from robojudo.pipeline.four_mode_pipeline import ControlMode
        from robojudo.pipeline.x2_locomanipulation_loco_mimic_pipeline import X2LocomanipulationLocoMimicPipeline

        self.directory = self.enterContext(tempfile.TemporaryDirectory(prefix="x2-xr-integration-"))
        self.children = []
        self.addCleanup(self.stop_children)
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))
            port = reserved.getsockname()[1]
        self.command_endpoint = f"tcp://127.0.0.1:{port}"
        if self.pelvis_ik:
            cfg = x2_xrobotoolkit_omnihand_pelvis_ik() if self.with_omnihand else x2_xrobotoolkit_pelvis_ik()
        else:
            cfg = x2_xrobotoolkit_omnihand() if self.with_omnihand else x2_xrobotoolkit()
        upper = next(item for item in cfg.ctrl if isinstance(item, UpperBodyZmqCtrlCfg))
        upper.endpoint = self.command_endpoint
        upper.feedback_endpoint = "tcp://127.0.0.1:*"
        upper.executor_collision_checking = self.collision_checking
        if self.with_omnihand:
            upper.hand_command_mode = self.finger_mode
        self.posture_endpoint = ""
        if self.pelvis_ik:
            from robojudo.controller.ctrl_cfgs import LocomanipulationPostureZmqCtrlCfg

            posture = next(item for item in cfg.ctrl if isinstance(item, LocomanipulationPostureZmqCtrlCfg))
            with socket.socket() as reserved:
                reserved.bind(("127.0.0.1", 0))
                posture.endpoint = f"tcp://127.0.0.1:{reserved.getsockname()[1]}"
            self.posture_endpoint = posture.endpoint
            cfg.ctrl = [upper, posture]
        else:
            cfg.ctrl = [upper]
        cfg.env.headless = True
        cfg.env.visualize_extras = False
        cfg.env.forward_kinematic = None
        cfg.env.update_with_fk = False
        cfg.device = "cpu"
        cfg.loco_policy.command_status_interval_s = 0.0
        self.pipeline = X2LocomanipulationLocoMimicPipeline(cfg)
        from robojudo.environment.utils.mujoco_viz import MujocoVisualizer

        self.markers = {}
        self.pipeline.env.visualizer = MujocoVisualizer(
            SimpleNamespace(add_marker=lambda **marker: self.markers.update({marker["id"]: marker}))
        )
        self.addCleanup(self.pipeline.env.shutdown)
        self.addCleanup(self.pipeline.ctrl_manager.close)
        self.receiver = self.pipeline.ctrl_manager.controllers[upper.ctrl_type].inst
        self.feedback_endpoint = self.receiver.feedback.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        self.pipeline.prepare()
        self.pipeline._enter_mode(ControlMode.JOINT_DEFAULT)
        for _ in range(self.pipeline._joint_default_steps):
            self.pipeline.step()
        self.assertTrue(self.pipeline._joint_default_complete)
        self.pipeline._enter_mode(ControlMode.RL_DEFAULT)

    def stop_children(self):
        for child in self.children:
            if child.stdin is not None:
                child.stdin.close()
            if child.poll() is None:
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:
                        child.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=3)

    def start_bridge(self):
        root = Path(__file__).resolve().parents[1]
        log = self.enterContext(open(Path(self.directory) / f"bridge-{len(self.children)}.log", "w"))
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(root / "third_party/xrobotoolkit_teleop")
        script_args = ["-c", BRIDGE_PROCESS]
        if self.pelvis_ik:
            script = Path(self.directory) / "bridge.py"
            script.write_text(BRIDGE_PROCESS)
            script_args = [str(script)]
        child = subprocess.Popen(
            [
                os.environ["XROBOTOOLKIT_TEST_PYTHON"],
                "-u",
                *script_args,
                self.command_endpoint,
                self.feedback_endpoint,
                os.environ.get("XROBOTOOLKIT_TEST_BACKEND", "lerobot" if self.pelvis_ik else "placo"),
                "x2_omnihand" if self.with_omnihand else "x2",
                str(self.collision_checking),
                self.posture_endpoint,
            ],
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
        )
        self.children.append(child)
        return child

    def cycles(self, duration):
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            started = time.monotonic()
            self.pipeline.step()
            self.assertEqual(self.pipeline.mode.value, "RL_DEFAULT")
            time.sleep(max(0, self.pipeline.dt - (time.monotonic() - started)))

    def wait_until(self, predicate, timeout=4):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.cycles(0.02)
            if predicate():
                return
        logs = "\n".join(path.read_text() for path in Path(self.directory).glob("*.log"))
        self.fail(f"two-process integration did not reach expected state:\n{logs}")

    @staticmethod
    def control(child, **values):
        child.stdin.write(json.dumps(values) + "\n")
        child.stdin.flush()


@unittest.skipUnless(
    os.environ.get("XROBOTOOLKIT_TEST_PYTHON"), "set XROBOTOOLKIT_TEST_PYTHON for two-process XR tests"
)
class XrobotoolkitIntegrationTests(XrobotoolkitFixture, unittest.TestCase):
    def test_tracking_loss_pending_return_and_bridge_restart(self):
        child = self.start_bridge()
        self.wait_until(lambda: self.receiver.feedback.command_id > 0)
        self.pipeline._set_upper_body_enabled(True)
        self.cycles(0.2)  # Require fresh feedback and released grips after activation.
        self.control(child, grip=1.0)
        self.wait_until(lambda: self.pipeline._upper_body_stream_was_fresh)
        initial = self.pipeline._upper_body_filtered.copy()
        self.control(child, z=-0.03)
        self.wait_until(lambda: np.max(np.abs(self.pipeline._upper_body_filtered - initial)) > 0.005)
        self.assertEqual(self.markers[6004]["label"], "left hand target")
        self.assertGreater(self.markers[6004]["rgba"][3], 0)
        self.assertEqual(self.markers[6012]["label"], "world (fixed)")
        self.control(child, freeze=True)
        self.cycles(0.35)
        held = self.pipeline._upper_body_filtered.copy()
        self.cycles(0.15)
        np.testing.assert_array_equal(self.pipeline._upper_body_filtered, held)
        self.assertFalse(self.pipeline._upper_body_stream_was_fresh)

        self.control(child, freeze=False)
        self.wait_until(lambda: self.pipeline._upper_body_stream_was_fresh)
        np.testing.assert_allclose(self.pipeline._upper_body_filtered, held, atol=1e-5, rtol=0)
        self.control(child, z=-0.05)
        self.wait_until(lambda: np.max(np.abs(self.pipeline._upper_body_filtered - held)) > 0.005)

        # Lose the bridge while actively tracking, independently of XR loss.
        child.terminate()
        child.wait(timeout=3)
        self.cycles(0.35)
        held = self.pipeline._upper_body_filtered.copy()
        self.cycles(0.15)
        np.testing.assert_array_equal(self.pipeline._upper_body_filtered, held)
        self.assertFalse(self.pipeline._upper_body_stream_was_fresh)
        self.pipeline._set_upper_body_enabled(False)
        self.cycles(0.15)
        self.assertEqual(self.receiver.feedback.intent, "HOME")
        np.testing.assert_array_equal(self.pipeline._upper_body_filtered, held)
        self.start_bridge()
        self.wait_until(
            lambda: np.max(np.abs(self.pipeline._upper_body_filtered - self.pipeline._upper_body_default)) < 1e-5
        )
        self.assertFalse(self.pipeline._upper_body_enabled)

    def test_slow_inference_still_accepts_fresh_arm_desired_targets(self):
        original = self.pipeline.policy.get_pd_target

        def slow_inference(obs):
            time.sleep(0.055)
            return original(obs)

        self.pipeline.policy.get_pd_target = slow_inference
        child = self.start_bridge()
        self.wait_until(lambda: self.receiver.feedback.command_id > 0)
        self.pipeline._set_upper_body_enabled(True)
        self.cycles(0.2)
        self.control(child, grip=1.0)
        self.wait_until(lambda: self.pipeline._upper_body_stream_was_fresh)
        initial = self.pipeline._upper_body_filtered.copy()
        self.control(child, z=-0.03)
        self.wait_until(lambda: np.max(np.abs(self.pipeline._upper_body_filtered - initial)) > 0.005)
        accepted = self.receiver.feedback.command_id
        self.wait_until(lambda: self.receiver.feedback.command_id >= accepted + 3)
        state = self.receiver.feedback._states[self.receiver.feedback.state_id][1]
        self.assertLess(max(state["joint_age_s"].values()), 0.02)
        self.assertGreater(state["pipeline_timing_ms"]["policy"], 50)


@unittest.skipUnless(os.environ.get("XROBOTOOLKIT_TEST_PYTHON"), "set XROBOTOOLKIT_TEST_PYTHON")
class XrobotoolkitOmniHandIntegrationTests(XrobotoolkitIntegrationTests):
    with_omnihand = True

    def test_delayed_hand_writes_do_not_pace_bridge_publishing(self):
        if self.collision_checking:
            self.skipTest("50 Hz timing check isolates hand I/O from collision sampling cost")
        env = self.pipeline.env

        class DelayedHands:
            takeover_requested = True

            def __init__(self):
                self.delay = 0.04
                self.pending = None
                self.applied = None

            def get_data(self):
                if self.pending is not None:
                    positions, identity, ready_at, deadline = self.pending
                    if time.monotonic() >= ready_at and ready_at <= deadline:
                        env.submit_hand_targets(positions)
                        self.applied = identity
                        self.pending = None
                snapshot = env.hand_snapshot()
                return dict(snapshot, joint_state_fresh=snapshot["fresh"], applied=self.applied)

            def submit(self, positions, identity, deadline):
                if self.pending is not None:
                    raise AssertionError("in-flight hand command was overwritten")
                self.applied = None
                self.pending = (positions.copy(), identity, time.monotonic() + self.delay, deadline)

            def acknowledge(self):
                pass

            def set_takeover_enabled(self, enabled):
                self.takeover_requested = enabled
                if not enabled:
                    self.reset()

            def invalidate_commands(self):
                self.reset()

            def reset(self):
                self.pending = self.applied = None

            def close(self):
                self.reset()

        hands = DelayedHands()
        self.receiver._hand_runtime = hands
        # Include a 100 ms hardware delay without intentionally expiring the
        # default 100 ms measurement lease during this scheduling test.
        self.receiver.cfg_ctrl.feedback_timeout_s = 0.3
        self.pipeline._set_upper_body_enabled(True)
        child = self.start_bridge()
        self.wait_until(lambda: self.receiver.feedback.command_id > 0)
        self.control(child, trigger=0.06)
        monitor = zmq.Context.instance().socket(zmq.SUB)
        self.addCleanup(monitor.close, linger=0)
        monitor.setsockopt(zmq.SUBSCRIBE, b"")
        monitor.connect(self.command_endpoint)
        self.cycles(0.15)
        for delay in (0.04, 0.06, 0.1):
            with self.subTest(hand_delay=delay):
                hands.delay = delay
                while monitor.poll(0):
                    monitor.recv()
                completed = self.receiver.feedback.command_id
                started = time.monotonic()
                self.cycles(0.8)
                elapsed = time.monotonic() - started
                frames = []
                while monitor.poll(0):
                    frame = monitor.recv_json()
                    if frame["type"] == ("upper_body_desired" if self.with_omnihand else "upper_body_desired"):
                        frames.append(frame)
                rate = len(frames) / elapsed
                print(f"Injected hand delay={delay * 1000:.0f} ms: bridge publish={rate:.1f} Hz")
                self.assertGreater(rate, 35.0)
                self.assertGreater(self.receiver.feedback.command_id, completed)
                self.assertGreater(len(frames), self.receiver.feedback.command_id - completed)
                self.assertEqual(len({frame["frame_id"] for frame in frames}), len(frames))
                self.assertIsNone(child.poll())

    def test_delayed_hand_writes_with_policy_checkpoints(self):
        self.test_delayed_hand_writes_do_not_pace_bridge_publishing()
        self.assertIn("checkpoint_before", self.pipeline._upper_body_timing_ms)
        self.assertIn("checkpoint_after", self.pipeline._upper_body_timing_ms)

    def test_independent_triggers_and_home_holds_fingers(self):
        child = self.start_bridge()
        self.wait_until(lambda: self.receiver.feedback.command_id > 0)
        self.pipeline._set_upper_body_enabled(True)
        self.cycles(0.25)
        self.control(child, trigger=0.06)
        self.wait_until(lambda: self.pipeline.env.hand_snapshot()["targets"]["L_index_mcp_joint"] > 0.04)
        self.wait_until(lambda: self.pipeline.env.hand_snapshot()["positions"]["L_index_mcp_joint"] > 0.03)
        self.assertEqual(self.pipeline.env.hand_snapshot()["targets"]["R_index_mcp_joint"], 0.0)
        self.control(child, right_trigger=0.05)
        self.wait_until(lambda: self.pipeline.env.hand_snapshot()["targets"]["R_index_mcp_joint"] > 0.04)
        self.wait_until(lambda: self.pipeline.env.hand_snapshot()["positions"]["R_index_mcp_joint"] > 0.03)
        self.control(child, grip=0.0, right_grip=0.0)
        self.cycles(0.15)
        held = self.pipeline.env.hand_snapshot()["targets"].copy()
        self.pipeline._set_upper_body_enabled(False)
        self.cycles(0.3)
        self.assertEqual(self.pipeline.env.hand_snapshot()["targets"], held)

        self.control(child, trigger=0.0)
        self.cycles(0.2)
        self.assertEqual(self.pipeline.env.hand_snapshot()["targets"], held)
        self.pipeline._set_upper_body_enabled(True)
        self.wait_until(lambda: self.pipeline.env.hand_snapshot()["targets"]["L_index_mcp_joint"] < 0.01)
        self.assertEqual(self.pipeline.env.hand_snapshot()["targets"]["R_index_mcp_joint"], held["R_index_mcp_joint"])
        self.control(child, right_trigger=0.0)
        self.wait_until(lambda: self.pipeline.env.hand_snapshot()["targets"]["R_index_mcp_joint"] < 0.01)


class XrobotoolkitCollisionDisabledIntegrationTests(XrobotoolkitIntegrationTests):
    collision_checking = False


class XrobotoolkitOmniHandCollisionDisabledIntegrationTests(XrobotoolkitOmniHandIntegrationTests):
    collision_checking = False


class XrobotoolkitFilteredOmniHandIntegrationTests(XrobotoolkitOmniHandIntegrationTests):
    finger_mode = "filtered"


class XrobotoolkitFilteredOmniHandCollisionDisabledIntegrationTests(XrobotoolkitFilteredOmniHandIntegrationTests):
    collision_checking = False


@unittest.skipUnless(os.environ.get("XROBOTOOLKIT_TEST_PYTHON"), "set XROBOTOOLKIT_TEST_PYTHON")
class XrobotoolkitPelvisIKFixture(XrobotoolkitFixture):
    pelvis_ik = True
    collision_checking = False

    def test_manual_posture_takeover_requires_grip_release(self):
        from robojudo.controller.ctrl_cfgs import KeyboardCtrlCfg
        from robojudo.controller.posture_source import PostureSourceArbiter

        manager = self.pipeline.ctrl_manager
        posture = manager.controllers.LocomanipulationPostureZmqCtrl
        keyboard_cfg = KeyboardCtrlCfg(posture_priority=300, posture_lease_timeout_s=0.1)
        keyboard_data = {"keyboard_event": [], "pressed_keys": []}
        keyboard = SimpleNamespace(
            get_data=lambda: keyboard_data,
            process_triggers=lambda data: (data, []),
            post_step_callback=lambda commands: None,
        )
        manager.controllers["KeyboardCtrl"] = {"inst": keyboard, "cfg": keyboard_cfg}
        manager.posture_source_arbiter = PostureSourceArbiter([keyboard_cfg, posture.cfg])

        child = self.start_bridge()
        self.wait_until(lambda: self.receiver.feedback.command_id > 0, timeout=15)
        self.pipeline._set_upper_body_enabled(True)
        self.wait_until(lambda: self.pipeline._upper_body_stream_was_fresh)
        self.control(child, grip=1.0)
        self.cycles(0.2)
        self.control(child, height=-0.1)
        policy = self.pipeline._inner_policy()
        self.wait_until(lambda: policy.posture_targets[0] < 0.57)
        keyboard_data["pressed_keys"] = ["r"]
        keyboard_data["keyboard_event"] = [{"name": "r", "type": "keyboard", "pressed": True}]
        self.cycles(0.02)
        keyboard_data["keyboard_event"] = []
        manual_target = policy.posture_targets[0]
        self.assertGreater(manual_target, 0.55)
        keyboard_data["pressed_keys"] = []
        self.control(child, height=-0.18)
        self.cycles(0.3)
        self.assertEqual(policy.posture_targets[0], manual_target)
        self.control(child, grip=0.0)
        self.cycles(0.15)
        self.control(child, grip=1.0)
        self.cycles(0.15)
        self.control(child, height=-0.23)
        self.wait_until(lambda: policy.posture_targets[0] < manual_target - 0.02)

    def test_slow_ik_keeps_height_live_and_drops_stale_yaw(self):
        child = self.start_bridge()
        self.wait_until(lambda: self.receiver.feedback.command_id > 0, timeout=15)
        self.pipeline._set_upper_body_enabled(True)
        self.wait_until(lambda: self.pipeline._upper_body_stream_was_fresh)
        policy = self.pipeline._inner_policy()
        self.control(child, grip=1.0)
        self.cycles(0.4)
        self.control(child, ik_delay=0.18)
        self.cycles(0.08)
        self.control(child, height=-0.1, twist=0.3)
        self.wait_until(lambda: policy.posture_targets[0] < 0.57)
        self.cycles(0.35)
        self.assertAlmostEqual(policy.posture_targets[1], 0.0, places=2)
        held = policy.cmd[3:5].copy()
        self.pipeline._set_upper_body_enabled(False)
        self.cycles(0.15)
        np.testing.assert_array_equal(policy.cmd[3:5], held)

    def test_short_home_press_completes_arm_return(self):
        if self.collision_checking:
            self.skipTest("large arm travel is exercised with collision checking disabled")
        child = self.start_bridge()
        self.wait_until(lambda: self.receiver.feedback.command_id > 0, timeout=15)
        self.pipeline._set_upper_body_enabled(True)
        self.wait_until(lambda: self.pipeline._upper_body_stream_was_fresh)
        self.control(child, grip=1.0)
        self.cycles(0.2)
        self.control(child, z=-0.22)
        defaults = self.pipeline._upper_body_default
        self.wait_until(lambda: np.max(np.abs(self.pipeline._upper_body_filtered - defaults)) > 0.32)
        self.control(child, home=True)
        self.cycles(0.08)
        self.control(child, home=False, grip=0.0)
        self.wait_until(lambda: np.max(np.abs(self.pipeline._upper_body_filtered - defaults)) < 0.025)

    def test_separate_streams_track_crouch_twist_and_home(self):
        child = self.start_bridge()
        self.wait_until(lambda: self.receiver.feedback.command_id > 0, timeout=15)
        self.pipeline._set_upper_body_enabled(True)
        self.wait_until(lambda: self.pipeline._upper_body_stream_was_fresh)
        policy = self.pipeline._inner_policy()
        if self.with_omnihand:
            self.control(child, trigger=0.02)
            self.wait_until(lambda: self.pipeline.env.hand_snapshot()["targets"]["L_index_mcp_joint"] > 0.01)
            self.control(child, grip=1.0)
        else:
            self.control(child, grip=1.0, trigger=None, right_trigger=None)
        self.cycles(0.4)
        self.control(child, height=-0.1, heading=0.25)
        self.wait_until(lambda: policy.posture_targets[0] < 0.57)
        self.assertAlmostEqual(policy.posture_targets[1], 0.0, places=2)
        for twist in (0.10, 0.17, 0.24):
            self.control(child, twist=twist)
            self.cycles(0.16)
        self.wait_until(lambda: abs(policy.posture_targets[1]) > 0.05)
        self.control(child, right_grip=1.0)
        self.cycles(0.15)
        self.control(child, grip=0.0)
        self.cycles(0.2)
        self.control(child, height=-0.17)
        self.wait_until(lambda: policy.posture_targets[0] < 0.53)
        self.control(child, freeze_body=True)
        self.cycles(0.4)
        held = policy.posture_targets
        self.control(child, height=-0.22)
        self.cycles(0.15)
        self.assertEqual(policy.posture_targets, held)
        self.control(child, home=True)
        self.wait_until(lambda: policy.posture_targets == (0.64, 0.0))
        self.control(child, home=False, freeze_body=False, height=-0.28)
        self.cycles(0.4)
        self.assertEqual(policy.posture_targets, (0.64, 0.0))
        self.pipeline._set_upper_body_enabled(False)
        self.cycles(0.15)
        self.assertFalse(self.pipeline.ctrl_manager.controllers.LocomanipulationPostureZmqCtrl.inst.get_data()["fresh"])


class XrobotoolkitArmOnlyPelvisIKIntegrationTests(XrobotoolkitPelvisIKFixture, unittest.TestCase):
    pass


class XrobotoolkitArmOnlyPelvisIKCollisionIntegrationTests(XrobotoolkitArmOnlyPelvisIKIntegrationTests):
    collision_checking = True


class XrobotoolkitPelvisIKIntegrationTests(XrobotoolkitPelvisIKFixture, unittest.TestCase):
    with_omnihand = True


class XrobotoolkitPelvisIKCollisionIntegrationTests(XrobotoolkitPelvisIKIntegrationTests):
    collision_checking = True


if __name__ == "__main__":
    unittest.main()
