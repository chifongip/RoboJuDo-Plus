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

class SyntheticXR:
    def __init__(self):
        self.grip = 0.0
        self.right_grip = 0.0
        self.trigger = 0.0
        self.right_trigger = 0.0
        self.z = 0.0
        self.freeze = False
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
        return False
    def close(self):
        pass

if os.environ.get("XROBOTOOLKIT_TEST_DIAGNOSTICS"):
    logging.basicConfig(level=logging.INFO)
xr = SyntheticXR()
settings = TeleopSettings(collision_checking=sys.argv[5] == "True")
bridge = RoboJuDoBridge(sys.argv[1], sys.argv[2], settings=settings, xr_client=xr,
                       ik_backend=sys.argv[3], robot=sys.argv[4])
try:
    while True:
        if select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline()
            if not line:
                break
            for name, value in json.loads(line).items():
                setattr(xr, name, value)
        bridge.feedback.poll(timeout=20)
        started = time.monotonic()
        if bridge.poll():
            time.sleep(max(0, 0.02 - (time.monotonic() - started)))
        if os.environ.get("XROBOTOOLKIT_TEST_DIAGNOSTICS"):
            bridge.report_xr_diagnostics()
finally:
    bridge.close()
"""


@unittest.skipUnless(
    os.environ.get("XROBOTOOLKIT_TEST_PYTHON"), "set XROBOTOOLKIT_TEST_PYTHON for two-process XR tests"
)
class XrobotoolkitIntegrationTests(unittest.TestCase):
    with_omnihand = False
    collision_checking = True

    def setUp(self):
        from robojudo.config.x2 import x2_xrobotoolkit, x2_xrobotoolkit_omnihand
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
        cfg = x2_xrobotoolkit_omnihand() if self.with_omnihand else x2_xrobotoolkit()
        upper = next(item for item in cfg.ctrl if isinstance(item, UpperBodyZmqCtrlCfg))
        upper.endpoint = self.command_endpoint
        upper.feedback_endpoint = "tcp://127.0.0.1:*"
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
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=3)
            if child.stdin is not None:
                child.stdin.close()

    def start_bridge(self):
        root = Path(__file__).resolve().parents[1]
        log = self.enterContext(open(Path(self.directory) / f"bridge-{len(self.children)}.log", "w"))
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(root / "third_party/xrobotoolkit_teleop")
        child = subprocess.Popen(
            [
                os.environ["XROBOTOOLKIT_TEST_PYTHON"],
                "-u",
                "-c",
                BRIDGE_PROCESS,
                self.command_endpoint,
                self.feedback_endpoint,
                os.environ.get("XROBOTOOLKIT_TEST_BACKEND", "placo"),
                "x2_omnihand" if self.with_omnihand else "x2",
                str(self.collision_checking),
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

    def test_slow_inference_still_accepts_fresh_arm_proposals(self):
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
        self.cycles(0.6)
        self.assertGreaterEqual(
            self.receiver.feedback.command_id - accepted,
            3,
            "\n".join(path.read_text() for path in Path(self.directory).glob("*.log")),
        )
        state = self.receiver.feedback._states[self.receiver.feedback.state_id][1]
        self.assertLess(max(state["joint_age_s"].values()), 0.02)
        self.assertGreater(state["pipeline_timing_ms"]["policy"], 50)


@unittest.skipUnless(os.environ.get("XROBOTOOLKIT_TEST_PYTHON"), "set XROBOTOOLKIT_TEST_PYTHON")
class XrobotoolkitOmniHandIntegrationTests(XrobotoolkitIntegrationTests):
    with_omnihand = True

    def test_independent_triggers_and_home_holds_fingers(self):
        child = self.start_bridge()
        self.wait_until(lambda: self.receiver.feedback.command_id > 0)
        self.pipeline._set_upper_body_enabled(True)
        self.cycles(0.25)
        self.control(child, trigger=0.4)
        self.wait_until(lambda: self.pipeline.env.hand_snapshot()["targets"]["L_index_mcp_joint"] > 0.04)
        self.wait_until(lambda: self.pipeline.env.hand_snapshot()["positions"]["L_index_mcp_joint"] > 0.03)
        self.assertEqual(self.pipeline.env.hand_snapshot()["targets"]["R_index_mcp_joint"], 0.0)
        self.control(child, right_trigger=0.3)
        self.wait_until(lambda: self.pipeline.env.hand_snapshot()["targets"]["R_index_mcp_joint"] > 0.04)
        self.wait_until(lambda: self.pipeline.env.hand_snapshot()["positions"]["R_index_mcp_joint"] > 0.03)
        self.control(child, grip=0.0, right_grip=0.0)
        self.cycles(0.15)
        held = self.pipeline.env.hand_snapshot()["targets"].copy()
        self.pipeline._set_upper_body_enabled(False)
        self.cycles(0.3)
        self.assertEqual(self.pipeline.env.hand_snapshot()["targets"], held)

        self.control(child, trigger=0.0, right_trigger=0.0)
        self.cycles(0.2)
        self.assertEqual(self.pipeline.env.hand_snapshot()["targets"], held)
        self.pipeline._set_upper_body_enabled(True)
        self.wait_until(lambda: self.pipeline.env.hand_snapshot()["targets"]["L_index_mcp_joint"] < 0.01)
        self.wait_until(lambda: self.pipeline.env.hand_snapshot()["targets"]["R_index_mcp_joint"] < 0.01)


class XrobotoolkitCollisionDisabledIntegrationTests(XrobotoolkitIntegrationTests):
    collision_checking = False


class XrobotoolkitOmniHandCollisionDisabledIntegrationTests(XrobotoolkitOmniHandIntegrationTests):
    collision_checking = False


if __name__ == "__main__":
    unittest.main()
