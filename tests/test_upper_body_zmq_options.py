"""Bounded transport options apply before connection without changing admission."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import zmq
from test_upper_body_feedback import MemorySocket, TransportExecutor

from robojudo.config.x2 import x2_xrobotoolkit, x2_xrobotoolkit_omnihand_real
from robojudo.controller.ctrl_cfgs import UpperBodyZmqCtrlCfg
from robojudo.controller.upper_body_zmq_ctrl import UpperBodyZmqCtrl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_pipeline import parse_args  # noqa: E402


class RecordingSocket(MemorySocket):
    def __init__(self):
        super().__init__()
        self.events = []

    def setsockopt(self, option, value):
        self.events.append(("option", option, value))

    def connect(self, endpoint):
        self.events.append(("connect", endpoint))

    def bind(self, endpoint):
        self.events.append(("bind", endpoint))


class RecordingContext:
    def __init__(self):
        self.sockets = []

    def socket(self, kind):
        socket = RecordingSocket()
        self.sockets.append(socket)
        return socket


class ZmqOptionTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch("robojudo.controller.xr_target_executor.XRTargetExecutor", TransportExecutor))

    def config(self, **kwargs):
        return UpperBodyZmqCtrlCfg(
            joint_names=["arm"], ema_alpha=0.0, feedback_endpoint="tcp://127.0.0.1:8571", **kwargs
        )

    def test_legacy_defaults_and_xr_command_queue_are_explicit(self):
        legacy = self.config()
        self.assertEqual(legacy.command_receive_hwm, 100)
        self.assertEqual(legacy.feedback_send_hwm, 1)
        for preset in (x2_xrobotoolkit, x2_xrobotoolkit_omnihand_real):
            controller = next(c for c in preset().ctrl if isinstance(c, UpperBodyZmqCtrlCfg))
            self.assertEqual(controller.command_receive_hwm, 8)
            self.assertEqual(controller.feedback_send_hwm, 1)

    def test_options_are_set_before_bind_connect_without_conflating_feedback(self):
        for hwm in (1, 8):
            with self.subTest(hwm=hwm):
                context = RecordingContext()
                cfg = self.config(command_receive_hwm=8, feedback_send_hwm=hwm)
                with patch("robojudo.controller.upper_body_zmq_ctrl.zmq.Context.instance", return_value=context):
                    controller = UpperBodyZmqCtrl(cfg)
                self.addCleanup(controller.close)
                command, feedback = context.sockets
                self.assertLess(
                    command.events.index(("option", zmq.RCVHWM, 8)), command.events.index(("connect", cfg.endpoint))
                )
                self.assertLess(
                    feedback.events.index(("option", zmq.SNDHWM, hwm)),
                    feedback.events.index(("bind", cfg.feedback_endpoint)),
                )
                self.assertFalse(
                    any(e[0] == "option" and e[1] == zmq.CONFLATE for socket in context.sockets for e in socket.events)
                )

    def test_queue_limits_reject_unbounded_or_invalid_values(self):
        for values in (
            {"command_receive_hwm": 0},
            {"command_receive_hwm": 101},
            {"feedback_send_hwm": 0},
            {"feedback_send_hwm": 9},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.config(**values)

    def test_launcher_rejects_removed_collision_option(self):
        with patch.object(sys, "argv", ["run_pipeline.py", "--xr-collision-checking"]), self.assertRaises(SystemExit):
            parse_args()
