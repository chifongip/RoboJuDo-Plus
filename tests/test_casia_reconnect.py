"""GR00T reconnect/session integration, without cameras or robot hardware."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import zmq


class MessageSocket:
    def __init__(self):
        self.messages = []

    def recv_json(self, flags=0):
        if not self.messages:
            raise zmq.Again()
        return self.messages.pop(0)

    def close(self, linger=0):
        pass


class HandRuntime:
    def __init__(self):
        from robojudo.controller.casia_hand_runtime import CASIA_JOINT_NAMES

        self.data = {
            "joint_names": list(CASIA_JOINT_NAMES),
            "joint_positions": np.full(20, 0.2),
            "connected": True,
            "joint_state_fresh": True,
            "connection_generation": 1,
        }
        self.commands = []
        self.gates = []

    def get_data(self):
        return self.data.copy()

    def set_takeover_enabled(self, enabled, *, return_to_default=False):
        self.gates.append((enabled, return_to_default))

    def set_joint_commands(self, *command):
        self.commands.append(command)

    def close(self):
        pass


class TestCasiaReconnect(unittest.TestCase):
    def setUp(self):
        from robojudo.controller.ctrl_cfgs import CasiaHandCfg, Gr00tZmqCtrlCfg
        from robojudo.controller.gr00t_zmq_ctrl import Gr00tZmqCtrl

        self.hand = HandRuntime()
        cfg = Gr00tZmqCtrlCfg(
            joint_names=["left_arm", "right_arm"],
            endpoint="tcp://127.0.0.1:18559",
            casia_hand=CasiaHandCfg(auto_reconnect=True),
            ema_alpha=0.0,
        )
        with patch("robojudo.controller.gr00t_zmq_ctrl.CasiaHandRuntime", return_value=self.hand):
            self.controller = Gr00tZmqCtrl(cfg)
        self.controller._socket.close()
        self.socket = MessageSocket()
        self.controller._socket = self.socket
        self.addCleanup(self.controller.close)
        self.controller.get_data()
        self.controller.set_takeover_enabled(True)

    def message(self, session=None):
        return {
            "stream_id": self.controller._observation_stream_id,
            "control_session": self.controller._control_session if session is None else session,
            "sequence": 1,
            "positions": {"left_arm": 0.8, "right_arm": -0.8, **dict.fromkeys(self.hand.data["joint_names"], 0.4)},
            "locomotion_command": [0.3, 0.0, 0.0, 0.7],
        }

    def pipeline(self):
        from robojudo.pipeline.four_mode_pipeline import ControlMode
        from robojudo.pipeline.g1_gr00t_locomanipulation_pipeline import G1Gr00tLocomanipulationPipeline

        pipeline = G1Gr00tLocomanipulationPipeline.__new__(G1Gr00tLocomanipulationPipeline)
        pipeline.mode = ControlMode.RL_DEFAULT
        pipeline._upper_body_enabled = True
        pipeline._upper_body_control_available = lambda: True
        pipeline._upper_body_cfg = self.controller.cfg_ctrl
        pipeline.ctrl_manager = SimpleNamespace(controllers={"Gr00tZmqCtrl": SimpleNamespace(inst=self.controller)})
        pipeline._upper_body_indices = np.asarray([0, 1])
        pipeline._upper_body_default = np.asarray([0.1, -0.1])
        pipeline._upper_body_filtered = np.asarray([0.8, -0.8])
        pipeline._upper_body_stream_was_fresh = True
        pipeline.env = SimpleNamespace(position_limits=np.asarray([[-2.0, 2.0], [-2.0, 2.0]]))
        pipeline.dt = 0.02
        return pipeline

    def test_outage_invalidates_actions_and_arms_return_to_default(self):
        self.socket.messages.append(self.message())
        self.assertTrue(self.controller.get_data()["fresh"])
        self.controller._observation_snapshot = (1, np.ones(22))
        count = len(self.hand.commands)
        self.hand.data["joint_state_fresh"] = False
        self.socket.messages.append(self.message())
        data = self.controller.get_data()
        self.assertFalse(data["fresh"])
        self.assertFalse(data["takeover_enabled"])
        self.assertTrue(self.controller._takeover_requested)
        self.assertIsNone(self.controller._observation_snapshot)
        self.assertEqual(data["joint_positions"], {})
        self.assertEqual(len(self.hand.commands), count)
        pipeline = self.pipeline()
        ctrl_data = {"Gr00tZmqCtrl": data}
        pipeline._prepare_gr00t_stream(ctrl_data)
        self.assertFalse(data["takeover_enabled"])
        self.assertTrue(pipeline._upper_body_enabled)
        target = pipeline._apply_upper_body_override(np.zeros(2), ctrl_data)
        np.testing.assert_allclose(target, [0.78, -0.78])
        for _ in range(50):
            target = pipeline._apply_upper_body_override(np.zeros(2), ctrl_data)
        np.testing.assert_allclose(target, [0.1, -0.1])

    def test_observation_snapshot_is_cleared_until_feedback_recovers(self):
        self.controller.cfg_ctrl.observation_enabled = True
        self.controller._joint_indices = np.asarray([0, 1])
        env_data = {"dof_pos": np.asarray([0.1, -0.1])}
        self.controller.get_data_with_hook({}, env_data)
        self.assertEqual(self.controller._observation_snapshot[1].shape, (22,))
        session = self.controller._control_session
        self.hand.data["joint_state_fresh"] = False
        self.controller.get_data_with_hook({}, env_data)
        self.assertIsNone(self.controller._observation_snapshot)
        self.hand.data.update(joint_state_fresh=True, connection_generation=2)
        self.controller.get_data_with_hook({}, env_data)
        self.assertEqual(self.controller._control_session, session + 1)
        self.assertEqual(self.controller._observation_snapshot[1].shape, (22,))

    def test_reconnect_advances_session_and_rejects_old_commands(self):
        old_message = self.message()
        self.hand.data.update(connected=False, joint_state_fresh=False)
        self.controller.get_data()
        self.hand.data.update(connected=True, joint_state_fresh=True, connection_generation=2)
        self.socket.messages.append(old_message)
        data = self.controller.get_data()
        self.assertEqual(data["control_session"], old_message["control_session"] + 1)
        self.assertTrue(data["takeover_enabled"])
        self.assertFalse(data["fresh"])
        self.assertEqual(self.hand.commands, [])
        self.socket.messages.append(self.message())
        self.assertTrue(self.controller.get_data()["fresh"])
        self.assertEqual(len(self.hand.commands), 1)

    def test_reconnection_between_ticks_still_invalidates_session(self):
        old_message = self.message()
        self.hand.data["connection_generation"] = 2
        self.socket.messages.append(old_message)
        data = self.controller.get_data()
        self.assertEqual(data["control_session"], old_message["control_session"] + 1)
        self.assertFalse(data["fresh"])
        self.assertEqual(self.hand.commands, [])

    def test_feedback_recovers_without_reopening_still_starts_new_session(self):
        session = self.controller._control_session
        self.hand.data["joint_state_fresh"] = False
        self.controller.get_data()
        self.hand.data["joint_state_fresh"] = True
        data = self.controller.get_data()
        self.assertEqual(data["control_session"], session + 1)
        self.assertFalse(data["fresh"])

    def test_damping_or_start_disable_while_offline_prevents_auto_takeover(self):
        from robojudo.pipeline.four_mode_pipeline import ControlMode

        pipeline = self.pipeline()
        self.hand.data.update(connected=False, joint_state_fresh=False)
        self.controller.get_data()
        session = self.controller._control_session
        pipeline.mode = ControlMode.DAMPING_DEFAULT
        pipeline._set_upper_body_enabled(False)
        self.assertFalse(self.controller._takeover_requested)
        self.hand.data.update(connected=True, joint_state_fresh=True, connection_generation=2)
        data = self.controller.get_data()
        self.assertFalse(data["takeover_enabled"])
        self.assertEqual(data["control_session"], session)
        self.assertEqual(self.hand.commands, [])

    def test_reconnect_is_enabled_only_in_selected_config_family(self):
        from robojudo.config.g1.g1_vla_cfg import (
            g1_23_casia_locomanipulation_stiff_real,
            g1_23_gr00t_locomanipulation_stiff_real,
        )

        self.assertTrue(g1_23_gr00t_locomanipulation_stiff_real().ctrl[-1].casia_hand.auto_reconnect)
        self.assertFalse(g1_23_casia_locomanipulation_stiff_real().ctrl[-1].casia_hand.auto_reconnect)
        from robojudo.controller.ctrl_cfgs import CasiaHandCfg

        self.assertFalse(CasiaHandCfg().auto_reconnect)
        with self.assertRaises(ValueError):
            CasiaHandCfg(reconnect_timeout_s=0.01)


if __name__ == "__main__":
    unittest.main()
