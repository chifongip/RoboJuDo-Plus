import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from robojudo.controller.casia_hand_runtime import CASIA_JOINT_NAMES
from robojudo.pipeline.upper_body_casia_hand_zmq_pipeline import UpperBodyCasiaHandZmqPipelineMixin
from robojudo.pipeline.upper_body_zmq_pipeline import UpperBodyZmqPipelineMixin


class TestPipeline(UpperBodyCasiaHandZmqPipelineMixin, UpperBodyZmqPipelineMixin):
    pass


class FakeRecorderClient:
    def __init__(self):
        self.samples = []

    def submit(self, **sample):
        self.samples.append(sample)


class TestUpperBodyCasiaHandZmqPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = TestPipeline.__new__(TestPipeline)
        self.pipeline._recorder_client = FakeRecorderClient()
        self.pipeline._upper_body_cfg = SimpleNamespace(joint_names=["left_arm", "right_arm"])
        self.pipeline._upper_body_indices = np.asarray([1, 3], dtype=np.int32)
        self.pipeline._upper_body_enabled = True
        self.pipeline._upper_body_stream_was_fresh = True
        self.pipeline._recording_active = True
        self.pipeline._recording_paused = False
        self.pipeline._upper_body_control_available = lambda: True
        self.pipeline._upper_body_casia_hand_ctrl_data = self.ctrl_data()
        self.env_data = SimpleNamespace(dof_pos=np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32))

    @staticmethod
    def ctrl_data(*, hand_fresh=True):
        return {
            "UpperBodyCasiaHandZmqCtrl": {
                "fresh": True,
                "joint_positions": {"left_arm": 0.1, "right_arm": 0.2},
                "casia_hand": {
                    "joint_names": ["left_thumb", "right_thumb"],
                    "joint_positions": np.asarray([0.2, 0.3], dtype=np.float32),
                    "joint_position_commands": np.asarray([0.4, 0.5], dtype=np.float32),
                    "fresh": hand_fresh,
                },
            }
        }

    def test_records_arm_and_hands_as_one_named_schema(self):
        self.pipeline._record_upper_body_sample(
            self.env_data,
            {"locomotion_command": np.zeros(4)},
            np.asarray([10.0, 11.0, 12.0, 13.0], dtype=np.float32),
            rl_active=True,
        )

        sample = self.pipeline._recorder_client.samples[0]
        self.assertEqual(sample["joint_names"], ["left_arm", "right_arm", "left_thumb", "right_thumb"])
        np.testing.assert_allclose(sample["joint_positions"], [1.0, 3.0, 0.2, 0.3])
        np.testing.assert_allclose(sample["joint_position_commands"], [11.0, 13.0, 0.4, 0.5])

    def test_skips_recording_until_hand_command_and_feedback_are_fresh(self):
        self.pipeline._upper_body_casia_hand_ctrl_data = self.ctrl_data(hand_fresh=False)
        self.pipeline._record_upper_body_sample(
            self.env_data,
            {"locomotion_command": np.zeros(4)},
            np.zeros(4),
            rl_active=True,
        )
        self.assertEqual(self.pipeline._recorder_client.samples, [])

    def test_takeover_state_is_forwarded_only_by_casia_pipeline(self):
        controller = Mock()
        self.pipeline.ctrl_manager = SimpleNamespace(
            controllers={"UpperBodyCasiaHandZmqCtrl": SimpleNamespace(inst=controller)}
        )
        self.pipeline._upper_body_enabled = False

        self.pipeline._set_upper_body_enabled(True)

        controller.set_takeover_enabled.assert_called_once_with(True)

    def test_casia_controller_stream_reuses_existing_arm_filter(self):
        self.pipeline._upper_body_indices = np.asarray([0, 1], dtype=np.int32)
        self.pipeline._upper_body_default = np.zeros(2, dtype=np.float32)
        self.pipeline._upper_body_filtered = np.zeros(2, dtype=np.float32)
        self.pipeline._upper_body_cfg = SimpleNamespace(
            joint_names=["left_arm", "right_arm"],
            ema_alpha=0.0,
        )
        self.pipeline.env = SimpleNamespace(
            position_limits=np.asarray([[-1.0, 1.0], [-1.0, 1.0]], dtype=np.float32)
        )

        result = self.pipeline._apply_pd_target_override(np.zeros(2, dtype=np.float32), self.ctrl_data())

        np.testing.assert_allclose(result, [0.1, 0.2])

    def test_g1_casia_configs_are_dedicated_and_preserve_arm_only_configs(self):
        from robojudo.config.g1.g1_cfg import (
            g1_23_casia_locomanipulation_default_real,
            g1_23_casia_locomanipulation_stiff_real,
            g1_23_locomanipulation_default_real,
            g1_29_casia_locomanipulation_stiff_real,
        )

        arm_only = g1_23_locomanipulation_default_real()
        self.assertEqual(arm_only.pipeline_type, "G1LocomanipulationPipeline")
        self.assertEqual(arm_only.ctrl[-1].ctrl_type, "UpperBodyZmqCtrl")

        cases = [
            (g1_23_casia_locomanipulation_default_real(), 10),
            (g1_23_casia_locomanipulation_stiff_real(), 10),
            (g1_29_casia_locomanipulation_stiff_real(), 14),
        ]
        for cfg, arm_joint_count in cases:
            with self.subTest(config=type(cfg).__name__):
                self.assertEqual(cfg.pipeline_type, "G1CasiaHandLocomanipulationPipeline")
                self.assertEqual(
                    [ctrl.ctrl_type for ctrl in cfg.ctrl],
                    ["UnitreeCtrl", "UpperBodyCasiaHandZmqCtrl"],
                )
                self.assertEqual(len(cfg.ctrl[-1].joint_names), arm_joint_count)
                self.assertEqual(len(cfg.ctrl[-1].joint_names) + len(CASIA_JOINT_NAMES), arm_joint_count + 20)
                self.assertEqual(cfg.ctrl[-1].endpoint, "tcp://127.0.0.1:8560")
                self.assertEqual(cfg.ctrl[-1].casia_hand.port_name, "/dev/ttyUSB0")


if __name__ == "__main__":
    unittest.main()
