import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from robojudo.pipeline.upper_body_hand_zmq_pipeline import UpperBodyHandZmqPipelineMixin
from robojudo.pipeline.upper_body_zmq_pipeline import UpperBodyZmqPipelineMixin


class TestPipeline(UpperBodyHandZmqPipelineMixin, UpperBodyZmqPipelineMixin):
    pass


class FakeRecorderClient:
    def __init__(self):
        self.samples = []

    def submit(self, **sample):
        self.samples.append(sample)


class TestUpperBodyHandZmqPipeline(unittest.TestCase):
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
        self.pipeline._upper_body_hand_ctrl_data = self.ctrl_data()
        self.env_data = SimpleNamespace(dof_pos=np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32))

    @staticmethod
    def ctrl_data(*, hand_fresh=True):
        return {
            "UpperBodyHandZmqCtrl": {
                "fresh": True,
                "joint_positions": {"left_arm": 0.1, "right_arm": 0.2},
                "omnihand": {
                    "joint_names": ["L_thumb", "R_thumb"],
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
        self.assertEqual(sample["joint_names"], ["left_arm", "right_arm", "L_thumb", "R_thumb"])
        np.testing.assert_allclose(sample["joint_positions"], [1.0, 3.0, 0.2, 0.3])
        np.testing.assert_allclose(sample["joint_position_commands"], [11.0, 13.0, 0.4, 0.5])

    def test_skips_recording_until_hand_command_and_feedback_are_fresh(self):
        self.pipeline._upper_body_hand_ctrl_data = self.ctrl_data(hand_fresh=False)
        self.pipeline._record_upper_body_sample(
            self.env_data,
            {"locomotion_command": np.zeros(4)},
            np.zeros(4),
            rl_active=True,
        )
        self.assertEqual(self.pipeline._recorder_client.samples, [])

    def test_takeover_state_is_forwarded_only_by_hand_pipeline(self):
        controller = Mock()
        self.pipeline.ctrl_manager = SimpleNamespace(
            controllers={"UpperBodyHandZmqCtrl": SimpleNamespace(inst=controller)}
        )
        self.pipeline._upper_body_enabled = False

        self.pipeline._set_upper_body_enabled(True)

        controller.set_takeover_enabled.assert_called_once_with(True)

    def test_hand_controller_stream_reuses_the_existing_arm_filter(self):
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

        result = self.pipeline._apply_pd_target_override(
            np.zeros(2, dtype=np.float32),
            self.ctrl_data(),
        )

        np.testing.assert_allclose(result, [0.1, 0.2])

    def test_x2_real_config_selects_only_the_dedicated_hand_path(self):
        from robojudo.config.x2 import x2_locomanipulation, x2_locomanipulation_real

        sim_cfg = x2_locomanipulation()
        real_cfg = x2_locomanipulation_real()
        self.assertEqual(sim_cfg.pipeline_type, "X2LocomanipulationPipeline")
        self.assertEqual(sim_cfg.ctrl[-1].ctrl_type, "UpperBodyZmqCtrl")
        self.assertFalse(hasattr(sim_cfg.ctrl[-1], "omnihand"))
        self.assertEqual(real_cfg.pipeline_type, "X2OmniHandLocomanipulationPipeline")
        self.assertEqual(
            [ctrl.ctrl_type for ctrl in real_cfg.ctrl],
            ["RosJoystickCtrl", "UpperBodyHandZmqCtrl"],
        )
        self.assertEqual(real_cfg.ctrl[-1].endpoint, "tcp://10.0.1.20:8560")
        self.assertEqual(real_cfg.ctrl[-1].omnihand.transport, "hcan")
        self.assertEqual(real_cfg.ctrl[-1].omnihand.left_adapter_index, 1)
        self.assertEqual(real_cfg.ctrl[-1].omnihand.right_adapter_index, 0)


if __name__ == "__main__":
    unittest.main()
