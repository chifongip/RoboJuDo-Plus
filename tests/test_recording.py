import unittest
from types import SimpleNamespace

import numpy as np

from robojudo.pipeline.upper_body_zmq_pipeline import UpperBodyZmqPipelineMixin
from robojudo.recording.record_cfgs import RecordCfg


class FakeRecorderClient:
    def __init__(self):
        self.samples = []
        self.finished = []

    def submit(self, **sample):
        self.samples.append(sample)

    def finish_episode(self, *, save):
        self.finished.append(save)


class TestRecordCfg(unittest.TestCase):
    def test_recording_disabled_by_default(self):
        self.assertFalse(RecordCfg().enabled)

    def test_recording_configuration_validation(self):
        with self.assertRaisesRegex(ValueError, "tcp"):
            RecordCfg(endpoint="ipc:///tmp/recorder")
        with self.assertRaisesRegex(ValueError, "task"):
            RecordCfg(task=" ")


class TestUpperBodyRecording(unittest.TestCase):
    def setUp(self):
        self.pipeline = UpperBodyZmqPipelineMixin.__new__(UpperBodyZmqPipelineMixin)
        self.pipeline._recorder_client = FakeRecorderClient()
        self.pipeline._upper_body_cfg = SimpleNamespace(joint_names=["left_arm", "right_arm"])
        self.pipeline._upper_body_indices = np.asarray([1, 3], dtype=np.int32)
        self.pipeline._upper_body_enabled = True
        self.pipeline._upper_body_stream_was_fresh = True
        self.pipeline._recording_active = True
        self.pipeline._recording_paused = False
        self.pipeline._upper_body_control_available = lambda: True
        self.pipeline._upper_body_enable_available = lambda: True
        self.env_data = SimpleNamespace(dof_pos=np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32))

    def test_records_feedback_final_targets_and_locomotion_command(self):
        pd_target = np.asarray([10.0, 11.0, 12.0, 13.0], dtype=np.float32)
        extras = {"locomotion_command": np.asarray([0.3, -0.2, 0.1, 0.62, 1.2], dtype=np.float32)}

        self.pipeline._record_upper_body_sample(self.env_data, extras, pd_target, rl_active=True)

        sample = self.pipeline._recorder_client.samples[0]
        self.assertEqual(sample["joint_names"], ["left_arm", "right_arm"])
        np.testing.assert_array_equal(sample["joint_positions"], [1.0, 3.0])
        np.testing.assert_array_equal(sample["joint_position_commands"], [11.0, 13.0])
        np.testing.assert_allclose(sample["velocity_height_command"], [0.3, -0.2, 0.1, 0.62])

    def test_does_not_record_stale_upper_body_stream(self):
        self.pipeline._upper_body_stream_was_fresh = False
        self.pipeline._record_upper_body_sample(
            self.env_data,
            {"locomotion_command": np.zeros(4)},
            np.zeros(4),
            rl_active=True,
        )
        self.assertEqual(self.pipeline._recorder_client.samples, [])

    def test_finishes_episode_when_control_becomes_unavailable(self):
        self.pipeline._record_upper_body_sample(self.env_data, {}, np.zeros(4), rl_active=False)
        self.assertEqual(self.pipeline._recorder_client.finished, [True])
        self.assertFalse(self.pipeline._recording_active)

    def test_recording_requires_explicit_start_and_supports_pause(self):
        self.pipeline._recording_active = False
        self.pipeline._toggle_recording_episode()
        self.assertTrue(self.pipeline._recording_active)

        self.pipeline._toggle_recording_pause()
        self.assertTrue(self.pipeline._recording_paused)
        self.pipeline._record_upper_body_sample(
            self.env_data,
            {"locomotion_command": np.zeros(4)},
            np.zeros(4),
            rl_active=True,
        )
        self.assertEqual(self.pipeline._recorder_client.samples, [])

        self.pipeline._toggle_recording_pause()
        self.assertFalse(self.pipeline._recording_paused)
        self.pipeline._toggle_recording_episode()
        self.assertFalse(self.pipeline._recording_active)
        self.assertEqual(self.pipeline._recorder_client.finished, [True])

    def test_record_start_requires_upper_body_takeover(self):
        self.pipeline._recording_active = False
        self.pipeline._upper_body_enabled = False
        self.pipeline._toggle_recording_episode()
        self.assertFalse(self.pipeline._recording_active)


if __name__ == "__main__":
    unittest.main()
