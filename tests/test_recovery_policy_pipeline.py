import unittest
from types import SimpleNamespace

import numpy as np


class TestRecoveryPolicyPipeline(unittest.TestCase):
    @staticmethod
    def _pipeline(mode, tilt_quat):
        from robojudo.pipeline.four_mode_pipeline import ControlMode
        from robojudo.pipeline.x2_locomanipulation_loco_mimic_pipeline import (
            X2LocomanipulationLocoMimicPipeline,
        )

        pipeline = X2LocomanipulationLocoMimicPipeline.__new__(X2LocomanipulationLocoMimicPipeline)
        pipeline.mode = mode
        pipeline.env = SimpleNamespace(base_quat=np.asarray(tilt_quat, dtype=np.float32))
        pipeline._shutdown_requested = False
        pipeline._joint_default_complete = True
        pipeline._upper_body_enabled = True
        pipeline._set_upper_body_enabled = lambda enabled: setattr(
            pipeline, "_upper_body_enabled", enabled
        )
        pipeline.policy_locomotion_mimic_flag = 0
        pipeline._recovery_return_in_progress = False
        pipeline.entered_modes = []

        def enter_mode(requested, force=False):
            del force
            pipeline.entered_modes.append(requested)
            pipeline.mode = requested
            return True

        pipeline._enter_mode = enter_mode
        pipeline.policy_manager = SimpleNamespace(
            activate_recovery=lambda: True,
            switch_to_loco=lambda callback_end=None: (
                setattr(pipeline, "return_callback", callback_end),
                True,
            )[1],
        )
        return pipeline

    def test_recovery_requires_fallen_joint_default_state(self):
        from robojudo.pipeline.four_mode_pipeline import ControlMode

        fallen = [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]
        upright = [0.0, 0.0, 0.0, 1.0]

        pipeline = self._pipeline(ControlMode.JOINT_DEFAULT, fallen)
        self.assertTrue(pipeline._try_enter_recovery())
        self.assertEqual(pipeline.mode, ControlMode.RECOVERY_DEFAULT)
        self.assertFalse(pipeline._upper_body_enabled)

        pipeline = self._pipeline(ControlMode.JOINT_DEFAULT, upright)
        self.assertFalse(pipeline._try_enter_recovery())
        self.assertEqual(pipeline.mode, ControlMode.JOINT_DEFAULT)

        pipeline = self._pipeline(ControlMode.DAMPING_DEFAULT, fallen)
        self.assertFalse(pipeline._try_enter_recovery())
        self.assertEqual(pipeline.mode, ControlMode.DAMPING_DEFAULT)

        pipeline = self._pipeline(ControlMode.JOINT_DEFAULT, fallen)
        pipeline._joint_default_complete = False
        self.assertFalse(pipeline._try_enter_recovery())
        self.assertEqual(pipeline.mode, ControlMode.JOINT_DEFAULT)

    def test_loco_return_requires_upright_and_completes_in_rl(self):
        from robojudo.pipeline.four_mode_pipeline import ControlMode

        fallen = [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]
        upright = [0.0, 0.0, 0.0, 1.0]

        pipeline = self._pipeline(ControlMode.RECOVERY_DEFAULT, fallen)
        self.assertFalse(pipeline._request_loco_from_recovery())
        self.assertFalse(hasattr(pipeline, "return_callback"))

        pipeline.env.base_quat = np.asarray(upright, dtype=np.float32)
        self.assertTrue(pipeline._request_loco_from_recovery())
        self.assertTrue(pipeline._recovery_return_in_progress)
        pipeline.return_callback()
        self.assertEqual(pipeline.mode, ControlMode.RL_DEFAULT)

    def test_fall_safety_is_suspended_only_during_recovery(self):
        from robojudo.pipeline.four_mode_pipeline import ControlMode, FourModePipelineMixin

        fallen = np.asarray([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], dtype=np.float32)
        pipeline = FourModePipelineMixin.__new__(FourModePipelineMixin)
        pipeline.env = SimpleNamespace(base_quat=fallen)
        pipeline.do_safety_check = True
        reasons = []
        pipeline._force_damping = reasons.append

        pipeline.mode = ControlMode.RECOVERY_DEFAULT
        pipeline._safety_check_before_command()
        self.assertEqual(reasons, [])

        pipeline.mode = ControlMode.RL_DEFAULT
        pipeline._safety_check_before_command()
        self.assertEqual(len(reasons), 1)

    def test_manual_passive_is_supported_while_fallen(self):
        from robojudo.pipeline.four_mode_pipeline import ControlMode, FourModePipelineMixin

        fallen = np.asarray([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], dtype=np.float32)
        pipeline = FourModePipelineMixin.__new__(FourModePipelineMixin)
        pipeline.mode = ControlMode.PASSIVE_DEFAULT
        pipeline.env = SimpleNamespace(base_quat=fallen)
        pipeline.do_safety_check = True
        pipeline._manual_mode_override = ControlMode.PASSIVE_DEFAULT
        reasons = []
        pipeline._force_damping = reasons.append
        pipeline._enter_mode = lambda requested: setattr(pipeline, "mode", requested) or True

        pipeline._safety_check_before_command()
        self.assertEqual(reasons, [])

        pipeline._manual_mode_override = None
        pipeline._safety_check_before_command()
        self.assertEqual(len(reasons), 1)

        reasons.clear()
        pipeline.mode = ControlMode.DAMPING_DEFAULT
        pipeline._process_commands(["[PASSIVE_DEFAULT]"])
        pipeline._safety_check_before_command()
        self.assertEqual(pipeline.mode, ControlMode.PASSIVE_DEFAULT)
        self.assertEqual(pipeline._manual_mode_override, ControlMode.PASSIVE_DEFAULT)
        self.assertEqual(reasons, [])

        pipeline._process_commands(["[DAMPING_DEFAULT]"])
        self.assertEqual(pipeline.mode, ControlMode.DAMPING_DEFAULT)
        self.assertIsNone(pipeline._manual_mode_override)

        pipeline._process_commands(["[JOINT_DEFAULT]"])
        pipeline._safety_check_before_command()
        self.assertEqual(pipeline.mode, ControlMode.JOINT_DEFAULT)
        self.assertEqual(pipeline._manual_mode_override, ControlMode.JOINT_DEFAULT)

    def test_nonfinite_orientation_cannot_activate_recovery(self):
        from robojudo.pipeline.four_mode_pipeline import ControlMode

        pipeline = self._pipeline(
            ControlMode.DAMPING_DEFAULT,
            [np.nan, 0.0, 0.0, 1.0],
        )
        self.assertFalse(pipeline._try_enter_recovery())
        self.assertEqual(pipeline.mode, ControlMode.DAMPING_DEFAULT)


if __name__ == "__main__":
    unittest.main()
