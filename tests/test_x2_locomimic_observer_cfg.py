import unittest

from robojudo.config import cfg_registry
from robojudo.config.config_manager import ConfigManager
from robojudo.config.x2 import x2_locomimic_real, x2_locomimic_real_observer
from robojudo.config.x2.env.x2_real_env_cfg import X2ArmObserverCfg
from robojudo.utils.x2_observer_reference import JOINT_NAMES


class TestX2LocomimicObserverConfig(unittest.TestCase):
    def test_registered_preset_preserves_base_behavior(self):
        self.assertIs(cfg_registry.get("x2_locomimic_real_observer"), x2_locomimic_real_observer)
        cfg = ConfigManager("x2_locomimic_real_observer").get_cfg()
        base = x2_locomimic_real()
        actual, expected = cfg.model_dump(), base.model_dump()
        actual["env"].pop("arm_observer")
        expected["env"].pop("arm_observer")
        self.assertEqual(actual, expected)
        self.assertFalse(cfg.env.is_sim)
        self.assertTrue(cfg.do_safety_check)

    def test_arm_limits_use_joint_names_and_compensation_defaults_off(self):
        cfg = x2_locomimic_real_observer()
        limits = dict(zip(cfg.env.dof.joint_names, cfg.env.dof.torque_limits, strict=True))
        observer = cfg.env.arm_observer
        self.assertEqual(observer.mode, "OFF")
        self.assertEqual(observer.actuator_limits_nm, [limits[name] for name in JOINT_NAMES])
        self.assertEqual(len(observer.actuator_limits_nm), 14)
        self.assertEqual(observer.correction_limits_nm, [min(8.0, 0.8 * limits[name]) for name in JOINT_NAMES])
        self.assertEqual(observer.slew_nm_s, 2.0)
        self.assertEqual(observer.release_slew_nm_s, 16.0)
        self.assertEqual(x2_locomimic_real().env.arm_observer.mode, "OFF")
        shadow = X2ArmObserverCfg.model_validate({**observer.model_dump(), "mode": "SHADOW"})
        self.assertEqual(shadow.correction_limits_nm, observer.correction_limits_nm)
        self.assertEqual(shadow.slew_nm_s, observer.slew_nm_s)
        self.assertEqual(shadow.release_slew_nm_s, observer.release_slew_nm_s)

    def test_mutations_do_not_leak_to_other_presets(self):
        cfg = x2_locomimic_real_observer()
        pristine = x2_locomimic_real_observer().model_dump()
        base = x2_locomimic_real().model_dump()
        cfg.env.arm_observer.actuator_limits_nm[0] = 1.0
        cfg.env.dof.torque_limits[0] = 1.0
        cfg.ctrl[-1].endpoint = "tcp://127.0.0.1:9999"
        cfg.ctrl[0].triggers.clear()
        self.assertEqual(x2_locomimic_real_observer().model_dump(), pristine)
        self.assertEqual(x2_locomimic_real().model_dump(), base)


if __name__ == "__main__":
    unittest.main()
