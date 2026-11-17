import unittest

from robojudo.config import cfg_registry
from robojudo.config.config_manager import ConfigManager
from robojudo.config.observer_cfg import JointObserverCfg
from robojudo.config.x2 import x2_locomimic_observer, x2_locomimic_real, x2_locomimic_real_observer
from robojudo.config.x2.observer_parameters import JOINT_NAMES


class TestX2LocomimicObserverConfig(unittest.TestCase):
    def test_simulation_and_deployment_use_identical_observer_parameters(self):
        simulation = x2_locomimic_observer().env.observer.model_dump()
        deployment = x2_locomimic_real_observer().env.observer.model_dump()
        self.assertEqual(simulation.pop("mode"), "ACTIVE")
        self.assertEqual(deployment.pop("mode"), "OFF")
        self.assertEqual(simulation["algorithm"], "nonlinear")
        self.assertEqual(simulation["slew_nm_s"], 2.0)
        self.assertEqual(simulation["release_slew_nm_s"], 16.0)
        self.assertEqual(simulation, deployment)

    def test_registered_preset_preserves_base_behavior(self):
        self.assertIs(cfg_registry.get("x2_locomimic_real_observer"), x2_locomimic_real_observer)
        cfg = ConfigManager("x2_locomimic_real_observer").get_cfg()
        base = x2_locomimic_real()
        actual, expected = cfg.model_dump(), base.model_dump()
        actual["env"].pop("observer")
        expected["env"].pop("observer")
        self.assertEqual(actual, expected)
        self.assertFalse(cfg.env.is_sim)
        self.assertTrue(cfg.do_safety_check)

    def test_arm_limits_use_joint_names_and_compensation_defaults_off(self):
        cfg = x2_locomimic_real_observer()
        limits = dict(zip(cfg.env.dof.joint_names, cfg.env.dof.torque_limits, strict=True))
        observer = cfg.env.observer
        self.assertEqual(observer.mode, "OFF")
        self.assertEqual(observer.actuator_limits_nm, [limits[name] for name in JOINT_NAMES])
        self.assertEqual(len(observer.actuator_limits_nm), 14)
        self.assertEqual(observer.correction_limit_fraction, 0.8)
        self.assertEqual(observer.correction_limit_nm, 0.0)
        self.assertEqual(observer.resolve_correction_limits(), [0.8 * limits[name] for name in JOINT_NAMES])
        self.assertEqual(observer.slew_nm_s, 2.0)
        self.assertEqual(observer.release_slew_nm_s, 16.0)
        self.assertIsNone(x2_locomimic_real().env.observer)
        shadow = JointObserverCfg.model_validate({**observer.model_dump(), "mode": "SHADOW"})
        self.assertEqual(shadow.resolve_correction_limits(), observer.resolve_correction_limits())
        self.assertEqual(shadow.slew_nm_s, observer.slew_nm_s)
        self.assertEqual(shadow.release_slew_nm_s, observer.release_slew_nm_s)

    def test_mutations_do_not_leak_to_other_presets(self):
        cfg = x2_locomimic_real_observer()
        pristine = x2_locomimic_real_observer().model_dump()
        base = x2_locomimic_real().model_dump()
        cfg.env.observer.actuator_limits_nm[0] = 1.0
        cfg.env.dof.torque_limits[0] = 1.0
        cfg.ctrl[-1].endpoint = "tcp://127.0.0.1:9999"
        cfg.ctrl[0].triggers.clear()
        self.assertEqual(x2_locomimic_real_observer().model_dump(), pristine)
        self.assertEqual(x2_locomimic_real().model_dump(), base)


if __name__ == "__main__":
    unittest.main()
