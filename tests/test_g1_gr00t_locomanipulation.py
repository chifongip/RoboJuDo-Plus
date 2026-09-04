import unittest

import numpy as np
from box import Box


class TestG1Gr00tLocomanipulation(unittest.TestCase):
    def test_configs_match_23_dof_profiles(self):
        from robojudo.config.g1.g1_cfg import (
            g1_23_gr00t_locomanipulation_default,
            g1_23_gr00t_locomanipulation_default_real,
            g1_23_gr00t_locomanipulation_stiff,
            g1_23_gr00t_locomanipulation_stiff_real,
        )

        cases = [
            (g1_23_gr00t_locomanipulation_default(), "default", "policy_23dof_default", False),
            (g1_23_gr00t_locomanipulation_stiff(), "stiff", "policy_23dof_stiff", False),
            (g1_23_gr00t_locomanipulation_default_real(), "default", "policy_23dof_default", True),
            (g1_23_gr00t_locomanipulation_stiff_real(), "stiff", "policy_23dof_stiff", True),
        ]
        for cfg, preset, model, is_real in cases:
            with self.subTest(config=type(cfg).__name__):
                self.assertEqual(cfg.pipeline_type, "G1Gr00tLocomanipulationPipeline")
                self.assertEqual(cfg.policy.policy_type, "G1Gr00tLocomanipulationPolicy")
                self.assertEqual(cfg.policy.pd_gain_preset, preset)
                self.assertEqual(cfg.policy.policy_name, model)
                self.assertEqual(cfg.policy.obs_dof.num_dofs, 23)
                self.assertEqual(cfg.policy.action_dof.num_dofs, 13)
                self.assertEqual(cfg.policy.num_obs, 360)
                self.assertEqual(cfg.ctrl[-1].ctrl_type, "Gr00tZmqCtrl")
                self.assertEqual(cfg.ctrl[-1].joint_names, cfg.policy.obs_dof.joint_names[13:])
                self.assertEqual(len(cfg.ctrl[-1].joint_names), 10)
                self.assertTrue(cfg.ctrl[-1].observation_enabled)
                self.assertEqual(cfg.ctrl[-1].observation_profile, "g1_23dof")
                self.assertEqual(cfg.ctrl[-1].camera.type, "realsense")
                self.assertEqual(cfg.ctrl[-1].casia_hand is not None, is_real)
                expected_controls = ["UnitreeCtrl", "Gr00tZmqCtrl"] if is_real else [
                    "JoystickCtrl",
                    "KeyboardCtrl",
                    "Gr00tZmqCtrl",
                ]
                self.assertEqual([controller.ctrl_type for controller in cfg.ctrl], expected_controls)

    def test_existing_23_dof_configs_keep_manual_command_path(self):
        from robojudo.config.g1.g1_cfg import (
            g1_23_locomanipulation_default,
            g1_23_locomanipulation_stiff_real,
        )

        sim = g1_23_locomanipulation_default()
        real = g1_23_locomanipulation_stiff_real()
        self.assertEqual(sim.pipeline_type, "G1LocomanipulationPipeline")
        self.assertEqual(sim.policy.policy_type, "G1LocomanipulationPolicy")
        self.assertEqual(sim.ctrl[-1].ctrl_type, "UpperBodyZmqCtrl")
        self.assertEqual(real.pipeline_type, "G1LocomanipulationPipeline")
        self.assertEqual(real.policy.policy_type, "G1LocomanipulationPolicy")
        self.assertEqual(real.ctrl[-1].ctrl_type, "UpperBodyZmqCtrl")

    def test_real_lower_body_models_accept_gr00t_commands(self):
        from robojudo.config.g1.policy.g1_gr00t_locomanipulation_policy_cfg import (
            G1Gr00tLocomanipulation23PolicyCfg,
        )
        from robojudo.policy.g1_gr00t_locomanipulation_policy import G1Gr00tLocomanipulationPolicy

        for preset, model in (("default", "policy_23dof_default"), ("stiff", "policy_23dof_stiff")):
            with self.subTest(preset=preset):
                cfg = G1Gr00tLocomanipulation23PolicyCfg(policy_name=model, pd_gain_preset=preset)
                policy = G1Gr00tLocomanipulationPolicy(cfg, "cpu")
                env_data = Box(
                    {
                        "base_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                        "base_ang_vel": np.zeros(3, dtype=np.float32),
                        "dof_pos": np.asarray(cfg.obs_dof.default_pos, dtype=np.float32),
                        "dof_vel": np.zeros(cfg.obs_dof.num_dofs, dtype=np.float32),
                    }
                )
                ctrl_data = Box(
                    {
                        "Gr00tZmqCtrl": {
                            "takeover_enabled": True,
                            "fresh": True,
                            "locomotion_command": [0.3, -0.1, 0.2, 0.9],
                        }
                    }
                )

                observation, extras = policy.get_observation(env_data, ctrl_data)
                action = policy.get_action(observation)

                self.assertEqual(observation.shape, (360,))
                self.assertEqual(action.shape, (13,))
                self.assertTrue(np.isfinite(action).all())
                np.testing.assert_allclose(extras["locomotion_command"], [0.3, -0.1, 0.2, 0.78, 0.0])

    def test_g1_pipeline_uses_shared_takeover_gate(self):
        from robojudo.pipeline.four_mode_pipeline import ControlMode
        from robojudo.pipeline.g1_gr00t_locomanipulation_pipeline import G1Gr00tLocomanipulationPipeline

        pipeline = G1Gr00tLocomanipulationPipeline.__new__(G1Gr00tLocomanipulationPipeline)
        pipeline.mode = ControlMode.RL_DEFAULT
        pipeline._upper_body_enabled = True
        pipeline._upper_body_control_available = lambda: True
        stream = {"fresh": True}
        ctrl_data = {"Gr00tZmqCtrl": stream}

        pipeline._prepare_gr00t_stream(ctrl_data)

        self.assertTrue(stream["takeover_enabled"])
        self.assertIs(ctrl_data["UpperBodyZmqCtrl"], stream)


if __name__ == "__main__":
    unittest.main()
