"""Headless MuJoCo regression with real lower-body ONNX inference."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from robojudo.config.g1.g1_cfg import g1_23_locomanipulation_stiff
from robojudo.config.g1.policy.g1_gr00t_locomanipulation_policy_cfg import G1Gr00tLocomanipulation23PolicyCfg
from robojudo.config.x2.x2_cfg import x2_gr00t_locomanipulation
from robojudo.environment.mujoco_env import MujocoEnv
from robojudo.pipeline.rl_pipeline import PolicyWrapper


class TestLocomanipulationHeightSimulation(unittest.TestCase):
    def test_height_rate_through_resets_and_source_changes(self):
        for cfg_type in (g1_23_locomanipulation_stiff, x2_gr00t_locomanipulation):
            with self.subTest(config=cfg_type.__name__):
                cfg = cfg_type()
                if cfg_type is g1_23_locomanipulation_stiff:
                    # Exercise GR00T policy inference in the supported G1 simulator
                    # without adding a deployment preset that requires physical hands.
                    cfg.policy = G1Gr00tLocomanipulation23PolicyCfg(**cfg.policy.model_dump(exclude={"policy_type"}))
                cfg.policy.command_status_interval_s = 0.0
                # Only replace the window; physics, observations, inference, and PD control run normally.
                viewer = SimpleNamespace(cam=SimpleNamespace(), is_alive=False, close=lambda: None)
                with patch("robojudo.environment.mujoco_env.mujoco_viewer.MujocoViewer", return_value=viewer):
                    env = MujocoEnv(cfg.env, "cpu")
                policy = PolicyWrapper(cfg.policy, cfg.env.dof, "cpu")
                try:
                    env.data.qpos[-env.num_dofs :] = env.default_pos
                    env.update()
                    # Release the support band for the closed-loop motion check.
                    if env.elastic_band is not None:
                        env.elastic_band.active = False
                    default = policy.base_height_default
                    previous = float(policy.cmd[3])
                    dt = env.control_dt
                    with (
                        patch("robojudo.policy.locomanipulation_policy.time.monotonic") as clock,
                        patch("builtins.print"),
                    ):
                        for step in range(400):
                            clock.return_value = 10.0 + step * dt
                            if step < 100:
                                data = {
                                    "LocomanipulationPostureZmqCtrl": {
                                        "height": default - 0.10,
                                        "waist_yaw": 0.0,
                                        "fresh": True,
                                    }
                                }
                            elif step < 150:
                                events = [{"type": "keyboard", "name": "r", "pressed": True}] if step == 100 else []
                                data = {"KeyboardCtrl": {"keyboard_event": events}}
                            elif step < 250:
                                data = {
                                    "Gr00tZmqCtrl": {
                                        "takeover_enabled": True,
                                        "fresh": not 200 <= step < 225,
                                        "locomotion_command": [
                                            0.0,
                                            0.0,
                                            0.0,
                                            default - 0.15 if step < 225 else default,
                                        ],
                                    }
                                }
                            else:
                                if step == 250:
                                    policy.reset()
                                    self.assertEqual(float(policy.cmd[3]), previous)
                                data = {}
                            observation, extras = policy.get_observation(env.get_data(), data)
                            height = float(extras["locomotion_command"][3])
                            self.assertLessEqual(
                                abs(height - previous), cfg.policy.max_height_command_rate_m_s * dt + 1e-7
                            )
                            target = policy._target_height
                            self.assertGreaterEqual(height, min(previous, target) - 1e-7)
                            self.assertLessEqual(height, max(previous, target) + 1e-7)
                            if step in (0, 250) or 200 <= step < 225:
                                self.assertEqual(height, previous)
                            pd_target = policy.get_pd_target(observation)
                            self.assertTrue(np.isfinite(observation).all())
                            self.assertTrue(np.isfinite(pd_target).all())
                            env.step(pd_target)
                            self.assertTrue(np.isfinite(env.data.qpos).all())
                            self.assertTrue(np.isfinite(env.data.qvel).all())
                            policy.post_step_callback()
                            previous = height
                        self.assertAlmostEqual(previous, default, places=6)
                finally:
                    env.shutdown()


if __name__ == "__main__":
    unittest.main()
