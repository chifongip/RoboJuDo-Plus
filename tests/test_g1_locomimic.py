import unittest
from types import SimpleNamespace

import numpy as np
from box import Box


class TestG1LocoMimic(unittest.TestCase):
    def test_configs_register_models_controls_and_safety(self):
        from robojudo.config.g1.g1_loco_mimic_cfg import (
            g1_23_locomanipulation_default_locomimic,
            g1_23_locomanipulation_default_locomimic_real,
            g1_23_locomanipulation_locomimic,
            g1_23_locomanipulation_locomimic_real,
            g1_29_locomanipulation_locomimic,
            g1_29_locomanipulation_locomimic_real,
        )

        cases = [
            (g1_23_locomanipulation_default_locomimic(), 23, 10, True, "default", "policy_23dof_default"),
            (g1_23_locomanipulation_locomimic(), 23, 10, True, "stiff", "policy_23dof_stiff"),
            (g1_29_locomanipulation_locomimic(), 29, 14, False, "stiff", "policy_29dof_stiff"),
        ]
        for cfg, num_dofs, upper_dofs, padded, pd_gain_preset, policy_name in cases:
            with self.subTest(config=type(cfg).__name__):
                self.assertEqual(cfg.pipeline_type, "G1LocoMimicPipeline")
                self.assertEqual(cfg.env.dof.num_dofs, num_dofs)
                self.assertEqual(cfg.loco_policy.pd_gain_preset, pd_gain_preset)
                self.assertEqual(cfg.loco_policy.policy_name, policy_name)
                self.assertEqual(cfg.joint_default_dof.stiffness, cfg.env.dof.stiffness)
                self.assertEqual(cfg.joint_default_dof.damping, cfg.env.dof.damping)
                self.assertEqual(cfg.upper_dof_num, upper_dofs)
                self.assertEqual(cfg.upper_dof_override_indices, list(range(-upper_dofs, 0)))
                self.assertEqual(cfg.ctrl[-1].joint_names, cfg.env.dof.joint_names[-upper_dofs:])
                self.assertEqual(cfg.ctrl[0].triggers["Back"], "[POLICY_LOCO]")
                self.assertEqual(cfg.ctrl[0].triggers["Start"], "[POLICY_MIMIC]")
                self.assertEqual(cfg.ctrl[0].triggers["RB"], "[POLICY_SWITCH],NEXT")
                self.assertEqual(cfg.ctrl[0].triggers["LB"], "[POLICY_SWITCH],LAST")
                self.assertEqual(cfg.ctrl[0].triggers["L"], "[UPPER_BODY_TOGGLE]")
                self.assertEqual(cfg.ctrl[1].triggers_extra["9"], "[ELASTIC_BAND_TOGGLE]")
                self.assertEqual(cfg.env.elastic_band.body_name, "torso_link")
                self.assertTrue(cfg.realign_on_policy_switch)
                self.assertEqual(
                    [
                        (policy.policy_name, policy.without_state_estimator, policy.max_timestep)
                        for policy in cfg.mimic_policies
                    ],
                    [("Jump_wose", True, 140), ("Dance_wose", True, 6574)],
                )
                self.assertTrue(all(policy.pad_missing_dofs is padded for policy in cfg.mimic_policies))

        for cfg, num_dofs in (
            (g1_23_locomanipulation_default_locomimic_real(), 23),
            (g1_23_locomanipulation_locomimic_real(), 23),
            (g1_29_locomanipulation_locomimic_real(), 29),
        ):
            with self.subTest(config=type(cfg).__name__):
                self.assertEqual(cfg.env.env_type, "UnitreeCppEnv")
                self.assertEqual(cfg.env.dof.num_dofs, num_dofs)
                self.assertEqual(cfg.env.unitree.command_timeout, 0.1)
                self.assertEqual(cfg.env.unitree.state_timeout, 0.1)
                self.assertEqual(cfg.ctrl[0].triggers["Select"], "[POLICY_LOCO]")
                self.assertEqual(cfg.ctrl[0].triggers["Start"], "[POLICY_MIMIC]")
                self.assertEqual(cfg.ctrl[0].triggers["R1"], "[POLICY_SWITCH],NEXT")
                self.assertEqual(cfg.ctrl[0].triggers["L1"], "[POLICY_SWITCH],LAST")
                self.assertEqual(cfg.ctrl[0].triggers["L2"], "[UPPER_BODY_TOGGLE]")
                self.assertEqual(cfg.ctrl[0].triggers["L1+R1+A"], "[SHUTDOWN]")
                self.assertTrue(cfg.do_safety_check)
                self.assertTrue(cfg.realign_on_policy_switch)

    def test_29_dof_beyondmimic_adapter_pads_native_23_dof_observations(self):
        from robojudo.config.g1.policy.g1_beyondmimic_policy_cfg import G1BeyondMimicPolicyCfg
        from robojudo.config.g1.policy.g1_locomanipulation_policy_cfg import G1Locomanipulation23ObsDoF
        from robojudo.pipeline.rl_pipeline import PolicyWrapper

        env_dof = G1Locomanipulation23ObsDoF.from_preset("stiff")
        cfg = G1BeyondMimicPolicyCfg(
            policy_name="Jump_wose",
            without_state_estimator=True,
            max_timestep=140,
            pad_missing_dofs=True,
        )
        wrapper = PolicyWrapper(cfg, env_dof, "cpu")
        self.addCleanup(wrapper.close_progress)

        env_pos = np.linspace(-0.5, 0.5, env_dof.num_dofs, dtype=np.float32)
        env_vel = np.linspace(0.1, 0.3, env_dof.num_dofs, dtype=np.float32)
        policy_pos, policy_vel = wrapper._adapt_observation_dofs(env_pos, env_vel)
        for index, name in enumerate(wrapper.cfg_obs_dof.joint_names):
            if name in env_dof.joint_names:
                env_index = env_dof.joint_names.index(name)
                self.assertEqual(policy_pos[index], env_pos[env_index])
                self.assertEqual(policy_vel[index], env_vel[env_index])
            else:
                self.assertEqual(policy_pos[index], wrapper.default_dof_pos[index])
                self.assertEqual(policy_vel[index], 0.0)

        env_data = Box(
            {
                "base_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                "base_ang_vel": np.zeros(3, dtype=np.float32),
                "base_lin_vel": np.zeros(3, dtype=np.float32),
                "torso_pos": np.asarray([0.0, 0.0, 0.8], dtype=np.float32),
                "torso_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                "dof_pos": np.asarray(env_dof.default_pos, dtype=np.float32),
                "dof_vel": np.zeros(env_dof.num_dofs, dtype=np.float32),
            }
        )
        obs, _ = wrapper.get_observation(env_data, Box({}))
        target = wrapper.get_pd_target(obs)
        self.assertEqual(target.shape, (23,))
        self.assertTrue(np.isfinite(target).all())
        self.assertEqual(wrapper.get_init_dof_pos().shape, (23,))
        wrapper.policy.timestep = 139
        wrapper.post_step_callback([])
        self.assertTrue(wrapper.policy.flag_motion_done)
        self.assertIsNone(wrapper.policy.pbar)

        dance = PolicyWrapper(
            G1BeyondMimicPolicyCfg(
                policy_name="Dance_wose",
                without_state_estimator=True,
                max_timestep=6574,
                pad_missing_dofs=True,
            ),
            env_dof,
            "cpu",
        )
        self.addCleanup(dance.close_progress)
        dance_obs, _ = dance.get_observation(env_data, Box({}))
        dance_target = dance.get_pd_target(dance_obs)
        self.assertEqual(dance_target.shape, (23,))
        self.assertTrue(np.isfinite(dance_obs).all())
        self.assertTrue(np.isfinite(dance_target).all())

    def test_missing_dof_padding_is_strict_by_default(self):
        from robojudo.config.g1.policy.g1_beyondmimic_policy_cfg import G1BeyondMimicPolicyCfg
        from robojudo.config.g1.policy.g1_locomanipulation_policy_cfg import G1Locomanipulation23ObsDoF
        from robojudo.pipeline.rl_pipeline import PolicyWrapper

        cfg = G1BeyondMimicPolicyCfg(policy_name="Jump_wose", without_state_estimator=True)
        with self.assertRaisesRegex(ValueError, "joints missing from environment"):
            PolicyWrapper(cfg, G1Locomanipulation23ObsDoF(), "cpu")

    def test_g1_uses_shared_transition_and_auto_return_behavior(self):
        from robojudo.pipeline.g1_loco_mimic_pipeline import G1LocoMimicPipeline
        from robojudo.pipeline.locomanipulation_loco_mimic_pipeline import (
            LocomanipulationLocoMimicPipelineMixin,
        )
        from robojudo.pipeline.rl_loco_mimic_pipeline import PolicyInterpManager

        self.assertTrue(issubclass(G1LocoMimicPipeline, LocomanipulationLocoMimicPipelineMixin))
        pipeline = G1LocoMimicPipeline.__new__(G1LocoMimicPipeline)
        manager = SimpleNamespace(
            current_policy_id=1,
            policy_loco_id=0,
            interp_state=PolicyInterpManager.InterpState.IDLE,
            return_requested=False,
        )
        manager.switch_to_loco = lambda: (setattr(manager, "return_requested", True), True)[1]
        pipeline.policy_manager = manager
        pipeline.policy_locomotion_mimic_flag = 1

        commands = []
        pipeline._process_policy_commands(commands, {"CALLBACK": ["[MOTION_DONE]"]})

        self.assertEqual(commands, ["[POLICY_LOCO]"])
        self.assertTrue(manager.return_requested)
        self.assertEqual(pipeline.policy_locomotion_mimic_flag, 0)

    def test_transition_frame_visualizes_with_the_policy_that_created_extras(self):
        from robojudo.pipeline.g1_loco_mimic_pipeline import G1LocoMimicPipeline

        calls = []
        old_policy = SimpleNamespace(
            post_step_callback=lambda commands: None,
            debug_viz=lambda visualizer, env_data, ctrl_data, extras: calls.append(extras["source"]),
        )
        new_policy = SimpleNamespace(
            debug_viz=lambda *args: self.fail("new policy received extras from the previous policy")
        )

        class SwitchingManager:
            policy = old_policy

            def step(self, env_data, ctrl_data):
                self.policy = new_policy

        pipeline = G1LocoMimicPipeline.__new__(G1LocoMimicPipeline)
        pipeline.timestep = 0
        pipeline.cfg = SimpleNamespace(debug=SimpleNamespace(log_obs=False))
        pipeline.visualizer = object()
        pipeline.policy_manager = SwitchingManager()
        pipeline.ctrl_manager = SimpleNamespace(post_step_callback=lambda ctrl_data: None)

        pipeline._post_mode_step(
            Box({}),
            Box({"COMMANDS": []}),
            {"source": "old-policy"},
            np.zeros(23),
            rl_active=True,
        )

        self.assertEqual(calls, ["old-policy"])
        self.assertIs(pipeline.policy, new_policy)

    def test_beyondmimic_debug_viz_skips_non_matching_extras(self):
        from robojudo.policy.beyondmimic_policy import BeyondMimicPolicy

        policy = BeyondMimicPolicy.__new__(BeyondMimicPolicy)
        visualizer = SimpleNamespace(draw_arrow=lambda *args, **kwargs: self.fail("unexpected draw"))

        policy.debug_viz(visualizer, Box({}), Box({}), {})

    def test_beyondmimic_rejects_state_estimator_mode_mismatch_and_wrong_observation_size(self):
        from robojudo.config.g1.policy.g1_beyondmimic_policy_cfg import G1BeyondMimicPolicyCfg
        from robojudo.policy.beyondmimic_policy import BeyondMimicPolicy

        with self.assertRaisesRegex(ValueError, "exported without state-estimator"):
            BeyondMimicPolicy(
                G1BeyondMimicPolicyCfg(policy_name="Jump_wose", without_state_estimator=False),
                "cpu",
            )

        policy = BeyondMimicPolicy(
            G1BeyondMimicPolicyCfg(policy_name="Dance_wose", without_state_estimator=True),
            "cpu",
        )
        self.addCleanup(policy.close_progress)
        with self.assertRaisesRegex(ValueError, r"expects observation shape \(154,\)"):
            policy.get_action(np.zeros(160, dtype=np.float32))

    def test_g1_policy_activation_realigns_without_resetting_elastic_band(self):
        from robojudo.pipeline.rl_loco_mimic_pipeline import PolicyInterpManager

        calls = []
        elastic_band = SimpleNamespace(active=False, rest_length=0.6)
        env = SimpleNamespace(elastic_band=elastic_band)
        env.reset_alignment = lambda: calls.append("realign")

        manager = PolicyInterpManager.__new__(PolicyInterpManager)
        manager.env = env
        manager.realign_on_policy_switch = True
        manager.set_policy = lambda policy_id, reset_env: calls.append(("policy", policy_id, reset_env))
        manager._activate_policy(2)

        self.assertEqual(calls, ["realign", ("policy", 2, False)])
        self.assertFalse(elastic_band.active)
        self.assertEqual(elastic_band.rest_length, 0.6)


if __name__ == "__main__":
    unittest.main()
