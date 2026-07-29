import unittest
from types import SimpleNamespace

import numpy as np
from box import Box


class FakePolicyEnv:
    def __init__(self, dof_cfg):
        self.dof_cfg = dof_cfg
        self.default_pos = np.asarray(dof_cfg.default_pos, dtype=np.float32)
        self.dof_pos = self.default_pos.copy()
        self.joint_names = list(dof_cfg.joint_names)
        self.reset_count = 0
        self.elastic_band = SimpleNamespace(active=False, rest_length=0.7)

    def reset(self):
        self.reset_count += 1
        self.elastic_band.active = True
        self.elastic_band.rest_length = 0.0

    def update_dof_cfg(self, override_cfg=None):
        self.last_dof_cfg = override_cfg

    def set_control_joint_names(self, joint_names):
        self.control_joint_names = list(joint_names)


class TestX2LocomanipulationLocoMimic(unittest.TestCase):
    def test_pipeline_composes_locomanipulation_loco_mimic_and_four_mode_behavior(self):
        from robojudo.pipeline.locomanipulation_loco_mimic_pipeline import (
            LocomanipulationLocoMimicPipelineMixin,
        )
        from robojudo.pipeline.rl_loco_mimic_pipeline import RlLocoMimicPipeline
        from robojudo.pipeline.x2_locomanipulation_loco_mimic_pipeline import (
            X2LocomanipulationLocoMimicPipeline,
        )
        from robojudo.pipeline.x2_locomanipulation_pipeline import X2FourModePipelineMixin

        self.assertTrue(
            issubclass(X2LocomanipulationLocoMimicPipeline, LocomanipulationLocoMimicPipelineMixin)
        )
        self.assertTrue(issubclass(X2LocomanipulationLocoMimicPipeline, X2FourModePipelineMixin))
        self.assertTrue(issubclass(X2LocomanipulationLocoMimicPipeline, RlLocoMimicPipeline))

    def test_configs_register_expected_policies_and_controls(self):
        from robojudo.config.x2 import x2_locomimic, x2_locomimic_real
        from robojudo.config.x2.env.x2_env_cfg import X2_ARM_JOINT_NAMES
        from robojudo.config.x2.pipeline.x2_loco_mimic_pipeline_cfg import (
            X2LocomanipulationLocoMimicPipelineCfg,
        )

        sim_cfg = x2_locomimic()
        real_cfg = x2_locomimic_real()

        self.assertIsInstance(sim_cfg, X2LocomanipulationLocoMimicPipelineCfg)
        self.assertEqual(sim_cfg.pipeline_type, "X2LocomanipulationLocoMimicPipeline")
        self.assertEqual(sim_cfg.env.env_type, "MujocoEnv")
        self.assertEqual(sim_cfg.env.sim_dt, 0.005)
        self.assertEqual(sim_cfg.env.sim_decimation, 4)
        self.assertEqual(sim_cfg.loco_policy.policy_type, "X2LocomanipulationPolicy")
        self.assertEqual(
            [policy.policy_type for policy in sim_cfg.mimic_policies],
            ["X2DeployPolicy", "X2DeployPolicy"],
        )
        self.assertIsNot(sim_cfg.mimic_policies[0], sim_cfg.mimic_policies[1])
        for policy in sim_cfg.mimic_policies:
            self.assertIn("x2_rl_deploy", policy.policy_file)
            self.assertEqual(policy.max_timestep, 2820)
        self.assertEqual(len(real_cfg.mimic_policies), 2)
        self.assertTrue(all(policy.max_timestep == 2820 for policy in real_cfg.mimic_policies))
        self.assertEqual(sim_cfg.upper_dof_num, 16)
        self.assertEqual(len(sim_cfg.upper_dof_pos_default), 16)
        self.assertEqual(sim_cfg.upper_dof_override_indices, list(range(-16, -2)))
        self.assertTrue(sim_cfg.realign_on_policy_switch)
        self.assertEqual(sim_cfg.ctrl[-1].joint_names, X2_ARM_JOINT_NAMES)
        self.assertEqual(sim_cfg.ctrl[0].triggers["Back"], "[POLICY_LOCO]")
        self.assertEqual(sim_cfg.ctrl[0].triggers["Start"], "[POLICY_MIMIC]")
        self.assertEqual(sim_cfg.ctrl[0].triggers["RB"], "[POLICY_SWITCH],NEXT")
        self.assertEqual(sim_cfg.ctrl[0].triggers["LB"], "[POLICY_SWITCH],LAST")
        self.assertEqual(sim_cfg.ctrl[0].triggers["L"], "[UPPER_BODY_TOGGLE]")
        self.assertEqual(sim_cfg.ctrl[0].triggers["R"], "[POLICY_RECOVERY]")
        self.assertEqual(sim_cfg.ctrl[1].triggers_extra["r"], "[POLICY_RECOVERY]")
        self.assertEqual(sim_cfg.recovery_policy.policy_type, "AmpRecoveryPolicy")
        self.assertEqual(sim_cfg.recovery_policy.action_dof.num_dofs, 29)
        self.assertTrue(sim_cfg.do_safety_check)
        self.assertEqual(sim_cfg.ctrl[1].triggers_extra["["], "[POLICY_MIMIC]")
        self.assertEqual(real_cfg.env.env_type, "AgiBotCppEnv")
        self.assertTrue(real_cfg.do_safety_check)
        self.assertTrue(real_cfg.realign_on_policy_switch)
        self.assertEqual(real_cfg.ctrl[0].triggers["Back"], "[POLICY_LOCO]")
        self.assertEqual(real_cfg.ctrl[0].triggers["RB"], "[POLICY_SWITCH],NEXT")
        self.assertEqual(real_cfg.ctrl[0].triggers["LB"], "[POLICY_SWITCH],LAST")
        self.assertEqual(real_cfg.ctrl[0].triggers["R"], "[POLICY_RECOVERY]")
        self.assertNotIn("LB+RB+Y", real_cfg.ctrl[0].triggers)

    def test_policy_manager_switches_both_directions_and_can_cancel(self):
        from robojudo.config.x2.policy.x2_amp_recovery_policy_cfg import X2AmpRecoveryPolicyCfg
        from robojudo.config.x2.policy.x2_deploy_policy_cfg import X2DeployPolicyCfg
        from robojudo.config.x2.policy.x2_locomanipulation_policy_cfg import (
            X2LocomanipulationEnvDoF,
            X2LocomanipulationPolicyCfg,
        )
        from robojudo.pipeline.rl_loco_mimic_pipeline import PolicyInterpManager

        env = FakePolicyEnv(X2LocomanipulationEnvDoF())
        manager = PolicyInterpManager(
            X2LocomanipulationPolicyCfg(),
            [X2DeployPolicyCfg(), X2DeployPolicyCfg()],
            env,
            cfg_policy_recovery=X2AmpRecoveryPolicyCfg(),
            loco_dof_pos=env.default_pos,
        )
        manager.DURATIONS_LOCO_MIMIC = [0, 2, 1]
        manager.DURATIONS_MIMIC_LOCO = [1, 2, 1]
        env_data = Box(
            {
                "base_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                "base_ang_vel": np.zeros(3, dtype=np.float32),
                "dof_pos": env.default_pos.copy(),
                "dof_vel": np.zeros(31, dtype=np.float32),
            }
        )
        ctrl_data = Box({})

        self.assertEqual(manager.policy_mimic_idx, 0)
        self.assertEqual(manager.policy_mimic_ids, [1, 2])
        self.assertEqual(manager.policy_recovery_id, 3)
        manager.toggle_mimic_policy(1)
        self.assertEqual(manager.policy_mimic_idx, 1)
        manager.toggle_mimic_policy(1)
        self.assertEqual(manager.policy_mimic_idx, 0)
        manager.toggle_mimic_policy(-1)
        self.assertEqual(manager.policy_mimic_idx, 1)

        manager.switch_to_mimic()
        for _ in range(5):
            manager.step(env_data, ctrl_data)
        self.assertEqual(manager.current_policy_id, manager.policy_mimic_ids[1])
        self.assertEqual(manager.interp_state, PolicyInterpManager.InterpState.IDLE)
        manager.toggle_mimic_policy(1)
        self.assertEqual(manager.policy_mimic_idx, 1)
        self.assertEqual(env.reset_count, 0)
        self.assertFalse(env.elastic_band.active)
        self.assertEqual(env.elastic_band.rest_length, 0.7)

        manager.switch_to_loco()
        for _ in range(5):
            manager.step(env_data, ctrl_data)
        self.assertEqual(manager.current_policy_id, manager.policy_loco_id)
        self.assertEqual(manager.interp_state, PolicyInterpManager.InterpState.IDLE)
        self.assertEqual(env.reset_count, 0)
        self.assertFalse(env.elastic_band.active)
        self.assertEqual(env.elastic_band.rest_length, 0.7)

        manager.toggle_mimic_policy(-1)
        self.assertEqual(manager.policy_mimic_idx, 0)
        manager.switch_to_mimic()
        manager.step(env_data, ctrl_data)
        manager.toggle_mimic_policy(1)
        self.assertEqual(manager.policy_mimic_idx, 0)
        manager.reset_to_loco(refresh_env=False)
        self.assertEqual(manager.current_policy_id, manager.policy_loco_id)
        self.assertEqual(manager.interp_state, PolicyInterpManager.InterpState.IDLE)
        self.assertFalse(manager.timer.has_pending())
        np.testing.assert_array_equal(manager.override_dof_pos, manager.loco_dof_pos)

        self.assertTrue(manager.activate_recovery())
        self.assertEqual(manager.current_policy_id, manager.policy_recovery_id)
        self.assertIs(env.last_dof_cfg, manager.policy.cfg_action_dof)
        self.assertEqual(env.reset_count, 0)
        completed = []
        self.assertTrue(manager.switch_to_loco(callback_end=lambda: completed.append(True)))
        for _ in range(5):
            manager.step(env_data, ctrl_data)
        self.assertEqual(manager.current_policy_id, manager.policy_loco_id)
        self.assertEqual(completed, [True])

    def test_zmq_is_available_only_for_idle_loco_and_disables_on_mimic(self):
        from robojudo.pipeline.rl_loco_mimic_pipeline import PolicyInterpManager
        from robojudo.pipeline.x2_locomanipulation_loco_mimic_pipeline import (
            X2LocomanipulationLocoMimicPipeline,
        )

        pipeline = X2LocomanipulationLocoMimicPipeline.__new__(X2LocomanipulationLocoMimicPipeline)
        manager = SimpleNamespace(
            current_policy_id=0,
            policy_loco_id=0,
            interp_state=PolicyInterpManager.InterpState.IDLE,
            switch_to_mimic=lambda: (setattr(manager, "mimic_requested", True), True)[1],
        )
        pipeline.policy_manager = manager
        pipeline.policy_locomotion_mimic_flag = 0
        pipeline._upper_body_cfg = object()
        pipeline._upper_body_enabled = True
        pipeline._upper_body_stream_was_fresh = True

        self.assertTrue(pipeline._upper_body_control_available())
        manager.interp_state = PolicyInterpManager.InterpState.IN_PROGRESS
        self.assertFalse(pipeline._upper_body_control_available())
        manager.interp_state = PolicyInterpManager.InterpState.IDLE

        pipeline._process_policy_commands(["[POLICY_MIMIC]"], {})
        self.assertFalse(pipeline._upper_body_enabled)
        self.assertTrue(manager.mimic_requested)
        self.assertEqual(pipeline.policy_locomotion_mimic_flag, 1)

    def test_zmq_filter_syncs_to_loco_target_after_mimic(self):
        from robojudo.pipeline.rl_loco_mimic_pipeline import PolicyInterpManager
        from robojudo.pipeline.x2_locomanipulation_loco_mimic_pipeline import (
            X2LocomanipulationLocoMimicPipeline,
        )

        pipeline = X2LocomanipulationLocoMimicPipeline.__new__(X2LocomanipulationLocoMimicPipeline)
        manager = SimpleNamespace(
            current_policy_id=1,
            policy_loco_id=0,
            interp_state=PolicyInterpManager.InterpState.IN_PROGRESS,
        )
        pipeline.policy_manager = manager
        pipeline._upper_body_cfg = SimpleNamespace(ema_alpha=0.9, joint_names=["left_arm", "right_arm"])
        pipeline._upper_body_indices = np.asarray([1, 3], dtype=np.int32)
        pipeline._upper_body_default = np.asarray([0.2, -0.2], dtype=np.float32)
        pipeline._upper_body_filtered = np.asarray([1.5, -1.5], dtype=np.float32)
        pipeline._upper_body_enabled = False
        pipeline._upper_body_stream_was_fresh = True
        pipeline._upper_body_override_was_available = True
        pipeline.env = SimpleNamespace(position_limits=np.asarray([[-2.0, 2.0]] * 4, dtype=np.float32))
        loco_target = np.asarray([0.0, 0.2, 0.0, -0.2], dtype=np.float32)
        pipeline._get_policy_step = lambda env_data, ctrl_data: (loco_target.copy(), {})
        ctrl_data = Box(
            {
                "UpperBodyZmqCtrl": {
                    "fresh": True,
                    "joint_positions": {"left_arm": 1.0, "right_arm": -1.0},
                }
            }
        )

        mimic_target, _ = pipeline._step_rl_policy(Box({}), ctrl_data, dry_run=True)
        np.testing.assert_array_equal(mimic_target, loco_target)
        np.testing.assert_array_equal(pipeline._upper_body_filtered, [1.5, -1.5])
        self.assertFalse(pipeline._upper_body_override_was_available)

        manager.current_policy_id = manager.policy_loco_id
        manager.interp_state = PolicyInterpManager.InterpState.IDLE
        resumed_target, _ = pipeline._step_rl_policy(Box({}), ctrl_data, dry_run=True)

        np.testing.assert_array_equal(resumed_target, loco_target)
        np.testing.assert_array_equal(pipeline._upper_body_filtered, pipeline._upper_body_default)
        self.assertFalse(pipeline._upper_body_enabled)
        self.assertFalse(pipeline._upper_body_stream_was_fresh)
        self.assertTrue(pipeline._upper_body_override_was_available)

    def test_disabling_zmq_during_loco_keeps_smooth_return_to_default(self):
        from robojudo.pipeline.rl_loco_mimic_pipeline import PolicyInterpManager
        from robojudo.pipeline.x2_locomanipulation_loco_mimic_pipeline import (
            X2LocomanipulationLocoMimicPipeline,
        )

        pipeline = X2LocomanipulationLocoMimicPipeline.__new__(X2LocomanipulationLocoMimicPipeline)
        pipeline.policy_manager = SimpleNamespace(
            current_policy_id=0,
            policy_loco_id=0,
            interp_state=PolicyInterpManager.InterpState.IDLE,
        )
        pipeline._upper_body_cfg = SimpleNamespace(ema_alpha=0.5, joint_names=["left_arm"])
        pipeline._upper_body_indices = np.asarray([1], dtype=np.int32)
        pipeline._upper_body_default = np.asarray([0.0], dtype=np.float32)
        pipeline._upper_body_filtered = np.asarray([1.0], dtype=np.float32)
        pipeline._upper_body_enabled = False
        pipeline._upper_body_stream_was_fresh = False
        pipeline._upper_body_override_was_available = True
        pipeline.env = SimpleNamespace(position_limits=np.asarray([[-2.0, 2.0]] * 2, dtype=np.float32))
        loco_target = np.asarray([0.0, 0.0], dtype=np.float32)
        pipeline._get_policy_step = lambda env_data, ctrl_data: (loco_target.copy(), {})

        pd_target, _ = pipeline._step_rl_policy(Box({}), Box({}), dry_run=True)

        self.assertEqual(pd_target[1], 0.5)
        self.assertEqual(pipeline._upper_body_filtered[0], 0.5)
        self.assertTrue(pipeline._upper_body_override_was_available)

    def test_leaving_rl_resets_loco_and_cancels_policy_state(self):
        from robojudo.pipeline.x2_locomanipulation_loco_mimic_pipeline import (
            X2LocomanipulationLocoMimicPipeline,
        )

        pipeline = X2LocomanipulationLocoMimicPipeline.__new__(X2LocomanipulationLocoMimicPipeline)
        progress = SimpleNamespace(close_calls=0)
        manager = SimpleNamespace(
            reset_to_loco_calls=[],
            policy=SimpleNamespace(
                close_progress=lambda: setattr(progress, "close_calls", progress.close_calls + 1)
            ),
        )
        manager.reset_to_loco = lambda refresh_env: manager.reset_to_loco_calls.append(refresh_env)
        pipeline.policy_manager = manager
        pipeline.policy_locomotion_mimic_flag = 1
        pipeline._upper_body_cfg = object()
        pipeline._upper_body_enabled = True
        pipeline._upper_body_stream_was_fresh = True

        pipeline._on_leave_rl()

        self.assertFalse(pipeline._upper_body_enabled)
        self.assertEqual(progress.close_calls, 1)
        self.assertEqual(manager.reset_to_loco_calls, [True])
        self.assertEqual(pipeline.policy_locomotion_mimic_flag, 0)
        self.assertFalse(pipeline._upper_body_override_was_available)

    def test_policy_time_and_switch_timer_advance_only_in_rl(self):
        from robojudo.pipeline.x2_locomanipulation_loco_mimic_pipeline import (
            X2LocomanipulationLocoMimicPipeline,
        )

        pipeline = X2LocomanipulationLocoMimicPipeline.__new__(X2LocomanipulationLocoMimicPipeline)
        calls = SimpleNamespace(policy_post=0, manager_step=0, ctrl_post=0)
        pipeline.timestep = 0
        pipeline.cfg = SimpleNamespace(debug=SimpleNamespace(log_obs=False))
        pipeline.visualizer = None
        pipeline.policy_manager = SimpleNamespace(
            policy=SimpleNamespace(
                post_step_callback=lambda commands: setattr(calls, "policy_post", calls.policy_post + 1)
            ),
            warmup_policy_indices=set(),
            step=lambda env_data, ctrl_data: setattr(calls, "manager_step", calls.manager_step + 1),
        )
        pipeline.ctrl_manager = SimpleNamespace(
            post_step_callback=lambda ctrl_data: setattr(calls, "ctrl_post", calls.ctrl_post + 1)
        )
        env_data = Box({})
        ctrl_data = Box({"COMMANDS": []})

        pipeline._post_mode_step(env_data, ctrl_data, {}, np.zeros(31), rl_active=False)
        self.assertEqual((calls.policy_post, calls.manager_step, calls.ctrl_post), (0, 0, 1))

        pipeline.policy_manager.warmup_policy_indices = {0, 1}
        pipeline._post_mode_step(env_data, ctrl_data, {}, np.zeros(31), rl_active=False)
        self.assertEqual((calls.policy_post, calls.manager_step, calls.ctrl_post), (0, 1, 2))

        pipeline.policy_manager.warmup_policy_indices.clear()
        pipeline._post_mode_step(env_data, ctrl_data, {}, np.zeros(31), rl_active=True)
        self.assertEqual((calls.policy_post, calls.manager_step, calls.ctrl_post), (1, 2, 3))

    def test_motion_done_callback_starts_return_to_loco(self):
        from robojudo.pipeline.x2_locomanipulation_loco_mimic_pipeline import (
            X2LocomanipulationLocoMimicPipeline,
        )

        pipeline = X2LocomanipulationLocoMimicPipeline.__new__(X2LocomanipulationLocoMimicPipeline)
        manager = SimpleNamespace(return_requested=False)
        manager.switch_to_loco = lambda: (setattr(manager, "return_requested", True), True)[1]
        pipeline.policy_manager = manager
        pipeline.policy_locomotion_mimic_flag = 1

        commands = []
        pipeline._process_policy_commands(commands, {"CALLBACK": ["[MOTION_DONE]"]})

        self.assertEqual(commands, ["[POLICY_LOCO]"])
        self.assertTrue(manager.return_requested)
        self.assertEqual(pipeline.policy_locomotion_mimic_flag, 0)


if __name__ == "__main__":
    unittest.main()
