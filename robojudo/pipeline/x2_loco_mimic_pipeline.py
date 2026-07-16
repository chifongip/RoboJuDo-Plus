import logging

from robojudo.pipeline import pipeline_registry
from robojudo.pipeline.rl_loco_mimic_pipeline import PolicyInterpManager, RlLocoMimicPipeline
from robojudo.pipeline.x2_deploy_pipeline import X2ModePipelineMixin

logger = logging.getLogger(__name__)


@pipeline_registry.register
class X2LocoMimicPipeline(X2ModePipelineMixin, RlLocoMimicPipeline):
    """X2 loco-mimic switching guarded by the X2 deployment mode state machine."""

    def __init__(self, cfg):
        self._upper_body_override_was_available = False
        super().__init__(cfg)

    def reset(self):
        self._upper_body_override_was_available = False
        super().reset()

    @staticmethod
    def _initial_policy_cfg(cfg):
        return cfg.loco_policy

    def _upper_body_action_joint_names(self) -> list[str]:
        return self.cfg.loco_policy.action_dof.joint_names

    def _upper_body_control_available(self) -> bool:
        return (
            self.policy_manager.current_policy_id == self.policy_manager.policy_loco_id
            and self.policy_manager.interp_state == PolicyInterpManager.InterpState.IDLE
        )

    def _reset_loco_policy_state(self):
        self._upper_body_override_was_available = False
        self.policy_manager.reset_to_loco(refresh_env=True)
        self.policy_locomotion_mimic_flag = 0

    def _on_enter_rl(self):
        self._reset_loco_policy_state()

    def _on_leave_rl(self):
        super()._on_leave_rl()
        self._reset_loco_policy_state()

    def _reset_policy_state(self):
        self._reset_loco_policy_state()

    def _step_rl_policy(self, env_data, ctrl_data, dry_run: bool):
        commands = ctrl_data.get("COMMANDS", [])
        if "[POLICY_MIMIC]" in commands:
            self._set_upper_body_enabled(False)

        pd_target, extras = self._get_policy_step(env_data, ctrl_data)
        upper_body_override_available = self._upper_body_control_available()
        if upper_body_override_available:
            if not self._upper_body_override_was_available and self._upper_body_indices.size:
                self._upper_body_filtered = pd_target[self._upper_body_indices].copy()
                self._upper_body_stream_was_fresh = False
            pd_target = self._apply_upper_body_override(pd_target, ctrl_data)
        self._upper_body_override_was_available = upper_body_override_available
        if not dry_run:
            self.env.step(pd_target, extras.get("hand_pose"))
        return pd_target, extras

    def _process_policy_commands(self, commands: list[str], extras):
        for callback in extras.get("CALLBACK", []):
            if callback == "[MOTION_DONE]" and self.policy_locomotion_mimic_flag == 1:
                commands.append("[POLICY_LOCO]")
                logger.info("Mimic motion done, switch to locomotion policy.")

        for command in commands:
            match command:
                case cmd if cmd.startswith("[POLICY_SWITCH]"):
                    switch_target = cmd.split(",")[1]
                    if switch_target == "NEXT":
                        self.policy_manager.toggle_mimic_policy(1)
                    elif switch_target == "LAST":
                        self.policy_manager.toggle_mimic_policy(-1)
                case "[POLICY_LOCO]":
                    if self.policy_manager.switch_to_loco():
                        self.policy_locomotion_mimic_flag = 0
                case "[POLICY_MIMIC]":
                    self._set_upper_body_enabled(False)
                    if self.policy_manager.switch_to_mimic():
                        self.policy_locomotion_mimic_flag = 1

    def _post_mode_step(self, env_data, ctrl_data, extras, pd_target, rl_active: bool):
        commands = ctrl_data.get("COMMANDS", [])
        policy_commands = [command for command in commands if command.startswith("[POLICY_")]
        if rl_active:
            self._process_policy_commands(commands, extras)
        elif policy_commands:
            logger.warning("Ignored policy switch outside RL_DEFAULT")

        self.timestep += 1
        self.ctrl_manager.post_step_callback(ctrl_data)
        if rl_active:
            self.policy.post_step_callback(commands)
            self.policy_manager.step(env_data, ctrl_data)
            if self.visualizer is not None:
                self.policy.debug_viz(self.visualizer, env_data, ctrl_data, extras)
        if self.cfg.debug.log_obs:
            self.debug_logger.log(
                env_data=env_data,
                ctrl_data=ctrl_data,
                extras=extras,
                pd_target=pd_target,
                timestep=self.timestep,
            )
