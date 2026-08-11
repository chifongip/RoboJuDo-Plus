import logging

from robojudo.pipeline.four_mode_pipeline import (
    FALL_TILT_THRESHOLD_RAD,
    ControlMode,
)
from robojudo.pipeline.rl_loco_mimic_pipeline import PolicyInterpManager

logger = logging.getLogger(__name__)


class LocomanipulationLocoMimicPipelineMixin:
    """Four-mode loco-mimic behavior shared by X2 and G1 Locomanipulation."""

    def __init__(self, cfg):
        self._upper_body_override_was_available = False
        self._recovery_return_in_progress = False
        super().__init__(cfg)

    def reset(self):
        self._upper_body_override_was_available = False
        self._recovery_return_in_progress = False
        if hasattr(self, "policy_manager") and hasattr(self.policy_manager.policy, "close_progress"):
            self.policy_manager.policy.close_progress()
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

    def _reset_loco_policy_state(self, close_active_progress: bool = True):
        self._upper_body_override_was_available = False
        self._recovery_return_in_progress = False
        if close_active_progress and hasattr(self.policy_manager.policy, "close_progress"):
            self.policy_manager.policy.close_progress()
        self.policy_manager.reset_to_loco(refresh_env=True)
        self.policy_locomotion_mimic_flag = 0

    def _on_enter_rl(self):
        if self._recovery_return_in_progress:
            self._recovery_return_in_progress = False
            self.policy_locomotion_mimic_flag = 0
            return
        self._reset_loco_policy_state()

    def _on_leave_rl(self):
        super()._on_leave_rl()
        self._reset_loco_policy_state(close_active_progress=False)

    def _reset_policy_state(self):
        self._reset_loco_policy_state()

    def _process_commands(self, commands: list[str]):
        if "[POLICY_RECOVERY]" in commands:
            self._try_enter_recovery()
        if self.mode == ControlMode.RECOVERY_DEFAULT:
            commands = [
                command
                for command in commands
                if command
                not in (
                    "[JOINT_DEFAULT]",
                    "[RL_DEFAULT]",
                    "[UPPER_BODY_TOGGLE]",
                )
            ]
        super()._process_commands(commands)

    def _try_enter_recovery(self) -> bool:
        if self._shutdown_requested:
            logger.warning("Ignored recovery request during shutdown")
            return False
        if self.mode != ControlMode.JOINT_DEFAULT:
            logger.warning("Ignored recovery request outside JOINT_DEFAULT")
            return False
        if not self._joint_default_complete:
            logger.warning(
                "Ignored recovery request: JOINT_DEFAULT interpolation has not completed"
            )
            return False
        try:
            tilt = self._robot_tilt()
        except FloatingPointError as exc:
            logger.error("Ignored recovery request: %s", exc)
            return False
        if tilt <= FALL_TILT_THRESHOLD_RAD:
            logger.warning(
                "Ignored recovery request: robot tilt %.3f rad does not exceed %.1f rad",
                tilt,
                FALL_TILT_THRESHOLD_RAD,
            )
            return False
        self._set_upper_body_enabled(False)
        try:
            activated = self.policy_manager.activate_recovery()
        except (FloatingPointError, RuntimeError, ValueError) as exc:
            self._force_damping(f"failed to activate recovery policy: {exc}")
            return False
        if not activated:
            return False
        self._manual_mode_override = None
        self.policy_locomotion_mimic_flag = 0
        self._recovery_return_in_progress = False
        return self._enter_mode(ControlMode.RECOVERY_DEFAULT)

    def _finish_recovery_to_loco(self):
        if self.mode != ControlMode.RECOVERY_DEFAULT:
            return
        self._recovery_return_in_progress = True
        if not self._enter_mode(ControlMode.RL_DEFAULT, force=True):
            self._recovery_return_in_progress = False

    def _request_loco_from_recovery(self) -> bool:
        try:
            tilt = self._robot_tilt()
        except FloatingPointError as exc:
            self._force_damping(str(exc))
            return False
        if tilt >= FALL_TILT_THRESHOLD_RAD:
            logger.warning(
                "Ignored locomotion request: robot tilt %.3f rad is not below %.1f rad",
                tilt,
                FALL_TILT_THRESHOLD_RAD,
            )
            return False
        if self.policy_manager.switch_to_loco(callback_end=self._finish_recovery_to_loco):
            self._recovery_return_in_progress = True
            return True
        return False

    def _step_rl_policy(self, env_data, ctrl_data, dry_run: bool):
        commands = ctrl_data.get("COMMANDS", [])
        if "[POLICY_MIMIC]" in commands:
            self._set_upper_body_enabled(False)

        pd_target, extras = self._get_policy_step(env_data, ctrl_data)
        upper_body_override_available = self._upper_body_control_available()
        if upper_body_override_available:
            if not self._upper_body_override_was_available and self._upper_body_indices.size:
                # Re-entering idle loco must start from the current policy target, not a
                # stale target received before mimic mode disabled the ZMQ stream.
                self._upper_body_filtered = pd_target[self._upper_body_indices].copy()
                self._upper_body_stream_was_fresh = False
            pd_target = self._apply_pd_target_override(pd_target, ctrl_data)
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
            if getattr(self, "mode", None) == ControlMode.RECOVERY_DEFAULT and (
                command.startswith("[POLICY_SWITCH]") or command == "[POLICY_MIMIC]"
            ):
                logger.warning("Ignored %s during recovery", command)
                continue
            match command:
                case cmd if cmd.startswith("[POLICY_SWITCH]"):
                    switch_target = cmd.split(",")[1]
                    if switch_target == "NEXT":
                        self.policy_manager.toggle_mimic_policy(1)
                    elif switch_target == "LAST":
                        self.policy_manager.toggle_mimic_policy(-1)
                case "[POLICY_LOCO]":
                    if getattr(self, "mode", None) == ControlMode.RECOVERY_DEFAULT:
                        self._request_loco_from_recovery()
                    elif self.policy_manager.switch_to_loco():
                        self.policy_locomotion_mimic_flag = 0
                case "[POLICY_MIMIC]":
                    if getattr(self, "mode", None) == ControlMode.RECOVERY_DEFAULT:
                        logger.warning("Ignored mimic request during recovery")
                        continue
                    self._set_upper_body_enabled(False)
                    if self.policy_manager.switch_to_mimic():
                        self.policy_locomotion_mimic_flag = 1
                case "[POLICY_RECOVERY]":
                    if getattr(self, "mode", None) == ControlMode.RECOVERY_DEFAULT:
                        logger.warning("Ignored repeated recovery request")

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
            if self.visualizer is not None:
                # extras belongs to the policy that produced this frame. Draw it before
                # policy_manager.step() can complete an interpolation and change policy.
                self.policy.debug_viz(self.visualizer, env_data, ctrl_data, extras)
        self._record_upper_body_sample(env_data, extras, pd_target, rl_active=rl_active)
        if rl_active or self.policy_manager.warmup_policy_indices:
            self.policy_manager.step(env_data, ctrl_data)
        if self.cfg.debug.log_obs:
            self.debug_logger.log(
                env_data=env_data,
                ctrl_data=ctrl_data,
                extras=extras,
                pd_target=pd_target,
                timestep=self.timestep,
            )
