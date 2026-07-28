import logging
from enum import Enum

import numpy as np

from robojudo.pipeline.upper_body_zmq_pipeline import UpperBodyZmqPipelineMixin
from robojudo.utils.util_func import get_gravity_orientation

logger = logging.getLogger(__name__)


class ControlMode(str, Enum):
    PASSIVE_DEFAULT = "PASSIVE_DEFAULT"
    DAMPING_DEFAULT = "DAMPING_DEFAULT"
    JOINT_DEFAULT = "JOINT_DEFAULT"
    RL_DEFAULT = "RL_DEFAULT"


MODE_COMMANDS = {f"[{mode.value}]": mode for mode in ControlMode}
ELASTIC_BAND_COMMANDS = {
    "[ELASTIC_BAND_LOWER]": "lower_elastic_band",
    "[ELASTIC_BAND_LIFT]": "lift_elastic_band",
    "[ELASTIC_BAND_TOGGLE]": "toggle_elastic_band",
}


class FourModePipelineMixin(UpperBodyZmqPipelineMixin):
    """Four deployment modes shared by robot-specific RL pipelines."""

    @property
    def _mode_label(self) -> str:
        return getattr(self, "_mode_robot_name", "ROBOT")

    @staticmethod
    def _initial_policy_cfg(cfg):
        return cfg.policy

    def __init__(self, cfg):
        self._mode_robot_name = cfg.robot.upper()
        self.mode = ControlMode.PASSIVE_DEFAULT
        self._joint_default_start = None
        self._joint_default_step = 0
        self._joint_default_complete = False
        self._shutdown_requested = False
        self.should_exit = False
        self._joint_default_dof = cfg.joint_default_dof
        self._joint_default_target = np.asarray(self._joint_default_dof.default_pos, dtype=np.float32)
        self._joint_default_stiffness = np.asarray(self._joint_default_dof.stiffness, dtype=np.float32)
        self._joint_default_damping = np.asarray(self._joint_default_dof.damping, dtype=np.float32)
        self._joint_default_steps = max(1, round(cfg.joint_default_duration * self._initial_policy_cfg(cfg).freq))
        self._default_damping = float(cfg.default_damping)
        super().__init__(cfg=cfg)

        if self._joint_default_dof.joint_names != self.env.joint_names:
            raise ValueError(
                f"{self._mode_label} JOINT_DEFAULT joint order must match the environment joint order"
            )
        self._rl_stiffness = self.stiffness_from_env()
        self._rl_damping = self.damping_from_env()
        self._enter_mode(ControlMode.PASSIVE_DEFAULT, force=True)

    def _upper_body_enable_available(self) -> bool:
        if self.mode != ControlMode.RL_DEFAULT:
            logger.warning("Ignored upper-body enable outside RL_DEFAULT")
            return False
        if not self._upper_body_control_available():
            logger.warning("Ignored upper-body enable while the active policy controls the arms")
            return False
        return True

    @property
    def _has_default_pose_mode(self) -> bool:
        # Startup and policy activation are owned by this mode state machine.
        return False

    def stiffness_from_env(self):
        return np.asarray(self.env.dof_cfg.stiffness, dtype=np.float32).copy()

    def damping_from_env(self):
        return np.asarray(self.env.dof_cfg.damping, dtype=np.float32).copy()

    def prepare(self, init_motor_angle=None, prepare_seconds=None):
        del init_motor_angle, prepare_seconds
        self._enter_mode(ControlMode.PASSIVE_DEFAULT, force=True)
        self.env.command_passive()
        logger.warning(
            "%s ready in PASSIVE_DEFAULT; select JOINT_DEFAULT before RL_DEFAULT",
            self._mode_label,
        )

    def _enter_mode(self, requested: ControlMode, force: bool = False) -> bool:
        if requested == self.mode and not force:
            return True
        if requested == ControlMode.RL_DEFAULT and not (
            self.mode == ControlMode.JOINT_DEFAULT and self._joint_default_complete
        ):
            logger.error("Rejected RL_DEFAULT: JOINT_DEFAULT interpolation has not completed")
            return False

        previous = self.mode
        if previous == ControlMode.RL_DEFAULT and requested != ControlMode.RL_DEFAULT:
            self._on_leave_rl()
        self.mode = requested
        if requested in (ControlMode.PASSIVE_DEFAULT, ControlMode.DAMPING_DEFAULT):
            self._set_upper_body_enabled(False)
            self._joint_default_complete = False
            self._joint_default_start = None
        elif requested == ControlMode.JOINT_DEFAULT:
            self._set_upper_body_enabled(False)
            self._joint_default_start = self.env.dof_pos.astype(np.float32)
            self._joint_default_step = 0
            self._joint_default_complete = False
            self.env.set_control_joint_names(self.env.joint_names)
            self.env.set_gains(self._joint_default_stiffness, self._joint_default_damping)
            self.env.arm_position_control()
        elif requested == ControlMode.RL_DEFAULT:
            self.env.set_control_joint_names(self.env.joint_names)
            self.env.set_gains(self._rl_stiffness, self._rl_damping)
            self.env.arm_position_control()
            self._on_enter_rl()
            if getattr(self, "_upper_body_indices", np.asarray([])).size:
                self._upper_body_filtered = self.env.dof_pos[self._upper_body_indices].astype(np.float32)
                self._upper_body_stream_was_fresh = False

        logger.warning(
            "%s mode: %s -> %s",
            self._mode_label,
            previous.value,
            requested.value,
        )
        return True

    def _on_enter_rl(self):
        self.policy.reset()
        inner = self._inner_policy()
        if hasattr(inner, "set_default_pose_mode"):
            inner.set_default_pose_mode(False)

    def _on_leave_rl(self):
        self._set_upper_body_enabled(False)
        if hasattr(self.policy, "close_progress"):
            self.policy.close_progress()

    def _reset_policy_state(self):
        self.policy.reset()

    def _force_damping(self, reason: str):
        logger.error("Forcing DAMPING_DEFAULT: %s", reason)
        self._enter_mode(ControlMode.DAMPING_DEFAULT, force=True)
        self.env.command_damping(self._default_damping)

    def _process_commands(self, commands: list[str]):
        if "[SHUTDOWN]" in commands:
            self._force_damping("shutdown requested")
            self.env.shutdown()
            self._shutdown_requested = True
            self.should_exit = True
            return
        if "[SIM_REBORN]" in commands and hasattr(self.env, "reborn"):
            self._set_upper_body_enabled(False)
            self.env.reborn()
            self._reset_policy_state()
            self._enter_mode(ControlMode.PASSIVE_DEFAULT, force=True)
            return

        for command in commands:
            if command == "[UPPER_BODY_TOGGLE]":
                self._toggle_upper_body()
                continue
            requested = MODE_COMMANDS.get(command)
            if requested is not None:
                self._enter_mode(requested)
                continue
            method_name = ELASTIC_BAND_COMMANDS.get(command)
            if method_name is not None:
                method = getattr(self.env, method_name, None)
                if method is None:
                    logger.warning("Ignored %s: environment has no ElasticBand", command)
                else:
                    method()

    def _safety_check_before_command(self):
        if not self.do_safety_check:
            return
        gravity = get_gravity_orientation(self.env.base_quat)
        tilt = np.arccos(np.clip(-gravity[2], -1.0, 1.0))
        if abs(tilt) > 1.0 and self.mode != ControlMode.DAMPING_DEFAULT:
            self._force_damping(f"robot tilt {tilt:.3f} rad exceeds 1.0 rad")

    def _step_joint_default(self):
        if self._joint_default_start is None:
            self._joint_default_start = self.env.dof_pos.astype(np.float32)
        self._joint_default_step = min(self._joint_default_step + 1, self._joint_default_steps)
        alpha = self._joint_default_step / self._joint_default_steps
        target = (1.0 - alpha) * self._joint_default_start + alpha * self._joint_default_target
        self.env.step(target)
        if self._joint_default_step == self._joint_default_steps and not self._joint_default_complete:
            self._joint_default_complete = True
            logger.warning("JOINT_DEFAULT interpolation complete; RL_DEFAULT is now enabled")
        return target

    def _step_rl_policy(self, env_data, ctrl_data, dry_run: bool):
        obs, extras = self.policy.get_observation(env_data, ctrl_data)
        pd_target = self.policy.get_pd_target(obs)
        pd_target = self._apply_pd_target_override(pd_target, ctrl_data)
        if not dry_run:
            self.env.step(pd_target, extras.get("hand_pose"))
        return pd_target, extras

    def _post_mode_step(self, env_data, ctrl_data, extras, pd_target, rl_active: bool):
        commands = ctrl_data.get("COMMANDS", [])
        self.timestep += 1
        self.ctrl_manager.post_step_callback(ctrl_data)
        self.policy.post_step_callback(commands)
        self._record_upper_body_sample(env_data, extras, pd_target, rl_active=rl_active)
        if self.visualizer is not None and rl_active:
            self.policy.debug_viz(self.visualizer, env_data, ctrl_data, extras)
        if self.cfg.debug.log_obs:
            self.debug_logger.log(
                env_data=env_data,
                ctrl_data=ctrl_data,
                extras=extras,
                pd_target=pd_target,
                timestep=self.timestep,
            )

    def step(self, dry_run=False):
        try:
            self.env.update()
        except RuntimeError as exc:
            if dry_run:
                raise
            self._force_damping(str(exc))
            return

        env_data = self.env.get_data()
        ctrl_data = self.ctrl_manager.get_ctrl_data(env_data)
        commands = ctrl_data.get("COMMANDS", [])
        self._process_commands(commands)
        self._safety_check_before_command()

        extras = {}
        pd_target = self.env.dof_pos
        try:
            if not self._shutdown_requested:
                if dry_run:
                    pd_target, extras = self._step_rl_policy(env_data, ctrl_data, dry_run=True)
                elif self.mode == ControlMode.PASSIVE_DEFAULT:
                    self.env.command_passive()
                elif self.mode == ControlMode.DAMPING_DEFAULT:
                    self.env.command_damping(self._default_damping)
                elif self.mode == ControlMode.JOINT_DEFAULT:
                    pd_target = self._step_joint_default()
                elif self.mode == ControlMode.RL_DEFAULT:
                    pd_target, extras = self._step_rl_policy(env_data, ctrl_data, dry_run=False)
        except (FloatingPointError, RuntimeError, ValueError) as exc:
            if dry_run:
                raise
            self._force_damping(str(exc))

        self._post_mode_step(
            env_data,
            ctrl_data,
            extras,
            pd_target,
            rl_active=self.mode == ControlMode.RL_DEFAULT,
        )
