import logging
from enum import Enum

import numpy as np

from robojudo.controller.ctrl_cfgs import UpperBodyZmqCtrlCfg
from robojudo.pipeline import pipeline_registry
from robojudo.pipeline.rl_pipeline import RlPipeline
from robojudo.utils.util_func import get_gravity_orientation

logger = logging.getLogger(__name__)


class X2ControlMode(str, Enum):
    PASSIVE_DEFAULT = "PASSIVE_DEFAULT"
    DAMPING_DEFAULT = "DAMPING_DEFAULT"
    JOINT_DEFAULT = "JOINT_DEFAULT"
    RL_DEFAULT = "RL_DEFAULT"


MODE_COMMANDS = {f"[{mode.value}]": mode for mode in X2ControlMode}
ELASTIC_BAND_COMMANDS = {
    "[ELASTIC_BAND_LOWER]": "lower_elastic_band",
    "[ELASTIC_BAND_LIFT]": "lift_elastic_band",
    "[ELASTIC_BAND_TOGGLE]": "toggle_elastic_band",
}


@pipeline_registry.register
class X2DeployPipeline(RlPipeline):
    """Four-mode X2 deployment pipeline matching the standalone controller."""

    def __init__(self, cfg):
        self.mode = X2ControlMode.PASSIVE_DEFAULT
        self._joint_default_start = None
        self._joint_default_step = 0
        self._joint_default_complete = False
        self._shutdown_requested = False
        self.should_exit = False
        self._joint_default_dof = cfg.joint_default_dof
        self._joint_default_target = np.asarray(self._joint_default_dof.default_pos, dtype=np.float32)
        self._joint_default_stiffness = np.asarray(self._joint_default_dof.stiffness, dtype=np.float32)
        self._joint_default_damping = np.asarray(self._joint_default_dof.damping, dtype=np.float32)
        self._joint_default_steps = max(1, round(cfg.joint_default_duration * cfg.policy.freq))
        self._default_damping = float(cfg.default_damping)
        self._upper_body_cfg = next(
            (ctrl_cfg for ctrl_cfg in cfg.ctrl if isinstance(ctrl_cfg, UpperBodyZmqCtrlCfg)),
            None,
        )
        self._upper_body_enabled = False
        self._upper_body_stream_was_fresh = False
        self._upper_body_indices = np.asarray([], dtype=np.int32)
        self._upper_body_default = np.asarray([], dtype=np.float32)
        self._upper_body_filtered = np.asarray([], dtype=np.float32)
        super().__init__(cfg=cfg)

        if self._joint_default_dof.joint_names != self.env.joint_names:
            raise ValueError("X2 JOINT_DEFAULT joint order must match the environment joint order")
        self._rl_stiffness = self.stiffness_from_env()
        self._rl_damping = self.damping_from_env()
        self._configure_upper_body_override()
        self._enter_mode(X2ControlMode.PASSIVE_DEFAULT, force=True)

    def _configure_upper_body_override(self):
        if self._upper_body_cfg is None:
            return
        missing = [name for name in self._upper_body_cfg.joint_names if name not in self.env.joint_names]
        if missing:
            raise ValueError(f"Upper-body ZMQ joints missing from X2 environment: {missing}")
        action_joints = set(self.cfg.policy.action_dof.joint_names)
        overlap = sorted(action_joints.intersection(self._upper_body_cfg.joint_names))
        if overlap:
            raise ValueError(f"Upper-body ZMQ joints overlap policy actions: {overlap}")

        self._upper_body_indices = np.asarray(
            [self.env.joint_names.index(name) for name in self._upper_body_cfg.joint_names],
            dtype=np.int32,
        )
        self._upper_body_default = self.env.default_pos[self._upper_body_indices].astype(np.float32)
        self._upper_body_filtered = self.env.dof_pos[self._upper_body_indices].astype(np.float32)

    def reset(self):
        self._upper_body_enabled = False
        self._upper_body_stream_was_fresh = False
        super().reset()
        if self._upper_body_indices.size:
            self._upper_body_filtered = self.env.dof_pos[self._upper_body_indices].astype(np.float32)

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
        self._enter_mode(X2ControlMode.PASSIVE_DEFAULT, force=True)
        self.env.command_passive()
        logger.warning("X2 ready in PASSIVE_DEFAULT; select JOINT_DEFAULT before RL_DEFAULT")

    def _enter_mode(self, requested: X2ControlMode, force: bool = False) -> bool:
        if requested == self.mode and not force:
            return True
        if requested == X2ControlMode.RL_DEFAULT and not (
            self.mode == X2ControlMode.JOINT_DEFAULT and self._joint_default_complete
        ):
            logger.error("Rejected RL_DEFAULT: JOINT_DEFAULT interpolation has not completed")
            return False

        previous = self.mode
        self.mode = requested
        if requested in (X2ControlMode.PASSIVE_DEFAULT, X2ControlMode.DAMPING_DEFAULT):
            self._set_upper_body_enabled(False)
            self._joint_default_complete = False
            self._joint_default_start = None
        elif requested == X2ControlMode.JOINT_DEFAULT:
            self._set_upper_body_enabled(False)
            self._joint_default_start = self.env.dof_pos.astype(np.float32)
            self._joint_default_step = 0
            self._joint_default_complete = False
            self.env.set_control_joint_names(self.env.joint_names)
            self.env.set_gains(self._joint_default_stiffness, self._joint_default_damping)
            self.env.arm_position_control()
        elif requested == X2ControlMode.RL_DEFAULT:
            self.env.set_control_joint_names(self.env.joint_names)
            self.env.set_gains(self._rl_stiffness, self._rl_damping)
            self.env.arm_position_control()
            self.policy.reset()
            inner = self._inner_policy()
            if hasattr(inner, "set_default_pose_mode"):
                inner.set_default_pose_mode(False)
            if getattr(self, "_upper_body_indices", np.asarray([])).size:
                self._upper_body_filtered = self.env.dof_pos[self._upper_body_indices].astype(np.float32)
                self._upper_body_stream_was_fresh = False

        logger.warning("X2 mode: %s -> %s", previous.value, requested.value)
        return True

    def _force_damping(self, reason: str):
        logger.error("Forcing DAMPING_DEFAULT: %s", reason)
        self._enter_mode(X2ControlMode.DAMPING_DEFAULT, force=True)
        self.env.command_damping(self._default_damping)

    def _set_upper_body_enabled(self, enabled: bool):
        enabled = bool(enabled and getattr(self, "_upper_body_cfg", None) is not None)
        if getattr(self, "_upper_body_enabled", False) == enabled:
            return
        self._upper_body_enabled = enabled
        self._upper_body_stream_was_fresh = False
        logger.warning("X2 upper-body ZMQ control %s", "enabled" if enabled else "disabled")

    def _toggle_upper_body(self):
        if self._upper_body_cfg is None:
            logger.warning("Ignored upper-body toggle: ZMQ controller is not configured")
        elif self._upper_body_enabled:
            self._set_upper_body_enabled(False)
        elif self.mode != X2ControlMode.RL_DEFAULT:
            logger.warning("Ignored upper-body enable outside RL_DEFAULT")
        else:
            self._set_upper_body_enabled(True)

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
            self.policy.reset()
            self._enter_mode(X2ControlMode.PASSIVE_DEFAULT, force=True)
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

    def _apply_upper_body_override(self, pd_target: np.ndarray, ctrl_data) -> np.ndarray:
        if self._upper_body_cfg is None:
            return pd_target

        desired = self._upper_body_default.copy()
        stream_data = ctrl_data.get("UpperBodyZmqCtrl", {})
        stream_is_fresh = bool(stream_data.get("fresh", False))
        if self._upper_body_enabled and stream_is_fresh:
            positions = stream_data.get("joint_positions", {})
            for local_index, name in enumerate(self._upper_body_cfg.joint_names):
                if name in positions:
                    desired[local_index] = positions[name]
            limits = self.env.position_limits[self._upper_body_indices]
            desired = np.clip(desired, limits[:, 0], limits[:, 1])
            if not self._upper_body_stream_was_fresh:
                logger.info("X2 upper-body ZMQ stream active")
        elif self._upper_body_enabled and self._upper_body_stream_was_fresh:
            logger.warning("X2 upper-body ZMQ stream timed out; returning to defaults")

        self._upper_body_stream_was_fresh = self._upper_body_enabled and stream_is_fresh
        alpha = self._upper_body_cfg.ema_alpha
        self._upper_body_filtered = alpha * self._upper_body_filtered + (1.0 - alpha) * desired
        snap = np.abs(self._upper_body_filtered - desired) < 0.001
        self._upper_body_filtered[snap] = desired[snap]

        target = np.asarray(pd_target, dtype=np.float32).copy()
        target[self._upper_body_indices] = self._upper_body_filtered
        return target

    def _safety_check_before_command(self):
        if not self.do_safety_check:
            return
        gravity = get_gravity_orientation(self.env.base_quat)
        tilt = np.arccos(np.clip(-gravity[2], -1.0, 1.0))
        if abs(tilt) > 1.0 and self.mode != X2ControlMode.DAMPING_DEFAULT:
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

    def step(self, dry_run=False):
        try:
            self.env.update()
        except RuntimeError as exc:
            if not dry_run:
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
            if not dry_run and not self._shutdown_requested:
                if self.mode == X2ControlMode.PASSIVE_DEFAULT:
                    self.env.command_passive()
                elif self.mode == X2ControlMode.DAMPING_DEFAULT:
                    self.env.command_damping(self._default_damping)
                elif self.mode == X2ControlMode.JOINT_DEFAULT:
                    pd_target = self._step_joint_default()
                elif self.mode == X2ControlMode.RL_DEFAULT:
                    obs, extras = self.policy.get_observation(env_data, ctrl_data)
                    pd_target = self.policy.get_pd_target(obs)
                    pd_target = self._apply_upper_body_override(pd_target, ctrl_data)
                    self.env.step(pd_target, extras.get("hand_pose"))
        except (FloatingPointError, RuntimeError, ValueError) as exc:
            if not dry_run:
                self._force_damping(str(exc))

        self.timestep += 1
        self.ctrl_manager.post_step_callback(ctrl_data)
        self.policy.post_step_callback(commands)
        if self.visualizer is not None and self.mode == X2ControlMode.RL_DEFAULT:
            self.policy.debug_viz(self.visualizer, env_data, ctrl_data, extras)
        if self.cfg.debug.log_obs:
            self.debug_logger.log(
                env_data=env_data,
                ctrl_data=ctrl_data,
                extras=extras,
                pd_target=pd_target,
                timestep=self.timestep,
            )
