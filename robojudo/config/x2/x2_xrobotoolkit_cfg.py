"""Opt-in XRoboToolkit arm control using the X2 observer loco-mimic runtime."""

from robojudo.config import cfg_registry
from robojudo.controller.ctrl_cfgs import (
    JoystickCtrlCfg,
    KeyboardCtrlCfg,
    LocomanipulationPostureZmqCtrlCfg,
    RosJoystickCtrlCfg,
    UpperBodyZmqCtrlCfg,
    VelocityZmqCtrlCfg,
)

from .x2_locomimic_observer_cfg import x2_locomimic_observer, x2_locomimic_real_observer


def _xr_controllers(base):
    controllers = []
    for cfg in base.ctrl:
        controller = cfg.model_copy(deep=True)
        if isinstance(cfg, UpperBodyZmqCtrlCfg):
            controller.feedback_endpoint = "tcp://127.0.0.1:8571"
            controller.visualize_targets = base.env.is_sim
            controller.ema_alpha = 0.0
            controller.max_joint_velocity_rad_s = 8.0
            controller.return_joint_velocity_rad_s = 1.0
        controllers.append(controller)
    return controllers


@cfg_registry.register
class x2_xrobotoolkit(x2_locomimic_observer):
    """X2 simulation with externally checked XR arm proposals and measured feedback."""

    ctrl: list[
        JoystickCtrlCfg
        | KeyboardCtrlCfg
        | RosJoystickCtrlCfg
        | UpperBodyZmqCtrlCfg
        | VelocityZmqCtrlCfg
        | LocomanipulationPostureZmqCtrlCfg
    ] = _xr_controllers(x2_locomimic_observer())


@cfg_registry.register
class x2_xrobotoolkit_real(x2_locomimic_real_observer):
    """X2 AimDK deployment with the same externally checked XR arm protocol."""

    ctrl: list[
        JoystickCtrlCfg
        | KeyboardCtrlCfg
        | RosJoystickCtrlCfg
        | UpperBodyZmqCtrlCfg
        | VelocityZmqCtrlCfg
        | LocomanipulationPostureZmqCtrlCfg
    ] = _xr_controllers(x2_locomimic_real_observer())
