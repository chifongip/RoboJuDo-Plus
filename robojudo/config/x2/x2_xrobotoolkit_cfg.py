"""Opt-in XRoboToolkit arm control using the existing X2 locomanipulation runtime."""

from robojudo.config import cfg_registry
from robojudo.controller.ctrl_cfgs import (
    JoystickCtrlCfg,
    KeyboardCtrlCfg,
    RosJoystickCtrlCfg,
    UpperBodyZmqCtrlCfg,
)

from .x2_cfg import x2_locomanipulation, x2_locomanipulation_real


def _xr_controllers(base):
    controllers = []
    for cfg in base.ctrl:
        values = cfg.model_dump()
        if isinstance(cfg, UpperBodyZmqCtrlCfg):
            values.update(
                feedback_endpoint="tcp://127.0.0.1:8571",
                visualize_targets=base.env.is_sim,
                ema_alpha=0.0,
                max_joint_velocity_rad_s=8.0,
                return_joint_velocity_rad_s=1.0,
            )
        controllers.append(type(cfg).model_validate(values))
    return controllers


@cfg_registry.register
class x2_xrobotoolkit(x2_locomanipulation):
    """X2 simulation with externally checked XR arm proposals and measured feedback."""

    ctrl: list[JoystickCtrlCfg | KeyboardCtrlCfg | RosJoystickCtrlCfg | UpperBodyZmqCtrlCfg] = _xr_controllers(
        x2_locomanipulation()
    )


@cfg_registry.register
class x2_xrobotoolkit_real(x2_locomanipulation_real):
    """X2 AimDK deployment with the same externally checked XR arm protocol."""

    ctrl: list[JoystickCtrlCfg | KeyboardCtrlCfg | RosJoystickCtrlCfg | UpperBodyZmqCtrlCfg] = _xr_controllers(
        x2_locomanipulation_real()
    )
