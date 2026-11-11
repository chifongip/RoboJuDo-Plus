"""Opt-in XRoboToolkit arm control using the X2 observer loco-mimic runtime."""

from robojudo.config import ASSETS_DIR, cfg_registry
from robojudo.controller.ctrl_cfgs import (
    JoystickCtrlCfg,
    KeyboardCtrlCfg,
    LocomanipulationPostureZmqCtrlCfg,
    RosJoystickCtrlCfg,
    UpperBodyOmniHandZmqCtrlCfg,
    UpperBodyZmqCtrlCfg,
    VelocityZmqCtrlCfg,
)

from .env.x2_mujuco_env_cfg import X2ObserverMujocoEnvCfg
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


def _xr_hand_controllers(base):
    controllers = []
    for cfg in base.ctrl:
        if isinstance(cfg, UpperBodyZmqCtrlCfg):
            controllers.append(
                UpperBodyOmniHandZmqCtrlCfg(
                    **{
                        **cfg.model_dump(),
                        "ctrl_type": "UpperBodyOmniHandZmqCtrl",
                    }
                )
            )
        else:
            controllers.append(cfg.model_copy(deep=True))
    return controllers


XRHandControllers = list[
    JoystickCtrlCfg
    | KeyboardCtrlCfg
    | RosJoystickCtrlCfg
    | UpperBodyOmniHandZmqCtrlCfg
    | VelocityZmqCtrlCfg
    | LocomanipulationPostureZmqCtrlCfg
]


@cfg_registry.register
class x2_xrobotoolkit_omnihand(x2_xrobotoolkit):
    """XR arms and independent trigger fingers in the floating X2 simulation."""

    ctrl: XRHandControllers = _xr_hand_controllers(x2_xrobotoolkit())
    env: X2ObserverMujocoEnvCfg = X2ObserverMujocoEnvCfg(
        **{
            **x2_xrobotoolkit().env.model_dump(),
            "env_type": "X2OmniHandMujocoEnv",
            "xml": str(ASSETS_DIR / "robots/x2/x2_omnihand.xml"),
        }
    )


@cfg_registry.register
class x2_xrobotoolkit_omnihand_real(x2_xrobotoolkit_real):
    """Robot-side HCAN selected by local adapter serials and exact hand identities."""

    ctrl: XRHandControllers = _xr_hand_controllers(x2_xrobotoolkit_real())
