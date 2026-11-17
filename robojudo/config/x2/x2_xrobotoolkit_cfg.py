"""X2 XR teleoperation presets sharing robot-owned arm and finger settings."""

from robojudo.config import ASSETS_DIR, cfg_registry
from robojudo.controller.ctrl_cfgs import (
    JoystickCtrlCfg,
    KeyboardCtrlCfg,
    LocomanipulationPostureZmqCtrlCfg,
    OmniHandCfg,
    RosJoystickCtrlCfg,
    UpperBodyOmniHandZmqCtrlCfg,
    UpperBodyZmqCtrlCfg,
    VelocityZmqCtrlCfg,
)

from .env.x2_mujuco_env_cfg import X2ObserverMujocoEnvCfg
from .x2_locomimic_observer_cfg import x2_locomimic_observer, x2_locomimic_real_observer

# Shared by all four presets. The bridge adopts the robot's collision policy.
XR_ARM_SETTINGS = {
    "executor_collision_checking": False,
    "ema_alpha": 0.0,
    "max_joint_velocity_rad_s": 8.0,
    "return_joint_velocity_rad_s": 1.0,
    "feedback_endpoint": "tcp://127.0.0.1:8571",
    "command_receive_hwm": 8,
}

# Shared by the simulation and hardware OmniHand presets.
XR_HAND_SETTINGS = {
    "hand_command_mode": "direct",  # "direct" or "filtered"; filtered uses the arm EMA setting.
    "hand_velocity_limit": 2.0,  # rad/s; filtered fingers are also bounded by model limits.
}

XRArmControllers = list[
    JoystickCtrlCfg
    | KeyboardCtrlCfg
    | RosJoystickCtrlCfg
    | UpperBodyZmqCtrlCfg
    | VelocityZmqCtrlCfg
    | LocomanipulationPostureZmqCtrlCfg
]
XRHandControllers = list[
    JoystickCtrlCfg
    | KeyboardCtrlCfg
    | RosJoystickCtrlCfg
    | UpperBodyOmniHandZmqCtrlCfg
    | VelocityZmqCtrlCfg
    | LocomanipulationPostureZmqCtrlCfg
]


def _xr_controllers(base, *, omnihand=False):
    controllers = []
    for cfg in base.ctrl:
        if not isinstance(cfg, UpperBodyZmqCtrlCfg):
            controllers.append(cfg.model_copy(deep=True))
            continue
        values = {**cfg.model_dump(), **XR_ARM_SETTINGS, "visualize_targets": base.env.is_sim}
        if omnihand:
            controllers.append(
                UpperBodyOmniHandZmqCtrlCfg(
                    **{
                        **values,
                        **XR_HAND_SETTINGS,
                        "ctrl_type": "UpperBodyOmniHandZmqCtrl",
                        "omnihand": OmniHandCfg(joint_state_fps=50.0),
                    }
                )
            )
        else:
            controllers.append(UpperBodyZmqCtrlCfg(**values))
    return controllers


@cfg_registry.register
class x2_xrobotoolkit(x2_locomimic_observer):
    """Arm-only XR simulation with measured feedback and target visualization."""

    ctrl: XRArmControllers = _xr_controllers(x2_locomimic_observer())


@cfg_registry.register
class x2_xrobotoolkit_real(x2_locomimic_real_observer):
    """Arm-only XR hardware deployment through AimDK."""

    ctrl: XRArmControllers = _xr_controllers(x2_locomimic_real_observer())


@cfg_registry.register
class x2_xrobotoolkit_omnihand(x2_xrobotoolkit):
    """XR arms and independent trigger fingers in the floating X2 simulation."""

    ctrl: XRHandControllers = _xr_controllers(x2_locomimic_observer(), omnihand=True)
    env: X2ObserverMujocoEnvCfg = x2_xrobotoolkit().env.model_copy(
        update={
            "env_type": "X2OmniHandMujocoEnv",
            "xml": str(ASSETS_DIR / "robots/x2/x2_omnihand.xml"),
        }
    )


@cfg_registry.register
class x2_xrobotoolkit_omnihand_real(x2_xrobotoolkit_real):
    """Robot-side HCAN selected by local adapter serials and exact hand identities."""

    ctrl: XRHandControllers = _xr_controllers(x2_locomimic_real_observer(), omnihand=True)
