"""X2 XR teleoperation presets sharing robot-owned arm and finger settings."""

from typing import TypeVar

from robojudo.config import ASSETS_DIR, cfg_registry
from robojudo.controller.ctrl_cfgs import (
    CtrlCfg,
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
from .policy.x2_locomanipulation_policy_cfg import X2LocomanipulationPolicyCfg
from .x2_locomimic_observer_cfg import x2_locomimic_observer, x2_locomimic_real_observer

# The bridge adopts the robot's collision policy.
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

XR_VELOCITY_SETTINGS = {
    "endpoint": "tcp://127.0.0.1:8558",
    "velocity_priority": 100,
    "xr_authorized": True,
}

XR_POSTURE_SETTINGS = {
    "endpoint": "tcp://127.0.0.1:8557",
    "posture_priority": 100,
    "xr_authorized": False,
}

PELVIS_IK_POSTURE_RATES = {
    "max_height_command_rate_m_s": 0.2,
    "max_waist_yaw_command_rate_rad_s": 1.0,
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

_ControllerCfg = TypeVar("_ControllerCfg", bound=CtrlCfg)


def _xr_controllers(base, *, omnihand=False):
    replacements = {
        cfg.ctrl_type: cfg
        for cfg in (
            VelocityZmqCtrlCfg(**XR_VELOCITY_SETTINGS),
            LocomanipulationPostureZmqCtrlCfg(**XR_POSTURE_SETTINGS),
        )
    }
    controllers = []
    for cfg in base.ctrl:
        replacement = replacements.pop(cfg.ctrl_type, None)
        if replacement is not None:
            controllers.append(replacement)
            continue
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
    controllers.extend(replacements.values())
    return controllers


def _pelvis_ik_controllers(controllers: list[_ControllerCfg]) -> list[_ControllerCfg]:
    return [
        cfg.model_copy(deep=True, update={"xr_authorized": True})
        if isinstance(cfg, LocomanipulationPostureZmqCtrlCfg)
        else cfg.model_copy(deep=True)
        for cfg in controllers
    ]


@cfg_registry.register
class x2_xrobotoolkit(x2_locomimic_observer):
    """Arm-only XR simulation with measured feedback and target visualization."""

    ctrl: XRArmControllers = _xr_controllers(x2_locomimic_observer())


@cfg_registry.register
class x2_xrobotoolkit_pelvis_ik(x2_xrobotoolkit):
    """Pelvis-relative XR arms and session-authorized posture in arm-only simulation."""

    loco_policy: X2LocomanipulationPolicyCfg = x2_xrobotoolkit().loco_policy.model_copy(update=PELVIS_IK_POSTURE_RATES)
    ctrl: XRArmControllers = _pelvis_ik_controllers(x2_xrobotoolkit().ctrl)


@cfg_registry.register
class x2_xrobotoolkit_real(x2_locomimic_real_observer):
    """Arm-only XR hardware deployment through AimDK."""

    ctrl: XRArmControllers = _xr_controllers(x2_locomimic_real_observer())


@cfg_registry.register
class x2_xrobotoolkit_pelvis_ik_real(x2_xrobotoolkit_real):
    """Pelvis-relative XR arms and session-authorized posture through AimDK."""

    loco_policy: X2LocomanipulationPolicyCfg = x2_xrobotoolkit_real().loco_policy.model_copy(
        update=PELVIS_IK_POSTURE_RATES
    )
    ctrl: XRArmControllers = _pelvis_ik_controllers(x2_xrobotoolkit_real().ctrl)


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
class x2_xrobotoolkit_omnihand_pelvis_ik(x2_xrobotoolkit_omnihand):
    """Pelvis-relative XR arms and session-authorized posture in simulation."""

    loco_policy: X2LocomanipulationPolicyCfg = x2_xrobotoolkit_omnihand().loco_policy.model_copy(
        update=PELVIS_IK_POSTURE_RATES
    )
    ctrl: XRHandControllers = _pelvis_ik_controllers(x2_xrobotoolkit_omnihand().ctrl)


@cfg_registry.register
class x2_xrobotoolkit_omnihand_real(x2_xrobotoolkit_real):
    """Robot-side HCAN selected by local adapter serials and exact hand identities."""

    ctrl: XRHandControllers = _xr_controllers(x2_locomimic_real_observer(), omnihand=True)


@cfg_registry.register
class x2_xrobotoolkit_omnihand_pelvis_ik_real(x2_xrobotoolkit_omnihand_real):
    """Pelvis-relative XR arms and posture with AimDK and robot-side OmniHands."""

    loco_policy: X2LocomanipulationPolicyCfg = x2_xrobotoolkit_omnihand_real().loco_policy.model_copy(
        update=PELVIS_IK_POSTURE_RATES
    )
    ctrl: XRHandControllers = _pelvis_ik_controllers(x2_xrobotoolkit_omnihand_real().ctrl)
