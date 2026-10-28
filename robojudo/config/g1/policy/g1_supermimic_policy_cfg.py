"""G1 SuperMimic deployment; gains match mjlab's training actuators."""

import re
from pathlib import Path
from typing import Literal

from pydantic import Field

from robojudo.policy.policy_cfgs import PolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig

from ..env.g1_env_cfg import G1_23_DOF_INDICES, G1_29DoF

# kp, kd, effort limit. These are the SuperMimic training
# settings, which differ from the AMP recovery and stock Unitree presets.
MOTOR_GROUPS = {
    r".*_(elbow|shoulder_pitch|shoulder_roll|shoulder_yaw|wrist_roll)_joint": (
        14.25062309787429,
        0.907222843292423,
        25.0,
    ),
    r"(.*_hip_pitch|.*_hip_yaw|waist_yaw)_joint": (40.17923863450712, 2.557889775413375, 88.0),
    r".*_(hip_roll|knee)_joint": (99.09842777666111, 6.308801853496639, 139.0),
    r".*_wrist_(pitch|yaw)_joint": (16.77832748089279, 1.06814150219, 5.0),
    r"(waist_pitch|waist_roll|.*_ankle_pitch|.*_ankle_roll)_joint": (
        28.50124619574858,
        1.814445686584846,
        50.0,
    ),
}


def motor_parameters(joint_names):
    result = []
    for name in joint_names:
        matches = [values for pattern, values in MOTOR_GROUPS.items() if re.fullmatch(pattern, name)]
        if len(matches) != 1:
            raise ValueError(f"No unique SuperMimic motor preset for {name}")
        result.append(matches[0])
    return result


_DOF = G1_29DoF()
_MOTORS = motor_parameters(_DOF.joint_names)
_DEFAULT = [
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,
    0.0,
    0.0,
    0.0,
    0.2,
    0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,
    0.2,
    -0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,
]


class G1SuperMimicDoF(DoFConfig):
    joint_names: list[str] = _DOF.joint_names
    default_pos: list[float] | None = _DEFAULT
    stiffness: list[float] | None = [m[0] for m in _MOTORS]
    damping: list[float] | None = [m[1] for m in _MOTORS]
    torque_limits: list[float] | None = [m[2] for m in _MOTORS]
    position_limits: list[list[float]] | None = _DOF.position_limits


class G1_23SuperMimicDoF(G1SuperMimicDoF):
    _subset: bool = True
    _subset_joint_names: list[str] | None = [_DOF.joint_names[i] for i in G1_23_DOF_INDICES]


class G1SuperMimicPolicyCfg(PolicyCfg):
    robot: str = "g1"
    disable_autoload: bool = True
    freq: Literal[50] = 50
    history_length: Literal[10] = 10
    obs_dof: DoFConfig = G1SuperMimicDoF()
    action_dof: DoFConfig = G1SuperMimicDoF()
    checkpoint: Path
    action_scales: list[float] = [0.25 * m[2] / m[0] for m in _MOTORS]
    torch_num_threads: int = Field(default=1, ge=1)

    @property
    def policy_file(self) -> str:
        return str(self.checkpoint)


class G1SuperMimicMotionPolicyCfg(G1SuperMimicPolicyCfg):
    policy_type: str = "SuperMimicMotionPolicy"
    motion_file: Path
    reference_dim: Literal[2680] = 2680
    motion_anchor_ori_b: Literal[True] = True
    loop_motion: bool = False


class G1SuperMimicVelocityPolicyCfg(G1SuperMimicPolicyCfg):
    policy_type: str = "SuperMimicVelocityPolicy"
    # Training metadata only: the deployed velocity actor takes history + command.
    reference_dim: Literal[730] = 730
    command_limits: tuple[float, float, float] = (1.0, 0.5, 1.0)
    keyboard_command: tuple[float, float, float] = (0.5, 0.3, 0.5)


class G1_23SuperMimicMotionPolicyCfg(G1SuperMimicMotionPolicyCfg):
    """Native 23-DoF full-body actor; the supplied checkpoint has no torso anchor."""

    obs_dof: DoFConfig = G1_23SuperMimicDoF()
    action_dof: DoFConfig = G1_23SuperMimicDoF()
    action_scales: list[float] = [0.25 * m[2] / m[0] for m in motor_parameters(G1_23SuperMimicDoF().joint_names)]
    reference_dim: Literal[2560] = 2560
    motion_anchor_ori_b: bool = False


class G1_23SuperMimicVelocityPolicyCfg(G1SuperMimicVelocityPolicyCfg):
    """Native 23-DoF velocity actor with 75-D proprioception per history frame."""

    obs_dof: DoFConfig = G1_23SuperMimicDoF()
    action_dof: DoFConfig = G1_23SuperMimicDoF()
    action_scales: list[float] = [0.25 * m[2] / m[0] for m in motor_parameters(G1_23SuperMimicDoF().joint_names)]
    # The uploaded velocity checkpoint was trained with full-body motion references.
    # Deployment consumes only proprioceptive history and the velocity command.
    reference_dim: Literal[2560] = 2560
