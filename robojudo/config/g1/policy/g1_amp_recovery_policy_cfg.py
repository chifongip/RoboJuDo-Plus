from robojudo.policy.policy_cfgs import AmpRecoveryPolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig

from ..env.g1_env_cfg import G1_23_DOF_INDICES, G1_29DoF

_G1_29_DOF = G1_29DoF()
_G1_23_JOINT_NAMES = [_G1_29_DOF.joint_names[index] for index in G1_23_DOF_INDICES]
_G1_POSITION_LIMITS = dict(zip(_G1_29_DOF.joint_names, _G1_29_DOF.position_limits, strict=True))

G1_AMP_RECOVERY_29_DEFAULT_POS = [
    *[-0.1, 0.0, 0.0, 0.3, -0.2, 0.0],
    *[-0.1, 0.0, 0.0, 0.3, -0.2, 0.0],
    *[0.0, 0.0, 0.0],
    *[0.35, 0.18, 0.0, 0.87, 0.0, 0.0, 0.0],
    *[0.35, -0.18, 0.0, 0.87, 0.0, 0.0, 0.0],
]

G1_AMP_RECOVERY_29_STIFFNESS = [
    *[99.09842777666111, 99.09842777666111, 40.17923863450712, 99.09842777666111, 28.50124619574858, 28.50124619574858],
    *[99.09842777666111, 99.09842777666111, 40.17923863450712, 99.09842777666111, 28.50124619574858, 28.50124619574858],
    *[40.17923863450712, 28.50124619574858, 28.50124619574858],
    *[14.25062309787429] * 5,
    *[8.611032447370201] * 2,
    *[14.25062309787429] * 5,
    *[8.611032447370201] * 2,
]
G1_AMP_RECOVERY_29_DAMPING = [
    *[6.308801853496639, 6.308801853496639, 2.557889775413375, 6.308801853496639, 1.814445686584846, 1.814445686584846],
    *[6.308801853496639, 6.308801853496639, 2.557889775413375, 6.308801853496639, 1.814445686584846, 1.814445686584846],
    *[2.557889775413375, 1.814445686584846, 1.814445686584846],
    *[0.907222843292423] * 5,
    *[0.548195351665136] * 2,
    *[0.907222843292423] * 5,
    *[0.548195351665136] * 2,
]
G1_AMP_RECOVERY_29_EFFORT_LIMITS = [
    *[139.0, 139.0, 88.0, 139.0, 50.0, 50.0],
    *[139.0, 139.0, 88.0, 139.0, 50.0, 50.0],
    *[88.0, 50.0, 50.0],
    *[25.0] * 5,
    *[10.0] * 2,
    *[25.0] * 5,
    *[10.0] * 2,
]
G1_AMP_RECOVERY_29_ACTION_SCALES = [
    0.25 * effort / stiffness
    for effort, stiffness in zip(
        G1_AMP_RECOVERY_29_EFFORT_LIMITS,
        G1_AMP_RECOVERY_29_STIFFNESS,
        strict=True,
    )
]

G1_AMP_RECOVERY_23_DEFAULT_POS = [
    *[-0.1, 0.0, 0.0, 0.3, -0.2, 0.0],
    *[-0.1, 0.0, 0.0, 0.3, -0.2, 0.0],
    0.0,
    *[0.35, 0.18, 0.0, 0.87, 0.0],
    *[0.35, -0.18, 0.0, 0.87, 0.0],
]
G1_AMP_RECOVERY_23_STIFFNESS = [
    *[40.17923863450712, 99.09842777666111, 40.17923863450712, 99.09842777666111, 28.50124619574858, 28.50124619574858],
    *[40.17923863450712, 99.09842777666111, 40.17923863450712, 99.09842777666111, 28.50124619574858, 28.50124619574858],
    40.17923863450712,
    *[14.25062309787429] * 10,
]
G1_AMP_RECOVERY_23_DAMPING = [
    *[2.557889775413375, 6.308801853496639, 2.557889775413375, 6.308801853496639, 1.814445686584846, 1.814445686584846],
    *[2.557889775413375, 6.308801853496639, 2.557889775413375, 6.308801853496639, 1.814445686584846, 1.814445686584846],
    2.557889775413375,
    *[0.907222843292423] * 10,
]
G1_AMP_RECOVERY_23_EFFORT_LIMITS = [
    *[88.0, 139.0, 88.0, 139.0, 50.0, 50.0],
    *[88.0, 139.0, 88.0, 139.0, 50.0, 50.0],
    88.0,
    *[25.0] * 10,
]
G1_AMP_RECOVERY_23_ACTION_SCALES = [
    0.25 * effort / stiffness
    for effort, stiffness in zip(
        G1_AMP_RECOVERY_23_EFFORT_LIMITS,
        G1_AMP_RECOVERY_23_STIFFNESS,
        strict=True,
    )
]


class G1AmpRecovery29DoF(DoFConfig):
    joint_names: list[str] = _G1_29_DOF.joint_names
    default_pos: list[float] | None = G1_AMP_RECOVERY_29_DEFAULT_POS
    stiffness: list[float] | None = G1_AMP_RECOVERY_29_STIFFNESS
    damping: list[float] | None = G1_AMP_RECOVERY_29_DAMPING
    torque_limits: list[float] | None = G1_AMP_RECOVERY_29_EFFORT_LIMITS
    position_limits: list[list[float]] | None = _G1_29_DOF.position_limits


class G1AmpRecovery23DoF(DoFConfig):
    joint_names: list[str] = _G1_23_JOINT_NAMES
    default_pos: list[float] | None = G1_AMP_RECOVERY_23_DEFAULT_POS
    stiffness: list[float] | None = G1_AMP_RECOVERY_23_STIFFNESS
    damping: list[float] | None = G1_AMP_RECOVERY_23_DAMPING
    torque_limits: list[float] | None = G1_AMP_RECOVERY_23_EFFORT_LIMITS
    position_limits: list[list[float]] | None = [_G1_POSITION_LIMITS[name] for name in joint_names]


class G1AmpRecoveryPolicyCfg(AmpRecoveryPolicyCfg):
    robot: str = "g1"
    policy_name: str = "policy_29dof"
    obs_dof: DoFConfig = G1AmpRecovery29DoF()
    action_dof: DoFConfig = obs_dof
    num_obs: int = 384
    action_scales: list[float] = G1_AMP_RECOVERY_29_ACTION_SCALES


class G1_23AmpRecoveryPolicyCfg(AmpRecoveryPolicyCfg):
    robot: str = "g1"
    policy_name: str = "policy_23dof"
    obs_dof: DoFConfig = G1AmpRecovery23DoF()
    action_dof: DoFConfig = obs_dof
    num_obs: int = 312
    action_scales: list[float] = G1_AMP_RECOVERY_23_ACTION_SCALES
