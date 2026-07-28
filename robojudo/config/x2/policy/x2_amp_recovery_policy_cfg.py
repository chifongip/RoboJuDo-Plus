from robojudo.policy.policy_cfgs import AmpRecoveryPolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig

from ..env.x2_env_cfg import X2_POSITION_LIMITS_BY_NAME

X2_AMP_RECOVERY_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_pitch_joint",
    "waist_roll_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
]
X2_AMP_RECOVERY_DEFAULT_POS = [
    *[-0.1, 0.0, 0.0, 0.3, -0.2, 0.0],
    *[-0.1, 0.0, 0.0, 0.3, -0.2, 0.0],
    *[0.0, 0.0, 0.0],
    *[0.35, 0.1, 0.0, -0.87, 0.0, 0.0, 0.0],
    *[0.35, -0.1, 0.0, -0.87, 0.0, 0.0, 0.0],
]
X2_AMP_RECOVERY_STIFFNESS = [
    *[120.0, 100.0, 100.0, 150.0, 40.0, 40.0],
    *[120.0, 100.0, 100.0, 150.0, 40.0, 40.0],
    *[40.18, 200.0, 200.0],
    *[50.0, 50.0, 50.0, 50.0, 20.0, 20.0, 20.0],
    *[50.0, 50.0, 50.0, 50.0, 20.0, 20.0, 20.0],
]
X2_AMP_RECOVERY_DAMPING = [
    *[5.0, 4.0, 4.0, 5.0, 2.0, 2.0],
    *[5.0, 4.0, 4.0, 5.0, 2.0, 2.0],
    *[2.56, 2.0, 2.0],
    *[3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0],
    *[3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0],
]
X2_AMP_RECOVERY_EFFORT_LIMITS = [
    *[118.0, 118.0, 118.0, 118.0, 36.0, 24.0],
    *[118.0, 118.0, 118.0, 118.0, 36.0, 24.0],
    *[118.0, 48.0, 48.0],
    *[36.0, 36.0, 24.0, 24.0, 24.0, 2.2, 2.2],
    *[36.0, 36.0, 24.0, 24.0, 24.0, 2.2, 2.2],
]

X2_AMP_RECOVERY_ACTION_SCALES = [
    0.25 * effort / stiffness
    for effort, stiffness in zip(
        X2_AMP_RECOVERY_EFFORT_LIMITS,
        X2_AMP_RECOVERY_STIFFNESS,
        strict=True,
    )
]


class X2AmpRecoveryDoF(DoFConfig):
    joint_names: list[str] = X2_AMP_RECOVERY_JOINT_NAMES
    default_pos: list[float] | None = X2_AMP_RECOVERY_DEFAULT_POS
    stiffness: list[float] | None = X2_AMP_RECOVERY_STIFFNESS
    damping: list[float] | None = X2_AMP_RECOVERY_DAMPING
    torque_limits: list[float] | None = X2_AMP_RECOVERY_EFFORT_LIMITS
    position_limits: list[list[float]] | None = [X2_POSITION_LIMITS_BY_NAME[name] for name in joint_names]


class X2AmpRecoveryPolicyCfg(AmpRecoveryPolicyCfg):
    robot: str = "x2"
    policy_name: str = "policy"
    obs_dof: DoFConfig = X2AmpRecoveryDoF()
    action_dof: DoFConfig = obs_dof
    num_obs: int = 384
    action_scales: list[float] = X2_AMP_RECOVERY_ACTION_SCALES
