from robojudo.config import ASSETS_DIR
from robojudo.policy.policy_cfgs import PolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig

from ..env.x2_env_cfg import X2_HEAD_JOINT_NAMES, X2_POSITION_LIMITS_BY_NAME


# Deployment constants captured in the training run's params/env.yaml. Keep the
# full-precision YAML values here; the ONNX metadata rounds several entries.
X2_LOCOMANIPULATION_JOINT_NAMES = [
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

X2_LOCOMANIPULATION_ACTION_JOINT_NAMES = X2_LOCOMANIPULATION_JOINT_NAMES[:15]

X2_LOCOMANIPULATION_DEFAULT_POS = [
    -0.1,
    0.0,
    0.0,
    0.3,
    -0.2,
    0.0,
    -0.1,
    0.0,
    0.0,
    0.3,
    -0.2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.35,
    0.1,
    0.0,
    -0.87,
    0.0,
    0.0,
    0.0,
    0.35,
    -0.1,
    0.0,
    -0.87,
    0.0,
    0.0,
    0.0,
]

X2_LOCOMANIPULATION_STIFFNESS = [
    120.0,
    100.0,
    100.0,
    150.0,
    40.0,
    40.0,
    120.0,
    100.0,
    100.0,
    150.0,
    40.0,
    40.0,
    40.18,
    200.0,
    200.0,
    50.0,
    50.0,
    50.0,
    50.0,
    20.0,
    20.0,
    20.0,
    50.0,
    50.0,
    50.0,
    50.0,
    20.0,
    20.0,
    20.0,
]

X2_LOCOMANIPULATION_DAMPING = [
    5.0,
    4.0,
    4.0,
    5.0,
    2.0,
    2.0,
    5.0,
    4.0,
    4.0,
    5.0,
    2.0,
    2.0,
    2.56,
    2.0,
    2.0,
    3.0,
    3.0,
    3.0,
    3.0,
    2.0,
    2.0,
    2.0,
    3.0,
    3.0,
    3.0,
    3.0,
    2.0,
    2.0,
    2.0,
]

X2_LOCOMANIPULATION_EFFORT_LIMITS = [
    118.0,
    118.0,
    118.0,
    118.0,
    36.0,
    24.0,
    118.0,
    118.0,
    118.0,
    118.0,
    36.0,
    24.0,
    118.0,
    48.0,
    48.0,
    36.0,
    36.0,
    24.0,
    24.0,
    24.0,
    2.2,
    2.2,
    36.0,
    36.0,
    24.0,
    24.0,
    24.0,
    2.2,
    2.2,
]

X2_LOCOMANIPULATION_ACTION_SCALES = [
    0.24583333333333332,
    0.295,
    0.295,
    0.19666666666666666,
    0.225,
    0.15,
    0.24583333333333332,
    0.295,
    0.295,
    0.19666666666666666,
    0.225,
    0.15,
    0.7341961174713788,
    0.06,
    0.06,
]


class X2LocomanipulationObsDoF(DoFConfig):
    joint_names: list[str] = X2_LOCOMANIPULATION_JOINT_NAMES
    default_pos: list[float] | None = X2_LOCOMANIPULATION_DEFAULT_POS
    stiffness: list[float] | None = X2_LOCOMANIPULATION_STIFFNESS
    damping: list[float] | None = X2_LOCOMANIPULATION_DAMPING
    torque_limits: list[float] | None = X2_LOCOMANIPULATION_EFFORT_LIMITS
    position_limits: list[list[float]] | None = [X2_POSITION_LIMITS_BY_NAME[name] for name in joint_names]


class X2LocomanipulationActionDoF(DoFConfig):
    joint_names: list[str] = X2_LOCOMANIPULATION_ACTION_JOINT_NAMES
    default_pos: list[float] | None = X2_LOCOMANIPULATION_DEFAULT_POS[:15]
    stiffness: list[float] | None = X2_LOCOMANIPULATION_STIFFNESS[:15]
    damping: list[float] | None = X2_LOCOMANIPULATION_DAMPING[:15]
    torque_limits: list[float] | None = X2_LOCOMANIPULATION_EFFORT_LIMITS[:15]
    position_limits: list[list[float]] | None = [X2_POSITION_LIMITS_BY_NAME[name] for name in joint_names]


class X2LocomanipulationEnvDoF(DoFConfig):
    """Training DoF parameters plus the X2's two non-policy head joints."""

    joint_names: list[str] = [*X2_LOCOMANIPULATION_JOINT_NAMES, *X2_HEAD_JOINT_NAMES]
    default_pos: list[float] | None = [*X2_LOCOMANIPULATION_DEFAULT_POS, 0.0, 0.0]
    stiffness: list[float] | None = [*X2_LOCOMANIPULATION_STIFFNESS, 20.0, 20.0]
    damping: list[float] | None = [*X2_LOCOMANIPULATION_DAMPING, 2.0, 2.0]
    torque_limits: list[float] | None = [*X2_LOCOMANIPULATION_EFFORT_LIMITS, 2.6, 0.6]
    position_limits: list[list[float]] | None = [X2_POSITION_LIMITS_BY_NAME[name] for name in joint_names]


class X2LocomanipulationPolicyCfg(PolicyCfg):
    policy_type: str = "X2LocomanipulationPolicy"
    robot: str = "x2"
    policy_name: str = "policy"
    disable_autoload: bool = True
    freq: int = 50

    obs_dof: DoFConfig = X2LocomanipulationObsDoF()
    action_dof: DoFConfig = X2LocomanipulationActionDoF()
    action_scale: float = 1.0  # Per-joint action_scales are applied by the policy.
    action_clip: float | None = 100.0
    action_beta: float = 1.0

    num_obs: int = 430
    history_length: int = 5
    history_obs_dims: dict[str, int] = {
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "command": 3,
        "base_height_command": 1,
        "waist_yaw_command": 1,
        "phase": 2,
        "joint_pos": 29,
        "joint_vel": 29,
        "actions": 15,
    }
    action_scales: list[float] = X2_LOCOMANIPULATION_ACTION_SCALES

    gait_period: float = 0.6
    standing_command_threshold: float = 0.1
    command_decay: float = 0.95
    height_step: float = 0.02
    waist_yaw_step: float = 0.1
    commands_map: list[list[float]] = [
        [-0.5, 0.0, 1.0],
        [0.5, 0.0, -0.5],
        [1.0, 0.0, -1.0],
        [0.40, 0.64, 0.66],
        [-1.5708, 0.0, 1.5708],
    ]

    @property
    def policy_file(self) -> str:
        return (ASSETS_DIR / f"models/x2/locomanipulation/{self.policy_name}.onnx").as_posix()
