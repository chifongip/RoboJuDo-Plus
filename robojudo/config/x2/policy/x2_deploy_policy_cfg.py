from robojudo.config import ASSETS_DIR
from robojudo.policy.policy_cfgs import PolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig

from ..env.x2_env_cfg import X2_POSITION_LIMITS_BY_NAME

X2_POLICY_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_roll_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
]
X2_POLICY_DEFAULT_POS = [
    -0.3120,
    -0.3120,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.6690,
    0.6690,
    0.2000,
    0.2000,
    -0.3630,
    -0.3630,
    0.2000,
    -0.2000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    -0.3000,
    -0.3000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
]
X2_POLICY_KPS = [
    120.0,
    120.0,
    40.1792,
    100.0,
    100.0,
    200.0,
    100.0,
    100.0,
    200.0,
    150.0,
    150.0,
    50.0,
    50.0,
    40.0,
    40.0,
    50.0,
    50.0,
    40.0,
    40.0,
    50.0,
    50.0,
    50.0,
    50.0,
    20.0,
    20.0,
    20.0,
    20.0,
    20.0,
    20.0,
]
X2_POLICY_KDS = [
    5.0,
    5.0,
    2.5579,
    4.0,
    4.0,
    2.0,
    4.0,
    4.0,
    2.0,
    5.0,
    5.0,
    3.0,
    3.0,
    2.0,
    2.0,
    3.0,
    3.0,
    2.0,
    2.0,
    3.0,
    3.0,
    3.0,
    3.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
]


class X2DeployDoF(DoFConfig):
    joint_names: list[str] = X2_POLICY_JOINT_NAMES
    default_pos: list[float] | None = X2_POLICY_DEFAULT_POS
    stiffness: list[float] | None = X2_POLICY_KPS
    damping: list[float] | None = X2_POLICY_KDS
    position_limits: list[list[float]] | None = [X2_POSITION_LIMITS_BY_NAME[name] for name in joint_names]


class X2DeployPolicyCfg(PolicyCfg):
    policy_type: str = "X2DeployPolicy"
    robot: str = "x2"
    policy_name: str = "kuailechongbai"
    disable_autoload: bool = True
    freq: int = 50

    obs_dof: DoFConfig = X2DeployDoF()
    action_dof: DoFConfig = X2DeployDoF()
    action_scale: float = 0.25
    action_clip: float | None = 100.0
    action_beta: float = 1.0

    num_obs: int = 151
    obs_clip: float = 100.0
    warmup_frames: int = 5
    phase_start_count: float = 1.0
    phase_end_count: float = 2820.0

    obs_scales: dict[str, float] = {
        "ang_vel": 0.25,
        "dof_pos": 1.0,
        "dof_vel": 0.05,
        "actions": 1.0,
    }

    @property
    def policy_file(self) -> str:
        return (ASSETS_DIR / f"models/x2/x2_rl_deploy/{self.policy_name}.onnx").as_posix()
