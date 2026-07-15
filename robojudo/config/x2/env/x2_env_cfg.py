from robojudo.config import ASSETS_DIR
from robojudo.environment.env_cfgs import EnvCfg
from robojudo.tools.tool_cfgs import DoFConfig, ForwardKinematicCfg

X2_LEG_JOINT_NAMES = [
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
]
X2_WAIST_JOINT_NAMES = [
    "waist_yaw_joint",
    "waist_pitch_joint",
    "waist_roll_joint",
]
X2_ARM_JOINT_NAMES = [
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
X2_HEAD_JOINT_NAMES = [
    "head_yaw_joint",
    "head_pitch_joint",
]
X2_FULL_JOINT_NAMES = [
    *X2_LEG_JOINT_NAMES,
    *X2_WAIST_JOINT_NAMES,
    *X2_ARM_JOINT_NAMES,
    *X2_HEAD_JOINT_NAMES,
]

X2_POSITION_LIMITS_BY_NAME = {
    "left_hip_pitch_joint": [-2.704, 2.556],
    "left_hip_roll_joint": [-0.235, 2.906],
    "left_hip_yaw_joint": [-1.684, 3.430],
    "left_knee_joint": [0.0, 2.4073],
    "left_ankle_pitch_joint": [-0.803, 0.453],
    "left_ankle_roll_joint": [-0.2625, 0.2625],
    "right_hip_pitch_joint": [-2.704, 2.556],
    "right_hip_roll_joint": [-2.906, 0.235],
    "right_hip_yaw_joint": [-3.430, 1.684],
    "right_knee_joint": [0.0, 2.4073],
    "right_ankle_pitch_joint": [-0.803, 0.453],
    "right_ankle_roll_joint": [-0.2625, 0.2625],
    "waist_yaw_joint": [-3.43, 2.382],
    "waist_pitch_joint": [-0.314, 0.314],
    "waist_roll_joint": [-0.488, 0.488],
    "left_shoulder_pitch_joint": [-3.08, 2.04],
    "left_shoulder_roll_joint": [-0.061, 2.993],
    "left_shoulder_yaw_joint": [-2.556, 2.556],
    "left_elbow_joint": [-2.3556, 0.0],
    "left_wrist_yaw_joint": [-2.556, 2.556],
    "left_wrist_pitch_joint": [-0.558, 0.558],
    "left_wrist_roll_joint": [-1.571, 0.724],
    "right_shoulder_pitch_joint": [-3.08, 2.04],
    "right_shoulder_roll_joint": [-2.993, 0.061],
    "right_shoulder_yaw_joint": [-2.556, 2.556],
    "right_elbow_joint": [-2.3556, 0.0],
    "right_wrist_yaw_joint": [-2.556, 2.556],
    "right_wrist_pitch_joint": [-0.558, 0.558],
    "right_wrist_roll_joint": [-0.724, 1.571],
    "head_yaw_joint": [-0.366, 0.366],
    "head_pitch_joint": [-0.3838, 0.3838],
}

X2_JOINT_DEFAULT_POSITION_BY_NAME = {
    "left_hip_pitch_joint": -0.05,
    "left_knee_joint": 0.1,
    "left_ankle_pitch_joint": -0.05,
    "right_hip_pitch_joint": -0.05,
    "right_knee_joint": 0.1,
    "right_ankle_pitch_joint": -0.05,
    "left_shoulder_pitch_joint": 0.4,
    "right_shoulder_pitch_joint": 0.4,
    "left_elbow_joint": -1.2,
    "right_elbow_joint": -1.2,
}
X2_JOINT_DEFAULT_GAINS_BY_NAME = {
    "left_hip_pitch_joint": (40.0, 4.0),
    "left_hip_roll_joint": (40.0, 4.0),
    "left_hip_yaw_joint": (30.0, 3.0),
    "left_knee_joint": (80.0, 8.0),
    "left_ankle_pitch_joint": (40.0, 4.0),
    "left_ankle_roll_joint": (20.0, 2.0),
    "right_hip_pitch_joint": (40.0, 4.0),
    "right_hip_roll_joint": (40.0, 4.0),
    "right_hip_yaw_joint": (30.0, 3.0),
    "right_knee_joint": (80.0, 8.0),
    "right_ankle_pitch_joint": (40.0, 4.0),
    "right_ankle_roll_joint": (20.0, 2.0),
    "left_shoulder_pitch_joint": (30.0, 1.0),
    "right_shoulder_pitch_joint": (30.0, 1.0),
    "left_elbow_joint": (50.0, 1.0),
    "right_elbow_joint": (50.0, 1.0),
    "left_wrist_yaw_joint": (50.0, 1.0),
    "right_wrist_yaw_joint": (50.0, 1.0),
    "waist_yaw_joint": (150.0, 3.0),
    "waist_pitch_joint": (300.0, 3.0),
    "waist_roll_joint": (300.0, 3.0),
}


class X2_31DoF(DoFConfig):
    joint_names: list[str] = X2_FULL_JOINT_NAMES
    default_pos: list[float] | None = [
        *[-0.24, 0.0, 0.0, 0.45, -0.21, 0.0],
        *[-0.24, 0.0, 0.0, 0.45, -0.21, 0.0],
        *[0.0, 0.0, 0.0],
        *[0.196, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        *[0.196, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        *[0.0, 0.0],
    ]
    stiffness: list[float] | None = [
        *[40.0, 40.0, 30.0, 80.0, 40.0, 20.0],
        *[40.0, 40.0, 30.0, 80.0, 40.0, 20.0],
        *[20.0, 20.0, 20.0],
        *[20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
        *[20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
        *[20.0, 20.0],
    ]
    damping: list[float] | None = [
        *[4.0, 4.0, 3.0, 8.0, 4.0, 2.0],
        *[4.0, 4.0, 3.0, 8.0, 4.0, 2.0],
        *[4.0, 4.0, 4.0],
        *[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        *[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        *[2.0, 2.0],
    ]
    torque_limits: list[float] | None = [
        *[120.0, 120.0, 120.0, 120.0, 36.0, 24.0],
        *[120.0, 120.0, 120.0, 120.0, 36.0, 24.0],
        *[120.0, 48.0, 48.0],
        *[36.0, 36.0, 24.0, 24.0, 24.0, 4.8, 4.8],
        *[36.0, 36.0, 24.0, 24.0, 24.0, 4.8, 4.8],
        *[2.6, 0.6],
    ]
    position_limits: list[list[float]] | None = [X2_POSITION_LIMITS_BY_NAME[name] for name in joint_names]


class X2JointDefaultDoF(DoFConfig):
    """Preparation pose and gains used by the working standalone X2 controller."""

    joint_names: list[str] = X2_FULL_JOINT_NAMES
    default_pos: list[float] | None = [X2_JOINT_DEFAULT_POSITION_BY_NAME.get(name, 0.0) for name in joint_names]
    stiffness: list[float] | None = [X2_JOINT_DEFAULT_GAINS_BY_NAME.get(name, (20.0, 1.0))[0] for name in joint_names]
    damping: list[float] | None = [X2_JOINT_DEFAULT_GAINS_BY_NAME.get(name, (20.0, 1.0))[1] for name in joint_names]
    position_limits: list[list[float]] | None = [X2_POSITION_LIMITS_BY_NAME[name] for name in joint_names]


class X2EnvCfg(EnvCfg):
    xml: str = (ASSETS_DIR / "robots/x2/x2.xml").as_posix()
    dof: DoFConfig = X2_31DoF()
    forward_kinematic: ForwardKinematicCfg | None = ForwardKinematicCfg(
        xml_path=xml,
        debug_viz=False,
        kinematic_joint_names=dof.joint_names,
    )
    update_with_fk: bool = True
    torso_name: str = "torso_link"
