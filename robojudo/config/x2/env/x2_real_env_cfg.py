from typing import Literal

from robojudo.environment.env_cfgs import AgiBotEnvCfg

from .x2_env_cfg import (
    X2_ARM_JOINT_NAMES,
    X2_HEAD_JOINT_NAMES,
    X2_LEG_JOINT_NAMES,
    X2_WAIST_JOINT_NAMES,
    X2EnvCfg,
)


class X2AimdkCfg(AgiBotEnvCfg.AimdkCfg):
    control_dt: float = 0.02
    publish_dt: float = 0.002
    command_timeout: float = 0.1
    shutdown_damping: float = 5.0
    shutdown_publish_duration: float = 0.2
    state_timeout: float = 0.1
    odometry_timeout: float = 0.1
    # AimDK calls this the "torso" IMU, but its deployment documentation
    # identifies it as the hip/pelvis IMU used for RL base observations.
    base_imu_topic: str = "/aima/hal/imu/torso/state"


class X2SuperOdomCfg(X2AimdkCfg):
    odometry_timeout: float = 0.3
    odometry_topic: str = "/laser_odometry"
    odometry_parent_frame: str = "map"
    odometry_child_frame: str = "lidar_chest_front"
    torso_to_odometry_sensor_position: list[float] = [
        0.102632855873251,
        0.0,
        0.181586916322065,
    ]
    torso_to_odometry_sensor_quaternion: list[float] = [
        -0.7071067811865476,
        0.0,
        0.0,
        0.7071067811865476,
    ]


class X2RealEnvCfg(X2EnvCfg, AgiBotEnvCfg):
    env_type: str = "AgiBotCppEnv"
    aimdk: AgiBotEnvCfg.AimdkCfg = X2AimdkCfg()
    odometry_type: Literal["NONE", "DUMMY", "AIMDK", "SUPERODOM"] = "AIMDK"

    leg_joint_names: list[str] = X2_LEG_JOINT_NAMES
    waist_joint_names: list[str] = X2_WAIST_JOINT_NAMES
    arm_joint_names: list[str] = X2_ARM_JOINT_NAMES
    head_joint_names: list[str] = X2_HEAD_JOINT_NAMES
