from typing import Literal

from pydantic import Field

from robojudo.environment.env_cfgs import AgiBotEnvCfg

from .x2_env_cfg import (
    X2_ARM_JOINT_NAMES,
    X2_HEAD_JOINT_NAMES,
    X2_LEG_JOINT_NAMES,
    X2_WAIST_JOINT_NAMES,
    X2EnvCfg,
)


class X2AimdkCfg(AgiBotEnvCfg.AimdkCfg):
    control_dt: float = Field(default=0.02, gt=0.0, allow_inf_nan=False)
    publish_dt: float = Field(default=0.002, gt=0.0, allow_inf_nan=False)
    command_timeout: float = Field(default=0.1, gt=0.0, allow_inf_nan=False)
    command_damping_timeout: float = Field(default=0.5, gt=0.0, allow_inf_nan=False)
    shutdown_damping: float = Field(default=5.0, ge=0.0, allow_inf_nan=False)
    shutdown_publish_duration: float = Field(default=0.2, ge=0.0, allow_inf_nan=False)
    state_timeout: float = Field(default=0.1, gt=0.0, allow_inf_nan=False)
    state_damping_timeout: float = Field(default=0.5, gt=0.0, allow_inf_nan=False)
    odometry_timeout: float = Field(default=0.1, gt=0.0, allow_inf_nan=False)
    odometry_damping_timeout: float = Field(default=0.5, gt=0.0, allow_inf_nan=False)
    # AimDK calls this the "torso" IMU, but its deployment documentation
    # identifies it as the hip/pelvis IMU used for RL base observations.
    base_imu_topic: str = "/aima/hal/imu/torso/state"


class X2SuperOdomCfg(X2AimdkCfg):
    odometry_timeout: float = Field(default=0.3, gt=0.0, allow_inf_nan=False)
    odometry_topic: str = "/laser_odometry"
    odometry_parent_frame: str = "map"
    odometry_child_frame: str = "lidar_chest_front"
    # SuperOdom initializes the LiDAR translation at zero. Rebase its converted
    # pelvis pose so policies consume displacement, not a fictitious ground height.
    odometry_position_mode: Literal["ABSOLUTE", "RELATIVE_START"] = "RELATIVE_START"
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
