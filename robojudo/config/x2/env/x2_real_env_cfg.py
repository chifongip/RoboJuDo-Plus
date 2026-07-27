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
    base_imu_topic: str = "/aima/hal/imu/torso/state"


class X2RealEnvCfg(X2EnvCfg, AgiBotEnvCfg):
    env_type: str = "AgiBotCppEnv"
    aimdk: AgiBotEnvCfg.AimdkCfg = X2AimdkCfg()

    leg_joint_names: list[str] = X2_LEG_JOINT_NAMES
    waist_joint_names: list[str] = X2_WAIST_JOINT_NAMES
    arm_joint_names: list[str] = X2_ARM_JOINT_NAMES
    head_joint_names: list[str] = X2_HEAD_JOINT_NAMES
