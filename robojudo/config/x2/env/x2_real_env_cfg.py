from typing import Literal

from pydantic import Field, model_validator

from robojudo.config import Config
from robojudo.environment.env_cfgs import AgiBotEnvCfg

from .x2_env_cfg import (
    X2_ARM_JOINT_NAMES,
    X2_HEAD_JOINT_NAMES,
    X2_LEG_JOINT_NAMES,
    X2_WAIST_JOINT_NAMES,
    X2EnvCfg,
)


class X2ArmObserverCfg(Config):
    """Active limits must come from deployment qualification, never simulation defaults.

    All vectors use the frozen reference's left-seven then right-seven order.
    """

    mode: Literal["OFF", "SHADOW", "ACTIVE"] = "OFF"
    actuator_limits_nm: list[float] | None = None
    correction_limits_nm: list[float] | None = None
    slew_nm_s: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    # None preserves the growth rate for removal. Faster release never enables
    # faster growth or an immediate change of correction sign.
    release_slew_nm_s: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_limits(self):
        import math

        for values in (self.actuator_limits_nm, self.correction_limits_nm):
            if values is not None and (len(values) != 14 or any(not math.isfinite(v) or v <= 0 for v in values)):
                raise ValueError("Observer limits must be 14 positive finite values in reference arm order")
        if self.mode == "ACTIVE" and any(
            value is None for value in (self.actuator_limits_nm, self.correction_limits_nm, self.slew_nm_s)
        ):
            raise ValueError("ACTIVE requires explicit hardware actuator/correction limits and slew")
        if self.actuator_limits_nm is not None and self.correction_limits_nm is not None:
            if any(c > a for c, a in zip(self.correction_limits_nm, self.actuator_limits_nm, strict=True)):
                raise ValueError("Correction limits must not exceed actuator limits")
        return self


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
    arm_observer: X2ArmObserverCfg = Field(default_factory=X2ArmObserverCfg)
    aimdk: AgiBotEnvCfg.AimdkCfg = X2AimdkCfg()
    odometry_type: Literal["NONE", "DUMMY", "AIMDK", "SUPERODOM"] = "AIMDK"

    leg_joint_names: list[str] = X2_LEG_JOINT_NAMES
    waist_joint_names: list[str] = X2_WAIST_JOINT_NAMES
    arm_joint_names: list[str] = X2_ARM_JOINT_NAMES
    head_joint_names: list[str] = X2_HEAD_JOINT_NAMES
