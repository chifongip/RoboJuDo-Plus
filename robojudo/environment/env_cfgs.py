from typing import Literal

from pydantic import Field, model_validator

from robojudo.config import Config
from robojudo.tools.tool_cfgs import DoFConfig, ForwardKinematicCfg, ZedOdometryCfg


class EnvCfg(Config):
    env_type: str  # name of the environment class
    is_sim: bool = False

    urdf: str | None = None
    xml: str
    body_names: list[str] | None = None

    dof: DoFConfig

    forward_kinematic: ForwardKinematicCfg | None = None
    update_with_fk: bool = False
    """Whether to update info from fk"""
    torso_name: str = "torso_link"
    """Name of the torso link, used in fk info extraction"""

    born_place_align: bool = True
    """Whether to align the born place to zero position and heading"""


class ElasticBandCfg(Config):
    """Virtual tension band applied to one MuJoCo body."""

    body_name: str
    anchor_point: tuple[float, float, float] = (0.0, 0.0, 3.0)
    stiffness: float = Field(default=200.0, ge=0.0)
    damping: float = Field(default=100.0, ge=0.0)
    rest_length: float = Field(default=0.0, ge=0.0)
    length_step: float = Field(default=0.1, gt=0.0)
    active: bool = True
    visualize: bool = True
    visual_radius: float = Field(default=0.015, gt=0.0)
    visual_rgba: tuple[float, float, float, float] = (0.95, 0.2, 0.05, 1.0)
    anchor_radius: float = Field(default=0.04, gt=0.0)


class MujocoEnvCfg(EnvCfg):
    env_type: str = "MujocoEnv"
    is_sim: bool = True
    # ====== ENV CONFIGURATION ======
    sim_duration: float = 60.0
    sim_dt: float = 0.001
    sim_decimation: int = 20

    visualize_extras: bool = True  # TODO: remove

    random_heading: bool = False
    """Randomize the robot's yaw heading on each spawn/reborn (useful for testing heading alignment)."""

    elastic_band: ElasticBandCfg | None = None
    """Optional simulation-only suspension band."""


class RobotEnvCfg(EnvCfg):
    env_type: str = "DummyEnv"
    is_sim: bool = False
    # ====== ENV CONFIGURATION ======
    act: bool = True

    odometry_type: Literal["NONE", "DUMMY", "ZED"] = "NONE"
    zed_cfg: ZedOdometryCfg | None = None
    """ZED odometry config, if odometry_type is "ZED", this must be set"""

    @model_validator(mode="after")
    def check_zed_config(self):
        if self.odometry_type == "ZED" and self.zed_cfg is None:
            raise ValueError("zed_cfg must be set if odometry_type is 'ZED'")
        return self


class UnitreeEnvCfg(RobotEnvCfg):
    """
    Configuration for Unitree Robot environment.
    """

    class UnitreeCfg(Config):
        """Unitree SDK configuration"""

        net_if: str = "eth0"
        """network interface to communicate with the robot"""

        robot: Literal["h1", "g1"]
        msg_type: Literal["hg", "go"]
        control_mode: str = "position"
        hand_type: Literal["Dex-3", "Inspire", "NONE"] = "NONE"

        lowcmd_topic: str = "rt/lowcmd"
        lowstate_topic: str = "rt/lowstate"

        enable_odometry: bool = False
        sport_state_topic: str = "rt/odommodestate"

        control_dt: float = 0.02
        """control command dt"""

    env_type: str = "UnitreeEnv"  # For unitree_sdk2py
    # env_type: str = "UnitreeCppEnv" # For unitree_cpp
    """UnitreeEnv for unitree_sdk2py, UnitreeCppEnv for unitree_cpp, check README for more details"""

    unitree: UnitreeCfg

    odometry_type: Literal["NONE", "DUMMY", "UNITREE", "ZED"] = "DUMMY"  # pyright: ignore[reportIncompatibleVariableOverride]

    joint2motor_idx: list[int] | None = None
    """Mapping from env dof to motor index, None for direct mapping"""
    weak_motor: list[int] = []

    hand_retarget: None = None  # TODO

    @model_validator(mode="after")
    def check_joint2motor_idx(self):
        if self.joint2motor_idx is not None and len(self.joint2motor_idx) != self.dof.num_dofs:
            raise ValueError("joint2motor_idx length must match dof.num_dofs")
        return self


class AgiBotEnvCfg(RobotEnvCfg):
    """
    Configuration for AgiBot/AimDK ROS 2 robot environments.
    """

    class AimdkCfg(Config):
        """AimDK ROS 2 topic and timing configuration."""

        control_dt: float = 0.02
        publish_dt: float = 0.002
        command_timeout: float = 0.1
        shutdown_damping: float = 5.0
        shutdown_publish_duration: float = 0.2
        state_timeout: float = 0.1
        base_imu_topic: str = "/aima/hal/imu/torso/state"

        leg_state_topic: str = "/aima/hal/joint/leg/state"
        waist_state_topic: str = "/aima/hal/joint/waist/state"
        arm_state_topic: str = "/aima/hal/joint/arm/state"
        head_state_topic: str = "/aima/hal/joint/head/state"

        leg_command_topic: str = "/aima/hal/joint/leg/command"
        waist_command_topic: str = "/aima/hal/joint/waist/command"
        arm_command_topic: str = "/aima/hal/joint/arm/command"
        head_command_topic: str = "/aima/hal/joint/head/command"

    env_type: str = "AgiBotCppEnv"
    aimdk: AimdkCfg = AimdkCfg()

    leg_joint_names: list[str] = []
    waist_joint_names: list[str] = []
    arm_joint_names: list[str] = []
    head_joint_names: list[str] = []
