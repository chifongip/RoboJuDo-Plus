from typing import Literal

from pydantic import model_validator

from robojudo.config.observer_cfg import JointObserverCfg
from robojudo.environment.env_cfgs import UnitreeEnvCfg

# from robojudo.tools.tool_cfgs import ZedOdometryCfg
from .g1_env_cfg import (
    G1_23_ARM_JOINT_NAMES,
    G1_23_DOF_INDICES,
    G1_29_ARM_JOINT_NAMES,
    G1_23DoF,
    G1_23EnvCfg,
    G1_29DoF,
    G1EnvCfg,
)


class G1UnitreeCfg(UnitreeEnvCfg.UnitreeCfg):
    robot: Literal["h1", "g1"] = "g1"

    msg_type: Literal["hg", "go"] = "hg"
    hand_type: Literal["Dex-3", "Inspire", "NONE"] = "NONE"

    enable_odometry: bool = True


class G1RealEnvCfg(G1EnvCfg, UnitreeEnvCfg):
    # env_type: str = UnitreeEnvCfg.model_fields["env_type"].default
    env_type: str = "UnitreeCppEnv"
    # ====== ENV CONFIGURATION ======
    unitree: UnitreeEnvCfg.UnitreeCfg = G1UnitreeCfg(
        net_if="eth0",
    )

    odometry_type: Literal["NONE", "DUMMY", "UNITREE", "ZED"] = "UNITREE"

    joint2motor_idx: list[int] | None = None  # list(range(0, 29))


class G1_23RealEnvCfg(G1_23EnvCfg, UnitreeEnvCfg):
    """Logical 23-DOF G1 using the standard 29-slot Unitree motor transport."""

    env_type: str = "UnitreeCppEnv"
    unitree: UnitreeEnvCfg.UnitreeCfg = G1UnitreeCfg(
        net_if="eth0",
    )
    odometry_type: Literal["NONE", "DUMMY", "UNITREE", "ZED"] = "UNITREE"
    motor_dof_count: int = 29
    joint2motor_idx: list[int] = G1_23_DOF_INDICES.copy()


class _G1ObserverRealEnvCfg(UnitreeEnvCfg):
    observer: JointObserverCfg

    @model_validator(mode="after")
    def check_observer_transport(self):
        motor_count = self.motor_dof_count or self.dof.num_dofs
        if (
            self.env_type != "UnitreeCppEnv"
            or self.unitree.robot != "g1"
            or self.unitree.msg_type != "hg"
            or self.unitree.control_mode != "position"
            or motor_count != 29
            or self.unitree.command_timeout <= 0
            or self.unitree.state_timeout <= 0
        ):
            raise ValueError("G1 observer requires the 29-slot Unitree position transport with both watchdogs")
        if self.dof.joint_names == G1_23DoF().joint_names and self.joint2motor_idx == G1_23_DOF_INDICES:
            arm_joints = G1_23_ARM_JOINT_NAMES
        elif self.dof.joint_names == G1_29DoF().joint_names and self.joint2motor_idx in (None, list(range(29))):
            arm_joints = G1_29_ARM_JOINT_NAMES
        else:
            raise ValueError("G1 observer requires the standard 23- or 29-DOF motor mapping")
        if set(self.observer.joint_names) != set(arm_joints):
            raise ValueError("G1 observer joints must match the arm joints for its DOF layout")
        self.observer.validate_hardware()
        return self


class G1ObserverRealEnvCfg(G1RealEnvCfg, _G1ObserverRealEnvCfg):
    """Real 29-DOF G1 with a configured arm observer."""


class G1_23ObserverRealEnvCfg(G1_23RealEnvCfg, _G1ObserverRealEnvCfg):
    """Real 23-DOF G1 with a configured arm observer."""


class G1WithHandRealEnvCfg(G1EnvCfg, UnitreeEnvCfg):
    # env_type: str = UnitreeEnvCfg.model_fields["env_type"].default
    env_type: str = "UnitreeCppEnv"
    # ====== ENV CONFIGURATION ======
    unitree: UnitreeEnvCfg.UnitreeCfg = G1UnitreeCfg(
        net_if="eth0",
        hand_type="Dex-3",
    )

    odometry_type: Literal["NONE", "DUMMY", "UNITREE", "ZED"] = "DUMMY"
    # zed_cfg: ZedOdometryCfg | None = ZedOdometryCfg(
    #     server_ip="192.168.123.167",
    #     pos_offset=[0.0, 0.0, 0.9],
    #     zero_align=True,
    # )

    joint2motor_idx: list[int] | None = None  # list(range(0, 29))
