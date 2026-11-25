from pydantic import model_validator

from robojudo.environment.env_cfgs import MujocoEnvCfg, ObserverMujocoEnvCfg

from .g1_env_cfg import G1_23_ARM_JOINT_NAMES, G1_29_ARM_JOINT_NAMES, G1_12EnvCfg, G1_23EnvCfg, G1EnvCfg


class G1MujocoEnvCfg(G1EnvCfg, MujocoEnvCfg):
    env_type: str = MujocoEnvCfg.model_fields["env_type"].default
    is_sim: bool = MujocoEnvCfg.model_fields["is_sim"].default
    # ====== ENV CONFIGURATION ======

    update_with_fk: bool = True


class G1ObserverMujocoEnvCfg(G1MujocoEnvCfg, ObserverMujocoEnvCfg):
    env_type: str = "ObserverMujocoEnv"

    @property
    def feedforward_joint_names(self) -> list[str]:
        return list(G1_29_ARM_JOINT_NAMES)

    @model_validator(mode="after")
    def check_arm_feedforward_limits(self):
        if set(self.feedforward_effort_limits) - set(G1_29_ARM_JOINT_NAMES):
            raise ValueError("G1 29-DOF feedforward is restricted to the 14 arm joints")
        return self


class G1_23MujocoEnvCfg(G1_23EnvCfg, MujocoEnvCfg):
    env_type: str = MujocoEnvCfg.model_fields["env_type"].default
    is_sim: bool = MujocoEnvCfg.model_fields["is_sim"].default
    # ====== ENV CONFIGURATION ======
    update_with_fk: bool = True

    @property
    def feedforward_joint_names(self) -> list[str]:
        return list(G1_23_ARM_JOINT_NAMES)

    @model_validator(mode="after")
    def check_arm_feedforward_limits(self):
        if set(self.feedforward_effort_limits) - set(G1_23_ARM_JOINT_NAMES):
            raise ValueError("G1 23-DOF feedforward is restricted to the 10 arm joints")
        return self


class G1_23ObserverMujocoEnvCfg(G1_23MujocoEnvCfg, ObserverMujocoEnvCfg):
    env_type: str = "ObserverMujocoEnv"


class G1_12MujocoEnvCfg(G1_12EnvCfg, MujocoEnvCfg):
    env_type: str = MujocoEnvCfg.model_fields["env_type"].default
    is_sim: bool = MujocoEnvCfg.model_fields["is_sim"].default
    # ====== ENV CONFIGURATION ======
    update_with_fk: bool = False
