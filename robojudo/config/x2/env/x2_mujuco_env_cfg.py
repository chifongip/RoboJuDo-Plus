from pydantic import model_validator

from robojudo.environment.env_cfgs import ElasticBandCfg, MujocoEnvCfg, ObserverMujocoEnvCfg

from .x2_env_cfg import X2_ARM_JOINT_NAMES, X2EnvCfg


class X2MujocoEnvCfg(X2EnvCfg, MujocoEnvCfg):
    env_type: str = MujocoEnvCfg.model_fields["env_type"].default
    is_sim: bool = MujocoEnvCfg.model_fields["is_sim"].default
    update_with_fk: bool = True
    elastic_band: ElasticBandCfg = ElasticBandCfg(body_name="torso_link")

    @property
    def feedforward_joint_names(self) -> list[str]:
        return X2_ARM_JOINT_NAMES

    @model_validator(mode="after")
    def check_arm_feedforward_limits(self):
        if set(self.feedforward_effort_limits) - set(X2_ARM_JOINT_NAMES):
            raise ValueError("X2 feedforward is restricted to the 14 arm joints")
        return self


class X2ObserverMujocoEnvCfg(X2MujocoEnvCfg, ObserverMujocoEnvCfg):
    env_type: str = "ObserverMujocoEnv"
