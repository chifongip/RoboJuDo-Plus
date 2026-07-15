from robojudo.environment.env_cfgs import ElasticBandCfg, MujocoEnvCfg

from .x2_env_cfg import X2EnvCfg


class X2MujocoEnvCfg(X2EnvCfg, MujocoEnvCfg):
    env_type: str = MujocoEnvCfg.model_fields["env_type"].default
    is_sim: bool = MujocoEnvCfg.model_fields["is_sim"].default
    update_with_fk: bool = True
    elastic_band: ElasticBandCfg = ElasticBandCfg(body_name="torso_link")
