"""G1 Locomanipulation arm observer presets."""

from robojudo.config import cfg_registry
from robojudo.config.observer_cfg import JointObserverCfg
from robojudo.environment.env_cfgs import ElasticBandCfg

from .env.g1_mujuco_env_cfg import G1_23ObserverMujocoEnvCfg
from .env.g1_real_env_cfg import G1_23ObserverRealEnvCfg, G1UnitreeCfg
from .g1_loco_mimic_cfg import g1_23_locomanipulation_locomimic, g1_23_locomanipulation_locomimic_real
from .observer_parameters import JOINT_NAMES, selected_observer_parameters
from .policy.g1_locomanipulation_policy_cfg import G1Locomanipulation23ObsDoF

_DOF = G1Locomanipulation23ObsDoF.from_preset("stiff")
_ACTUATOR_LIMITS_BY_NAME = dict(zip(_DOF.joint_names, _DOF.torque_limits, strict=True))
_OBSERVER_PARAMETERS = selected_observer_parameters()


@cfg_registry.register
class g1_23_locomanipulation_locomimic_observer(g1_23_locomanipulation_locomimic):
    """Stiff 23-DOF G1 Locomanipulation with arm disturbance correction in MuJoCo."""

    env: G1_23ObserverMujocoEnvCfg = G1_23ObserverMujocoEnvCfg(
        dof=_DOF,
        sim_dt=0.005,
        sim_decimation=4,
        elastic_band=ElasticBandCfg(body_name="torso_link"),
        feedforward_effort_limits={name: _ACTUATOR_LIMITS_BY_NAME[name] for name in JOINT_NAMES},
        observer=JointObserverCfg(
            algorithm="nonlinear",
            mode="ACTIVE",
            joint_names=list(JOINT_NAMES),
            **_OBSERVER_PARAMETERS,
            actuator_limits_nm=[_ACTUATOR_LIMITS_BY_NAME[name] for name in JOINT_NAMES],
            correction_limit_fraction=0.8,
            correction_limit_nm=0.0,
            slew_nm_s=2.0,
            release_slew_nm_s=16.0,
        ),
    )


@cfg_registry.register
class g1_23_locomanipulation_locomimic_real_observer(g1_23_locomanipulation_locomimic_real):
    """Stiff 23-DOF G1 with optional hardware observation, OFF by default."""

    env: G1_23ObserverRealEnvCfg = G1_23ObserverRealEnvCfg(
        dof=_DOF,
        unitree=G1UnitreeCfg(net_if="eth0", command_timeout=0.1, state_timeout=0.1, shutdown_damping=5.0),
        observer=JointObserverCfg(
            algorithm="nonlinear",
            mode="OFF",
            joint_names=list(JOINT_NAMES),
            **_OBSERVER_PARAMETERS,
            actuator_limits_nm=[_ACTUATOR_LIMITS_BY_NAME[name] for name in JOINT_NAMES],
            correction_limit_fraction=0.8,
            correction_limit_nm=0.0,
            slew_nm_s=2.0,
            release_slew_nm_s=16.0,
        ),
    )
