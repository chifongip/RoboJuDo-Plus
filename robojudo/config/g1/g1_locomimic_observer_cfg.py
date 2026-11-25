"""G1 Locomanipulation arm observer presets."""

from robojudo.config import cfg_registry
from robojudo.config.observer_cfg import JointObserverCfg
from robojudo.environment.env_cfgs import ElasticBandCfg

from .env.g1_mujuco_env_cfg import G1_23ObserverMujocoEnvCfg, G1ObserverMujocoEnvCfg
from .env.g1_real_env_cfg import G1_23ObserverRealEnvCfg, G1ObserverRealEnvCfg, G1UnitreeCfg
from .g1_loco_mimic_cfg import (
    g1_23_locomanipulation_locomimic,
    g1_23_locomanipulation_locomimic_real,
    g1_29_locomanipulation_locomimic,
    g1_29_locomanipulation_locomimic_real,
)
from .observer_parameters import JOINT_NAMES as JOINT_NAMES_29
from .observer_parameters import selected_observer_parameters as selected_observer_parameters_29
from .observer_parameters_23 import JOINT_NAMES as JOINT_NAMES_23
from .observer_parameters_23 import selected_observer_parameters as selected_observer_parameters_23
from .policy.g1_locomanipulation_policy_cfg import G1Locomanipulation23ObsDoF, G1Locomanipulation29ObsDoF

_DOF_23 = G1Locomanipulation23ObsDoF.from_preset("stiff")
_ACTUATOR_LIMITS_23_BY_NAME = dict(zip(_DOF_23.joint_names, _DOF_23.torque_limits, strict=True))
_OBSERVER_PARAMETERS_23 = selected_observer_parameters_23()
_DOF_29 = G1Locomanipulation29ObsDoF.from_preset("stiff")
_ACTUATOR_LIMITS_29_BY_NAME = dict(zip(_DOF_29.joint_names, _DOF_29.torque_limits, strict=True))
_OBSERVER_PARAMETERS_29 = selected_observer_parameters_29()


@cfg_registry.register
class g1_23_locomanipulation_locomimic_observer(g1_23_locomanipulation_locomimic):
    """Stiff 23-DOF G1 Locomanipulation with arm disturbance correction in MuJoCo."""

    env: G1_23ObserverMujocoEnvCfg = G1_23ObserverMujocoEnvCfg(
        dof=_DOF_23,
        sim_dt=0.005,
        sim_decimation=4,
        elastic_band=ElasticBandCfg(body_name="torso_link"),
        feedforward_effort_limits={name: _ACTUATOR_LIMITS_23_BY_NAME[name] for name in JOINT_NAMES_23},
        observer=JointObserverCfg(
            algorithm="nonlinear",
            mode="ACTIVE",
            joint_names=list(JOINT_NAMES_23),
            **_OBSERVER_PARAMETERS_23,
            actuator_limits_nm=[_ACTUATOR_LIMITS_23_BY_NAME[name] for name in JOINT_NAMES_23],
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
        dof=_DOF_23,
        unitree=G1UnitreeCfg(net_if="eth0", command_timeout=0.1, state_timeout=0.1, shutdown_damping=5.0),
        observer=JointObserverCfg(
            algorithm="nonlinear",
            mode="OFF",
            joint_names=list(JOINT_NAMES_23),
            **_OBSERVER_PARAMETERS_23,
            actuator_limits_nm=[_ACTUATOR_LIMITS_23_BY_NAME[name] for name in JOINT_NAMES_23],
            correction_limit_fraction=0.8,
            correction_limit_nm=0.0,
            slew_nm_s=2.0,
            release_slew_nm_s=16.0,
        ),
    )


@cfg_registry.register
class g1_29_locomanipulation_locomimic_observer(g1_29_locomanipulation_locomimic):
    """Stiff 29-DOF G1 Locomanipulation with arm disturbance correction in MuJoCo."""

    env: G1ObserverMujocoEnvCfg = G1ObserverMujocoEnvCfg(
        dof=_DOF_29,
        sim_dt=0.005,
        sim_decimation=4,
        elastic_band=ElasticBandCfg(body_name="torso_link"),
        feedforward_effort_limits={name: _ACTUATOR_LIMITS_29_BY_NAME[name] for name in JOINT_NAMES_29},
        observer=JointObserverCfg(
            algorithm="nonlinear",
            mode="ACTIVE",
            joint_names=list(JOINT_NAMES_29),
            **_OBSERVER_PARAMETERS_29,
            actuator_limits_nm=[_ACTUATOR_LIMITS_29_BY_NAME[name] for name in JOINT_NAMES_29],
            correction_limit_fraction=0.8,
            correction_limit_nm=0.0,
            slew_nm_s=2.0,
            release_slew_nm_s=16.0,
        ),
    )


@cfg_registry.register
class g1_29_locomanipulation_locomimic_real_observer(g1_29_locomanipulation_locomimic_real):
    """Stiff 29-DOF G1 with optional hardware observation, OFF by default."""

    env: G1ObserverRealEnvCfg = G1ObserverRealEnvCfg(
        dof=_DOF_29,
        unitree=G1UnitreeCfg(net_if="eth0", command_timeout=0.1, state_timeout=0.1, shutdown_damping=5.0),
        observer=JointObserverCfg(
            algorithm="nonlinear",
            mode="OFF",
            joint_names=list(JOINT_NAMES_29),
            **_OBSERVER_PARAMETERS_29,
            actuator_limits_nm=[_ACTUATOR_LIMITS_29_BY_NAME[name] for name in JOINT_NAMES_29],
            correction_limit_fraction=0.8,
            correction_limit_nm=0.0,
            slew_nm_s=2.0,
            release_slew_nm_s=16.0,
        ),
    )
