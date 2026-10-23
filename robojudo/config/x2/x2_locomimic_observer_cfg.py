"""Separate simulation and deployment presets for the X2 arm observer."""

from robojudo.config import cfg_registry
from robojudo.environment.env_cfgs import JointObserverCfg
from robojudo.utils.x2_observer_reference import JOINT_NAMES, selected_observer_parameters

from .env.x2_mujuco_env_cfg import X2ObserverMujocoEnvCfg
from .env.x2_real_env_cfg import X2AimdkCfg, X2ArmObserverCfg, X2RealEnvCfg
from .policy.x2_locomanipulation_policy_cfg import X2LocomanipulationEnvDoF
from .x2_cfg import x2_locomimic, x2_locomimic_real

_DOF = X2LocomanipulationEnvDoF()
_ACTUATOR_LIMITS_BY_NAME = dict(zip(_DOF.joint_names, _DOF.torque_limits, strict=True))
_CORRECTION_LIMITS_BY_NAME = {name: min(8.0, 0.8 * _ACTUATOR_LIMITS_BY_NAME[name]) for name in JOINT_NAMES}
_OBSERVER_PARAMETERS = selected_observer_parameters()
_OBSERVER_PARAMETERS.pop("method")


@cfg_registry.register
class x2_locomimic_observer(x2_locomimic):
    """X2 locomotion with arm compensation, Sim2Sim."""

    env: X2ObserverMujocoEnvCfg = X2ObserverMujocoEnvCfg(
        dof=X2LocomanipulationEnvDoF(),
        sim_dt=0.005,
        sim_decimation=4,
        feedforward_effort_limits=_CORRECTION_LIMITS_BY_NAME,
        observer=JointObserverCfg(
            mode="ACTIVE",
            joint_names=list(JOINT_NAMES),
            **_OBSERVER_PARAMETERS,
            actuator_limits_nm=[_ACTUATOR_LIMITS_BY_NAME[name] for name in JOINT_NAMES],
            correction_limits_nm=[_CORRECTION_LIMITS_BY_NAME[name] for name in JOINT_NAMES],
            slew_nm_s=2.0,
        ),
    )


@cfg_registry.register
class x2_locomimic_real_observer(x2_locomimic_real):
    """X2 position control with optional arm observer, Sim2Real."""

    env: X2RealEnvCfg = X2RealEnvCfg(
        dof=X2LocomanipulationEnvDoF(),
        odometry_type="NONE",
        aimdk=X2AimdkCfg(startup_state_timeout=10.0),
        arm_observer=X2ArmObserverCfg(
            mode="OFF",
            # Configured bounds do not establish continuous hardware ratings or firmware clipping limits.
            actuator_limits_nm=[_ACTUATOR_LIMITS_BY_NAME[name] for name in JOINT_NAMES],
            # Simulation settings for optional SHADOW evaluation; not hardware-qualified ACTIVE limits.
            correction_limits_nm=[_CORRECTION_LIMITS_BY_NAME[name] for name in JOINT_NAMES],
            slew_nm_s=2.0,
        ),
    )
