from robojudo.config import cfg_registry
from robojudo.controller.ctrl_cfgs import JoystickCtrlCfg, KeyboardCtrlCfg, UpperBodyZmqCtrlCfg
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg

from .env.x2_env_cfg import X2_ARM_JOINT_NAMES, X2JointDefaultDoF
from .env.x2_mujuco_env_cfg import X2MujocoEnvCfg
from .env.x2_real_env_cfg import X2RealEnvCfg
from .pipeline.x2_loco_mimic_pipeline_cfg import X2LocomanipulationLocoMimicPipelineCfg
from .policy.x2_beyondmimic_policy_cfg import X2BeyondMimicPolicyCfg
from .policy.x2_deploy_policy_cfg import X2DeployPolicyCfg
from .policy.x2_locomanipulation_policy_cfg import (
    X2LocomanipulationEnvDoF,
    X2LocomanipulationPolicyCfg,
)


@cfg_registry.register
class x2(RlPipelineCfg):
    """
    AgiBot X2 configuration, X2 deploy policy, Sim2Sim.
    Add future X2 deployment presets in this module to mirror g1_cfg.py.
    """

    robot: str = "x2"
    pipeline_type: str = "X2LocomanipulationPipeline"
    env: X2MujocoEnvCfg = X2MujocoEnvCfg()
    ctrl: list[JoystickCtrlCfg | KeyboardCtrlCfg] = [
        JoystickCtrlCfg(
            triggers={
                "A": "[PASSIVE_DEFAULT]",
                "B": "[DAMPING_DEFAULT]",
                "Y": "[JOINT_DEFAULT]",
                "X": "[RL_DEFAULT]",
                "LB+RB+A": "[SHUTDOWN]",
                "LB+RB+Y": "[SIM_REBORN]",
            }
        ),
        KeyboardCtrlCfg(
            triggers={
                "k": "[PASSIVE_DEFAULT]",
                "l": "[DAMPING_DEFAULT]",
                "i": "[JOINT_DEFAULT]",
                "j": "[RL_DEFAULT]",
                "7": "[ELASTIC_BAND_LOWER]",
                "8": "[ELASTIC_BAND_LIFT]",
                "9": "[ELASTIC_BAND_TOGGLE]",
            }
        ),
    ]
    policy: X2DeployPolicyCfg = X2DeployPolicyCfg()
    joint_default_dof: X2JointDefaultDoF = X2JointDefaultDoF()
    joint_default_duration: float = 1.5
    default_damping: float = 5.0


@cfg_registry.register
class x2_real(x2):
    """
    AgiBot X2 configuration, X2 deploy policy, Sim2Real through AimDK.
    """

    env: X2RealEnvCfg = X2RealEnvCfg()
    ctrl: list[JoystickCtrlCfg] = [
        JoystickCtrlCfg(
            triggers={
                "A": "[PASSIVE_DEFAULT]",
                "B": "[DAMPING_DEFAULT]",
                "Y": "[JOINT_DEFAULT]",
                "X": "[RL_DEFAULT]",
                "LB+RB+A": "[SHUTDOWN]",
            }
        )
    ]
    do_safety_check: bool = True


@cfg_registry.register
class x2_beyondmimic(x2):
    """Shared BeyondMimic runtime with a 29-DoF no-state X2 export, Sim2Sim."""

    policy: X2BeyondMimicPolicyCfg = X2BeyondMimicPolicyCfg()


@cfg_registry.register
class x2_beyondmimic_real(x2_beyondmimic):
    """Shared BeyondMimic runtime with torso-IMU orientation, Sim2Real through AimDK."""

    env: X2RealEnvCfg = X2RealEnvCfg()
    ctrl: list[JoystickCtrlCfg] = [
        JoystickCtrlCfg(
            triggers={
                "A": "[PASSIVE_DEFAULT]",
                "B": "[DAMPING_DEFAULT]",
                "Y": "[JOINT_DEFAULT]",
                "X": "[RL_DEFAULT]",
                "LB+RB+A": "[SHUTDOWN]",
            }
        )
    ]
    do_safety_check: bool = True


@cfg_registry.register
class x2_locomanipulation(x2):
    """AgiBot X2 locomanipulation policy, Sim2Sim with recorded training parameters."""

    env: X2MujocoEnvCfg = X2MujocoEnvCfg(
        dof=X2LocomanipulationEnvDoF(),
        sim_dt=0.005,
        sim_decimation=4,
    )
    policy: X2LocomanipulationPolicyCfg = X2LocomanipulationPolicyCfg()
    joint_default_dof: X2LocomanipulationEnvDoF = X2LocomanipulationEnvDoF()
    ctrl: list[JoystickCtrlCfg | KeyboardCtrlCfg | UpperBodyZmqCtrlCfg] = [
        JoystickCtrlCfg(
            triggers={
                "A": "[PASSIVE_DEFAULT]",
                "B": "[DAMPING_DEFAULT]",
                "Y": "[JOINT_DEFAULT]",
                "X": "[RL_DEFAULT]",
                "Start": "[UPPER_BODY_TOGGLE]",
                "LB+RB+A": "[SHUTDOWN]",
                "LB+RB+Y": "[SIM_REBORN]",
            }
        ),
        KeyboardCtrlCfg(
            triggers={
                "k": "[PASSIVE_DEFAULT]",
                "l": "[DAMPING_DEFAULT]",
                "i": "[JOINT_DEFAULT]",
                "j": "[RL_DEFAULT]",
                "7": "[ELASTIC_BAND_LOWER]",
                "8": "[ELASTIC_BAND_LIFT]",
                "9": "[ELASTIC_BAND_TOGGLE]",
                "t": "[UPPER_BODY_TOGGLE]",
            }
        ),
        UpperBodyZmqCtrlCfg(joint_names=X2_ARM_JOINT_NAMES),
    ]


@cfg_registry.register
class x2_locomanipulation_real(x2_locomanipulation):
    """AgiBot X2 locomanipulation policy, Sim2Real through AimDK."""

    env: X2RealEnvCfg = X2RealEnvCfg(dof=X2LocomanipulationEnvDoF())
    ctrl: list[JoystickCtrlCfg | UpperBodyZmqCtrlCfg] = [
        JoystickCtrlCfg(
            triggers={
                "A": "[PASSIVE_DEFAULT]",
                "B": "[DAMPING_DEFAULT]",
                "Y": "[JOINT_DEFAULT]",
                "X": "[RL_DEFAULT]",
                "Start": "[UPPER_BODY_TOGGLE]",
                "LB+RB+A": "[SHUTDOWN]",
            }
        ),
        UpperBodyZmqCtrlCfg(joint_names=X2_ARM_JOINT_NAMES),
    ]
    do_safety_check: bool = True


@cfg_registry.register
class x2_locomimic(X2LocomanipulationLocoMimicPipelineCfg):
    """X2 locomanipulation locomotion with x2_rl_deploy as the test mimic, Sim2Sim."""

    env: X2MujocoEnvCfg = X2MujocoEnvCfg(
        dof=X2LocomanipulationEnvDoF(),
        sim_dt=0.005,
        sim_decimation=4,
    )
    ctrl: list[JoystickCtrlCfg | KeyboardCtrlCfg | UpperBodyZmqCtrlCfg] = [
        JoystickCtrlCfg(
            triggers={
                "A": "[PASSIVE_DEFAULT]",
                "B": "[DAMPING_DEFAULT]",
                "Y": "[JOINT_DEFAULT]",
                "X": "[RL_DEFAULT]",
                "Back": "[POLICY_LOCO]",
                "Start": "[POLICY_MIMIC]",
                "RB": "[POLICY_SWITCH],NEXT",
                "LB": "[POLICY_SWITCH],LAST",
                "L": "[UPPER_BODY_TOGGLE]",
                "LB+RB+A": "[SHUTDOWN]",
                "LB+RB+Y": "[SIM_REBORN]",
            }
        ),
        KeyboardCtrlCfg(
            triggers_extra={
                "k": "[PASSIVE_DEFAULT]",
                "l": "[DAMPING_DEFAULT]",
                "i": "[JOINT_DEFAULT]",
                "j": "[RL_DEFAULT]",
                "]": "[POLICY_LOCO]",
                "[": "[POLICY_MIMIC]",
                ";": "[POLICY_SWITCH],NEXT",
                "'": "[POLICY_SWITCH],LAST",
                "t": "[UPPER_BODY_TOGGLE]",
                "7": "[ELASTIC_BAND_LOWER]",
                "8": "[ELASTIC_BAND_LIFT]",
                "9": "[ELASTIC_BAND_TOGGLE]",
            }
        ),
        UpperBodyZmqCtrlCfg(joint_names=X2_ARM_JOINT_NAMES),
    ]
    loco_policy: X2LocomanipulationPolicyCfg = X2LocomanipulationPolicyCfg()
    mimic_policies: list[X2DeployPolicyCfg] = [
        X2DeployPolicyCfg(max_timestep=2820),
        X2DeployPolicyCfg(max_timestep=2820),
    ]


@cfg_registry.register
class x2_locomimic_real(x2_locomimic):
    """X2 locomanipulation locomotion with x2_rl_deploy as the test mimic, Sim2Real."""

    env: X2RealEnvCfg = X2RealEnvCfg(dof=X2LocomanipulationEnvDoF())
    ctrl: list[JoystickCtrlCfg | UpperBodyZmqCtrlCfg] = [
        JoystickCtrlCfg(
            triggers={
                "A": "[PASSIVE_DEFAULT]",
                "B": "[DAMPING_DEFAULT]",
                "Y": "[JOINT_DEFAULT]",
                "X": "[RL_DEFAULT]",
                "Back": "[POLICY_LOCO]",
                "Start": "[POLICY_MIMIC]",
                "RB": "[POLICY_SWITCH],NEXT",
                "LB": "[POLICY_SWITCH],LAST",
                "L": "[UPPER_BODY_TOGGLE]",
                "LB+RB+A": "[SHUTDOWN]",
            }
        ),
        UpperBodyZmqCtrlCfg(joint_names=X2_ARM_JOINT_NAMES),
    ]
    do_safety_check: bool = True

@cfg_registry.register
class x2_locomimic_beyondmimic(X2LocomanipulationLocoMimicPipelineCfg):
    """X2 locomanipulation locomotion with x2_rl_deploy as the test mimic, Sim2Sim."""

    env: X2MujocoEnvCfg = X2MujocoEnvCfg(
        dof=X2LocomanipulationEnvDoF(),
        sim_dt=0.005,
        sim_decimation=4,
    )
    ctrl: list[JoystickCtrlCfg | KeyboardCtrlCfg | UpperBodyZmqCtrlCfg] = [
        JoystickCtrlCfg(
            triggers={
                "A": "[PASSIVE_DEFAULT]",
                "B": "[DAMPING_DEFAULT]",
                "Y": "[JOINT_DEFAULT]",
                "X": "[RL_DEFAULT]",
                "Back": "[POLICY_LOCO]",
                "Start": "[POLICY_MIMIC]",
                "RB": "[POLICY_SWITCH],NEXT",
                "LB": "[POLICY_SWITCH],LAST",
                "L": "[UPPER_BODY_TOGGLE]",
                "LB+RB+A": "[SHUTDOWN]",
                "LB+RB+Y": "[SIM_REBORN]",
            }
        ),
        KeyboardCtrlCfg(
            triggers_extra={
                "k": "[PASSIVE_DEFAULT]",
                "l": "[DAMPING_DEFAULT]",
                "i": "[JOINT_DEFAULT]",
                "j": "[RL_DEFAULT]",
                "]": "[POLICY_LOCO]",
                "[": "[POLICY_MIMIC]",
                ";": "[POLICY_SWITCH],NEXT",
                "'": "[POLICY_SWITCH],LAST",
                "t": "[UPPER_BODY_TOGGLE]",
                "7": "[ELASTIC_BAND_LOWER]",
                "8": "[ELASTIC_BAND_LIFT]",
                "9": "[ELASTIC_BAND_TOGGLE]",
            }
        ),
        UpperBodyZmqCtrlCfg(joint_names=X2_ARM_JOINT_NAMES),
    ]
    loco_policy: X2LocomanipulationPolicyCfg = X2LocomanipulationPolicyCfg()
    mimic_policies: list[X2BeyondMimicPolicyCfg] = [
        X2BeyondMimicPolicyCfg(max_timestep=1800, policy_name="Walk1_subject1_wose", without_state_estimator=True),
    ]

@cfg_registry.register
class x2_locomimic_beyondmimic_real(x2_locomimic_beyondmimic):
    """X2 locomanipulation locomotion with x2_rl_deploy as the test mimic, Sim2Real."""

    env: X2RealEnvCfg = X2RealEnvCfg(dof=X2LocomanipulationEnvDoF())
    ctrl: list[JoystickCtrlCfg | UpperBodyZmqCtrlCfg] = [
        JoystickCtrlCfg(
            triggers={
                "A": "[PASSIVE_DEFAULT]",
                "B": "[DAMPING_DEFAULT]",
                "Y": "[JOINT_DEFAULT]",
                "X": "[RL_DEFAULT]",
                "Back": "[POLICY_LOCO]",
                "Start": "[POLICY_MIMIC]",
                "RB": "[POLICY_SWITCH],NEXT",
                "LB": "[POLICY_SWITCH],LAST",
                "L": "[UPPER_BODY_TOGGLE]",
                "LB+RB+A": "[SHUTDOWN]",
            }
        ),
        UpperBodyZmqCtrlCfg(joint_names=X2_ARM_JOINT_NAMES),
    ]
    do_safety_check: bool = True
