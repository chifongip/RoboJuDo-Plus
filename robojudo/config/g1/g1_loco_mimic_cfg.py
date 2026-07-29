from robojudo.config import cfg_registry
from robojudo.controller.ctrl_cfgs import (
    JoystickCtrlCfg,  # noqa: F401
    KeyboardCtrlCfg,  # noqa: F401
    UnitreeCtrlCfg,  # noqa: F401
    UpperBodyZmqCtrlCfg,
)
from robojudo.environment.env_cfgs import ElasticBandCfg
from robojudo.pipeline.pipeline_cfgs import (
    RlLocoMimicPipelineCfg,  # noqa: F401
    RlMultiPolicyPipelineCfg,  # noqa: F401
    RlPipelineCfg,  # noqa: F401
)

from .ctrl.g1_beyondmimic_ctrl_cfg import G1BeyondmimicCtrlCfg  # noqa: F401
from .ctrl.g1_motion_ctrl_cfg import (  # noqa: F401
    G1MotionCtrlCfg,
    G1MotionH2HCtrlCfg,
    G1MotionKungfuBotCtrlCfg,
    G1MotionTwistCtrlCfg,
)
from .ctrl.g1_twist_redis_ctrl_cfg import G1TwistRedisCtrlCfg  # noqa: F401
from .env.g1_dummy_env_cfg import G1DummyEnvCfg  # noqa: F401
from .env.g1_mujuco_env_cfg import G1_12MujocoEnvCfg, G1_23MujocoEnvCfg, G1MujocoEnvCfg  # noqa: F401
from .env.g1_real_env_cfg import G1_23RealEnvCfg, G1RealEnvCfg, G1UnitreeCfg  # noqa: F401
from .pipeline.g1_loco_mimic_pipeline_cfg import (
    G1Locomanipulation23LocoMimicPipelineCfg,
    G1Locomanipulation29LocoMimicPipelineCfg,
    G1RlLocoMimicPipelineCfg,
)
from .policy.g1_amp_recovery_policy_cfg import G1_23AmpRecoveryPolicyCfg, G1AmpRecoveryPolicyCfg
from .policy.g1_amo_policy_cfg import G1AmoPolicyCfg  # noqa: F401
from .policy.g1_asap_policy_cfg import G1AsapLocoPolicyCfg, G1AsapPolicyCfg  # noqa: F401
from .policy.g1_beyondmimic_policy_cfg import G1BeyondMimicPolicyCfg  # noqa: F401
from .policy.g1_h2h_policy_cfg import G1H2HPolicyCfg  # noqa: F401
from .policy.g1_kungfubot_policy_cfg import G1KungfuBotGeneralPolicyCfg, G1KungfuBotPolicyCfg  # noqa: F401
from .policy.g1_locomanipulation_policy_cfg import (
    G1Locomanipulation23ObsDoF,
    G1Locomanipulation23PolicyCfg,
    G1Locomanipulation29ObsDoF,
    G1Locomanipulation29PolicyCfg,
)
from .policy.g1_smooth_policy_cfg import G1SmoothPolicyCfg  # noqa: F401
from .policy.g1_twist_policy_cfg import G1TwistPolicyCfg  # noqa: F401
from .policy.g1_unitree_policy_cfg import G1UnitreePolicyCfg, G1UnitreeWoGaitPolicyCfg  # noqa: F401

# ================= LocoMotion + MotionMimic Policy Switch Configs ================= #


def _g1_locomanipulation_mimic_policies(pad_missing_dofs: bool):
    return [
        G1BeyondMimicPolicyCfg(
            policy_name="Violin",
            without_state_estimator=False,
            max_timestep=610,
            pad_missing_dofs=pad_missing_dofs,
        ),
        G1BeyondMimicPolicyCfg(
            policy_name="Waltz",
            without_state_estimator=False,
            max_timestep=940,
            pad_missing_dofs=pad_missing_dofs,
        ),
        G1BeyondMimicPolicyCfg(
            policy_name="Jump_wose",
            without_state_estimator=True,
            max_timestep=140,
            pad_missing_dofs=pad_missing_dofs,
        ),
        G1BeyondMimicPolicyCfg(
            policy_name="Dance_wose",
            without_state_estimator=True,
            max_timestep=6574,
            pad_missing_dofs=pad_missing_dofs,
        ),
    ]


def _g1_locomanipulation_locomimic_sim_ctrl(joint_names: list[str]):
    return [
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
                "R": "[POLICY_RECOVERY]",
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
                "r": "[POLICY_RECOVERY]",
                "7": "[ELASTIC_BAND_LOWER]",
                "8": "[ELASTIC_BAND_LIFT]",
                "9": "[ELASTIC_BAND_TOGGLE]",
            }
        ),
        UpperBodyZmqCtrlCfg(joint_names=joint_names),
    ]


def _g1_locomanipulation_locomimic_real_ctrl(joint_names: list[str]):
    return [
        UnitreeCtrlCfg(
            combination_init_buttons=["L1", "R1"],
            triggers={
                "A": "[PASSIVE_DEFAULT]",
                "B": "[DAMPING_DEFAULT]",
                "Y": "[JOINT_DEFAULT]",
                "X": "[RL_DEFAULT]",
                "Select": "[POLICY_LOCO]",
                "Start": "[POLICY_MIMIC]",
                "R1": "[POLICY_SWITCH],NEXT",
                "L1": "[POLICY_SWITCH],LAST",
                "L2": "[UPPER_BODY_TOGGLE]",
                "R2": "[POLICY_RECOVERY]",
                "L1+R1+A": "[SHUTDOWN]",
            },
        ),
        UpperBodyZmqCtrlCfg(joint_names=joint_names),
    ]


@cfg_registry.register
class g1_23_locomanipulation_locomimic(G1Locomanipulation23LocoMimicPipelineCfg):
    """23-DOF Locomanipulation with Jump and Dance BeyondMimic policies, Sim2Sim."""

    env: G1_23MujocoEnvCfg = G1_23MujocoEnvCfg(
        dof=G1Locomanipulation23ObsDoF.from_preset("stiff"),
        sim_dt=0.005,
        sim_decimation=4,
        elastic_band=ElasticBandCfg(body_name="torso_link"),
    )
    ctrl: list[JoystickCtrlCfg | KeyboardCtrlCfg | UpperBodyZmqCtrlCfg] = (
        _g1_locomanipulation_locomimic_sim_ctrl(G1Locomanipulation23ObsDoF().joint_names[13:])
    )
    loco_policy: G1Locomanipulation23PolicyCfg = G1Locomanipulation23PolicyCfg()
    mimic_policies: list[G1BeyondMimicPolicyCfg] = _g1_locomanipulation_mimic_policies(True)
    recovery_policy: G1_23AmpRecoveryPolicyCfg = G1_23AmpRecoveryPolicyCfg()


@cfg_registry.register
class g1_23_locomanipulation_default_locomimic(g1_23_locomanipulation_locomimic):
    """23-DOF default-gain Locomanipulation with BeyondMimic policies, Sim2Sim."""

    env: G1_23MujocoEnvCfg = G1_23MujocoEnvCfg(
        dof=G1Locomanipulation23ObsDoF.from_preset("default"),
        sim_dt=0.005,
        sim_decimation=4,
        elastic_band=ElasticBandCfg(body_name="torso_link"),
    )
    loco_policy: G1Locomanipulation23PolicyCfg = G1Locomanipulation23PolicyCfg(
        policy_name="policy_23dof_default",
        pd_gain_preset="default",
    )
    joint_default_dof: G1Locomanipulation23ObsDoF = G1Locomanipulation23ObsDoF.from_preset("default")
    upper_dof_num: int = 10
    upper_dof_pos_default: list[float] = joint_default_dof.default_pos[-upper_dof_num:]


@cfg_registry.register
class g1_29_locomanipulation_locomimic(G1Locomanipulation29LocoMimicPipelineCfg):
    """29-DOF Locomanipulation with Jump and Dance BeyondMimic policies, Sim2Sim."""

    env: G1MujocoEnvCfg = G1MujocoEnvCfg(
        dof=G1Locomanipulation29ObsDoF.from_preset("stiff"),
        sim_dt=0.005,
        sim_decimation=4,
        elastic_band=ElasticBandCfg(body_name="torso_link"),
    )
    ctrl: list[JoystickCtrlCfg | KeyboardCtrlCfg | UpperBodyZmqCtrlCfg] = (
        _g1_locomanipulation_locomimic_sim_ctrl(G1Locomanipulation29ObsDoF().joint_names[15:])
    )
    loco_policy: G1Locomanipulation29PolicyCfg = G1Locomanipulation29PolicyCfg()
    mimic_policies: list[G1BeyondMimicPolicyCfg] = _g1_locomanipulation_mimic_policies(False)
    recovery_policy: G1AmpRecoveryPolicyCfg = G1AmpRecoveryPolicyCfg()


@cfg_registry.register
class g1_23_locomanipulation_locomimic_real(g1_23_locomanipulation_locomimic):
    """23-DOF Locomanipulation loco-mimic configuration for a real G1."""

    env: G1_23RealEnvCfg = G1_23RealEnvCfg(
        dof=G1Locomanipulation23ObsDoF.from_preset("stiff"),
        unitree=G1UnitreeCfg(
            net_if="eth0",
            command_timeout=0.1,
            state_timeout=0.1,
            shutdown_damping=5.0,
        ),
    )
    ctrl: list[UnitreeCtrlCfg | UpperBodyZmqCtrlCfg] = _g1_locomanipulation_locomimic_real_ctrl(
        G1Locomanipulation23ObsDoF().joint_names[13:]
    )
    do_safety_check: bool = True


@cfg_registry.register
class g1_23_locomanipulation_default_locomimic_real(g1_23_locomanipulation_default_locomimic):
    """23-DOF default-gain Locomanipulation loco-mimic configuration for a real G1."""

    env: G1_23RealEnvCfg = G1_23RealEnvCfg(
        dof=G1Locomanipulation23ObsDoF.from_preset("default"),
        unitree=G1UnitreeCfg(
            net_if="eth0",
            command_timeout=0.1,
            state_timeout=0.1,
            shutdown_damping=5.0,
        ),
    )
    ctrl: list[UnitreeCtrlCfg | UpperBodyZmqCtrlCfg] = _g1_locomanipulation_locomimic_real_ctrl(
        G1Locomanipulation23ObsDoF().joint_names[13:]
    )
    do_safety_check: bool = True


@cfg_registry.register
class g1_29_locomanipulation_locomimic_real(g1_29_locomanipulation_locomimic):
    """29-DOF Locomanipulation loco-mimic configuration for a real G1."""

    env: G1RealEnvCfg = G1RealEnvCfg(
        dof=G1Locomanipulation29ObsDoF.from_preset("stiff"),
        unitree=G1UnitreeCfg(
            net_if="eth0",
            command_timeout=0.1,
            state_timeout=0.1,
            shutdown_damping=5.0,
        ),
    )
    ctrl: list[UnitreeCtrlCfg | UpperBodyZmqCtrlCfg] = _g1_locomanipulation_locomimic_real_ctrl(
        G1Locomanipulation29ObsDoF().joint_names[15:]
    )
    do_safety_check: bool = True


@cfg_registry.register
class g1_locomimic_beyondmimic(G1RlLocoMimicPipelineCfg):
    """
    Smooth switch between multiple BeyondMimic policies, Sim2Sim.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()
    ctrl: list[KeyboardCtrlCfg | JoystickCtrlCfg] = [
        KeyboardCtrlCfg(
            triggers={
                "i": "[SIM_REBORN]",
                "o": "[SHUTDOWN]",
                "]": "[POLICY_LOCO]",
                "[": "[POLICY_MIMIC]",
                ";": "[POLICY_SWITCH],NEXT",
                "'": "[POLICY_SWITCH],LAST",
            }
        ),
        # JoystickCtrlCfg(
        #     combination_init_buttons=[],
        #     triggers={
        #         "A": "[SHUTDOWN]",
        #         "Back": "[POLICY_LOCO]",
        #         "Start": "[POLICY_MIMIC]",
        #         "RB": "[POLICY_SWITCH],NEXT",
        #         "LB": "[POLICY_SWITCH],LAST",
        #     },
        # ),
    ]

    loco_policy: G1AmoPolicyCfg = G1AmoPolicyCfg()
    # loco_policy: G1AsapLocoPolicyCfg = G1AsapLocoPolicyCfg()
    # loco_policy: G1UnitreePolicyCfg = G1UnitreePolicyCfg()
    # loco_policy: G1UnitreeWoGaitPolicyCfg = G1UnitreeWoGaitPolicyCfg()
    """Any LocoMotion policy, as init"""

    mimic_policies: list[G1BeyondMimicPolicyCfg] = [
        G1BeyondMimicPolicyCfg(policy_name="Dance_wose", without_state_estimator=True),
        G1BeyondMimicPolicyCfg(policy_name="Violin", without_state_estimator=False, max_timestep=500),
        G1BeyondMimicPolicyCfg(policy_name="Waltz", without_state_estimator=False, max_timestep=850),
    ]


@cfg_registry.register
class g1_locomimic_asap(G1RlLocoMimicPipelineCfg):
    """
    Unitree G1 robot configuration, ASAP Locomotion + Deepmimic, Sim2Sim.
    Dynamic switch, keyboard control.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg(forward_kinematic=None, update_with_fk=False, born_place_align=True)

    ctrl: list[KeyboardCtrlCfg | JoystickCtrlCfg] = [  # note: the ranking of controllers matters
        KeyboardCtrlCfg(
            triggers={
                "i": "[SIM_REBORN]",
                "o": "[SHUTDOWN]",
                "]": "[POLICY_LOCO]",
                "[": "[POLICY_MIMIC]",
                ";": "[POLICY_SWITCH],NEXT",
                "'": "[POLICY_SWITCH],LAST",
            }
        ),
        # JoystickCtrlCfg(
        #     combination_init_buttons=[],
        #     triggers={
        #         "A": "[SHUTDOWN]",
        #         "Back": "[POLICY_LOCO]",
        #         "Start": "[POLICY_MIMIC]",
        #         "RB": "[POLICY_SWITCH],NEXT",
        #         "LB": "[POLICY_SWITCH],LAST",
        #     },
        # ),
    ]

    loco_policy: G1AsapLocoPolicyCfg = G1AsapLocoPolicyCfg()

    # fmt: off
    mimic_policies: list[G1AsapPolicyCfg] = [
        G1AsapPolicyCfg(), # default CR7_level1
        G1AsapPolicyCfg(
            policy_name="robomimic",
            relative_path="dance_0605.onnx",
            motion_length_s=18.0,
            start_upper_body_dof_pos = [
                0, 0, 0,
                0.35, 0.18, 0, 0.87, 
                0.35, -0.18, 0, 0.87,
            ],
        ),
        G1KungfuBotPolicyCfg(),
    ]
    # fmt: on


# ================= LocoMimic Policy Switch Sim2real Configs ================= #


@cfg_registry.register
class g1_locomimic_beyondmimic_real(g1_locomimic_beyondmimic):
    """
    Locomotion + Beyondmimic, Sim2Real.
    Warning: Make sure the policy is stable for real robot before using it.
    """

    env: G1RealEnvCfg = G1RealEnvCfg(
        unitree=G1UnitreeCfg(
            net_if="eth0",  # note: change to your network interface
        ),
    )
    ctrl: list[UnitreeCtrlCfg] = [
        UnitreeCtrlCfg(
            combination_init_buttons=[],
            triggers={
                "A": "[SHUTDOWN]",
                "Select": "[POLICY_LOCO]",
                "Start": "[POLICY_MIMIC]",
                "R1": "[POLICY_SWITCH],NEXT",
                "L1": "[POLICY_SWITCH],LAST",
            },
        ),
    ]

    do_safety_check: bool = True  # enable safety check for real robot


@cfg_registry.register
class g1_locomimic_asap_real(g1_locomimic_asap):
    """
    ASAP Locomotion + Deepmimic, Sim2Real.
    Warning: Make sure the policy is stable for real robot before using it.
    """

    # env: G1DummyEnvCfg = G1DummyEnvCfg()
    env: G1RealEnvCfg = G1RealEnvCfg(
        unitree=G1UnitreeCfg(
            net_if="eth0",  # note: change to your network interface
        ),
    )

    ctrl: list[UnitreeCtrlCfg] = [
        UnitreeCtrlCfg(
            combination_init_buttons=[],
            triggers={
                "A": "[SHUTDOWN]",
                "Select": "[POLICY_LOCO]",
                "Start": "[POLICY_MIMIC]",
                "R1": "[POLICY_SWITCH],NEXT",
                "L1": "[POLICY_SWITCH],LAST",
            },
        ),
    ]

    do_safety_check: bool = True  # enable safety check for real robot


# ================= ASAP Policy  ================= #
@cfg_registry.register
class g1_locomimic_asap_full(G1RlLocoMimicPipelineCfg):
    """
    Exact reproduce of the original ASAP code.
    You need to download the model files from the official repo and put them in assets/models/g1/asap
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg(forward_kinematic=None, update_with_fk=False, born_place_align=True)

    ctrl: list[KeyboardCtrlCfg | JoystickCtrlCfg] = [  # note: the ranking of controllers matters
        KeyboardCtrlCfg(
            triggers={
                "i": "[SIM_REBORN]",
                "o": "[SHUTDOWN]",
                "]": "[POLICY_LOCO]",
                "[": "[POLICY_MIMIC]",
                ";": "[POLICY_SWITCH],NEXT",
                "'": "[POLICY_SWITCH],LAST",
            }
        ),
    ]

    loco_policy: G1AsapLocoPolicyCfg = G1AsapLocoPolicyCfg()

    mimic_policies: list[G1AsapPolicyCfg] = []

    def __init__(self, **data) -> None:
        super().__init__(**data)
        # add all the asap policies in asap.yaml
        from pathlib import Path

        import yaml

        asap_config = yaml.safe_load(open(Path(__file__).parent / "asap.yaml"))
        for plicy_name, relative_path in asap_config["mimic_models"].items():
            start_upper_body_dof_pos = asap_config["start_upper_body_dof_pos"].get(plicy_name, None)
            # remove some joints that are not in the g1 23-dof model
            if start_upper_body_dof_pos is not None:
                start_upper_body_dof_pos = [start_upper_body_dof_pos[i] for i in [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13]]
            motion_length_s = asap_config["motion_length_s"].get(plicy_name, 10.0)
            self.mimic_policies.append(
                G1AsapPolicyCfg(
                    policy_name=plicy_name,
                    relative_path=relative_path,
                    start_upper_body_dof_pos=start_upper_body_dof_pos,
                    motion_length_s=motion_length_s,
                )
            )
