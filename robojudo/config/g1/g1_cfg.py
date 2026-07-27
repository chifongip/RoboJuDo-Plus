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
from .policy.g1_amo_policy_cfg import G1AmoPolicyCfg  # noqa: F401
from .policy.g1_asap_policy_cfg import G1AsapLocoPolicyCfg, G1AsapPolicyCfg  # noqa: F401
from .policy.g1_beyondmimic_policy_cfg import (  # noqa: F401
    G1_23BeyondMimicPolicyCfg,
    G1BeyondMimicPolicyCfg,
)
from .policy.g1_h2h_policy_cfg import G1H2HPolicyCfg  # noqa: F401
from .policy.g1_kungfubot_policy_cfg import G1KungfuBotGeneralPolicyCfg, G1KungfuBotPolicyCfg  # noqa: F401
from .policy.g1_locomanipulation_policy_cfg import (
    G1Locomanipulation23ObsDoF,
    G1Locomanipulation23PolicyCfg,
    G1Locomanipulation29ObsDoF,
    G1Locomanipulation29PolicyCfg,
)
from .policy.g1_protomotions_tracker_cfg import ProtoMotionsTrackerPolicyCfg  # noqa: F401
from .policy.g1_smooth_policy_cfg import G1SmoothPolicyCfg  # noqa: F401
from .policy.g1_twist_policy_cfg import G1TwistPolicyCfg  # noqa: F401
from .policy.g1_unitree_policy_cfg import G1UnitreePolicyCfg, G1UnitreeWoGaitPolicyCfg  # noqa: F401


# ======================== Basic Configs ======================== #
@cfg_registry.register
class g1(RlPipelineCfg):
    """
    Unitree G1 robot configuration, Unitree Policy, Sim2Sim.
    You can modify to play with other policies and controllers.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()
    # env: G1_23MujocoEnvCfg = G1_23MujocoEnvCfg()
    # env: G1_12MujocoEnvCfg = G1_12MujocoEnvCfg()

    ctrl: list[JoystickCtrlCfg | KeyboardCtrlCfg] = [  # note: the ranking of controllers matters
        JoystickCtrlCfg(),
        # KeyboardCtrlCfg(),
    ]

    policy: G1UnitreePolicyCfg = G1UnitreePolicyCfg()
    # policy: G1UnitreeWoGaitPolicyCfg = G1UnitreeWoGaitPolicyCfg()
    # policy: G1AmoPolicyCfg = G1AmoPolicyCfg()

    # run_fullspeed: bool = env.is_sim


@cfg_registry.register
class g1_real(g1):
    """
    Unitree G1 robot, Unitree Policy, Sim2Real.
    To extend the sim2sim config to sim2real, just need to change the env to real env.
    """

    # env: G1DummyEnvCfg = G1DummyEnvCfg()
    env: G1RealEnvCfg = G1RealEnvCfg(
        # env_type="UnitreeEnv",  # For unitree_sdk2py
        env_type="UnitreeCppEnv",  # For unitree_cpp, check README for more details
        unitree=G1UnitreeCfg(
            net_if="eth0",  # note: change to your network interface
        ),
    )

    ctrl: list[UnitreeCtrlCfg] = [
        UnitreeCtrlCfg(),
    ]

    do_safety_check: bool = True  # enable safety check for real robot


@cfg_registry.register
class g1_switch(RlMultiPolicyPipelineCfg):
    """
    Example of multi-policy pipeline configuration.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()

    ctrl: list[KeyboardCtrlCfg | JoystickCtrlCfg] = [
        # KeyboardCtrlCfg(
        #     triggers_extra={
        #         "Key.tab": "[POLICY_TOGGLE]",
        #     }
        # ),
        JoystickCtrlCfg(
            triggers_extra={
                "RB+Down": "[POLICY_SWITCH],0",
                "RB+Up": "[POLICY_SWITCH],1",
            }
        ),
    ]

    policies: list[G1UnitreePolicyCfg | G1AmoPolicyCfg] = [
        G1UnitreePolicyCfg(),
        G1AmoPolicyCfg(),
    ]


@cfg_registry.register
class g1_locomimic(RlLocoMimicPipelineCfg):
    """
    Example of loco mimic pipeline configuration.
    You can switch between loco and mimic policies during runtime, with interpolation.
    === Check more fancy locomimic examples in g1_loco_mimic_cfg.py ===
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()

    ctrl: list[KeyboardCtrlCfg | JoystickCtrlCfg] = [
        KeyboardCtrlCfg(
            triggers_extra={
                "]": "[POLICY_LOCO]",
                "[": "[POLICY_MIMIC]",
            }
        ),
        JoystickCtrlCfg(
            triggers_extra={
                "RB+Down": "[POLICY_LOCO]",
                "RB+Up": "[POLICY_MIMIC]",
            }
        ),
    ]

    loco_policy: G1UnitreePolicyCfg = G1UnitreePolicyCfg()
    mimic_policies: list[G1AsapPolicyCfg] = [
        G1AsapPolicyCfg(),
    ]


# ======================== Configs for supported Policy ======================== #


def _g1_locomanipulation_sim_ctrl(
    joint_names: list[str],
) -> list[JoystickCtrlCfg | KeyboardCtrlCfg | UpperBodyZmqCtrlCfg]:
    return [
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
        UpperBodyZmqCtrlCfg(joint_names=joint_names),
    ]


@cfg_registry.register
class g1_23_locomanipulation_default(RlPipelineCfg):
    """G1 23-DOF Locomanipulation with recorded default PD gains, Sim2Sim."""

    robot: str = "g1"
    pipeline_type: str = "G1LocomanipulationPipeline"
    env: G1_23MujocoEnvCfg = G1_23MujocoEnvCfg(
        dof=G1Locomanipulation23ObsDoF.from_preset("default"),
        sim_dt=0.005,
        sim_decimation=4,
        elastic_band=ElasticBandCfg(body_name="torso_link"),
    )
    ctrl: list[JoystickCtrlCfg | KeyboardCtrlCfg | UpperBodyZmqCtrlCfg] = _g1_locomanipulation_sim_ctrl(
        G1Locomanipulation23ObsDoF().joint_names[13:]
    )
    policy: G1Locomanipulation23PolicyCfg = G1Locomanipulation23PolicyCfg(
        policy_name="policy_23dof_default",
        pd_gain_preset="default",
    )
    joint_default_dof: G1Locomanipulation23ObsDoF = G1Locomanipulation23ObsDoF.from_preset("default")
    joint_default_duration: float = 1.5
    default_damping: float = 5.0


@cfg_registry.register
class g1_23_locomanipulation_stiff(g1_23_locomanipulation_default):
    """G1 23-DOF Locomanipulation with recorded stiff PD gains, Sim2Sim."""

    env: G1_23MujocoEnvCfg = G1_23MujocoEnvCfg(
        dof=G1Locomanipulation23ObsDoF.from_preset("stiff"),
        sim_dt=0.005,
        sim_decimation=4,
        elastic_band=ElasticBandCfg(body_name="torso_link"),
    )
    policy: G1Locomanipulation23PolicyCfg = G1Locomanipulation23PolicyCfg(
        policy_name="policy_23dof_stiff",
        pd_gain_preset="stiff",
    )
    joint_default_dof: G1Locomanipulation23ObsDoF = G1Locomanipulation23ObsDoF.from_preset("stiff")


@cfg_registry.register
class g1_29_locomanipulation_stiff(RlPipelineCfg):
    """G1 29-DOF Locomanipulation with recorded stiff PD gains, Sim2Sim."""

    robot: str = "g1"
    pipeline_type: str = "G1LocomanipulationPipeline"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg(
        dof=G1Locomanipulation29ObsDoF.from_preset("stiff"),
        sim_dt=0.005,
        sim_decimation=4,
        elastic_band=ElasticBandCfg(body_name="torso_link"),
    )
    ctrl: list[JoystickCtrlCfg | KeyboardCtrlCfg | UpperBodyZmqCtrlCfg] = _g1_locomanipulation_sim_ctrl(
        G1Locomanipulation29ObsDoF().joint_names[15:]
    )
    policy: G1Locomanipulation29PolicyCfg = G1Locomanipulation29PolicyCfg()
    joint_default_dof: G1Locomanipulation29ObsDoF = G1Locomanipulation29ObsDoF.from_preset("stiff")
    joint_default_duration: float = 1.5
    default_damping: float = 5.0


def _g1_locomanipulation_real_ctrl(
    joint_names: list[str],
) -> list[UnitreeCtrlCfg | UpperBodyZmqCtrlCfg]:
    return [
        UnitreeCtrlCfg(
            combination_init_buttons=["L1", "R1"],
            triggers={
                "A": "[PASSIVE_DEFAULT]",
                "B": "[DAMPING_DEFAULT]",
                "Y": "[JOINT_DEFAULT]",
                "X": "[RL_DEFAULT]",
                "Start": "[UPPER_BODY_TOGGLE]",
                "L1+R1+A": "[SHUTDOWN]",
            },
        ),
        UpperBodyZmqCtrlCfg(joint_names=joint_names),
    ]


@cfg_registry.register
class g1_23_locomanipulation_default_real(g1_23_locomanipulation_default):
    """G1 Locomanipulation default-gain policy using the logical 23-DOF layout."""

    env: G1_23RealEnvCfg = G1_23RealEnvCfg(
        dof=G1Locomanipulation23ObsDoF.from_preset("default"),
        unitree=G1UnitreeCfg(
            net_if="eth0",
            command_timeout=0.1,
            state_timeout=0.1,
            shutdown_damping=5.0,
        ),
    )
    ctrl: list[UnitreeCtrlCfg | UpperBodyZmqCtrlCfg] = _g1_locomanipulation_real_ctrl(
        G1Locomanipulation23ObsDoF().joint_names[13:]
    )
    do_safety_check: bool = True


@cfg_registry.register
class g1_23_locomanipulation_stiff_real(g1_23_locomanipulation_stiff):
    """G1 Locomanipulation stiff-gain policy using the logical 23-DOF layout."""

    env: G1_23RealEnvCfg = G1_23RealEnvCfg(
        dof=G1Locomanipulation23ObsDoF.from_preset("stiff"),
        unitree=G1UnitreeCfg(
            net_if="enp58s0",
            command_timeout=0.1,
            state_timeout=0.1,
            shutdown_damping=5.0,
        ),
    )
    ctrl: list[UnitreeCtrlCfg | UpperBodyZmqCtrlCfg] = _g1_locomanipulation_real_ctrl(
        G1Locomanipulation23ObsDoF().joint_names[13:]
    )
    do_safety_check: bool = True


@cfg_registry.register
class g1_29_locomanipulation_stiff_real(g1_29_locomanipulation_stiff):
    """G1 Locomanipulation stiff-gain policy on a native 29-DOF robot."""

    env: G1RealEnvCfg = G1RealEnvCfg(
        dof=G1Locomanipulation29ObsDoF.from_preset("stiff"),
        unitree=G1UnitreeCfg(
            net_if="eth0",
            command_timeout=0.1,
            state_timeout=0.1,
            shutdown_damping=5.0,
        ),
    )
    ctrl: list[UnitreeCtrlCfg | UpperBodyZmqCtrlCfg] = _g1_locomanipulation_real_ctrl(
        G1Locomanipulation29ObsDoF().joint_names[15:]
    )
    do_safety_check: bool = True


@cfg_registry.register
class g1_h2h(RlPipelineCfg):
    """
    Human2Humanoid
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()
    ctrl: list[KeyboardCtrlCfg | G1MotionH2HCtrlCfg] = [
        KeyboardCtrlCfg(),
        G1MotionH2HCtrlCfg(),
    ]

    policy: G1H2HPolicyCfg = G1H2HPolicyCfg()


@cfg_registry.register
class g1_beyondmimic(RlPipelineCfg):
    """
    BeyondMimic Policy, support both with and without state estimator.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()
    ctrl: list[KeyboardCtrlCfg] = [
        KeyboardCtrlCfg(),
    ]

    policy: G1BeyondMimicPolicyCfg = G1BeyondMimicPolicyCfg(
        policy_name="Jump_wose",
        without_state_estimator=True,
        use_modelmeta_config=True,  # use robot dof config from modelmeta
        use_motion_from_model=True,  # use motion from onnx model
        max_timestep=140,
    )


@cfg_registry.register
class g1_23_beyondmimic(RlPipelineCfg):
    """Native 23-DoF G1 motion tracking policy, Sim2Sim."""

    realign_on_policy_switch: bool = True
    robot: str = "g1"
    env: G1_23MujocoEnvCfg = G1_23MujocoEnvCfg()
    ctrl: list[KeyboardCtrlCfg] = [
        KeyboardCtrlCfg()
    ]
    policy: G1_23BeyondMimicPolicyCfg = G1_23BeyondMimicPolicyCfg()


@cfg_registry.register
class g1_23_beyondmimic_real(g1_23_beyondmimic):
    """Native 23-DoF G1 motion tracking policy, Sim2Real."""
    realign_on_policy_switch: bool = True
    env: G1_23RealEnvCfg = G1_23RealEnvCfg()
    ctrl: list[UnitreeCtrlCfg] = [
        UnitreeCtrlCfg()
    ]
    do_safety_check: bool = True


@cfg_registry.register
class g1_beyondmimic_with_ctrl(RlPipelineCfg):
    """
    BeyondMimic with External BeyondMimicCtrl as motion source.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()
    ctrl: list[KeyboardCtrlCfg | G1BeyondmimicCtrlCfg] = [
        KeyboardCtrlCfg(),
        G1BeyondmimicCtrlCfg(
            motion_name="dance1_subject2",  # you can put your own motion file in assets/motions/g1
        ),
    ]

    policy: G1BeyondMimicPolicyCfg = G1BeyondMimicPolicyCfg(
        policy_name="Dance_wose",
        use_motion_from_model=False,  # use motion from BeyondmimicCtrl instead of the onnx
    )


@cfg_registry.register
class g1_asap(RlPipelineCfg):
    """
    Unitree G1 robot configuration, ASAP Policy, Sim2Sim.
    You can modify to play with other policies and controllers.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg(forward_kinematic=None, update_with_fk=False, born_place_align=True)

    ctrl: list[JoystickCtrlCfg | KeyboardCtrlCfg] = [  # note: the ranking of controllers matters
        # JoystickCtrlCfg(),
        KeyboardCtrlCfg(triggers={"i": "[SIM_REBORN]", "o": "[SHUTDOWN]", "r": "[MOTION_RESET]"}),
    ]

    policy: G1AsapPolicyCfg = G1AsapPolicyCfg()
    """You can also try other models, from ASAP, RoboMimic, KungfuBot(PBHC)"""
    # policy: G1KungfuBotPolicyCfg = G1KungfuBotPolicyCfg() # KungfuBot horse_squat
    # # fmt: off
    # policy: G1AsapPolicyCfg = G1AsapPolicyCfg(
    #     policy_name="robomimic",
    #     relative_path="dance_0605.onnx",
    #     motion_length_s=18.0,
    #     start_upper_body_dof_pos = [
    #         0, 0, 0,
    #         0.35, 0.18, 0, 0.87,
    #         0.35, -0.18, 0, 0.87,
    #     ],
    # )
    # # fmt: on


@cfg_registry.register
class g1_asap_loco(RlPipelineCfg):
    """
    Unitree G1 robot configuration, ASAP Locomotion Policy, Sim2Sim.
    You can modify to play with other policies and controllers.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg(forward_kinematic=None, update_with_fk=False, born_place_align=False)

    ctrl: list[JoystickCtrlCfg | KeyboardCtrlCfg] = [  # note: the ranking of controllers matters
        # JoystickCtrlCfg(),
        KeyboardCtrlCfg(
            triggers={
                "i": "[SIM_REBORN]",
                "o": "[SHUTDOWN]",
            }
        ),
    ]

    policy: G1AsapLocoPolicyCfg = G1AsapLocoPolicyCfg()


@cfg_registry.register
class g1_kungfubot2(RlPipelineCfg):
    """
    PBHC KungfuBot2 General Policy
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()
    ctrl: list[KeyboardCtrlCfg | G1MotionKungfuBotCtrlCfg] = [
        KeyboardCtrlCfg(),
        G1MotionKungfuBotCtrlCfg(
            motion_name="kungfubot/Horse-stance_pose",  # put motion files in assets/motions/g1/phc/kungfubot
        ),
    ]

    policy: G1KungfuBotGeneralPolicyCfg = G1KungfuBotGeneralPolicyCfg(
        policy_name="horse_test_43000",  # this is a test model trained with only one motion
        compatibility_old_version=True,  # for old version of kungfubot general policy (before 2025-11-13 bugfix #68)
    )


@cfg_registry.register
class g1_twist(RlPipelineCfg):
    """
    Unitree G1 robot configuration, TWIST Policy, Sim2Sim.
    TwistRedisCtrl for the original repo of high level motion stream over redis.
    MotionTwistCtrl for built-in motion control.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg(forward_kinematic=None, update_with_fk=False, born_place_align=False)

    ctrl: list[G1TwistRedisCtrlCfg | G1MotionTwistCtrlCfg] = [  # note: the ranking of controllers matters
        G1TwistRedisCtrlCfg(redis_host="localhost"),  # with hign level motion lib through redis
        # G1MotionTwistCtrlCfg(), # with built-in motion ctrl
    ]

    policy: G1TwistPolicyCfg = G1TwistPolicyCfg()


# ======================== Fancy Example Configs ======================== #


@cfg_registry.register
class g1_switch_beyondmimic(RlMultiPolicyPipelineCfg):
    """
    Switch between multiple BeyondMimic policies. Withour Interpolation.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()
    ctrl: list[KeyboardCtrlCfg | JoystickCtrlCfg] = [
        KeyboardCtrlCfg(
            triggers_extra={
                "Key.tab": "[POLICY_TOGGLE]",
                "!": "[POLICY_SWITCH],0",  # note: with shift
                "@": "[POLICY_SWITCH],1",  # note: with shift
                "#": "[POLICY_SWITCH],2",  # note: with shift
                "$": "[POLICY_SWITCH],3",  # note: with shift
            }
        ),
        JoystickCtrlCfg(
            triggers_extra={
                "RB+Down": "[POLICY_SWITCH],0",
                "RB+Left": "[POLICY_SWITCH],1",
                "RB+Up": "[POLICY_SWITCH],2",
                "RB+Right": "[POLICY_SWITCH],3",
            }
        ),
    ]

    policies: list[G1AmoPolicyCfg | G1BeyondMimicPolicyCfg] = [
        G1AmoPolicyCfg(),
        G1BeyondMimicPolicyCfg(policy_name="Violin", without_state_estimator=False, max_timestep=500),
        G1BeyondMimicPolicyCfg(policy_name="Waltz", without_state_estimator=False, max_timestep=850),
        G1BeyondMimicPolicyCfg(policy_name="Dance_wose", without_state_estimator=True),
    ]


# ======================== ProtoMotions Tracker ======================== #


@cfg_registry.register
class g1_protomotions_tracker(RlPipelineCfg):
    """ProtoMotions tracker with cached 50fps motion.

    Uses the standard RoboJuDo G1 MuJoCo environment with ``born_place_align``
    disabled (our policy handles heading alignment itself). ``random_heading``
    is on so we exercise the policy's heading-alignment recompute on each spawn.

    Use ``scripts/run_tracker_pipeline.py`` — it parses ``--onnx-path`` /
    ``--motion-path`` / ``--motion-index``, which the generic ``run_pipeline.py``
    does not.

    Usage::

        python scripts/run_tracker_pipeline.py -c g1_protomotions_tracker \\
            --motion-path assets/motions/g1/g1_bones_seed_mini.pt \\
            --motion-index 0
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg(
        born_place_align=False,
        random_heading=True,
    )
    ctrl: list[KeyboardCtrlCfg] = [
        KeyboardCtrlCfg(
            triggers={
                "r": "[MOTION_RESET]",
                "i": "[SIM_REBORN]",
                "o": "[SHUTDOWN]",
                "<": "[MOTION_FADE_IN]",
                ">": "[MOTION_FADE_OUT]",
            },
        ),
    ]

    policy: ProtoMotionsTrackerPolicyCfg = ProtoMotionsTrackerPolicyCfg()


@cfg_registry.register
class g1_protomotions_tracker_real(g1_protomotions_tracker):
    """ProtoMotions tracker on real G1 hardware.

    Use ``scripts/run_tracker_pipeline.py`` — it parses ``--onnx-path`` /
    ``--motion-path`` / ``--motion-index``, which the generic ``run_pipeline.py``
    does not.

    Usage::

        python scripts/run_tracker_pipeline.py -c g1_protomotions_tracker_real \\
            --motion-path assets/motions/g1/g1_bones_seed_mini.pt \\
            --motion-index 0
    """

    env: G1RealEnvCfg = G1RealEnvCfg(
        env_type="UnitreeCppEnv",
        unitree=G1UnitreeCfg(
            net_if="eth0",  # note: change to your network interface
        ),
        born_place_align=False,
    )
    ctrl: list[UnitreeCtrlCfg] = [
        UnitreeCtrlCfg(),
    ]
    do_safety_check: bool = True


# TIPS: check g1_loco_mimic_cfg.py for more complex examples
