"""G1 SuperMimic policies and loco/mimic switching in simulation and on hardware."""

from robojudo.config import ASSETS_DIR, ROOT_DIR, cfg_registry
from robojudo.controller.ctrl_cfgs import KeyboardCtrlCfg, UnitreeCtrlCfg, UpperBodyZmqCtrlCfg
from robojudo.pipeline.pipeline_cfgs import RlLocoMimicPipelineCfg, RlPipelineCfg

from .env.g1_mujuco_env_cfg import G1_23MujocoEnvCfg, G1MujocoEnvCfg
from .env.g1_real_env_cfg import G1_23RealEnvCfg, G1UnitreeCfg
from .policy.g1_locomanipulation_policy_cfg import G1Locomanipulation23ObsDoF, G1Locomanipulation23PolicyCfg
from .policy.g1_supermimic_policy_cfg import (
    G1_23SuperMimicDoF,
    G1_23SuperMimicMotionPolicyCfg,
    G1_23SuperMimicVelocityPolicyCfg,
    G1SuperMimicDoF,
    G1SuperMimicMotionPolicyCfg,
    G1SuperMimicVelocityPolicyCfg,
)

# 29-DoF checkpoints and motion use the sibling training checkout.
_SUPERMIMIC_DIR = ROOT_DIR.parent / "supermimic_mjlab"
_MOTION_CHECKPOINT = _SUPERMIMIC_DIR / "checkpoints/best/policy_best.pt"
_VELOCITY_CHECKPOINT = _SUPERMIMIC_DIR / "outputs/velocity_policy_last.pt"
_MOTION_FILE = _SUPERMIMIC_DIR / "src/assets/motions/g1/lafan1_npz/dance1_subject1.npz"

# 23-DoF assets are local to this repository; edit paths here for all entry points.
_MOTION23_CHECKPOINT = ASSETS_DIR / "models/g1/supermimic/g1_23dof/policy_best.pt"
_VELOCITY23_CHECKPOINT = ASSETS_DIR / "models/g1/supermimic/g1_23dof/velocity_policy_last.pt"
_MOTION23_FILE = ASSETS_DIR / "motions/g1/supermimic/g1_23dof/General_A5_-_Pick_Up_Box_stageii.npz"


@cfg_registry.register
class g1_supermimic_motion(RlPipelineCfg):
    """2680-D full-body reference plus torso motion anchor, simulation only."""

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg(dof=G1SuperMimicDoF(), sim_dt=0.005, sim_decimation=4)
    ctrl: list[KeyboardCtrlCfg] = [KeyboardCtrlCfg()]
    policy: G1SuperMimicMotionPolicyCfg = G1SuperMimicMotionPolicyCfg(
        checkpoint=_MOTION_CHECKPOINT, motion_file=_MOTION_FILE
    )


@cfg_registry.register
class g1_supermimic_velocity(RlPipelineCfg):
    """Reference-free velocity actor from the 730-D-reference training configuration."""

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg(dof=G1SuperMimicDoF(), sim_dt=0.005, sim_decimation=4)
    ctrl: list[KeyboardCtrlCfg] = [KeyboardCtrlCfg()]
    policy: G1SuperMimicVelocityPolicyCfg = G1SuperMimicVelocityPolicyCfg(checkpoint=_VELOCITY_CHECKPOINT)


@cfg_registry.register
class g1_supermimic_velocity_locomimic(RlLocoMimicPipelineCfg):
    """Start in velocity control; '[' enters motion tracking and ']' returns.

    Both actors control all 29 joints. The base config's empty upper-body
    overrides preserve the velocity actor's complete output.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg(dof=G1SuperMimicDoF(), sim_dt=0.005, sim_decimation=4)
    realign_on_policy_switch: bool = True
    ctrl: list[KeyboardCtrlCfg] = [
        KeyboardCtrlCfg(
            triggers={
                "Key.esc": "[SHUTDOWN]",
                "`": "[SIM_REBORN]",
                "[": "[POLICY_MIMIC]",
                "]": "[POLICY_LOCO]",
                ";": "[POLICY_SWITCH],NEXT",
                "'": "[POLICY_SWITCH],LAST",
            }
        )
    ]
    loco_policy: G1SuperMimicVelocityPolicyCfg = G1SuperMimicVelocityPolicyCfg(checkpoint=_VELOCITY_CHECKPOINT)
    mimic_policies: list[G1SuperMimicMotionPolicyCfg] = [
        G1SuperMimicMotionPolicyCfg(checkpoint=_MOTION_CHECKPOINT, motion_file=_MOTION_FILE)
    ]


@cfg_registry.register
class g1_23_supermimic_motion(RlPipelineCfg):
    """Native 23-DoF motion tracking with 2560-D references and 75-D proprioception."""

    robot: str = "g1"
    env: G1_23MujocoEnvCfg = G1_23MujocoEnvCfg(dof=G1_23SuperMimicDoF(), sim_dt=0.005, sim_decimation=4)
    ctrl: list[KeyboardCtrlCfg] = [KeyboardCtrlCfg()]
    policy: G1_23SuperMimicMotionPolicyCfg = G1_23SuperMimicMotionPolicyCfg(
        checkpoint=_MOTION23_CHECKPOINT, motion_file=_MOTION23_FILE
    )


@cfg_registry.register
class g1_23_supermimic_velocity(RlPipelineCfg):
    """Native 23-DoF velocity tracking with 10 x 75-D history and a 3-D command."""

    robot: str = "g1"
    env: G1_23MujocoEnvCfg = G1_23MujocoEnvCfg(dof=G1_23SuperMimicDoF(), sim_dt=0.005, sim_decimation=4)
    ctrl: list[KeyboardCtrlCfg] = [KeyboardCtrlCfg()]
    policy: G1_23SuperMimicVelocityPolicyCfg = G1_23SuperMimicVelocityPolicyCfg(checkpoint=_VELOCITY23_CHECKPOINT)


@cfg_registry.register
class g1_23_supermimic_velocity_locomimic(RlLocoMimicPipelineCfg):
    """SuperMimic velocity locomotion and motion tracking, both controlling all 23 joints."""

    robot: str = "g1"
    env: G1_23MujocoEnvCfg = G1_23MujocoEnvCfg(dof=G1_23SuperMimicDoF(), sim_dt=0.005, sim_decimation=4)
    realign_on_policy_switch: bool = True
    ctrl: list[KeyboardCtrlCfg] = [
        KeyboardCtrlCfg(
            triggers={
                "Key.esc": "[SHUTDOWN]",
                "`": "[SIM_REBORN]",
                "[": "[POLICY_MIMIC]",
                "]": "[POLICY_LOCO]",
                ";": "[POLICY_SWITCH],NEXT",
                "'": "[POLICY_SWITCH],LAST",
            }
        )
    ]
    loco_policy: G1_23SuperMimicVelocityPolicyCfg = G1_23SuperMimicVelocityPolicyCfg(checkpoint=_VELOCITY23_CHECKPOINT)
    mimic_policies: list[G1_23SuperMimicMotionPolicyCfg] = [
        G1_23SuperMimicMotionPolicyCfg(checkpoint=_MOTION23_CHECKPOINT, motion_file=_MOTION23_FILE)
    ]
    upper_dof_num: int = 0
    upper_dof_pos_default: list[float] | None = None
    upper_dof_override_indices: list[int] | None = []


@cfg_registry.register
class g1_23_maniloco_supermimic(RlLocoMimicPipelineCfg):
    """Locomanipulation locomotion plus 23-DoF SuperMimic motion tracking."""

    robot: str = "g1"
    env: G1_23MujocoEnvCfg = G1_23MujocoEnvCfg(dof=G1Locomanipulation23ObsDoF(), sim_dt=0.005, sim_decimation=4)
    realign_on_policy_switch: bool = True
    ctrl: list[KeyboardCtrlCfg] = [
        KeyboardCtrlCfg(
            triggers={
                "Key.esc": "[SHUTDOWN]",
                "`": "[SIM_REBORN]",
                "[": "[POLICY_MIMIC]",
                "]": "[POLICY_LOCO]",
                ";": "[POLICY_SWITCH],NEXT",
                "'": "[POLICY_SWITCH],LAST",
            }
        )
    ]
    loco_policy: G1Locomanipulation23PolicyCfg = G1Locomanipulation23PolicyCfg()
    mimic_policies: list[G1_23SuperMimicMotionPolicyCfg] = [
        G1_23SuperMimicMotionPolicyCfg(checkpoint=_MOTION23_CHECKPOINT, motion_file=_MOTION23_FILE)
    ]
    # Locomanipulation controls legs + waist yaw; hold/interpolate only the arms.
    # For a full-body locomotion policy, set upper_dof_num=0 and clear both
    # upper_dof_pos_default and upper_dof_override_indices.
    upper_dof_num: int = 10
    upper_dof_pos_default: list[float] | None = G1Locomanipulation23ObsDoF().default_pos[-10:]
    upper_dof_override_indices: list[int] | None = list(range(-10, 0))


@cfg_registry.register
class g1_23_supermimic_velocity_locomimic_real(g1_23_supermimic_velocity_locomimic):
    """23-DoF SuperMimic velocity/motion switching with G1 four-mode hardware control."""

    pipeline_type: str = "G1LocomanipulationLocoMimicPipeline"
    env: G1_23RealEnvCfg = G1_23RealEnvCfg(
        dof=G1_23SuperMimicDoF(),
        unitree=G1UnitreeCfg(
            net_if="enp58s0",  # Set to the robot-facing network interface.
            command_timeout=0.1,
            state_timeout=0.1,
            shutdown_damping=5.0,
        ),
    )
    ctrl: list[UnitreeCtrlCfg] = [
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
                "L1+R1+A": "[SHUTDOWN]",
            },
        )
    ]
    joint_default_dof: G1_23SuperMimicDoF = G1_23SuperMimicDoF()
    joint_default_duration: float = 1.5
    default_damping: float = 5.0
    do_safety_check: bool = True


@cfg_registry.register
class g1_23_maniloco_supermimic_real(g1_23_maniloco_supermimic):
    """23-DoF Locomanipulation/SuperMimic switching with G1 four-mode hardware control."""

    pipeline_type: str = "G1LocomanipulationLocoMimicPipeline"
    env: G1_23RealEnvCfg = G1_23RealEnvCfg(
        dof=G1Locomanipulation23ObsDoF.from_preset("stiff"),
        unitree=G1UnitreeCfg(
            net_if="eth0",  # Set to the robot-facing network interface.
            command_timeout=0.1,
            state_timeout=0.1,
            shutdown_damping=5.0,
        ),
    )
    ctrl: list[UnitreeCtrlCfg | UpperBodyZmqCtrlCfg] = [
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
                "L1+R1+A": "[SHUTDOWN]",
            },
        ),
        UpperBodyZmqCtrlCfg(joint_names=G1Locomanipulation23ObsDoF().joint_names[13:]),
    ]
    joint_default_dof: G1Locomanipulation23ObsDoF = G1Locomanipulation23ObsDoF.from_preset("stiff")
    joint_default_duration: float = 1.5
    default_damping: float = 5.0
    do_safety_check: bool = True
