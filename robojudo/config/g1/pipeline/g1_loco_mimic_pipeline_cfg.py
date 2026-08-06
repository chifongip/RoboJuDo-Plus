from robojudo.pipeline.pipeline_cfgs import RlLocoMimicPipelineCfg

from ..policy.g1_locomanipulation_policy_cfg import (
    G1Locomanipulation23ObsDoF,
    G1Locomanipulation29ObsDoF,
)


class G1RlLocoMimicPipelineCfg(RlLocoMimicPipelineCfg):
    """Base configuration for the G1 loco-mimic pipeline."""

    robot: str = "g1"
    realign_on_policy_switch: bool = True

    upper_dof_num: int = 17
    # fmt: off
    upper_dof_pos_default: list[float] | None = [
        0.0, 0.0, 0.0,
        0.0, 0.3, 0.0, 1.0, 0.0, 0.0, 0.0,
        0.0, -0.3, 0.0, 1.0, 0.0, 0.0, 0.0,
    ]
    """Default positions of the upper body DOFs."""
    upper_dof_override_indices: list[int] | None = [
        -17,
        -14, -13, -12, -11, -10, -9, -8,
        -7, -6, -5, -4, -3, -2, -1,
    ]
    """Indices of upper-body DOFs to override; waist roll and pitch are excluded."""
    # fmt: on


class G1LocomanipulationLocoMimicPipelineCfg(RlLocoMimicPipelineCfg):
    """Shared G1 Locomanipulation loco-mimic deployment settings."""

    robot: str = "g1"
    pipeline_type: str = "G1LocomanipulationLocoMimicPipeline"
    joint_default_duration: float = 1.5
    default_damping: float = 5.0
    do_safety_check: bool = True
    realign_on_policy_switch: bool = True


class G1Locomanipulation23LocoMimicPipelineCfg(G1LocomanipulationLocoMimicPipelineCfg):
    """Logical 23-DOF G1 Locomanipulation loco-mimic deployment settings."""

    joint_default_dof: G1Locomanipulation23ObsDoF = G1Locomanipulation23ObsDoF.from_preset("stiff")
    upper_dof_num: int = 10
    upper_dof_pos_default: list[float] = joint_default_dof.default_pos[-upper_dof_num:]
    upper_dof_override_indices: list[int] = list(range(-upper_dof_num, 0))


class G1Locomanipulation29LocoMimicPipelineCfg(G1LocomanipulationLocoMimicPipelineCfg):
    """Native 29-DOF G1 Locomanipulation loco-mimic deployment settings."""

    joint_default_dof: G1Locomanipulation29ObsDoF = G1Locomanipulation29ObsDoF.from_preset("stiff")
    upper_dof_num: int = 14
    upper_dof_pos_default: list[float] = joint_default_dof.default_pos[-upper_dof_num:]
    upper_dof_override_indices: list[int] = list(range(-upper_dof_num, 0))


__all__ = [
    "G1Locomanipulation23LocoMimicPipelineCfg",
    "G1Locomanipulation29LocoMimicPipelineCfg",
    "G1LocomanipulationLocoMimicPipelineCfg",
    "G1RlLocoMimicPipelineCfg",
]
