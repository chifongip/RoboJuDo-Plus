from robojudo.pipeline.pipeline_cfgs import RlLocoMimicPipelineCfg

from ..env.x2_env_cfg import X2_ARM_JOINT_NAMES
from ..policy.x2_locomanipulation_policy_cfg import X2LocomanipulationEnvDoF

_X2_LOCO_MIMIC_ENV_DOF = X2LocomanipulationEnvDoF()


class X2LocomanipulationLocoMimicPipelineCfg(RlLocoMimicPipelineCfg):
    """X2 Locomanipulation loco-mimic configuration with four deployment modes."""

    robot: str = "x2"
    pipeline_type: str = "X2LocomanipulationLocoMimicPipeline"
    joint_default_dof: X2LocomanipulationEnvDoF = _X2_LOCO_MIMIC_ENV_DOF
    joint_default_duration: float = 1.5
    default_damping: float = 5.0
    do_safety_check: bool = True
    realign_on_policy_switch: bool = True

    # The locomotion policy controls the first 15 joints. The remaining 14 arm
    # defaults and two head defaults make up the final 16 environment DoFs.
    upper_dof_num: int = 16
    upper_dof_pos_default: list[float] = _X2_LOCO_MIMIC_ENV_DOF.default_pos[-upper_dof_num:]
    upper_dof_override_indices: list[int] = [
        _X2_LOCO_MIMIC_ENV_DOF.joint_names.index(name) - _X2_LOCO_MIMIC_ENV_DOF.num_dofs
        for name in X2_ARM_JOINT_NAMES
    ]


__all__ = ["X2LocomanipulationLocoMimicPipelineCfg"]
