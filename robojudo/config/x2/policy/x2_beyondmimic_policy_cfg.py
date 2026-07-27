from robojudo.policy.policy_cfgs import BeyondMimicPolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig

from ..env.x2_env_cfg import X2_31DoF, X2_HEAD_JOINT_NAMES

_X2_ENV_DOF = X2_31DoF()
_X2_TRACKING_JOINT_NAMES = [
    name for name in _X2_ENV_DOF.joint_names if name not in X2_HEAD_JOINT_NAMES
]


class X2BeyondMimicDoF(X2_31DoF):
    """29-DoF tracking layout with the two physical head joints held by the environment."""

    _subset: bool = True
    _subset_joint_names: list[str] | None = _X2_TRACKING_JOINT_NAMES


class X2BeyondMimicPolicyCfg(BeyondMimicPolicyCfg):
    policy_type: str = "X2BeyondMimicPolicy"
    robot: str = "x2"
    policy_name: str = "Walk1_subject1_wose"

    obs_dof: DoFConfig = X2BeyondMimicDoF()
    action_dof: DoFConfig = obs_dof
    # Bootstrap value; replaced by ONNX action_scale metadata at runtime.
    action_scales: list[float] = [1.0] * 29

    without_state_estimator: bool = True
    override_robot_anchor_pos: bool = True
    use_modelmeta_config: bool = True
    use_motion_from_model: bool = True
