from robojudo.config.x2.policy.x2_beyondmimic_policy_cfg import X2BeyondMimicPolicyCfg
from robojudo.policy import policy_registry
from robojudo.policy.beyondmimic_policy import BeyondMimicPolicyBase
import logging
logger = logging.getLogger(__name__)


@policy_registry.register
class X2BeyondMimicPolicy(BeyondMimicPolicyBase):
    """X2 tracking policy with a torso-IMU fallback for no-state deployment."""

    cfg_policy: X2BeyondMimicPolicyCfg

