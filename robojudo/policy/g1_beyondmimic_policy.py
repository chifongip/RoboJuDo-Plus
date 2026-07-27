from robojudo.config.g1.policy.g1_beyondmimic_policy_cfg import G1BeyondMimicPolicyCfg
from robojudo.policy import policy_registry
from robojudo.policy.beyondmimic_policy import BeyondMimicPolicyBase


@policy_registry.register
class G1BeyondMimicPolicy(BeyondMimicPolicyBase):
    """G1 entry point for the shared BeyondMimic tracking runtime."""

    cfg_policy: G1BeyondMimicPolicyCfg
