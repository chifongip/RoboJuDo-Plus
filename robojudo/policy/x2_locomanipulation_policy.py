from robojudo.config.x2.policy.x2_locomanipulation_policy_cfg import X2LocomanipulationPolicyCfg
from robojudo.policy import policy_registry
from robojudo.policy.locomanipulation_policy import LocomanipulationPolicyBase


@policy_registry.register
class X2LocomanipulationPolicy(LocomanipulationPolicyBase):
    cfg_policy: X2LocomanipulationPolicyCfg
