from robojudo.config.g1.policy.g1_locomanipulation_policy_cfg import G1LocomanipulationPolicyCfg
from robojudo.policy import policy_registry
from robojudo.policy.locomanipulation_policy import LocomanipulationPolicyBase


@policy_registry.register
class G1LocomanipulationPolicy(LocomanipulationPolicyBase):
    cfg_policy: G1LocomanipulationPolicyCfg
