from robojudo.config.x2.policy.x2_gr00t_locomanipulation_policy_cfg import (
    X2Gr00tLocomanipulationPolicyCfg,
)
from robojudo.policy import policy_registry
from robojudo.policy.gr00t_locomanipulation_policy import Gr00tLocomanipulationPolicyMixin
from robojudo.policy.x2_locomanipulation_policy import X2LocomanipulationPolicy


@policy_registry.register
class X2Gr00tLocomanipulationPolicy(Gr00tLocomanipulationPolicyMixin, X2LocomanipulationPolicy):
    """Use GR00T velocity and height outputs instead of manual axes."""

    cfg_policy: X2Gr00tLocomanipulationPolicyCfg
