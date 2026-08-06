from robojudo.config.g1.policy.g1_gr00t_locomanipulation_policy_cfg import (
    G1Gr00tLocomanipulation23PolicyCfg,
)
from robojudo.policy import policy_registry
from robojudo.policy.g1_locomanipulation_policy import G1LocomanipulationPolicy
from robojudo.policy.gr00t_locomanipulation_policy import Gr00tLocomanipulationPolicyMixin


@policy_registry.register
class G1Gr00tLocomanipulationPolicy(Gr00tLocomanipulationPolicyMixin, G1LocomanipulationPolicy):
    """G1 23-DoF Locomanipulation with GR00T velocity and height commands."""

    cfg_policy: G1Gr00tLocomanipulation23PolicyCfg
