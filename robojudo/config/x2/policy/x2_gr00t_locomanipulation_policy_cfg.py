from .x2_locomanipulation_policy_cfg import X2LocomanipulationPolicyCfg


class X2Gr00tLocomanipulationPolicyCfg(X2LocomanipulationPolicyCfg):
    """X2 lower-body policy driven by atomic GR00T high-level commands."""

    policy_type: str = "X2Gr00tLocomanipulationPolicy"
