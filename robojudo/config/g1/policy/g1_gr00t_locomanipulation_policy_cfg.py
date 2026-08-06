from .g1_locomanipulation_policy_cfg import G1Locomanipulation23PolicyCfg


class G1Gr00tLocomanipulation23PolicyCfg(G1Locomanipulation23PolicyCfg):
    """G1 23-DoF lower-body policy driven by GR00T high-level commands."""

    policy_type: str = "G1Gr00tLocomanipulationPolicy"
