from robojudo.utils.module_registry import Registry

from .base_pipeline import Pipeline
from .pipeline_cfgs import PipelineCfg

pipeline_registry = Registry(package="robojudo.pipeline", base_class=Pipeline)


__all__ = [
    "Pipeline",
    "PipelineCfg",
    "pipeline_registry",
]


def __getattr__(name: str) -> type[Pipeline]:
    try:
        pipeline_class = pipeline_registry.get(name)
    except Exception as e:
        raise AttributeError(f"module {__name__} has no attribute {name}") from e
    print(f"[Pipeline] Dynamic import of Pipeline: {name}")
    globals()[name] = pipeline_class
    return pipeline_class


# ===== Declare all your custom pipelines here =====
pipeline_registry.add("RlPipeline", ".rl_pipeline")
pipeline_registry.add("RlMultiPolicyPipeline", ".rl_multi_policy_pipeline")
pipeline_registry.add("RlLocoMimicPipeline", ".rl_loco_mimic_pipeline")
pipeline_registry.add("X2LocomanipulationPipeline", ".x2_locomanipulation_pipeline")
pipeline_registry.add("X2Gr00tLocomanipulationPipeline", ".x2_gr00t_locomanipulation_pipeline")
pipeline_registry.add(
    "X2LocomanipulationLocoMimicPipeline",
    ".x2_locomanipulation_loco_mimic_pipeline",
)
pipeline_registry.add("G1LocomanipulationPipeline", ".g1_locomanipulation_pipeline")
pipeline_registry.add("G1Gr00tLocomanipulationPipeline", ".g1_gr00t_locomanipulation_pipeline")
pipeline_registry.add(
    "G1LocomanipulationLocoMimicPipeline",
    ".g1_locomanipulation_loco_mimic_pipeline",
)
