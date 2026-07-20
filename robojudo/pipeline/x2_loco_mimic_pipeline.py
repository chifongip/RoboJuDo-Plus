from robojudo.pipeline import pipeline_registry
from robojudo.pipeline.locomanipulation_loco_mimic_pipeline import LocomanipulationLocoMimicPipelineMixin
from robojudo.pipeline.rl_loco_mimic_pipeline import RlLocoMimicPipeline
from robojudo.pipeline.x2_deploy_pipeline import X2ModePipelineMixin


@pipeline_registry.register
class X2LocoMimicPipeline(LocomanipulationLocoMimicPipelineMixin, X2ModePipelineMixin, RlLocoMimicPipeline):
    """X2 loco-mimic switching guarded by the X2 deployment mode state machine."""
