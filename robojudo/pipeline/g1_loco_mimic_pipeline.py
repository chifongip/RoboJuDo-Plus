from robojudo.pipeline import pipeline_registry
from robojudo.pipeline.g1_locomanipulation_pipeline import G1ModePipelineMixin
from robojudo.pipeline.locomanipulation_loco_mimic_pipeline import LocomanipulationLocoMimicPipelineMixin
from robojudo.pipeline.rl_loco_mimic_pipeline import RlLocoMimicPipeline


@pipeline_registry.register
class G1LocoMimicPipeline(LocomanipulationLocoMimicPipelineMixin, G1ModePipelineMixin, RlLocoMimicPipeline):
    """G1 Locomanipulation loco-mimic pipeline with four deployment modes."""
