from robojudo.pipeline import pipeline_registry
from robojudo.pipeline.gr00t_locomanipulation_pipeline import Gr00tLocomanipulationPipelineMixin
from robojudo.pipeline.x2_locomanipulation_pipeline import X2LocomanipulationPipeline


@pipeline_registry.register
class X2Gr00tLocomanipulationPipeline(Gr00tLocomanipulationPipelineMixin, X2LocomanipulationPipeline):
    """Couple atomic GR00T arm and locomotion commands to one takeover gate."""
