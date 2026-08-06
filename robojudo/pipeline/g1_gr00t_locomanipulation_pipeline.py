from robojudo.pipeline import pipeline_registry
from robojudo.pipeline.g1_locomanipulation_pipeline import G1LocomanipulationPipeline
from robojudo.pipeline.gr00t_locomanipulation_pipeline import Gr00tLocomanipulationPipelineMixin


@pipeline_registry.register
class G1Gr00tLocomanipulationPipeline(Gr00tLocomanipulationPipelineMixin, G1LocomanipulationPipeline):
    """G1 23-DoF Locomanipulation with one gated GR00T command stream."""
