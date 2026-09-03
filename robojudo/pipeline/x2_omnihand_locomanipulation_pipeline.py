from robojudo.pipeline import pipeline_registry
from robojudo.pipeline.upper_body_omnihand_zmq_pipeline import UpperBodyOmniHandZmqPipelineMixin
from robojudo.pipeline.x2_locomanipulation_pipeline import X2LocomanipulationPipeline


@pipeline_registry.register
class X2OmniHandLocomanipulationPipeline(UpperBodyOmniHandZmqPipelineMixin, X2LocomanipulationPipeline):
    """X2 real pipeline with synchronized arm targets and direct dual OmniHand control."""


__all__ = ["X2OmniHandLocomanipulationPipeline"]
