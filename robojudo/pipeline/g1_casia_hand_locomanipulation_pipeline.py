from robojudo.pipeline import pipeline_registry
from robojudo.pipeline.g1_locomanipulation_pipeline import G1LocomanipulationPipeline
from robojudo.pipeline.upper_body_casia_hand_zmq_pipeline import UpperBodyCasiaHandZmqPipelineMixin


@pipeline_registry.register
class G1CasiaHandLocomanipulationPipeline(UpperBodyCasiaHandZmqPipelineMixin, G1LocomanipulationPipeline):
    """G1 real pipeline with synchronized arm targets and direct dual CASIA Hand control."""


__all__ = ["G1CasiaHandLocomanipulationPipeline"]
