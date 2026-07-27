from robojudo.pipeline import pipeline_registry
from robojudo.pipeline.locomanipulation_loco_mimic_pipeline import LocomanipulationLocoMimicPipelineMixin
from robojudo.pipeline.rl_loco_mimic_pipeline import RlLocoMimicPipeline
from robojudo.pipeline.x2_locomanipulation_pipeline import X2FourModePipelineMixin


@pipeline_registry.register
class X2LocomanipulationLocoMimicPipeline(
    LocomanipulationLocoMimicPipelineMixin,
    X2FourModePipelineMixin,
    RlLocoMimicPipeline,
):
    """Compose X2 Locomanipulation, loco-mimic, four-mode, and upper-body ZMQ behavior.

    Inherited capabilities:
    - ``RlLocoMimicPipeline``: locomotion/mimic policy runtime and interpolation.
    - ``LocomanipulationLocoMimicPipelineMixin``: Locomanipulation-aware policy switching.
    - ``X2FourModePipelineMixin``: X2 four-mode deployment control.
    - ``UpperBodyZmqPipelineMixin`` (transitive): optional upper-body ZMQ override.
    """


__all__ = ["X2LocomanipulationLocoMimicPipeline"]
