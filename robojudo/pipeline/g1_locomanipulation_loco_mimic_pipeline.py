from robojudo.pipeline import pipeline_registry
from robojudo.pipeline.g1_locomanipulation_pipeline import G1FourModePipelineMixin
from robojudo.pipeline.locomanipulation_loco_mimic_pipeline import LocomanipulationLocoMimicPipelineMixin
from robojudo.pipeline.rl_loco_mimic_pipeline import RlLocoMimicPipeline


@pipeline_registry.register
class G1LocomanipulationLocoMimicPipeline(
    LocomanipulationLocoMimicPipelineMixin,
    G1FourModePipelineMixin,
    RlLocoMimicPipeline,
):
    """Compose G1 Locomanipulation, loco-mimic, four-mode, and upper-body ZMQ behavior.

    Inherited capabilities:
    - ``RlLocoMimicPipeline``: locomotion/mimic policy runtime and interpolation.
    - ``LocomanipulationLocoMimicPipelineMixin``: Locomanipulation-aware policy switching.
    - ``G1FourModePipelineMixin``: G1 four-mode deployment control.
    - ``UpperBodyZmqPipelineMixin`` (transitive): optional upper-body ZMQ override.
    """


__all__ = ["G1LocomanipulationLocoMimicPipeline"]
