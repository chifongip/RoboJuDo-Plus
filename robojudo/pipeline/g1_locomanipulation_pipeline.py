from robojudo.pipeline import pipeline_registry
from robojudo.pipeline.four_mode_pipeline import ControlMode, FourModePipelineMixin
from robojudo.pipeline.rl_pipeline import RlPipeline

G1ControlMode = ControlMode


class G1ModePipelineMixin(FourModePipelineMixin):
    """G1 compatibility wrapper for the shared four-mode pipeline."""


@pipeline_registry.register
class G1LocomanipulationPipeline(G1ModePipelineMixin, RlPipeline):
    """Four-mode G1 Locomanipulation with opt-in upper-body ZMQ targets."""


__all__ = ["G1ControlMode", "G1LocomanipulationPipeline", "G1ModePipelineMixin"]
