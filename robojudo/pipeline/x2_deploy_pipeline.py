from robojudo.pipeline import pipeline_registry
from robojudo.pipeline.four_mode_pipeline import (
    ELASTIC_BAND_COMMANDS,
    MODE_COMMANDS,
    ControlMode,
    FourModePipelineMixin,
)
from robojudo.pipeline.rl_pipeline import RlPipeline

X2ControlMode = ControlMode


class X2ModePipelineMixin(FourModePipelineMixin):
    """Compatibility wrapper for the shared four-mode deployment pipeline."""


@pipeline_registry.register
class X2DeployPipeline(X2ModePipelineMixin, RlPipeline):
    """Four-mode X2 deployment pipeline matching the standalone controller."""


__all__ = [
    "ELASTIC_BAND_COMMANDS",
    "MODE_COMMANDS",
    "X2ControlMode",
    "X2DeployPipeline",
    "X2ModePipelineMixin",
]
