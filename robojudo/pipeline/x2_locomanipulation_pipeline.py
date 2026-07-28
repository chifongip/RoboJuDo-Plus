from robojudo.pipeline import pipeline_registry
from robojudo.pipeline.four_mode_pipeline import FourModePipelineMixin
from robojudo.pipeline.rl_pipeline import RlPipeline


class X2FourModePipelineMixin(FourModePipelineMixin):
    """X2 extension point for four-mode control and its upper-body ZMQ support.

    Inherited capabilities:
    - ``FourModePipelineMixin``: PASSIVE, DAMPING, JOINT, and RL mode control.
    - ``UpperBodyZmqPipelineMixin`` (transitive): filtered upper-body ZMQ targets.
    """


@pipeline_registry.register
class X2LocomanipulationPipeline(X2FourModePipelineMixin, RlPipeline):
    """Compose the X2 Locomanipulation, four-mode, and upper-body ZMQ pipeline.

    Inherited capabilities:
    - ``RlPipeline``: single-policy Locomanipulation inference and environment loop.
    - ``X2FourModePipelineMixin``: X2 four-mode deployment control.
    - ``UpperBodyZmqPipelineMixin`` (transitive): optional upper-body ZMQ override.
    """


__all__ = ["X2FourModePipelineMixin", "X2LocomanipulationPipeline"]
