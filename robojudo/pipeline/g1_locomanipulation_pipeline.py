from robojudo.pipeline import pipeline_registry
from robojudo.pipeline.four_mode_pipeline import FourModePipelineMixin
from robojudo.pipeline.rl_pipeline import RlPipeline


class G1FourModePipelineMixin(FourModePipelineMixin):
    """G1 extension point for four-mode control and its upper-body ZMQ support.

    Inherited capabilities:
    - ``FourModePipelineMixin``: PASSIVE, DAMPING, JOINT, and RL mode control.
    - ``UpperBodyZmqPipelineMixin`` (transitive): filtered upper-body ZMQ targets.
    """

    def _supports_observer_policy(self, policy) -> bool:
        from robojudo.policy.g1_locomanipulation_policy import G1LocomanipulationPolicy

        return isinstance(policy, G1LocomanipulationPolicy)


@pipeline_registry.register
class G1LocomanipulationPipeline(G1FourModePipelineMixin, RlPipeline):
    """Compose the G1 Locomanipulation, four-mode, and upper-body ZMQ pipeline.

    Inherited capabilities:
    - ``RlPipeline``: single-policy Locomanipulation inference and environment loop.
    - ``G1FourModePipelineMixin``: G1 four-mode deployment control.
    - ``UpperBodyZmqPipelineMixin`` (transitive): optional upper-body ZMQ override.
    """


__all__ = ["G1FourModePipelineMixin", "G1LocomanipulationPipeline"]
