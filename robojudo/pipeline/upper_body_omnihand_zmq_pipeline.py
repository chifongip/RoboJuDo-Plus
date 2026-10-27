"""OmniHand specialization of the shared upper-body hand pipeline."""

from robojudo.pipeline.upper_body_hand_zmq_pipeline import UpperBodyHandZmqPipelineMixinBase


class UpperBodyOmniHandZmqPipelineMixin(UpperBodyHandZmqPipelineMixinBase):
    """Add physical OmniHand takeover and recording to an upper-body pipeline."""

    hand_controller_type = "UpperBodyOmniHandZmqCtrl"
    hand_data_key = "omnihand"
    hand_display_name = "OmniHand"


__all__ = ["UpperBodyOmniHandZmqPipelineMixin"]
