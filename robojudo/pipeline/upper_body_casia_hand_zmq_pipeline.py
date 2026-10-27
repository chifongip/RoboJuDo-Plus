"""CASIA Hand specialization of the shared upper-body hand pipeline."""

from robojudo.pipeline.upper_body_hand_zmq_pipeline import UpperBodyHandZmqPipelineMixinBase


class UpperBodyCasiaHandZmqPipelineMixin(UpperBodyHandZmqPipelineMixinBase):
    """Add physical CASIA Hand takeover and recording to an upper-body pipeline."""

    hand_controller_type = "UpperBodyCasiaHandZmqCtrl"
    hand_data_key = "casia_hand"
    hand_display_name = "CASIA Hand"


__all__ = ["UpperBodyCasiaHandZmqPipelineMixin"]
