"""Synchronized upper-body and dual-OmniHand controller."""

from robojudo.controller import ctrl_registry
from robojudo.controller.ctrl_cfgs import UpperBodyOmniHandZmqCtrlCfg
from robojudo.controller.omnihand_runtime import (
    OMNIHAND_LEFT_JOINT_NAMES,
    OMNIHAND_RIGHT_JOINT_NAMES,
    OmniHandRuntime,
)
from robojudo.controller.upper_body_hand_zmq_ctrl import UpperBodyHandZmqCtrlBase


@ctrl_registry.register
class UpperBodyOmniHandZmqCtrl(UpperBodyHandZmqCtrlBase):
    """Drive both physical OmniHands from synchronized arm-and-hand frames."""

    cfg_ctrl: UpperBodyOmniHandZmqCtrlCfg
    hand_type = "omnihand"
    hand_data_key = "omnihand"
    hand_display_name = "OmniHand"
    left_hand_joint_names = OMNIHAND_LEFT_JOINT_NAMES
    right_hand_joint_names = OMNIHAND_RIGHT_JOINT_NAMES

    def _create_hand_runtime(self, cfg_ctrl: UpperBodyOmniHandZmqCtrlCfg):
        return OmniHandRuntime(cfg_ctrl.omnihand)


__all__ = ["UpperBodyOmniHandZmqCtrl"]
