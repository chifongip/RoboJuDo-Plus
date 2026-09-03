"""Synchronized upper-body and dual-CASIA-Hand controller."""

from robojudo.controller import ctrl_registry
from robojudo.controller.casia_hand_runtime import (
    CASIA_LEFT_JOINT_NAMES,
    CASIA_RIGHT_JOINT_NAMES,
    CasiaHandRuntime,
)
from robojudo.controller.ctrl_cfgs import UpperBodyCasiaHandZmqCtrlCfg
from robojudo.controller.upper_body_hand_zmq_ctrl import UpperBodyHandZmqCtrlBase


@ctrl_registry.register
class UpperBodyCasiaHandZmqCtrl(UpperBodyHandZmqCtrlBase):
    """Drive both physical CASIA Hands from synchronized arm-and-hand frames."""

    cfg_ctrl: UpperBodyCasiaHandZmqCtrlCfg
    hand_type = "casia"
    hand_data_key = "casia_hand"
    hand_display_name = "CASIA Hand"
    left_hand_joint_names = CASIA_LEFT_JOINT_NAMES
    right_hand_joint_names = CASIA_RIGHT_JOINT_NAMES

    def _create_hand_runtime(self, cfg_ctrl: UpperBodyCasiaHandZmqCtrlCfg):
        return CasiaHandRuntime(cfg_ctrl.casia_hand)


__all__ = ["UpperBodyCasiaHandZmqCtrl"]
