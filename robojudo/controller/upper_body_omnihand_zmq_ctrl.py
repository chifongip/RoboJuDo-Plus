"""Upper-body and dual-OmniHand control for synchronized frames or checked feedback."""

import json
import os
from pathlib import Path

import numpy as np

from robojudo.controller import ctrl_registry
from robojudo.controller.ctrl_cfgs import UpperBodyOmniHandZmqCtrlCfg
from robojudo.controller.omnihand_runtime import (
    OMNIHAND_LEFT_JOINT_NAMES,
    OMNIHAND_LEFT_LIMITS,
    OMNIHAND_RIGHT_JOINT_NAMES,
    OMNIHAND_RIGHT_LIMITS,
    OmniHandRuntime,
)
from robojudo.controller.upper_body_hand_zmq_ctrl import UpperBodyHandZmqCtrlBase


@ctrl_registry.register
class UpperBodyOmniHandZmqCtrl(UpperBodyHandZmqCtrlBase):
    """Receive synchronized or feedback-checked arm and finger targets."""

    cfg_ctrl: UpperBodyOmniHandZmqCtrlCfg
    hand_type = "omnihand"
    hand_data_key = "omnihand"
    hand_display_name = "OmniHand"
    left_hand_joint_names = OMNIHAND_LEFT_JOINT_NAMES
    right_hand_joint_names = OMNIHAND_RIGHT_JOINT_NAMES

    def _create_hand_runtime(self, cfg_ctrl: UpperBodyOmniHandZmqCtrlCfg):
        hand_cfg = cfg_ctrl.omnihand
        config_path = cfg_ctrl.serial_config or os.environ.get("ROBOJUDO_OMNIHAND_CONFIG")
        if config_path:
            values = json.loads(Path(config_path).read_text())
            allowed = {f"{side}_{kind}_serial" for side in ("left", "right") for kind in ("hcan", "hand")}
            if not isinstance(values, dict) or set(values) != allowed:
                raise ValueError("local hand configuration must contain exactly the four serial fields")
            hand_cfg = type(hand_cfg).model_validate({**hand_cfg.model_dump(), **values})
        if cfg_ctrl.feedback_endpoint is not None:
            hand_cfg = type(hand_cfg).model_validate(
                {
                    **hand_cfg.model_dump(),
                    "joint_state_timeout_s": min(hand_cfg.joint_state_timeout_s, cfg_ctrl.feedback_timeout_s),
                }
            )
        return OmniHandRuntime(hand_cfg, require_serials=cfg_ctrl.feedback_endpoint is not None)

    def hand_feedback_payload(self):
        snapshot = self.hand_snapshot()
        names = [*self.left_hand_joint_names, *self.right_hand_joint_names]
        limits = np.concatenate((OMNIHAND_LEFT_LIMITS, OMNIHAND_RIGHT_LIMITS))
        return {
            "positions": snapshot["positions"],
            "sample_times": snapshot["sample_times"],
            "targets": snapshot["targets"],
            "defaults": snapshot["targets"],
            "limits": dict(zip(names, limits.tolist(), strict=True)),
        }


__all__ = ["UpperBodyOmniHandZmqCtrl"]
