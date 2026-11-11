"""Shared physical-hand takeover and recorder composition."""


class UpperBodyHandZmqPipelineMixinBase:
    """Connect one hand-specific controller to upper-body control and recording."""

    hand_controller_type: str
    hand_data_key: str
    hand_display_name: str

    def _post_mode_step(self, env_data, ctrl_data, extras, pd_target, rl_active: bool):
        self._upper_body_hand_ctrl_data = ctrl_data
        try:
            return super()._post_mode_step(env_data, ctrl_data, extras, pd_target, rl_active)
        finally:
            self._upper_body_hand_ctrl_data = None

    def _upper_body_record_hand_data(self):
        feedback_data = super()._upper_body_record_hand_data()
        if feedback_data is not None:
            return feedback_data
        ctrl_data = getattr(self, "_upper_body_hand_ctrl_data", None) or {}
        return ctrl_data.get(self.hand_controller_type, {}).get(self.hand_data_key) or {}


__all__ = ["UpperBodyHandZmqPipelineMixinBase"]
