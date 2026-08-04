import logging
import time

from robojudo.controller import Controller, ctrl_registry
from robojudo.controller.ctrl_cfgs import RosJoystickCtrlCfg
from robojudo.controller.joystick_ctrl import JoystickCtrl
from robojudo.controller.utils.ros_joystick import RosJoyTranslator, neutral_axes

logger = logging.getLogger(__name__)


@ctrl_registry.register
class RosJoystickCtrl(JoystickCtrl):
    """Read ``sensor_msgs/msg/Joy`` through the Python-version-neutral C++ binding."""

    cfg_ctrl: RosJoystickCtrlCfg

    def __init__(self, cfg_ctrl: RosJoystickCtrlCfg, env=None, device="cpu"):
        Controller.__init__(self, cfg_ctrl=cfg_ctrl, env=env, device=device)
        try:
            from ros2_joy_cpp import JoySubscriber
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "RosJoystickCtrl requires the native ros2_joy_cpp package. Source ROS 2 and run "
                "`python submodule_install.py ros2_joy_cpp` with RoboJuDo's Python interpreter."
            ) from exc

        self._translator = RosJoyTranslator(cfg_ctrl.profile)
        self._subscriber = JoySubscriber(topic=cfg_ctrl.topic, queue_capacity=cfg_ctrl.queue_capacity)
        self._last_dropped_samples = 0
        self._invalid_log_at = float("-inf")
        self._drop_log_at = float("-inf")
        self._stale = False
        self.reset()
        logger.info("RosJoystickCtrl subscribed to %s with profile %s", cfg_ctrl.topic, cfg_ctrl.profile)

    def reset(self):
        self.combination_init_buttons = self.cfg_ctrl.combination_init_buttons
        self.onhold_buttons = set()
        self.used_combination_buttons = set()
        self._translator.reset()
        self._stale = False
        self._subscriber.poll()

    def close(self):
        subscriber = getattr(self, "_subscriber", None)
        if subscriber is not None:
            subscriber.close()

    @staticmethod
    def _sample_timestamp(sample, fallback: float) -> float:
        stamp = float(sample.stamp_sec) + float(sample.stamp_nanosec) * 1e-9
        return stamp if stamp > 0.0 else fallback

    def _log_invalid(self, fields: list[str], now: float):
        if fields and now - self._invalid_log_at >= 1.0:
            logger.warning("ROS Joy sample has missing or invalid fields: %s", ", ".join(sorted(set(fields))))
            self._invalid_log_at = now

    def get_data(self):
        now_mono = time.monotonic()
        now_wall = time.time()
        result = self._subscriber.poll()
        events = []

        if result.dropped_samples > self._last_dropped_samples:
            if now_mono - self._drop_log_at >= 1.0:
                logger.warning(
                    "ROS Joy native queue dropped %d sample(s)",
                    result.dropped_samples - self._last_dropped_samples,
                )
                self._drop_log_at = now_mono
            self._last_dropped_samples = result.dropped_samples

        fresh = result.has_received and result.age_s <= self.cfg_ctrl.timeout_s
        if fresh:
            for sample in result.samples:
                translated = self._translator.translate(
                    sample.axes,
                    sample.buttons,
                    self._sample_timestamp(sample, now_wall),
                )
                events.extend(translated.events)
                self._log_invalid(translated.invalid_fields, now_mono)
            self._stale = False
        elif result.has_received and not self._stale:
            events.extend(self._translator.release_all(now_wall))
            self._stale = True

        axes = self._translator.axes.copy() if fresh else neutral_axes()
        return {"axes": axes, "button_event": events}

    def post_step_callback(self, commands: list[str] | None = None):
        if commands and "[SHUTDOWN]" in commands:
            self.close()
