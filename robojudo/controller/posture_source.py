import logging
import time

from robojudo.controller.velocity_source import JOYSTICK_SOURCE_TYPES, KEYBOARD_VELOCITY_KEYS

logger = logging.getLogger(__name__)

POSTURE_SOURCE_KEY = "POSTURE_SOURCE"
POSTURE_ZMQ_SOURCE_TYPE = "LocomanipulationPostureZmqCtrl"
KEYBOARD_POSTURE_KEYS = frozenset({"r", "f", "z", "c", "x"})
JOYSTICK_POSTURE_BUTTONS = frozenset({"Up", "Down", "Left", "Right", "Back", "Select", "F1"})
KEYBOARD_MANUAL_CONTROL_KEYS = KEYBOARD_VELOCITY_KEYS | KEYBOARD_POSTURE_KEYS


class PostureSourceArbiter:
    """Select the highest-priority active body-height and waist-yaw source."""

    def __init__(self, cfg_ctrls: list):
        self._configured = any(cfg.ctrl_type == POSTURE_ZMQ_SOURCE_TYPE for cfg in cfg_ctrls)
        source_types = {*JOYSTICK_SOURCE_TYPES, "KeyboardCtrl", POSTURE_ZMQ_SOURCE_TYPE}
        source_cfgs = [cfg for cfg in cfg_ctrls if cfg.ctrl_type in source_types] if self._configured else []
        source_names = [cfg.ctrl_type for cfg in source_cfgs]
        duplicates = sorted({name for name in source_names if source_names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Posture source controller types must be unique; duplicated: {', '.join(duplicates)}")
        self.cfg_by_type = {cfg.ctrl_type: cfg for cfg in source_cfgs}
        self._validate_priorities()
        self.reset()

    @property
    def configured(self) -> bool:
        return self._configured

    def _validate_priorities(self):
        if len(self.cfg_by_type) <= 1:
            return

        missing = [name for name, cfg in self.cfg_by_type.items() if cfg.posture_priority is None]
        if missing:
            raise ValueError(
                "Multiple posture sources require explicit posture_priority values; "
                f"missing for: {', '.join(missing)}"
            )

        priorities: dict[int, str] = {}
        for name, cfg in self.cfg_by_type.items():
            priority = cfg.posture_priority
            if priority in priorities:
                raise ValueError(
                    "Posture source priorities must be unique; "
                    f"{priorities[priority]} and {name} both use {priority}"
                )
            priorities[priority] = name

    def reset(self):
        self._lease_expires_at = {name: float("-inf") for name in self.cfg_by_type}
        self.selected_source: str | None = None

    def _manual_active(self, name: str, ctrl_entry, cfg, now: float) -> bool:
        fresh = bool(ctrl_entry.get("fresh", True))
        if not fresh:
            self._lease_expires_at[name] = float("-inf")
            return False

        if name in JOYSTICK_SOURCE_TYPES:
            manual_event = any(
                event.get("type") == "button"
                and event.get("pressed", False)
                and event.get("name") in JOYSTICK_POSTURE_BUTTONS
                for event in ctrl_entry.get("button_event", [])
            )
            axes = ctrl_entry.get("axes", {})
            manual_event |= any(
                abs(float(axes.get(axis_name, 0.0))) > cfg.velocity_activity_deadzone
                for axis_name in ("LeftX", "LeftY", "RightX")
            )
        else:
            pressed_keys = set(ctrl_entry.get("pressed_keys", []))
            manual_event = bool(KEYBOARD_MANUAL_CONTROL_KEYS.intersection(pressed_keys)) or any(
                event.get("type") == "keyboard"
                and event.get("name") in KEYBOARD_MANUAL_CONTROL_KEYS
                for event in ctrl_entry.get("keyboard_event", [])
            )
        if manual_event:
            self._lease_expires_at[name] = now + cfg.posture_lease_timeout_s
        return now <= self._lease_expires_at[name]

    def update(self, ctrl_data, now: float | None = None) -> str | None:
        if not self._configured:
            return None

        now = time.monotonic() if now is None else now
        active_sources = []
        for name, cfg in self.cfg_by_type.items():
            ctrl_entry = ctrl_data.get(name, {})
            if name == POSTURE_ZMQ_SOURCE_TYPE:
                active = bool(ctrl_entry.get("fresh", False))
            else:
                active = self._manual_active(name, ctrl_entry, cfg, now)
            if active:
                priority = cfg.posture_priority if cfg.posture_priority is not None else 0
                active_sources.append((priority, name))

        selected = max(active_sources)[1] if active_sources else None
        if selected != self.selected_source:
            logger.info("Posture control source changed: %s -> %s", self.selected_source, selected)
            self.selected_source = selected
        return selected


def get_selected_posture_source(ctrl_data) -> str | None:
    """Return the arbiter-selected posture source with a single-source compatibility fallback."""

    if POSTURE_SOURCE_KEY in ctrl_data:
        return ctrl_data[POSTURE_SOURCE_KEY]
    available = [
        name
        for name in ctrl_data
        if name == POSTURE_ZMQ_SOURCE_TYPE or name in JOYSTICK_SOURCE_TYPES or name == "KeyboardCtrl"
    ]
    if POSTURE_ZMQ_SOURCE_TYPE not in available:
        return None
    if len(available) > 1:
        raise ValueError(f"Multiple posture sources require {POSTURE_SOURCE_KEY} arbitration metadata")
    return POSTURE_ZMQ_SOURCE_TYPE
