import logging
import time

from robojudo.controller.ctrl_cfgs import VelocitySourceCfg

logger = logging.getLogger(__name__)

VELOCITY_SOURCE_KEY = "VELOCITY_SOURCE"
JOYSTICK_SOURCE_TYPES = frozenset({"JoystickCtrl", "RosJoystickCtrl", "UnitreeCtrl"})
KEYBOARD_VELOCITY_KEYS = frozenset({"w", "s", "a", "d", "q", "e"})


class VelocitySourceArbiter:
    """Select the highest-priority active locomotion velocity source."""

    def __init__(self, cfg_ctrls: list):
        source_cfgs = [cfg for cfg in cfg_ctrls if isinstance(cfg, VelocitySourceCfg)]
        source_names = [cfg.ctrl_type for cfg in source_cfgs]
        duplicates = sorted({name for name in source_names if source_names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Velocity source controller types must be unique; duplicated: {', '.join(duplicates)}")
        self.cfg_by_type = {cfg.ctrl_type: cfg for cfg in source_cfgs}
        self._validate_priorities()
        self.reset()

    def _validate_priorities(self):
        if len(self.cfg_by_type) <= 1:
            return

        missing = [name for name, cfg in self.cfg_by_type.items() if cfg.velocity_priority is None]
        if missing:
            raise ValueError(
                "Multiple velocity sources require explicit velocity_priority values; "
                f"missing for: {', '.join(missing)}"
            )

        priorities: dict[int, str] = {}
        for name, cfg in self.cfg_by_type.items():
            priority = cfg.velocity_priority
            if priority in priorities:
                raise ValueError(
                    "Velocity source priorities must be unique; "
                    f"{priorities[priority]} and {name} both use {priority}"
                )
            priorities[priority] = name

    def reset(self):
        self._lease_expires_at = {name: float("-inf") for name in self.cfg_by_type}
        self._keyboard_held_keys: set[str] = set()
        self.selected_source: str | None = None

    @staticmethod
    def _joystick_moving(ctrl_entry, deadzone: float) -> bool:
        axes = ctrl_entry.get("axes", {})
        return any(abs(float(axes.get(name, 0.0))) > deadzone for name in ("LeftX", "LeftY", "RightX"))

    def _joystick_active(self, name: str, ctrl_entry, cfg, now: float) -> bool:
        fresh = bool(ctrl_entry.get("fresh", True))
        if fresh and self._joystick_moving(ctrl_entry, cfg.velocity_activity_deadzone):
            self._lease_expires_at[name] = now + cfg.velocity_lease_timeout_s
        return fresh and now <= self._lease_expires_at[name]

    def _keyboard_active(self, name: str, ctrl_entry, cfg, now: float) -> bool:
        if not ctrl_entry.get("fresh", True):
            self._keyboard_held_keys.clear()
            self._lease_expires_at[name] = float("-inf")
            return False
        if "pressed_keys" in ctrl_entry:
            self._keyboard_held_keys = set(KEYBOARD_VELOCITY_KEYS.intersection(ctrl_entry["pressed_keys"]))
        relevant_event = False
        for event in ctrl_entry.get("keyboard_event", []):
            if event.get("type") != "keyboard" or event.get("name") not in KEYBOARD_VELOCITY_KEYS:
                continue
            relevant_event = True
            if event.get("pressed", False):
                self._keyboard_held_keys.add(event["name"])
            else:
                self._keyboard_held_keys.discard(event["name"])

        if relevant_event or self._keyboard_held_keys:
            self._lease_expires_at[name] = now + cfg.velocity_lease_timeout_s
        return now <= self._lease_expires_at[name]

    def update(self, ctrl_data, now: float | None = None) -> str | None:
        now = time.monotonic() if now is None else now
        active_sources = []
        for name, cfg in self.cfg_by_type.items():
            ctrl_entry = ctrl_data.get(name, {})
            if name in JOYSTICK_SOURCE_TYPES:
                active = self._joystick_active(name, ctrl_entry, cfg, now)
            elif name == "KeyboardCtrl":
                active = self._keyboard_active(name, ctrl_entry, cfg, now)
            elif name == "VelocityZmqCtrl":
                active = bool(ctrl_entry.get("fresh", False))
            else:
                active = False

            if active:
                priority = cfg.velocity_priority if cfg.velocity_priority is not None else 0
                active_sources.append((priority, name))

        selected = max(active_sources)[1] if active_sources else None
        if selected != self.selected_source:
            logger.info("Velocity control source changed: %s -> %s", self.selected_source, selected)
            self.selected_source = selected
        return selected


def get_selected_velocity_source(ctrl_data) -> str | None:
    """Return the arbiter-selected source, with a single-source compatibility fallback."""

    if VELOCITY_SOURCE_KEY in ctrl_data:
        return ctrl_data[VELOCITY_SOURCE_KEY]
    available = [
        name
        for name in ctrl_data
        if name == "VelocityZmqCtrl" or name in JOYSTICK_SOURCE_TYPES or name == "KeyboardCtrl"
    ]
    if len(available) > 1:
        raise ValueError(f"Multiple velocity sources require {VELOCITY_SOURCE_KEY} arbitration metadata")
    return available[0] if available else None


def get_pressed_velocity_keys(ctrl_entry) -> set[str]:
    """Return currently pressed locomotion keys from controller state or press events."""

    if "pressed_keys" in ctrl_entry:
        return set(KEYBOARD_VELOCITY_KEYS.intersection(ctrl_entry["pressed_keys"]))
    return {
        event["name"]
        for event in ctrl_entry.get("keyboard_event", [])
        if event.get("type") == "keyboard"
        and event.get("pressed", False)
        and event.get("name") in KEYBOARD_VELOCITY_KEYS
    }
