import math
from dataclasses import dataclass
from numbers import Real

AXIS_NAMES = ("LeftX", "LeftY", "RightX", "RightY", "LT", "RT")

BUTTON_MAPS = {
    "xbox": {
        0: "A",
        1: "B",
        2: "X",
        3: "Y",
        4: "LB",
        5: "RB",
        6: "Back",
        7: "Start",
        8: "Xbox",
        9: "L",
        10: "R",
    },
    "xbox_bluetooth": {
        0: "A",
        1: "B",
        3: "X",
        4: "Y",
        6: "LB",
        7: "RB",
        11: "Start",
        13: "L",
        14: "R",
        15: "Back",
    },
    "ps5": {
        0: "A",  # Cross
        1: "B",  # Circle
        2: "Y",  # Triangle
        3: "X",  # Square
        4: "LB",  # L1
        5: "RB",  # R1
        6: "LT",  # L2 digital button
        7: "RT",  # R2 digital button
        8: "Back",  # Create
        9: "Start",  # Options
        10: "Xbox",  # PS
        11: "L",  # L3
        12: "R",  # R3
    },
    # PS5 Bluetooth calibration: face buttons 0-3, shoulders/triggers 4-7,
    # Create/Options 8-9, and stick clicks 11-12.
    "ps5_bluetooth": {
        0: "A",  # Cross
        1: "B",  # Circle
        2: "Y",  # Triangle
        3: "X",  # Square
        4: "LB",  # L1
        5: "RB",  # R1
        6: "LT",  # L2 digital button
        7: "RT",  # R2 digital button
        8: "Back",  # Create
        9: "Start",  # Options
        10: "Xbox",  # PS
        11: "L",  # L3
        12: "R",  # R3
    },
    # Jetson PS5 Bluetooth calibration: face buttons 0-3, Create/Options 4/6,
    # stick clicks 7-8, and shoulders 9-10.
    "ps5_bluetooth_jetson": {
        0: "A",  # Cross
        1: "B",  # Circle
        2: "X",  # Square
        3: "Y",  # Triangle
        4: "Back",  # Create
        6: "Start",  # Options
        7: "L",  # L3
        8: "R",  # R3
        9: "LB",  # L1
        10: "RB",  # R1
    },
}

# This Jetson joy_node layout sends D-pad directions as individual buttons,
# rather than the two D-pad axes used by the other supported profiles.
DPAD_BUTTON_MAPS = {
    "ps5_bluetooth_jetson": {11: "Up", 12: "Down", 13: "Left", 14: "Right"},
}

# Measured joy_node layouts report horizontal sticks with right=-1.
# RoboJuDo's pygame joystick contract uses right=+1, so normalize the sign here.
AXIS_SIGNS = {
    "xbox": {"LeftX": -1.0, "LeftY": 1.0, "RightX": -1.0, "RightY": 1.0},
    "xbox_bluetooth": {"LeftX": -1.0, "LeftY": 1.0, "RightX": -1.0, "RightY": 1.0},
    "ps5": {"LeftX": -1.0, "LeftY": 1.0, "RightX": -1.0, "RightY": 1.0},
    "ps5_bluetooth": {"LeftX": -1.0, "LeftY": 1.0, "RightX": -1.0, "RightY": 1.0},
    "ps5_bluetooth_jetson": {"LeftX": -1.0, "LeftY": 1.0, "RightX": -1.0, "RightY": 1.0},
}

AXIS_MAPS = {
    "xbox": {"LeftX": 0, "LeftY": 1, "LT": 2, "RightX": 3, "RightY": 4, "RT": 5, "DPadX": 6, "DPadY": 7},
    "xbox_bluetooth": {
        "LeftX": 0,
        "LeftY": 1,
        "RightX": 2,
        "RightY": 3,
        "RT": 4,
        "LT": 5,
        "DPadX": 6,
        "DPadY": 7,
    },
    "ps5": {"LeftX": 0, "LeftY": 1, "LT": 2, "RightX": 3, "RightY": 4, "RT": 5, "DPadX": 6, "DPadY": 7},
    "ps5_bluetooth": {
        "LeftX": 0,
        "LeftY": 1,
        "LT": 2,
        "RightX": 3,
        "RightY": 4,
        "RT": 5,
        "DPadX": 6,
        "DPadY": 7,
    },
    "ps5_bluetooth_jetson": {"LeftX": 0, "LeftY": 1, "RightX": 2, "RightY": 3, "LT": 4, "RT": 5},
}


def neutral_axes() -> dict[str, float]:
    return {name: 0.0 for name in AXIS_NAMES}


@dataclass
class TranslationResult:
    axes: dict[str, float]
    events: list[dict]
    invalid_fields: list[str]


class RosJoyTranslator:
    """Translate raw joy_node arrays into RoboJuDo's canonical joystick data."""

    def __init__(self, profile: str):
        if profile not in BUTTON_MAPS:
            raise ValueError(f"Unsupported ROS joystick profile: {profile}")
        self.profile = profile
        self.button_map = BUTTON_MAPS[profile]
        self.axis_signs = AXIS_SIGNS[profile]
        self.axis_map = AXIS_MAPS[profile]
        self.reset()

    def reset(self):
        self.axes = neutral_axes()
        self._pressed: dict[str, bool] = {
            **{name: False for name in self.button_map.values()},
            "Up": False,
            "Down": False,
            "Left": False,
            "Right": False,
        }

    @staticmethod
    def _axis(values, index: int, default: float, name: str, invalid_fields: list[str]) -> float:
        if index >= len(values):
            invalid_fields.append(name)
            return default
        value = values[index]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
            invalid_fields.append(name)
            return default
        return max(-1.0, min(1.0, float(value)))

    @staticmethod
    def _button(values, index: int, name: str, invalid_fields: list[str]) -> bool:
        if index >= len(values):
            invalid_fields.append(name)
            return False
        value = values[index]
        if isinstance(value, bool):
            return value
        if not isinstance(value, Real) or not math.isfinite(float(value)):
            invalid_fields.append(name)
            return False
        return bool(value)

    @staticmethod
    def _trigger(value: float) -> float:
        # joy_node reports common gamepad triggers as +1 released and -1 fully pressed.
        return round((1.0 - value) * 0.5, 3)

    def translate(self, raw_axes, raw_buttons, timestamp: float) -> TranslationResult:
        invalid_fields: list[str] = []
        left_x_index = self.axis_map["LeftX"]
        left_y_index = self.axis_map["LeftY"]
        left_trigger_index = self.axis_map["LT"]
        right_x_index = self.axis_map["RightX"]
        right_y_index = self.axis_map["RightY"]
        right_trigger_index = self.axis_map["RT"]
        left_x = self._axis(raw_axes, left_x_index, 0.0, f"axes[{left_x_index}]/LeftX", invalid_fields)
        left_y = self._axis(raw_axes, left_y_index, 0.0, f"axes[{left_y_index}]/LeftY", invalid_fields)
        left_trigger = self._axis(
            raw_axes, left_trigger_index, 1.0, f"axes[{left_trigger_index}]/LT", invalid_fields
        )
        right_x = self._axis(raw_axes, right_x_index, 0.0, f"axes[{right_x_index}]/RightX", invalid_fields)
        right_y = self._axis(raw_axes, right_y_index, 0.0, f"axes[{right_y_index}]/RightY", invalid_fields)
        right_trigger = self._axis(
            raw_axes, right_trigger_index, 1.0, f"axes[{right_trigger_index}]/RT", invalid_fields
        )
        self.axes = {
            "LeftX": round(left_x * self.axis_signs["LeftX"], 3),
            "LeftY": round(left_y * self.axis_signs["LeftY"], 3),
            "RightX": round(right_x * self.axis_signs["RightX"], 3),
            "RightY": round(right_y * self.axis_signs["RightY"], 3),
            "LT": self._trigger(left_trigger),
            "RT": self._trigger(right_trigger),
        }

        current = {
            name: self._button(raw_buttons, index, f"buttons[{index}]/{name}", invalid_fields)
            for index, name in self.button_map.items()
        }
        dpad_button_map = DPAD_BUTTON_MAPS.get(self.profile)
        if dpad_button_map is not None:
            current.update(
                {
                    name: self._button(raw_buttons, index, f"buttons[{index}]/{name}", invalid_fields)
                    for index, name in dpad_button_map.items()
                }
            )
        else:
            dpad_x_index = self.axis_map["DPadX"]
            dpad_y_index = self.axis_map["DPadY"]
            dpad_x = self._axis(raw_axes, dpad_x_index, 0.0, f"axes[{dpad_x_index}]/DPadX", invalid_fields)
            dpad_y = self._axis(raw_axes, dpad_y_index, 0.0, f"axes[{dpad_y_index}]/DPadY", invalid_fields)
            current.update(
                {
                    "Left": dpad_x > 0.5,
                    "Right": dpad_x < -0.5,
                    "Up": dpad_y > 0.5,
                    "Down": dpad_y < -0.5,
                }
            )

        events = []
        for name, pressed in current.items():
            if pressed != self._pressed[name]:
                events.append(
                    {
                        "type": "button",
                        "name": name,
                        "pressed": pressed,
                        "timestamp": timestamp,
                    }
                )
        self._pressed = current
        return TranslationResult(self.axes.copy(), events, invalid_fields)

    def release_all(self, timestamp: float) -> list[dict]:
        events = [
            {
                "type": "button",
                "name": name,
                "pressed": False,
                "timestamp": timestamp,
            }
            for name, pressed in self._pressed.items()
            if pressed
        ]
        for name in self._pressed:
            self._pressed[name] = False
        self.axes = neutral_axes()
        return events
