import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

from robojudo.controller.ctrl_cfgs import RosJoystickCtrlCfg
from robojudo.controller.ros_joystick_ctrl import RosJoystickCtrl
from robojudo.controller.utils.ros_joystick import RosJoyTranslator, neutral_axes


def sample(axes=None, buttons=None, stamp=100.0):
    return SimpleNamespace(
        axes=list(axes or [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        buttons=list(buttons or [0] * 13),
        stamp_sec=int(stamp),
        stamp_nanosec=int((stamp % 1.0) * 1e9),
    )


class FakeSubscriber:
    def __init__(self, results):
        self.results = list(results)
        self.closed = False

    def poll(self):
        return self.results.pop(0)

    def close(self):
        self.closed = True


class TestRosJoyTranslator(unittest.TestCase):
    def test_xbox_axes_triggers_buttons_and_dpad(self):
        translator = RosJoyTranslator("xbox")
        buttons = [0] * 11
        buttons[0] = 1
        result = translator.translate(
            [0.25, -0.5, -1.0, -0.75, 0.4, 0.0, 1.0, 1.0],
            buttons,
            12.5,
        )

        self.assertEqual(
            result.axes,
            {"LeftX": -0.25, "LeftY": -0.5, "RightX": 0.75, "RightY": 0.4, "LT": 1.0, "RT": 0.5},
        )
        self.assertEqual(
            [(event["name"], event["pressed"]) for event in result.events],
            [("A", True), ("Left", True), ("Up", True)],
        )
        self.assertEqual(result.invalid_fields, [])

        unchanged = translator.translate(
            [0.25, -0.5, -1.0, -0.75, 0.4, 0.0, 1.0, 1.0],
            buttons,
            13.0,
        )
        self.assertEqual(unchanged.events, [])

    def test_calibrated_horizontal_stick_directions_are_normalized(self):
        for profile, button_count in (("xbox", 11), ("ps5", 13)):
            with self.subTest(profile=profile):
                translator = RosJoyTranslator(profile)
                result = translator.translate(
                    [-1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 0.0, 0.0],
                    [0] * button_count,
                    1.0,
                )
                self.assertEqual(result.axes["LeftX"], 1.0)
                self.assertEqual(result.axes["LeftY"], 1.0)
                self.assertEqual(result.axes["RightX"], 1.0)
                self.assertEqual(result.axes["RightY"], 1.0)

    def test_calibrated_button_index_maps(self):
        expected = {
            "xbox": {0: "A", 1: "B", 2: "X", 3: "Y", 4: "LB", 5: "RB", 6: "Back", 7: "Start", 9: "L", 10: "R"},
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
                0: "A",
                1: "B",
                2: "Y",
                3: "X",
                4: "LB",
                5: "RB",
                6: "LT",
                7: "RT",
                8: "Back",
                9: "Start",
                11: "L",
                12: "R",
            },
        }
        for profile, expected_map in expected.items():
            with self.subTest(profile=profile):
                translator = RosJoyTranslator(profile)
                self.assertEqual(
                    {index: translator.button_map[index] for index in expected_map},
                    expected_map,
                )

    def test_xbox_bluetooth_calibrated_axis_layout(self):
        translator = RosJoyTranslator("xbox_bluetooth")
        result = translator.translate(
            [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0],
            [0] * 17,
            1.0,
        )
        self.assertEqual(
            result.axes,
            {"LeftX": 1.0, "LeftY": 1.0, "RightX": 1.0, "RightY": 1.0, "LT": 0.0, "RT": 1.0},
        )
        self.assertEqual(
            [(event["name"], event["pressed"]) for event in result.events],
            [("Left", True), ("Down", True)],
        )

    def test_ps5_face_buttons_use_canonical_names(self):
        translator = RosJoyTranslator("ps5")
        buttons = [0] * 13
        buttons[2] = 1  # Triangle
        result = translator.translate([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0], buttons, 1.0)
        self.assertEqual([(event["name"], event["pressed"]) for event in result.events], [("Y", True)])

        buttons[2] = 0
        buttons[3] = 1  # Square
        result = translator.translate([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0], buttons, 2.0)
        self.assertEqual(
            [(event["name"], event["pressed"]) for event in result.events],
            [("Y", False), ("X", True)],
        )

    def test_invalid_and_short_samples_are_neutral(self):
        translator = RosJoyTranslator("xbox")
        result = translator.translate([float("nan")], [], 1.0)
        self.assertEqual(result.axes, neutral_axes())
        self.assertEqual(result.events, [])
        self.assertIn("axes[0]/LeftX", result.invalid_fields)
        self.assertIn("buttons[0]/A", result.invalid_fields)


class TestRosJoystickCtrl(unittest.TestCase):
    @staticmethod
    def make_controller(results):
        controller = RosJoystickCtrl.__new__(RosJoystickCtrl)
        controller.cfg_ctrl = SimpleNamespace(timeout_s=0.5)
        controller._translator = RosJoyTranslator("xbox")
        controller._subscriber = FakeSubscriber(results)
        controller._last_dropped_samples = 0
        controller._invalid_log_at = float("-inf")
        controller._drop_log_at = float("-inf")
        controller._stale = False
        return controller

    def test_config_defaults_and_validation(self):
        cfg = RosJoystickCtrlCfg()
        self.assertEqual(cfg.ctrl_type, "RosJoystickCtrl")
        self.assertEqual(cfg.topic, "/joy")
        self.assertEqual(cfg.profile, "xbox")
        self.assertEqual(cfg.timeout_s, 0.5)

        with self.assertRaises(ValidationError):
            RosJoystickCtrlCfg(profile="unknown")
        self.assertEqual(RosJoystickCtrlCfg(profile="xbox_bluetooth").profile, "xbox_bluetooth")
        with self.assertRaises(ValidationError):
            RosJoystickCtrlCfg(topic=" ")
        with self.assertRaises(ValidationError):
            RosJoystickCtrlCfg(timeout_s=0.0)

    def test_fresh_state_then_stale_release(self):
        buttons = [0] * 11
        buttons[0] = 1
        results = [
            SimpleNamespace(samples=[sample(buttons=buttons)], has_received=True, age_s=0.01, dropped_samples=0),
            SimpleNamespace(samples=[], has_received=True, age_s=0.51, dropped_samples=0),
            SimpleNamespace(samples=[], has_received=True, age_s=1.0, dropped_samples=0),
        ]
        controller = self.make_controller(results)

        with patch("robojudo.controller.ros_joystick_ctrl.time.monotonic", return_value=1.0):
            with patch("robojudo.controller.ros_joystick_ctrl.time.time", return_value=200.0):
                fresh = controller.get_data()
        self.assertEqual(fresh["button_event"][0]["name"], "A")
        self.assertTrue(fresh["button_event"][0]["pressed"])

        with patch("robojudo.controller.ros_joystick_ctrl.time.monotonic", return_value=2.0):
            with patch("robojudo.controller.ros_joystick_ctrl.time.time", return_value=201.0):
                stale = controller.get_data()
        self.assertEqual(stale["axes"], neutral_axes())
        self.assertEqual([(event["name"], event["pressed"]) for event in stale["button_event"]], [("A", False)])

        with patch("robojudo.controller.ros_joystick_ctrl.time.monotonic", return_value=3.0):
            with patch("robojudo.controller.ros_joystick_ctrl.time.time", return_value=202.0):
                still_stale = controller.get_data()
        self.assertEqual(still_stale["button_event"], [])

    def test_close_delegates_to_native_subscriber(self):
        controller = self.make_controller([])
        controller.close()
        self.assertTrue(controller._subscriber.closed)

    def test_missing_native_extension_has_actionable_error(self):
        with patch.dict(sys.modules, {"ros2_joy_cpp": None}):
            with self.assertRaisesRegex(RuntimeError, "submodule_install.py ros2_joy_cpp"):
                RosJoystickCtrl(RosJoystickCtrlCfg())


if __name__ == "__main__":
    unittest.main()
