import unittest
from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch


class TestJoystickCtrl(unittest.TestCase):
    @staticmethod
    def _controller():
        from robojudo.controller.joystick_ctrl import JoystickCtrl

        controller = JoystickCtrl.__new__(JoystickCtrl)
        controller.triggers = {
            "RB": "[POLICY_SWITCH],NEXT",
            "LB": "[POLICY_SWITCH],LAST",
            "LB+RB+A": "[SHUTDOWN]",
        }
        controller.combination_init_buttons = ["LB", "RB"]
        controller.onhold_buttons = set()
        controller.used_combination_buttons = set()
        return controller

    @staticmethod
    def _button(name, pressed):
        return {"type": "button", "name": name, "pressed": pressed}

    def test_standalone_combination_button_triggers_on_release(self):
        controller = self._controller()
        ctrl_data = {
            "button_event": [
                self._button("RB", True),
                self._button("RB", False),
                self._button("LB", True),
                self._button("LB", False),
            ]
        }

        ctrl_data, commands = controller.process_triggers(ctrl_data)

        self.assertEqual(commands, ["[POLICY_SWITCH],NEXT", "[POLICY_SWITCH],LAST"])
        self.assertEqual(
            ctrl_data["button_event"],
            [self._button("RB", True), self._button("LB", True)],
        )
        self.assertEqual(controller.onhold_buttons, set())

    def test_recognized_chord_suppresses_standalone_release_triggers(self):
        controller = self._controller()
        ctrl_data = {
            "button_event": [
                self._button("LB", True),
                self._button("RB", True),
                self._button("A", True),
                self._button("A", False),
                self._button("RB", False),
                self._button("LB", False),
            ]
        }

        ctrl_data, commands = controller.process_triggers(ctrl_data)

        self.assertEqual(commands, ["[SHUTDOWN]"])
        self.assertNotIn(self._button("A", True), ctrl_data["button_event"])
        self.assertEqual(controller.onhold_buttons, set())
        self.assertEqual(controller.used_combination_buttons, set())

    def test_recording_combinations_are_installed_for_standard_joystick(self):
        controller = self._controller()
        controller.cfg_ctrl = SimpleNamespace(ctrl_type="JoystickCtrl")
        controller._install_recording_triggers()
        self.assertEqual(controller.triggers["LB+RB+Start"], "[RECORD_START_STOP]")
        self.assertEqual(controller.triggers["LB+RB+Back"], "[RECORD_PAUSE_RESUME]")

    def test_recording_combinations_are_installed_for_unitree_remote(self):
        controller = self._controller()
        controller.combination_init_buttons = ["L1", "R1"]
        controller.cfg_ctrl = SimpleNamespace(ctrl_type="UnitreeCtrl")
        controller._install_recording_triggers()
        self.assertEqual(controller.triggers["L1+R1+Start"], "[RECORD_START_STOP]")
        self.assertEqual(controller.triggers["L1+R1+Select"], "[RECORD_PAUSE_RESUME]")

    def test_stale_axis_state_is_neutralized(self):
        from robojudo.controller.joystick_ctrl import JoystickCtrl

        controller = JoystickCtrl.__new__(JoystickCtrl)
        controller.cfg_ctrl = SimpleNamespace(timeout_s=0.5)
        controller.axes_names = ["LeftX", "LeftY", "RightX", "RightY"]
        controller.state_queue = Queue()
        controller.event_queue = Queue()
        controller._last_received_at = None
        controller.last_state = {
            "axes": {"LeftX": 0.0, "LeftY": 0.0, "RightX": 0.0, "RightY": 0.0}
        }
        controller.state_queue.put(
            {
                "axes": {"LeftX": 0.0, "LeftY": 0.8, "RightX": 0.0, "RightY": 0.0}
            }
        )

        with patch("robojudo.controller.joystick_ctrl.time.monotonic", side_effect=[1.0, 1.1]):
            fresh = controller.get_data()
        with patch("robojudo.controller.joystick_ctrl.time.monotonic", return_value=1.6):
            stale = controller.get_data()

        self.assertTrue(fresh["fresh"])
        self.assertEqual(fresh["axes"]["LeftY"], 0.8)
        self.assertFalse(stale["fresh"])
        self.assertTrue(all(value == 0.0 for value in stale["axes"].values()))


if __name__ == "__main__":
    unittest.main()
