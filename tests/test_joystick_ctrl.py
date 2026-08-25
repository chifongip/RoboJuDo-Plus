import unittest


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

if __name__ == "__main__":
    unittest.main()
