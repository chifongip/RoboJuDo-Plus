import unittest

from robojudo.controller.ctrl_cfgs import JoystickCtrlCfg, KeyboardCtrlCfg, LocomanipulationPostureZmqCtrlCfg
from robojudo.controller.posture_source import PostureSourceArbiter, get_selected_posture_source


def joystick(*events, fresh=True, axes=None):
    return {"button_event": list(events), "fresh": fresh, "axes": axes or {}}


def keyboard(*events, fresh=True, pressed_keys=None):
    return {"keyboard_event": list(events), "fresh": fresh, "pressed_keys": pressed_keys or []}


def button(name, pressed=True):
    return {"type": "button", "name": name, "pressed": pressed}


def key(name, pressed=True):
    return {"type": "keyboard", "name": name, "pressed": pressed}


class TestPostureSourcePriority(unittest.TestCase):
    @staticmethod
    def arbiter():
        return PostureSourceArbiter(
            [
                LocomanipulationPostureZmqCtrlCfg(posture_priority=100),
                KeyboardCtrlCfg(posture_priority=200),
                JoystickCtrlCfg(posture_priority=300),
            ]
        )

    def test_multiple_sources_require_unique_explicit_priorities(self):
        with self.assertRaisesRegex(ValueError, "explicit posture_priority"):
            PostureSourceArbiter([LocomanipulationPostureZmqCtrlCfg(), JoystickCtrlCfg()])
        with self.assertRaisesRegex(ValueError, "must be unique"):
            PostureSourceArbiter(
                [LocomanipulationPostureZmqCtrlCfg(posture_priority=100), JoystickCtrlCfg(posture_priority=100)]
            )

    def test_manual_controllers_do_not_need_posture_priorities_without_posture_zmq(self):
        arbiter = PostureSourceArbiter([JoystickCtrlCfg(), KeyboardCtrlCfg()])
        data = {"JoystickCtrl": joystick(button("Up")), "KeyboardCtrl": keyboard()}
        self.assertIsNone(arbiter.update(data, now=0.0))

    def test_joystick_then_keyboard_then_zmq_priority_chain(self):
        arbiter = self.arbiter()
        data = {
            "JoystickCtrl": joystick(button("Up")),
            "KeyboardCtrl": keyboard(key("r")),
            "LocomanipulationPostureZmqCtrl": {"fresh": True},
        }
        self.assertEqual(arbiter.update(data, now=1.0), "JoystickCtrl")

        data["JoystickCtrl"] = joystick()
        self.assertEqual(arbiter.update(data, now=1.5001), "KeyboardCtrl")

        data["KeyboardCtrl"] = keyboard()
        self.assertEqual(arbiter.update(data, now=2.0002), "LocomanipulationPostureZmqCtrl")

    def test_stale_manual_source_releases_its_posture_lease(self):
        arbiter = self.arbiter()
        data = {
            "JoystickCtrl": joystick(button("Left")),
            "KeyboardCtrl": keyboard(),
            "LocomanipulationPostureZmqCtrl": {"fresh": True},
        }
        self.assertEqual(arbiter.update(data, now=0.0), "JoystickCtrl")
        data["JoystickCtrl"] = joystick(fresh=False)
        self.assertEqual(arbiter.update(data, now=0.1), "LocomanipulationPostureZmqCtrl")

    def test_active_manual_velocity_input_overrides_fresh_zmq(self):
        arbiter = self.arbiter()
        data = {
            "JoystickCtrl": joystick(axes={"LeftY": 0.2}),
            "KeyboardCtrl": keyboard(pressed_keys=["w"]),
            "LocomanipulationPostureZmqCtrl": {"fresh": True},
        }
        self.assertEqual(arbiter.update(data, now=0.0), "JoystickCtrl")

        data["JoystickCtrl"] = joystick()
        self.assertEqual(arbiter.update(data, now=0.5001), "KeyboardCtrl")

        data["KeyboardCtrl"] = keyboard()
        self.assertEqual(arbiter.update(data, now=1.0002), "LocomanipulationPostureZmqCtrl")

    def test_raw_multiple_sources_require_posture_arbitration_metadata(self):
        with self.assertRaisesRegex(ValueError, "require POSTURE_SOURCE"):
            get_selected_posture_source(
                {"KeyboardCtrl": keyboard(key("r")), "LocomanipulationPostureZmqCtrl": {"fresh": True}}
            )


if __name__ == "__main__":
    unittest.main()
