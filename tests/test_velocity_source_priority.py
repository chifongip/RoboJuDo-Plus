import unittest

from robojudo.controller.ctrl_cfgs import JoystickCtrlCfg, KeyboardCtrlCfg, VelocityZmqCtrlCfg
from robojudo.controller.velocity_source import VelocitySourceArbiter, get_selected_velocity_source


def joystick(x=0.0, y=0.0, yaw=0.0, fresh=True):
    return {
        "axes": {"LeftX": x, "LeftY": y, "RightX": yaw, "RightY": 0.0},
        "button_event": [],
        "fresh": fresh,
    }


def keyboard(*events, pressed_keys=()):
    return {"keyboard_event": list(events), "pressed_keys": list(pressed_keys)}


def key(name, pressed):
    return {"type": "keyboard", "name": name, "pressed": pressed}


def zmq(fresh=True):
    return {"fresh": fresh}


class TestVelocitySourcePriority(unittest.TestCase):
    @staticmethod
    def arbiter():
        return VelocitySourceArbiter(
            [
                VelocityZmqCtrlCfg(velocity_priority=100),
                KeyboardCtrlCfg(velocity_priority=200),
                JoystickCtrlCfg(velocity_priority=300),
            ]
        )

    def test_multiple_sources_require_unique_explicit_priorities(self):
        with self.assertRaisesRegex(ValueError, "explicit velocity_priority"):
            VelocitySourceArbiter([JoystickCtrlCfg(), KeyboardCtrlCfg()])
        with self.assertRaisesRegex(ValueError, "must be unique"):
            VelocitySourceArbiter(
                [JoystickCtrlCfg(velocity_priority=1), KeyboardCtrlCfg(velocity_priority=1)]
            )

    def test_duplicate_velocity_controller_types_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "controller types must be unique"):
            VelocitySourceArbiter(
                [JoystickCtrlCfg(velocity_priority=300), JoystickCtrlCfg(velocity_priority=200)]
            )

    def test_continuous_neutral_joystick_does_not_block_fresh_zero_zmq(self):
        arbiter = self.arbiter()
        data = {
            "JoystickCtrl": joystick(),
            "KeyboardCtrl": keyboard(),
            "VelocityZmqCtrl": zmq(),
        }
        self.assertEqual(arbiter.update(data, now=0.0), "VelocityZmqCtrl")
        self.assertEqual(arbiter.update(data, now=10.0), "VelocityZmqCtrl")

    def test_zero_deadzone_still_treats_exactly_neutral_joystick_as_idle(self):
        arbiter = VelocitySourceArbiter(
            [
                JoystickCtrlCfg(velocity_priority=300, velocity_activity_deadzone=0.0),
                VelocityZmqCtrlCfg(velocity_priority=100),
            ]
        )
        data = {"JoystickCtrl": joystick(), "VelocityZmqCtrl": zmq()}
        self.assertEqual(arbiter.update(data, now=0.0), "VelocityZmqCtrl")

    def test_joystick_zero_holds_lease_then_releases_to_zmq(self):
        arbiter = self.arbiter()
        data = {
            "JoystickCtrl": joystick(y=0.5),
            "KeyboardCtrl": keyboard(),
            "VelocityZmqCtrl": zmq(),
        }
        self.assertEqual(arbiter.update(data, now=1.0), "JoystickCtrl")

        data["JoystickCtrl"] = joystick()
        self.assertEqual(arbiter.update(data, now=1.5), "JoystickCtrl")
        self.assertEqual(arbiter.update(data, now=1.5001), "VelocityZmqCtrl")

    def test_joystick_then_keyboard_then_zmq_priority_chain(self):
        arbiter = self.arbiter()
        data = {
            "JoystickCtrl": joystick(y=0.5),
            "KeyboardCtrl": keyboard(key("w", True), pressed_keys=["w"]),
            "VelocityZmqCtrl": zmq(),
        }
        self.assertEqual(arbiter.update(data, now=2.0), "JoystickCtrl")

        data["JoystickCtrl"] = joystick()
        data["KeyboardCtrl"] = keyboard(pressed_keys=["w"])
        self.assertEqual(arbiter.update(data, now=2.5001), "KeyboardCtrl")

        data["KeyboardCtrl"] = keyboard(key("w", False))
        self.assertEqual(arbiter.update(data, now=2.6), "KeyboardCtrl")
        data["KeyboardCtrl"] = keyboard()
        self.assertEqual(arbiter.update(data, now=3.1001), "VelocityZmqCtrl")

    def test_stale_joystick_drops_its_unexpired_lease(self):
        arbiter = self.arbiter()
        data = {
            "JoystickCtrl": joystick(y=0.5),
            "KeyboardCtrl": keyboard(),
            "VelocityZmqCtrl": zmq(),
        }
        self.assertEqual(arbiter.update(data, now=0.0), "JoystickCtrl")
        data["JoystickCtrl"] = joystick(y=0.5, fresh=False)
        self.assertEqual(arbiter.update(data, now=0.1), "VelocityZmqCtrl")

    def test_dead_keyboard_listener_drops_held_key_lease(self):
        arbiter = self.arbiter()
        data = {
            "JoystickCtrl": joystick(),
            "KeyboardCtrl": {**keyboard(key("w", True), pressed_keys=["w"]), "fresh": True},
            "VelocityZmqCtrl": zmq(),
        }
        self.assertEqual(arbiter.update(data, now=0.0), "KeyboardCtrl")
        data["KeyboardCtrl"] = {**keyboard(pressed_keys=["w"]), "fresh": False}
        self.assertEqual(arbiter.update(data, now=0.1), "VelocityZmqCtrl")

    def test_config_order_does_not_affect_selection(self):
        arbiter = VelocitySourceArbiter(
            [JoystickCtrlCfg(velocity_priority=300), VelocityZmqCtrlCfg(velocity_priority=100)]
        )
        data = {"VelocityZmqCtrl": zmq(), "JoystickCtrl": joystick(y=0.2)}
        self.assertEqual(arbiter.update(data, now=0.0), "JoystickCtrl")

    def test_multiple_raw_sources_require_arbitration_metadata(self):
        with self.assertRaisesRegex(ValueError, "require VELOCITY_SOURCE"):
            get_selected_velocity_source({"VelocityZmqCtrl": zmq(), "JoystickCtrl": joystick()})

    def test_all_registered_configs_have_valid_velocity_priorities(self):
        from robojudo.config import cfg_registry

        for name in cfg_registry.types:
            cfg = cfg_registry.get(name)()
            if hasattr(cfg, "ctrl"):
                with self.subTest(config=name):
                    VelocitySourceArbiter(cfg.ctrl)


if __name__ == "__main__":
    unittest.main()
