# Controller

**Controller** is the component that gives `ctrl_data` to the robot.

## [Controller](#controller)

`Controller` is the base class for all controllers. It defines the interface that all controllers must implement.

---
We provide the following controllers:
- [JoystickCtrl](#controller--joystickctrl)
- [RosJoystickCtrl](#controller--rosjoystickctrl)
- [UnitreeCtrl](#controller--unitreectrl)
- [KeyboardCtrl](#controller--keyboardctrl)
- [MotionCtrl](#controller--motionctrl)
- [BeyondmimicCtrl](#controller--beyondmimicctrl)

## [Controller](#controller) > [JoystickCtrl](#controller--joystickctrl)

`JoystickCtrl` is the controller that controls the robot using the joystick. It is a subclass of `Controller` and implements the interface defined in `Controller`.

script:
  - [joystick_ctrl.py](../robojudo/controller/joystick_ctrl.py)

Example data of Xbox Joystick with Linear Triggers:

`ctrl_data`:`dict`, the control data.
  - `axes`: `dict[str, float]`:
    - `LeftX`: left axes x value. Range: [-1, 1]
    - `LeftY`: left axes y value. Range: [-1, 1]
    - `RightX`: Right axes x value. Range: [-1, 1]
    - `RightY`: Right axes y value. Range: [-1, 1]
    - `LT`: left trigger value. Range: [0, 1]
    - `RT`: right trigger value. Range: [0, 1]
  - `button_event`: `list[dict]`:
    `dict`:
      - `name`: the name of the button. like `A`, `B`, `X`, `Y`...
      - `press`: whether the button is pressed. `bool`. `True` for `press`, `False` for `release`
      - `timestamp`: the time when the button event occurs. `float`
      - `type`: the type of the button event.

`command`: `list` of commands when `triggers` or `triggers_extra` are detected.

**example**:
```json
{'axes': {'LeftX': 0.0, 'LeftY': 0.0, 'RightX': 0.0, 'RightY': 0.0, 'LT': 0.0, 'RT': 0.0}, 'button_event': [{'type': 'button', 'name': 'A', 'pressed': False, 'timestamp': 1758886189.6776087}]}
```

You can set Hotkeys in JoystickCtrlCfg:
```python
JoystickCtrlCfg(
    triggers_extra={
        "RB+Down": "[POLICY_SWITCH],0",
        "LB+RB+A": "COMBO_TEST",
    }
),
```
when you press the `RB+Down` button, the command will be `["[POLICY_SWITCH],0"]`.

💡We have joystick mapping config for different platforms and Joystick types. 

> For KEY name and more details, please refer to the [joystick.py](../robojudo/controller/utils/joystick.py)

## [Controller](#controller) > [RosJoystickCtrl](#controller--rosjoystickctrl)

`RosJoystickCtrl` is a drop-in alternative to `JoystickCtrl` that subscribes to the ROS 2 `/joy` topic published by
the `joy_node` executable from the `joy` package. It uses a small `rclcpp`/pybind11 extension so RoboJuDo can keep
running on Python 3.11 even when the ROS 2 distribution's Python packages were built for Python 3.10.

Build the extension for the active RoboJuDo interpreter after sourcing ROS 2:

```bash
source /opt/ros/humble/setup.bash
python submodule_install.py ros2_joy_cpp
```

Then start the ROS driver in another terminal:

```bash
source /opt/ros/humble/setup.bash
ros2 run joy joy_node
```

Select the raw `joy_node` layout explicitly in the pipeline configuration:

```python
from robojudo.controller.ctrl_cfgs import RosJoystickCtrlCfg

RosJoystickCtrlCfg(
    profile="xbox",  # "xbox", "xbox_bluetooth", "ps5", "ps5_bluetooth", or "ps5_bluetooth_jetson"
    topic="/joy",
    timeout_s=0.5,
    triggers_extra={"RB+Down": "[POLICY_SWITCH],0"},
)
```

All profiles expose the same `axes` and `button_event` schema as `JoystickCtrl`. For PS5 controllers, Cross,
Circle, Square, and Triangle are normalized to `A`, `B`, `X`, and `Y`; Create, Options, and PS are normalized to
`Back`, `Start`, and `Xbox`. The D-pad is emitted as `Up`, `Down`, `Left`, and `Right` button events.

Xbox controllers use different raw layouts over USB and Bluetooth. Select `profile="xbox"` for USB and
`profile="xbox_bluetooth"` for Bluetooth.

PS5 controllers calibrated over Bluetooth use the same raw layout as the PS5 profile: axes 0/1 and 3/4 are the
left/right sticks, axes 2/5 are L2/R2, axes 6/7 are the D-pad, and buttons 0-12 use the standard PS5 mapping.
Select `profile="ps5_bluetooth"` when using that connection.

On Jetson, PS5 Bluetooth has a different layout: axes 0-3 are the sticks, axes 4/5 are L2/R2, and the D-pad uses
buttons 11-14. Select `profile="ps5_bluetooth_jetson"` for this layout.

The controller returns neutral axes until the first message. If `/joy` becomes stale, it neutralizes all axes and
emits release events for held buttons once. `joy_node` does not include a device identity in `sensor_msgs/msg/Joy`,
so the controller profile cannot be detected reliably from the message itself.

To measure the raw layout of a specific controller and connection mode, run the interactive calibration script while
`joy_node` is publishing:

```bash
python scripts/calibrate_ros_joystick.py --profile ps5 --connection usb
python scripts/calibrate_ros_joystick.py --profile ps5 --connection bluetooth
python scripts/calibrate_ros_joystick.py --profile xbox --connection bluetooth
```

The script walks through each control and writes a JSON report containing the neutral state and every raw button or
axis that changed. USB and Bluetooth layouts should be calibrated separately if both connection modes will be used.

## [Controller](#controller) > [UnitreeCtrl](#controller--unitreectrl)

`UnitreeCtrl` is the controller that controls the robot using the `UnitreeG1` controller. It is a subclass of `Controller` and implements the interface defined in `Controller`. 

> ⚠️ If you don't connect to a Unitree robot, `UnitreeCtrl` won't work.

script:
- [unitree_ctrl.py](../robojudo/controller/unitree_ctrl.py)

`ctrl_data` and `command` are the same as `JoystickCtrl`.

## [Controller](#controller) > [KeyboardCtrl](#controller--keyboardctrl)

`KeyboardCtrl` is the controller that controls the robot using the keyboard. It is a subclass of `Controller` and implements the interface defined in `Controller`.

script:
  - [keyboard_ctrl.py](../robojudo/controller/keyboard_ctrl.py)
  
`ctrl_data`:`list[dict]`, the control data.
  `dict`:
   - `type`: the type of the input. `str`. `keyboard`
   - `name`: the name of the input key. `str`. like `a`, `Key.ctrl_l`, `Key.space`, `\x06`...
   - `pressed`: the value of the input. `bool`. `True` for `press`, `False` for `release`.
   - `timestamp`: the time when the input event occurs. `float`

`command`: `list`, only generate when you set `triggers`, otherwise, the command will be `[]`.

**example**:
```
[{'type': 'keyboard', 'name': 's', 'pressed': True, 'timestamp': 1758888074.643119}]
```

Similarly, you can set command triggers:
```python
KeyboardCtrl(
  cfg_ctrl=KeyboardCtrlCfg(
      triggers_extra={
          "Key.space": "[TEST]",
          "\x01": "[CTRL_A]",
      }
    )
  )
```
when you press the `Ctrl+A` button, the command will be `[CTRL_A]`.

💡For more details, please refer to the [keyboard.py](../robojudo/controller/utils/keyboard.py)


## [Controller](#controller) > [MotionCtrl](#controller--motionctrl)

`MotionCtrl` is the controller that controls the robot using the motion. It is a subclass of `Controller` and implements the interface defined in `Controller`.

`MotionCtrl` usually is used on mimic task to provide refer motion.

`ctrl_data`:`dict`, the control data.
  - `_motion_track_bodies_extend_id`: `int`: the id of the extended body.
  - `_robot_track_bodies_extend_id`: `int`: the id of the extended body.
  - `rg_pos_t`: `np.ndarray`: link position.
  - `body_vel_t`: `np.ndarray`: link velocity.
  - `root_pos`: `np.ndarray`: root position.
  - `root_vel`: `np.ndarray`: root velocity.
  - `root_rot`: `np.ndarray`: root rotation. w-last quat.
  - `root_ang_vel`: `np.ndarray`: root angular velocity.
  - `hand_pose`(Optional): `np.ndarray`: hand pose.

`MotionCtrl` can be set by `MotionCtrlCfg`. For instance, `G1MotionCtrlCfg`:

```python
G1MotionCtrlCfg(
    motion_name="amass_all",
)
```

Your motion file should be placed in the `assets/resources/motions/{robot}/phc` directory.

## [Controller](#controller) > [BeyondMimicCtrl](#controller--beyondmimicctrl)

`BeyondMimicCtrl` is the controller for `BeyondMimicPolicy`. It is a subclass of `Controller` and implements the interface defined in `Controller`.

You don't need to master `BeyondMimicCtrl`. It is just designed for `BeyondMimicPolicy`. You can set it with config:

```python
G1BeyondmimicCtrlCfg(
  motion_name="dance1_subject2", # only when policy: use_motion_from_model=False
)
```
