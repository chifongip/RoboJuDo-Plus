# X2 Sim2Real Deployment

The `x2_real` configuration runs the 29-joint X2 ONNX policy through the 31-joint AimDK robot interface.
RoboJuDo reorders joint state by name before constructing the `1x151` observation, then maps the `1x29` policy output
back to the robot-native order. The two head joints are not policy-driven; position control actively holds them at their
configured defaults.

The additional `x2_locomanipulation` and `x2_locomanipulation_real` configurations run the X2 locomanipulation policy.
That policy observes 29 joints with a five-frame `1x430` history and controls the 12 leg plus three waist joints. The
remaining arm and head joints hold the defaults recorded with the training run unless the optional arm-only ZMQ
override is enabled. Its PD gains, effort limits, action scales, and default pose come from the run's saved
`params/env.yaml`; rounded ONNX metadata is used only as a consistency check.

## Prerequisites

Initialize and build the pinned AimDK SDK, then install the ROS 2 extension with the managed installer:

```bash
source /opt/ros/humble/setup.bash
conda activate robojudo_test
python submodule_install.py aimdk
source third_party/aimdk/install/setup.bash
```

The installer builds only `aimdk_msgs`, installs the Python/CMake build tools into the active environment, and installs
`aimdk_cpp` without PEP 517 build isolation. Source the generated AimDK setup file in every shell used for real X2
deployment. Existing changes inside the AimDK submodule are preserved; use `python submodule_install.py --clean aimdk`
only when you intentionally want to discard them.

Verify the native backend explicitly after installation:

```bash
ROBOJUDO_REQUIRE_AIMDK=1 python -m unittest discover -s tests
```

This strict test mode fails instead of skipping when `aimdk_cpp` or one of its ROS 2 libraries cannot be loaded.

Confirm that all four joint-state topics and `/aima/hal/imu/torso/state` are updating before enabling commands. Missing or stale state prevents activation and triggers damping.

## Control Modes

The X2 pipeline starts in `PASSIVE_DEFAULT` and uses the same four operating modes as the standalone controller:

| Mode | Joystick | Behavior |
| --- | --- | --- |
| `PASSIVE_DEFAULT` | `A` | Zero stiffness and damping on all 31 joints. Use only while the robot is supported. |
| `DAMPING_DEFAULT` | `B` | Zero stiffness and damping `5.0` on all 31 joints. |
| `JOINT_DEFAULT` | `Y` | Interpolate all joints to the standalone preparation pose over 1.5 seconds. |
| `RL_DEFAULT` | `X` | Reset and warm up ONNX, then run the 29-joint policy from timestep 1. |

`RL_DEFAULT` is rejected until `JOINT_DEFAULT` completes. Entering passive or damping invalidates preparation, so joint preparation must run again before restarting RL. `LB+RB+A` requests damping followed by process shutdown. In simulation, `LB+RB+Y` resets the model and returns to passive mode.

## Simulation ElasticBand

The `x2` MuJoCo configuration starts with a virtual tension band attached to `torso_link`. Its default world anchor is `[0, 0, 3]`, stiffness is `200 N/m`, damping is `100 Ns/m`, and rest length is `0 m`. The band force is applied before every physics substep in all four control modes. The viewer renders the band as an orange capsule with a sphere at the world anchor; both are hidden when the band is released.

- `7`: lower the robot by increasing rest length by `0.1 m`.
- `8`: lift the robot by decreasing rest length by `0.1 m`, clamped at zero.
- `9`: release or reactivate the band.

The settings are defined by `elastic_band` in `X2MujocoEnvCfg`. Set `visualize=False` to hide the geometry while retaining the force, or adjust `visual_radius`, `visual_rgba`, and `anchor_radius` to change its appearance. The band is simulation-only; `x2_real` does not register these keyboard controls or apply an external force.

The locomanipulation policy uses the left stick for planar velocity, the right stick for yaw, D-pad up/down for body
height, and D-pad left/right for waist yaw. Keyboard controls are `W/S`, `A/D`, and `Q/E` for velocity, `R/F` for
height, `Z/C` for waist yaw, and `X` to reset commands. Its deployment command limits are x `[-0.5, 1.0]`, y
`[-0.5, 0.5]`, yaw `[-1.0, 1.0]`, height `[0.40, 0.66]`, and waist yaw `[-1.5708, 1.5708]`.

## Locomanipulation Upper-Body ZMQ Control

The two locomanipulation presets subscribe to `tcp://127.0.0.1:8559` for optional arm targets. The publisher must bind
that endpoint and send JSON objects in radians using the following envelope:

```json
{
  "positions": {
    "left_shoulder_pitch_joint": 0.35,
    "right_elbow_joint": -0.87
  }
}
```

Updates may contain any non-empty subset of the 14 arm joints: shoulder pitch/roll/yaw, elbow, and wrist yaw/pitch/roll
for the left and right sides. Values from earlier partial updates remain active while messages are fresh. Waist joints
remain policy-controlled, and the head remains at its configured default pose. Unknown joints, malformed envelopes,
and non-finite values cause the complete message to be rejected.

External arm control starts disabled. Press joystick `Start` in `RL_DEFAULT`, or release keyboard `T` in simulation, to
toggle it. Targets are clamped to the X2 joint limits and filtered at 50 Hz with an EMA alpha of `0.95`. If no valid
message arrives for `0.25 s`, or external control is disabled, the arms smoothly return to the recorded locomanipulation
pose. A fresh message resumes control after a timeout without another toggle. Leaving `RL_DEFAULT`, resetting the
simulation, or entering a safety mode disables the override.

Use the interactive predefined-pose publisher to exercise the interface:

```bash
python scripts/test_upper_body_zmq.py
```

Press `0`–`5` to select the default, forward, raised, wide, carrying, or left-wave pose, and press `q` to stop. Run
`python scripts/test_upper_body_zmq.py --help` for endpoint, frequency, and initial-pose options. A minimal custom
publisher is:

```python
import time

import zmq

context = zmq.Context()
publisher = context.socket(zmq.PUB)
publisher.bind("tcp://127.0.0.1:8559")
time.sleep(0.2)  # Allow the subscriber connection to settle.
publisher.send_json(
    {
        "positions": {
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_pitch_joint": 0.35,
        }
    }
)
```

## Startup Sequence

Run the real configuration only with the robot supported and an operator holding the emergency stop:

```bash
python scripts/run_pipeline.py -c x2_real
```

Use the corresponding locomanipulation presets for this policy:

```bash
python scripts/run_pipeline.py -c x2_locomanipulation
python scripts/run_pipeline.py -c x2_locomanipulation_real
```

To test interpolated switching from locomanipulation locomotion to the `x2_rl_deploy` mimic policy, use:

```bash
python scripts/run_pipeline.py -c x2_locomimic
python scripts/run_pipeline.py -c x2_locomimic_real
```

The loco-mimic presets retain the four X2 control modes. Enter `JOINT_DEFAULT`, wait for completion, and then enter
`RL_DEFAULT`; policy inference and switch timers remain stopped in every other mode. Joystick `Back` selects loco,
`Start` selects mimic, and left-stick click `L` toggles streamed upper-body targets while loco is idle. Release the
right bumper (`RB`/physical R1) to select the next mimic policy or the left bumper (`LB`/physical L1) to select the
previous one while loco is active and interpolation is idle. Bumpers used in a recognized chord do not also change the
selected policy, so `LB+RB+A` remains dedicated to shutdown. In simulation,
keyboard `]`, `[`, and `T` provide the same controls. Starting a mimic transition or leaving `RL_DEFAULT` disables
upper-body streaming. After returning to loco, explicitly toggle it back on. At its configured ONNX phase limit of
2820, each test mimic reports `MOTION_DONE` and the pipeline automatically starts the return-to-loco interpolation. A
manual return remains available through joystick `Back` or keyboard `]`.

With the robot supported, select `JOINT_DEFAULT` and wait for the completion log. Then select `RL_DEFAULT`. ONNX time does not advance in passive, damping, or joint-preparation modes.

The policy runs at 50 Hz. `aimdk_cpp` republishes the active mode at 500 Hz, matching the standalone X2 controller. If position commands stop for 100 ms, the backend latches damping. Position control cannot resume until a new mode transition explicitly re-arms it. Stale state, a non-finite target, excessive tilt, and shutdown also force damping.

## Preflight Checks

- Verify the startup log contains all 29 policy-to-environment joint mappings.
- Keep the robot supported for the first default-pose and low-amplitude policy trials.
- Confirm joystick `B` causes damping before attempting unsupported operation.
- Do not deploy if targets are repeatedly clamped or control-loop frame drops are reported.
