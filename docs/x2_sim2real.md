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

The `x2`, `x2_real`, and locomanipulation presets use `X2LocomanipulationPipeline`, which owns X2's four deployment
modes. The `x2_locomimic` presets use `X2LocomanipulationLocoMimicPipeline`, which adds locomotion/mimic policy
interpolation to the same mode state machine.

## Deployment CPU Overhead

The locomanipulation command display updates at 5 Hz by default, independently of
the 50 Hz policy and 500 Hz AimDK publication rates. Set the policy's
`command_status_interval_s` to `0.0` to disable the display or `0.02` to restore the
former display rate. For `x2_locomimic_real`, this setting belongs to `loco_policy`.
Command message buffers and state telemetry storage are reused by the native
backend; rebuild `packages/aimdk_cpp` after updating it to use these optimizations.
Joint selection, validation, freshness deadlines, and watchdog behavior remain
unchanged. Native regression test instructions are in
[`packages/aimdk_cpp/README.md`](../packages/aimdk_cpp/README.md#development-checks);
unset `FASTRTPS_DEFAULT_PROFILES_FILE` before running ROS-related tests.

## Prerequisites

Initialize and build the pinned AimDK SDK, then install the ROS 2 extension with the managed installer:

```bash
source /opt/ros/humble/setup.bash
conda activate robojudo-plus
python submodule_install.py aimdk
python submodule_install.py omnihand_sdk
source third_party/aimdk/install/setup.bash
```

The installer initializes missing AimDK SDK and `aimdk_cpp` backend submodules, preserves existing worktrees, builds
only `aimdk_msgs`, installs the Python/CMake build tools into the active environment, and installs `aimdk_cpp` without
PEP 517 build isolation. Source the generated AimDK setup file in every shell used for real X2 deployment. Use
`python submodule_install.py --clean aimdk` only when you intentionally want to discard AimDK C++ backend changes and
restore its repository-pinned revision.

`x2_omnihand_locomanipulation_real` embeds direct dual OmniHand Pro control. The OmniHand installer only installs the
platform-matched Python wheel; it does not modify `/usr/local`. On first use, install USB-CAN permissions with
`sudo bash third_party/omnihand_sdk/linux/x64/setup_udev.sh` (or `linux/aarch64`) and log out and back in. The default HCAN
mapping is left adapter index 1 and right adapter index 0; HCAN channel zero is fixed internally and is not configurable.
RoboJuDo subscribes to dex teleop's synchronized arm-and-hand stream on port 8560, splits each complete frame inside
`UpperBodyOmniHandZmqCtrl`, and calls the SDK directly. This dedicated controller and its
`X2OmniHandLocomanipulationPipeline` leave the existing arm-only controller, simulation pipeline, and GR00T path
unchanged. The old per-hand ports 5555/5556 and a standalone OmniHand ZMQ server are not used.

Verify the native backend explicitly after installation:

```bash
ROBOJUDO_REQUIRE_AIMDK=1 python -m unittest discover -s tests
```

This strict test mode fails instead of skipping when `aimdk_cpp` or one of its ROS 2 libraries cannot be loaded.

Confirm that all four joint-state topics and `/aima/hal/imu/torso/state` are updating before enabling commands.
The standard real X2 presets use `odometry_type="NONE"`; the BeyondMimic loco-mimic preset uses `"DUMMY"`. Both
require fresh joint and IMU state only. When a configuration enables `odometry_type="AIMDK"`, it requires
`nav_msgs/msg/Odometry` on
`/aima/mc/leg_odometry`; the SuperOdom loco-mimic preset requires `/laser_odometry`. Missing or stale required state
prevents activation and triggers damping.

At startup, `aimdk.startup_state_timeout` bounds the wait for the first complete, valid joint and IMU state
snapshot; it defaults to `2.0` seconds. Increase it when the robot's state publishers are expected to come up after
the deployment process. Keep `aimdk.state_timeout` at its control-safety value because it governs freshness after
startup.

To capture the exact inputs to the freshness decision, run the passive monitor in a second terminal with the same
deployment config:

```bash
python scripts/monitor_x2_state.py --config x2_real --output /tmp/x2-state.jsonl
```

The monitor forces its AimDK controller to `act=False`; it subscribes with the same native callbacks and sensor-data
QoS but never publishes robot commands. It prints health transitions and a callback summary every second by default, and writes
one JSON object per line containing IMU age, missing or stale joint names and ages, odometry validity, age, and rejection
context, plus per-topic callback rate, receive age, maximum inter-arrival gap, and sequence-gap diagnostics. The joint
topic records also retain their last header and measurement stamps and joint-name list. Those stamps are raw values: do
not compare them to this computer's time unless the onboard and deployment computers are synchronized. Omit `--output`
to create a UTC-timestamped file in the current directory. Use the actual deployment preset so odometry topics, expected
frames, joint names, and timeouts match. Set `aimdk.telemetry_window_sec` to change the one-second callback-rate window.
Reinstall `packages/aimdk_cpp` before using the monitor after pulling this
diagnostic API. Existing output files are preserved by default; pass `--append` to add another run or `--overwrite` to
replace one explicitly.

### Stress state subscriptions without commanding the robot

Use the passive subscriber stress probe to test whether adding independent ROS/DDS consumers reproduces
state-delivery gaps. Use the system Python because ROS Humble's `rclpy` extension is built for Python 3.10 on the
robot computer:

```bash
source /opt/ros/humble/setup.bash
source /home/ubuntu/aimdk/install/setup.bash
/usr/bin/python3 scripts/stress_x2_state_subscribers.py \
  --processes 4 --duration 120 --output /tmp/x2-subscriber-baseline.jsonl
```

Each process has its own ROS context/DDS participant and subscribes to the leg, waist, arm, head, and torso-IMU state
topics. All subscriptions use best-effort, volatile `KEEP_LAST(1)` QoS. The probe creates no publishers.

First run it without the manipulation launch, then repeat while `box_pick_place.launch.py` is running and write to a
different output file. Increase `--processes` gradually (for example 1, 2, 4, 8) rather than starting with a large
fan-out. A receive gap at or above 0.1 seconds is printed immediately. Interpret the paired values as follows:

- `receive_gap` large and `header_gap` similarly large: depth-1 delivery skipped to a recent sample, or samples were
  lost before delivery.
- `receive_gap` large but `header_gap` about 0.002 seconds: an old sample was delayed before the DDS reader history or
  was already taken into an executor callback queue. `KEEP_LAST(1)` limits the reader history, not those other queues.

The test isolates the effect of additional state readers. It does not reproduce the high-volume publishers created
by MoveIt, so a clean subscriber-only run does not rule out DDS transmit congestion from the manipulation stack.

The exception raised by the running deployment now includes the controller's in-process freshness snapshot and is
authoritative if its result differs from the independently subscribed monitor. A `frame_mismatch` odometry rejection
means the received parent or child frame differs from the configured values; `invalid_values_or_quaternion` means the
sample contained non-finite data or a near-zero quaternion. Odometry covariance entry 0 at or above `0.5` is reported
as degenerate and therefore invalid for the freshness check.

The leg-odometry publisher reports its pose in `leg_odom` for
`child_frame_id=lidar_imu_chest_front`. RoboJuDo therefore uses the pose directly as the
measured torso pose. The message twist follows the ROS `Odometry` convention and is already
expressed in the child/body frame; it is passed to the policy without another heading
rotation. Reinstall `aimdk_cpp` after changing or updating this integration because the
subscriber is implemented in the native extension.

## Control Modes

`X2LocomanipulationPipeline` starts in `PASSIVE_DEFAULT` and owns the four operating modes:

| Mode | Joystick | Behavior |
| --- | --- | --- |
| `PASSIVE_DEFAULT` | `A` | Zero stiffness and damping on all 31 joints. Use only while the robot is supported. |
| `DAMPING_DEFAULT` | `B` | Zero stiffness and damping `5.0` on all 31 joints. |
| `JOINT_DEFAULT` | `Y` | Interpolate all joints to the configured default pose over 1.5 seconds. |
| `RL_DEFAULT` | `X` | Reset and warm up ONNX, then run the 29-joint policy from timestep 1. |

`RL_DEFAULT` is rejected until `JOINT_DEFAULT` completes. Entering passive or damping invalidates preparation, so joint preparation must run again before restarting RL. `LB+RB+A` requests damping followed by process shutdown. In simulation, `LB+RB+Y` resets the model and returns to passive mode.

The X2 loco-mimic presets add AMP fall recovery. After a fall, select `JOINT_DEFAULT`, then press the joystick
right-stick button (`R`), or `r` on the simulation keyboard, to start recovery. An operator-selected `JOINT_DEFAULT`
is retained despite fall tilt so this sequence can complete. The recovery command is enabled only after the
joint-default interpolation finishes, matching the existing `JOINT_DEFAULT` to `RL_DEFAULT` gate. After the robot is
upright with tilt below `1.0 rad`, select loco with `Back` to interpolate directly into `RL_DEFAULT`. Recovery cannot
be entered while upright or from another mode; passive, damping, and shutdown remain immediate emergency exits.
A robot that starts or respawns fallen retains `PASSIVE_DEFAULT`; startup tilt alone does not force damping. An
explicit operator `PASSIVE_DEFAULT` selection also remains active despite the fall tilt until another mode is
requested.

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

The simulation locomanipulation preset subscribes to `tcp://127.0.0.1:8559` for optional arm-only targets. The
publisher must bind that endpoint and send JSON objects in radians using the following envelope:

```json
{
  "positions": {
    "left_shoulder_pitch_joint": 0.35,
    "right_elbow_joint": -0.87
  }
}
```

Each message must be one UTF-8 JSON frame, with no topic prefix or multipart envelope,
and at most 16 KiB. Updates may contain any non-empty subset of the 14 arm joints:
shoulder pitch/roll/yaw, elbow, and wrist yaw/pitch/roll for the left and right sides.
Each joint has its own `timeout_s` lease: updating one joint does not keep another
joint's old target alive. Expired or missing joints smoothly return to their
configured defaults. Waist joints remain policy-controlled, and the head remains
at its configured default pose. Unknown joints, duplicate JSON keys, malformed
envelopes, booleans and non-finite values reject the entire message without
refreshing any lease. Large finite positions are clipped without overflowing the
filter's float32 storage.

`x2_locomanipulation_real` instead connects to dex teleop at `tcp://10.0.1.20:8560` and requires one complete
`synchronized_teleop_frame` containing all 14 arm joints and both sets of 12 OmniHand joints. It rejects incomplete,
locally timed-out, duplicate/out-of-order, `sim2sim`, or incorrectly ordered frames atomically.

External arm control starts disabled. Press joystick `Start` in `RL_DEFAULT`, or release keyboard `T` in simulation, to
toggle it. Targets are clamped to the X2 joint limits and filtered at 50 Hz with an EMA alpha of `0.95`. If no valid
message arrives for `0.25 s`, or external control is disabled, the arms smoothly return to the recorded locomanipulation
pose. A fresh message resumes control after a timeout without another toggle.
Receive timestamps are checked again after policy inference, immediately before
target filtering. Leaving `RL_DEFAULT`, resetting the simulation, or entering a
safety mode disables the override. Toggle edges reset the receiver session and
flush pending messages; the data read before an enable edge cannot activate the
new session. The publisher must continue sending afterward.
After smoothing, `UpperBodyZmqCtrlCfg.max_joint_velocity_rad_s` limits each commanded joint to `1.0 rad/s` by default
(`0.02 rad` per cycle at 50 Hz). This applies to live commands, reconnection, and returns on timeout or disable.
The limit uses the configured control timestep and bounds commanded position changes, not measured speed or torque.
Validate changes to this setting in simulation first. GR00T uses the same `1.0 rad/s` default.

The receiver processes at most 100 frames per control poll. If it exhausts that
budget, or polling stalls longer than the timeout, it clears the command cache,
discards queued frames until the queue is observed empty, and requires a later
packet before resuming. This intentionally favors default-pose fallback over
replaying a queue after a stall. PUB/SUB delivery is best effort: continuously
publish current snapshots rather than relying on one-shot updates.

The inspected MoveIt adapter in `x2_ws` binds a PUB socket, uses a send high-water
mark of one, and emits complete 14-arm `positions` snapshots at a configured
50 Hz. It creates the direct arm HAL publisher only when `command_transport` is
`ros_topic`. Use `command_transport:=zmq` for the RoboJuDo route: the inspected
`box_pick_place.launch.py` defaults to `zmq`, while `real_robot.launch.py` defaults
to `ros_topic`. Keep one arm HAL command owner. The subscriber's endpoint must
name the publisher host; `127.0.0.1` is correct only when both run on the same host.

The path is: MoveIt/ros2_control → ZMQ PUB → `UpperBodyZmqCtrl` → `CtrlManager` →
mode gate and EMA/velocity limiting → final environment position clipping →
optional arm observer → atomic AimDK position/arm-effort submission. Only the
14 named arm targets are overridden. Leg/waist policy targets and head ownership
are preserved. The observer uses the final clipped arm targets, not raw ZMQ data.

Legacy `positions` messages contain no source timestamp, sequence or delivery
acknowledgement. Local receive-age and session checks cannot establish source
age, detect an otherwise valid delayed/replayed packet, or prove physical command
application. A stronger guarantee requires a coordinated publisher/receiver
protocol change and an explicit clock/session contract; adding an arbitrary
extra JSON field currently provides no such guarantee. Actual network timing,
robot feedback and firmware behavior still require deployment qualification.

Hardware-free checks cover malformed/duplicate/multipart payloads, per-joint
expiry, control-loop stalls and bounded draining. The loopback integration tests
use ephemeral localhost ports, actual PUB/SUB and `CtrlManager`, X2 filtering and
observer code, and a fake native adapter; they exercise timeout, reconnect,
post-inference expiry and enable-session invalidation without robot commands:

```bash
python -m unittest discover -s tests -p 'test_upper_body_zmq*.py'
```

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
publisher.setsockopt(zmq.SNDHWM, 1)
publisher.setsockopt(zmq.LINGER, 0)
publisher.bind("tcp://127.0.0.1:8559")
message = {
    "positions": {
        "left_shoulder_pitch_joint": 0.35,
        "right_shoulder_pitch_joint": 0.35,
    }
}
try:
    while True:
        publisher.send_json(message)
        time.sleep(0.02)  # Continuously refresh the commanded joints at about 50 Hz.
except KeyboardInterrupt:
    pass
finally:
    publisher.close()
    context.term()
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

`X2LocomanipulationLocoMimicPipeline` retains the same four X2 control modes. Enter `JOINT_DEFAULT`, wait for
completion, and then enter `RL_DEFAULT`; policy inference and switch timers remain stopped in every other mode.
Joystick `Back` selects loco,
`Start` selects mimic, and left-stick click `L` toggles streamed upper-body targets while loco is idle. Release the
right bumper (`RB`/physical R1) to select the next mimic policy or the left bumper (`LB`/physical L1) to select the
previous one while loco is active and interpolation is idle. Bumpers used in a recognized chord do not also change the
selected policy, so `LB+RB+A` remains dedicated to shutdown. In simulation,
keyboard `]`, `[`, and `T` provide the same controls. Starting a mimic transition or leaving `RL_DEFAULT` disables
upper-body streaming. After returning to loco, explicitly toggle it back on. At its configured ONNX phase limit of
2820, each test mimic reports `MOTION_DONE` and the pipeline automatically starts the return-to-loco interpolation. A
manual return remains available through joystick `Back` or keyboard `]`.

With the robot supported, select `JOINT_DEFAULT` and wait for the completion log. Then select `RL_DEFAULT`. ONNX time does not advance in passive, damping, or joint-preparation modes.

The policy runs at 50 Hz. `aimdk_cpp` republishes the active mode at 500 Hz. A command, IMU/joint, or required odometry stream that is 100 ms old enters a measured-position hold: the backend continues publishing the most recently measured joint positions with the active PD gains, rather than dropping stiffness. A fault persisting for 500 ms latches damping. Recovery from hold requires fresh required state and a command produced after that recovery; recovery from latched damping requires an explicit position-control re-arm. The X2 runners report a 220 ms host frame drop but leave its response to this backend state machine, so that warning cannot bypass the hard deadline. Non-finite targets, excessive tilt, and shutdown also force damping; non-finite joint or IMU samples are rejected without refreshing their state timestamps. State subscriptions keep only the latest sensor sample and isolate joint, IMU, and odometry callbacks with separate state locks on a blocking multi-threaded executor so high-rate joint traffic cannot starve another safety stream. Validate the 500 ms hard deadline on the supported deployment host before unattended operation, especially when the X2 MoveIt manipulation stack is sharing CPU resources.

## Preflight Checks

- Verify the startup log contains all 29 policy-to-environment joint mappings.
- Keep the robot supported for the first default-pose and low-amplitude policy trials.
- Confirm joystick `B` causes damping before attempting unsupported operation.
- Do not deploy if targets are repeatedly clamped or control-loop frame drops are reported.
