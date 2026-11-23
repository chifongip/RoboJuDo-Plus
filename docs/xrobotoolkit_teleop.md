# X2 XR teleoperation with RoboJuDo and XRoboToolkit

RoboJuDo runs the X2 simulation or AimDK hardware pipeline, manages robot modes and locomotion, and validates
and applies commands. The XR Bridge reads headset and controller data from XRoboToolkit PC Service, solves arm IK
using measured robot feedback, and publishes desired targets over ZMQ. Update RoboJuDo and its XRoboToolkit
submodule together so the protocol and model assets match.

## Install both environments

Set up RoboJuDo using the [repository installation guide](../README.md). From the RoboJuDo-Plus root,
initialize the pinned XRoboToolkit submodule and create the Python 3.11 bridge environment.

```bash
git submodule update --init --recursive third_party/xrobotoolkit_teleop
python scripts/install_xrobotoolkit_teleop.py
```

The installer requires Conda, Linux x86_64/aarch64, and a system C++ compiler. It installs pinned native dependencies,
Placo and LeRobot, and verifies imports and solver construction without opening XR or a viewer. Native build logs
and the manifest are under `$CONDA_PREFIX/share/robojudo-xr/`. An existing environment requires `--reuse`; use
`--name NAME` for another environment or `--jobs 1` on a memory-limited machine. Activate the resulting Conda environment
before running the bridge so its native-runtime settings take effect.

For **all XR presets**, install and verify numeric executor support with the RoboJuDo environment's interpreter:

```bash
python scripts/install_xr_executor.py
```

This installs the numeric Pinocchio/Coal dependencies and the shared XR model package. The installer verifies
both profiles. Simulation and hardware use the same executor and model identity checks.
Rerun the executor installer after updating the submodule to refresh the installed dependencies.

Install and start XRoboToolkit PC Service separately, connect the headset/controllers, and enable headset pose streaming.

## Choose a profile and tracking mode

Choose one row. Use the simulation preset first, then the corresponding AimDK preset after completing [hardware
preparation](#prepare-x2-hardware).

| Tracking mode | Bridge `--robot` | Simulation preset | AimDK preset |
|---|---|---|---|
| Fixed heading or waist | `x2` | `x2_xrobotoolkit` | `x2_xrobotoolkit_real` |
| Fixed heading or waist | `x2_omnihand` | `x2_xrobotoolkit_omnihand` | `x2_xrobotoolkit_omnihand_real` |
| Pelvis IK | `x2` | `x2_xrobotoolkit_pelvis_ik` | `x2_xrobotoolkit_pelvis_ik_real` |
| Pelvis IK | `x2_omnihand` | `x2_xrobotoolkit_omnihand_pelvis_ik` | `x2_xrobotoolkit_omnihand_pelvis_ik_real` |

| Bridge `--tracking-reference` | XR input and behavior | IK backend |
|---|---|---|
| `fixed-heading` (default) | Captures horizontal headset heading when you engage the grips. Subsequent head motion leaves the reference fixed. | Placo (default) or LeRobot |
| `waist` | Uses PICO full-body tracking to compensate for shared pelvis and controller motion. Changes the arm reference while posture stays under RoboJuDo control. | Placo (default) or LeRobot |
| `pelvis-ik` | Uses PICO full-body tracking to control crouch and waist yaw alongside pelvis-relative arm targets. | Placo (default) or LeRobot |

Complete PICO's native full-body calibration for `waist` and `pelvis-ik`.
Waist-relative tracking checks both XR packet and body-frame freshness.
See the [PICO calibration and body-tracking diagnostics](../third_party/xrobotoolkit_teleop/docs/agibot_x2.md#waist-relative-reference).

## Launch RoboJuDo and the XR Bridge

In the RoboJuDo environment, run the preset from the table. This example starts arm-only pelvis IK in simulation.

```bash
python scripts/run_pipeline.py -c x2_xrobotoolkit_pelvis_ik
```

In a second terminal, activate the XR environment and select the matching bridge profile and tracking mode.

```bash
conda activate robojudo-xr
python -m xrobotoolkit_teleop.integrations.robojudo --robot x2 \
  --tracking-reference pelvis-ik --ik-backend placo --scale-factor 1.0
```

For the default fixed-heading mode, use the corresponding preset and omit `--tracking-reference`.
For waist-relative tracking, use that same preset with `--tracking-reference waist`.
Use `--ik-backend lerobot` to select LeRobot without changing the RoboJuDo preset.
For a list of bridge options, run `python -m xrobotoolkit_teleop.integrations.robojudo --help`.

### Controls

With the robot supported and the work area clear on hardware, enter `JOINT_DEFAULT`, wait for interpolation,
then enter `RL_DEFAULT` and enable XR tracking.

| Action | Simulation keyboard | Simulation joystick | AimDK joystick |
|---|---|---|---|
| Enter `JOINT_DEFAULT` | `i` | `Y` | `RB+Y` |
| Enter `RL_DEFAULT` | `j` | `X` | `RB+X` |
| Toggle XR tracking | `t` | `L` | `L` |
| Select locomotion | `]` | `Back` | `Back` |
| Select mimic | `[` | `Start` | `Start` |
| Enter damping | `l` | `B` | `RB+B` |

XR tracking requires an active locomotion policy and completed interpolation. Release both XR grips before
engaging. Hold a grip to move its arm and release it to hold the issued target. With OmniHand, each trigger
controls the corresponding fingers independently of the arm grips. XR `Y` requests HOME. Release both
grips before resuming tracking. XR-button HOME requires a live XR stream. RoboJuDo can request arm HOME
with XR offline while the bridge and robot feedback are available. In simulation, torso and hand target
markers can include rejected requests.

### Pelvis IK controls

Either grip engages posture and its own arm. Keep one grip held while switching hands to preserve the
posture reference. Crouch or rise to change the height target. Twist your shoulders relative to your pelvis
to change preferred waist yaw. Both IK backends try that yaw first. If an active hand misses its goal,
the solver can adjust yaw while favoring the operator's preference. Hand targets
follow the robot pelvis as it crouches. Simulation and hardware limit posture commands to **0.2 m/s** in
height and **1.0 rad/s** in waist yaw.

Height can update while arm IK is running. The bridge adopts a solved waist target after RoboJuDo acknowledges
the corresponding upper-body command. When collision checking is enabled, RoboJuDo validates the combined
waist and upper-body transition before submission.

RoboJuDo gives manual posture input priority. Release both XR grips after manual takeover before re-engaging
posture. Releasing both grips or losing body tracking stops new posture targets. The policy can finish ramping
toward the last accepted target. Disabling XR freezes the posture command at its current value. XR `Y` requests
arm and posture defaults, and the arm return continues until feedback confirms convergence. OmniHand fingers
hold their issued targets during HOME.

## Prepare X2 hardware

Complete the AimDK and joystick setup, and source the AimDK environment on the robot host before starting
the selected hardware preset. The XR hardware presets default to observer compensation mode `OFF`. For the
initial hardware check, verify small motions, grip release, HOME, damping, and tracking and feedback loss.
With OmniHand, also verify each trigger independently. [X2 AimDK setup](x2_sim2real.md)

### OmniHand Pro through robot-side HCAN

For physical OmniHands, install the SDK in the robot's RoboJuDo environment and configure USB permissions.

```bash
python submodule_install.py omnihand_sdk
sudo bash third_party/omnihand_sdk/linux/aarch64/setup_udev.sh
# Use linux/x64 on x86_64; log out/in after initial udev setup.
```

Create a robot-local `.local/omnihand.json`:

```json
{
  "left_hcan_serial": "LEFT_ADAPTER_HW_SER",
  "right_hcan_serial": "RIGHT_ADAPTER_HW_SER",
  "left_hand_serial": "EXACT_LEFT_VENDOR_SERIAL",
  "right_hand_serial": "EXACT_RIGHT_VENDOR_SERIAL"
}
```

HCAN adapters belong to the robot host. The bridge does not open adapters. Exact adapter and hand identities,
finite measurements and the configured device ID must verify before takeover; missing serials stop startup.
Stop all other SDK clients, then run the read-only identity check:

```bash
python scripts/check_omnihand_hcan.py --config .local/omnihand.json
export ROBOJUDO_OMNIHAND_CONFIG="$PWD/.local/omnihand.json"
```

## Configure endpoints

The defaults support RoboJuDo and the XR Bridge on the same host. Run one bridge publisher per command
stream. For separate hosts, replace `XR_HOST` and `ROBOT_HOST` with reachable addresses.

| Stream | Default port | RoboJuDo controller and setting | XR Bridge option |
|---|---|---|---|
| Arm targets and optional finger targets | 8559 | `UpperBodyZmqCtrl` or `UpperBodyOmniHandZmqCtrl`, `endpoint = tcp://XR_HOST:8559` | `--command-endpoint 'tcp://*:8559'` |
| Robot feedback | 8571 | Same upper-body controller, `feedback_endpoint = tcp://*:8571` | `--feedback-endpoint tcp://ROBOT_HOST:8571` |
| Pelvis IK posture | 8557 | `LocomanipulationPostureZmqCtrl`, `endpoint = tcp://XR_HOST:8557` | `--posture-endpoint 'tcp://*:8557'` |

The bridge binds the target streams and connects to feedback. RoboJuDo binds feedback and connects to the
target streams.

## Tune XR Bridge and executor settings

| Setting | Default |
|---|---:|
| Control rate | 50 Hz |
| Hand target translation multiplier in all tracking modes (XR Bridge `--scale-factor`) | 1.0 |
| Arm / return speed cap | 8.0 / 1.0 rad/s |
| Grip activation | greater than 0.9 |
| XR / robot measurement freshness | 0.25 / 0.1 s |
| Collision checking | Disabled; explicitly opt in |
| Collision margin / activation distance | 0.02 / 0.08 m |

Applicable model limits and robot caps also bound arm motion.

Configure fingers in `XR_HAND_SETTINGS` in `robojudo/config/x2/x2_xrobotoolkit_cfg.py`:

```python
"hand_command_mode": "direct",  # "direct" (default) or "filtered"
"hand_velocity_limit": 2.0,      # rad/s, applied in filtered mode
```

These settings apply to all OmniHand presets. Restart RoboJuDo after changing them. The bridge validates
and echoes the robot-owned settings through the executor handshake.

- **Direct:** issue final finger targets within position bounds; device motor settings govern movement.
- **Filtered:** apply the same robot-side EMA and independent per-joint velocity filter as the arms, using the
  smaller of `hand_velocity_limit` and each joint's model velocity limit. Both use the existing `ema_alpha`
  (0.0 in XR presets), position limits, and float32 rounding.

The bridge sends final desired targets in either mode. Filtering advances once per fresh desired frame from
the completed issued baseline, without accumulating movement allowance during gaps or pending writes. HOME
holds fingers in both modes; collision checks and recording use actual issued targets after filtering.
Software target ramps do not guarantee physical finger speed or stopping behavior. Revoking commands does not
cancel movement toward a target already accepted by the device.

Configure collision checking for all X2 XR presets in `XR_ARM_SETTINGS` in the same file:

```python
"executor_collision_checking": False,  # Set True to enable collision checking.
```

Restart RoboJuDo after changing collision checking. The bridge reads the profile identity and collision policy
from validated robot feedback and initializes the corresponding IK checks before publishing targets.

Model mismatches, malformed policy, or enabled-check initialization failures block commands. Initialization and
IK time count toward measurement freshness. Large direct finger changes can exceed the bounded collision-sampling
budget and hold; use gradual trigger movement when checks are enabled. The executor does not search for smaller
steps. Numeric collision defaults are shared internally. Disabling collision checks retains finite-value,
position-limit, freshness, authorization and applicable velocity checks.

## Command completion, recovery, and recording

Arm-only commands complete after successful arm submission. OmniHand commands also require both hand writes.
Both profiles advance the completed-command baseline and retain fresh waiting frames after completion.

On OmniHand hardware, one SDK process owns both adapters and uses a separate I/O thread for each hand.
Completion reports successful SDK calls. Physical motion can still be in progress. Partial failures revoke
outstanding authorization. The SDK receive and startup budgets are 200 ms and 15 s, respectively. Feedback
freshness uses its own deadline.

Tracking loss or stale feedback holds issued targets. Transient recovery rebases XR references; startup, explicit
mode changes and bridge restarts require grip release. Explicit tracking disable requests bounded arm HOME; if the
bridge is offline, arms hold until fresh feedback permits return. Whole-robot safety states take priority.

Recording (`--record`, with the configured recorder service) stores fresh acknowledged tracking commands and measured
positions, excluding HOME and outage holds. Combined command recording uses the completed issued baseline, including
completion observed between control ticks.

## Read the runtime logs

The bridge and RoboJuDo log INFO summaries once per second during operation.

```text
XR: send=49.9 Hz | completion RTT=32.4 ms avg / 48.1 ms max | tracking
RoboJuDo: receive=49.8 Hz | control=50.0 Hz | complete=31.2 Hz
```

`send` counts desired commands successfully published by the bridge, excluding visualization-only frames.
`receive` counts valid desired commands accepted by RoboJuDo; `control` counts completed pipeline iterations,
excluding idle feedback servicing; `complete` counts commands acknowledged after successful arm submission and,
for OmniHand, both hand writes. A running 50 Hz pipeline can complete fewer than 50 fresh commands per second.

Completion RTT runs from bridge publication to the first matching completion feedback received, including the
return feedback path. It uses only the bridge's monotonic clock and existing command identifiers. Duplicate
feedback is counted once; missing, superseded, and unobserved acknowledgements produce no latency sample.
The log displays `RTT=n/a` when a window has no samples. Pending timestamp storage is bounded and cleared on
session/generation changes; logging does not gate command execution.

RTT does not establish physical arrival or firmware acceptance. One-way publication-to-receipt latency requires
synchronized clocks and is not reported. These summaries replace the local bridge loop-processing latency log.

## Validate the setup

In the RoboJuDo environment, run the installer and executor tests and the two-process simulation suite.

```bash
python -m unittest discover -s tests -p test_install_xr_executor.py
python -m unittest discover -s tests -p 'test_xr*.py'
XROBOTOOLKIT_TEST_PYTHON=/path/to/robojudo-xr/bin/python \
  python -m unittest discover -s tests -p test_xrobotoolkit_integration.py
# Set XROBOTOOLKIT_TEST_BACKEND=placo or lerobot to run every case with one backend.
```

In the XR environment, run the bridge tests from the RoboJuDo-Plus root.

```bash
python -m unittest discover -s third_party/xrobotoolkit_teleop/tests -p 'test_robojudo*.py'
python -m unittest discover -s third_party/xrobotoolkit_teleop/tests -p test_pelvis_ik.py
```

Physical acceptance remains required: verify provisional wrist mounts, joint ordering/signs, exact hand identities,
low-speed operation, clutch release, HOME finger hold, recording, and XR/bridge/adapter interruptions. Simulation
has no hand contacts and does not qualify grasp physics, hardware latency or 50 completed commands per second.
Do not increase freshness timeouts to hide stale hardware measurements.

## Migrate older configurations

| Removed setup | Release replacement |
|---|---|
| `_omnihand_real_direct`, `_direct_low_latency`, `_direct_executor` presets | `x2_xrobotoolkit_omnihand_real` |
| `--authoritative-executor`, schema switches | Always robot-authoritative; profile selects optional hands |
| Trace, queue, completion-observation and advanced timing flags | Internal release defaults |
| Checked arm/OmniHand proposals (schemas 1/2) | Desired targets (schema 3); update both peers |
| Bridge `--collision-checking` / launcher `--xr-collision-checking` | `XR_ARM_SETTINGS["executor_collision_checking"]` in `x2_xrobotoolkit_cfg.py` |
| Hand velocity CLI override | `hand_velocity_limit` in the shared configuration, for filtered fingers |
| `integrations.robojudo_x2` compatibility entrypoint | `integrations.robojudo` |
| X2 retargeter/settings compatibility aliases | `ExternalStateRetargeter` and `TeleopSettings` |

Non-XR controller protocols and standalone upstream demos retain their own behavior.
