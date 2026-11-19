# XRoboToolkit integration

RoboJuDo owns robot modes, walking, hardware, recording, and executable targets. A separate XR bridge reads
XRoboToolkit PC Service input and solves arm IK using measured robot feedback. Both arm-only and OmniHand XR send desired targets to RoboJuDo's robot-authoritative executor.
The robot profile selects the model and optional hand capability; both use the same protocol.
Update RoboJuDo and its XR submodule together.

## Install

Initialize the repository's pinned submodules. Keep RoboJuDo in its existing environment. From the repository root,
create the separate Python 3.11 XR environment with both supported IK backends:

```bash
python scripts/install_xrobotoolkit_teleop.py
```

The installer requires Conda, Linux x86_64/aarch64, and a system C++ compiler. It installs pinned native dependencies,
Placo and LeRobot, and verifies imports and solver construction without opening XR or a viewer. Native build logs
and the manifest are under `$CONDA_PREFIX/share/robojudo-xr/`. An existing environment requires `--reuse`; use
`--name NAME` for another environment or `--jobs 1` on a memory-limited machine. Reactivate after installation
so the installer's Conda native-runtime setting takes effect. No manual `PYTHONPATH` overlay is required.

For **all XR presets**, install and verify numeric executor support with the RoboJuDo environment's interpreter:

```bash
python scripts/install_xr_executor.py
```

This installs the numeric Pinocchio/Coal dependencies and the shared XR model package without installing XR SDK/IK
packages into RoboJuDo. The installer verifies both profiles. Simulation and hardware use the same executor and model identity checks.
Rerun the executor installer after updating the submodule to clear older SDK/IK dependency errors in `pip check`.

Install and start XRoboToolkit PC Service separately, connect the headset/controllers, and enable headset pose streaming.

## Run simulation

Use two terminals from the repository root:

```bash
# RoboJuDo environment: arm-only
python scripts/run_pipeline.py -c x2_xrobotoolkit
# Or arms and OmniHand
python scripts/run_pipeline.py -c x2_xrobotoolkit_omnihand
```

```bash
# XR environment: match the selected robot profile
conda activate robojudo-xr
python -m xrobotoolkit_teleop.integrations.robojudo --robot x2
# Or arms and OmniHand
python -m xrobotoolkit_teleop.integrations.robojudo --robot x2_omnihand
```

Placo is the default IK backend; use `--ik-backend lerobot` to select LeRobot. Both use the same XR environment.
The simulation displays torso and requested hand frames; rejected requests may still appear as display-only targets.

Select `JOINT_DEFAULT` with keyboard `i` / joystick `Y`, wait for interpolation, then select `RL_DEFAULT` with `j` / `X`.
Toggle arm tracking with `t` / joystick `L`. Release both XR grips before engagement. Hold a grip to move that arm;
release to hold its issued target. XR `Y` returns both arms toward defaults. Joystick `Back` selects locomotion and
`Start` selects mimic; XR tracking requires available locomotion control after interpolation.

Targets are relative to the measured torso. Headset horizontal heading is captured at engagement; subsequent head
motion does not move targets. OmniHand triggers control fingers independently of arm grips. HOME holds finger targets.

### Pelvis IK with OmniHand

Enable XR body tracking, then launch the pelvis IK simulation and bridge from separate environments:

```bash
# RoboJuDo environment
python scripts/run_pipeline.py -c x2_xrobotoolkit_omnihand_pelvis_ik

# XR environment
conda activate robojudo-xr
python -m xrobotoolkit_teleop.integrations.robojudo --robot x2_omnihand --ik-backend lerobot \
  --tracking-reference pelvis-ik
```

Enter `JOINT_DEFAULT`, wait for interpolation, enter `RL_DEFAULT`, and enable XR tracking. Release both grips
before engaging. Either grip engages posture and its own arm. Keeping one grip held while handing off preserves
the posture reference. Triggers control fingers independently of grips.

Crouch or rise to move the height target. Twist your shoulders relative to your pelvis to change preferred waist
yaw. The solver can adjust yaw when an active hand misses its goal. Hand goals follow the robot pelvis during a
crouch. The locomotion policy applies its configured height and waist slew rates.

The bridge sends posture on the separate loopback port 8557 (`--posture-endpoint`) and sends arms and fingers
on the existing upper-body stream. Height can update while arm IK is running. RoboJuDo validates posture against
the current XR session and gives manual posture input priority. After manual takeover, release both grips before
engaging posture again. Releasing both grips or losing body tracking stops new posture targets. The policy can
finish ramping to the last accepted target. Explicitly disabling XR freezes the posture command at its current
value. Press XR `Y` to request arm and posture defaults while the fingers hold their issued targets. The arm
return continues after the button is released until robot feedback confirms convergence. Release both grips
before resuming tracking.

The bridge adopts a solved waist target after RoboJuDo acknowledges the corresponding upper-body frame.
With collision checking enabled, RoboJuDo checks the executable waist, arm, and finger transition together
before submission. An expired or rejected transition holds the previous safe targets.

This preset requires the paired XRoboToolkit revision and the X2 OmniHand simulation assets.

## Deployment settings

The bridge CLI exposes `--robot`, `--ik-backend`, `--tracking-reference`, `--command-endpoint`,
`--feedback-endpoint`, `--posture-endpoint`, `--scale-factor`, `--arm-velocity-limit`, and
`--return-velocity-limit`. Upper-body endpoints default to loopback ports 8559
(commands) and 8571 (feedback); edit the corresponding RoboJuDo controller endpoints for a multi-host deployment.
Run only one bridge publisher.

| Setting | Release default |
|---|---:|
| Control rate | 50 Hz |
| Motion scale | 1.0 |
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

This shared configuration applies to both existing OmniHand simulation and hardware presets. There is no
additional enable switch, launcher flag, or special preset. Restart RoboJuDo after changing it; the bridge
validates and echoes the robot-owned settings through the existing executor handshake.

- **Direct:** issue final finger targets within position bounds; device motor settings govern movement.
- **Filtered:** apply the same robot-side EMA and independent per-joint velocity filter as the arms, using the
  smaller of `hand_velocity_limit` and each joint's model velocity limit. Both use the existing `ema_alpha`
  (0.0 in XR presets), position limits, and float32 rounding.

The bridge sends final desired targets in either mode. Filtering advances once per fresh desired frame from
the completed issued baseline, without accumulating movement allowance during gaps or pending writes. HOME
holds fingers in both modes; collision checks and recording use actual issued targets after filtering.
Software target ramps do not guarantee physical finger speed or stopping behavior. Revoking commands does not
cancel movement toward a target already accepted by the device.

Configure collision checking for all four presets in `XR_ARM_SETTINGS` in the same file:

```python
"executor_collision_checking": False,  # Set True to enable collision checking.
```

RoboJuDo owns this setting. Start the existing preset normally; neither the launcher nor the bridge has a
collision override flag.

Start the bridge normally. Validated feedback supplies the profile identity and collision policy; the bridge
initializes the corresponding IK checks before publishing desired targets. The policy is fixed for the robot
session. Changing it requires restarting RoboJuDo; a bridge cannot independently disable robot checks.

Model mismatches, malformed policy, or enabled-check initialization failures block commands. Initialization and
IK time count toward measurement freshness. Large direct finger changes can exceed the bounded collision-sampling
budget and hold; use gradual trigger movement when checks are enabled. The executor does not search for smaller
steps. Numeric collision defaults are shared internally. Disabling collision checks retains finite-value,
position-limit, freshness, authorization and applicable velocity checks.

## Real robot and hands

Follow [X2 AimDK setup](x2_sim2real.md), then use `x2_xrobotoolkit_real` or `x2_xrobotoolkit_omnihand_real`.
For pelvis IK use `x2_xrobotoolkit_omnihand_pelvis_ik_real` with the LeRobot bridge below. Real joystick
preparation uses `RB+Y` and `RB+X`. Real presets inherit observer compensation mode `OFF`.

For physical OmniHands, install the SDK in the robot's RoboJuDo environment and configure USB permissions:

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
python scripts/run_pipeline.py -c x2_xrobotoolkit_omnihand_real
```

One spawned SDK process owns both adapters, with independent per-hand I/O threads. A combined command completes only
after successful arm submission and both hand writes. A returned SDK call does not prove firmware acceptance,
physical arrival or simultaneous movement. Partial failures invalidate outstanding authorization; they cannot roll
back physical motion. Default receive/startup budgets remain 200 ms / 15 s and do not relax measurement freshness.

Tracking loss or stale feedback holds issued targets. Transient recovery rebases XR references; startup, explicit
mode changes and bridge restarts require grip release. Explicit tracking disable requests bounded arm HOME; if the
bridge is offline, arms hold until fresh feedback permits return. Whole-robot safety states take priority.

Arm-only commands complete after successful arm submission; OmniHand commands also require both hand writes.
Both advance the same completed-command baseline, retaining fresh waiting desired frames after completion.

Recording (`--record`, with the configured recorder service) stores fresh acknowledged tracking commands and measured
positions, excluding HOME and outage holds. Combined command recording uses the completed issued baseline, including
completion observed between control ticks.

### Pelvis IK on X2 with OmniHand

Complete the AimDK and HCAN setup above and enable XR body tracking. Source the AimDK setup file in the RoboJuDo
environment on the robot host and start the hardware preset.

```bash
export ROBOJUDO_OMNIHAND_CONFIG="$PWD/.local/omnihand.json"
python scripts/run_pipeline.py -c x2_xrobotoolkit_omnihand_pelvis_ik_real
```

Start the pelvis IK bridge in a separate XR environment.

```bash
conda activate robojudo-xr
python -m xrobotoolkit_teleop.integrations.robojudo --robot x2_omnihand --ik-backend lerobot \
  --tracking-reference pelvis-ik
```

These commands use loopback endpoints. For a bridge on another host, use the following endpoint settings with
the actual hostnames.

| Stream | Robot controller setting | XR bridge argument |
|---|---|---|
| Posture | `LocomanipulationPostureZmqCtrl.endpoint = tcp://XR_HOST:8557` | `--posture-endpoint 'tcp://*:8557'` |
| Arms and fingers | `UpperBodyOmniHandZmqCtrl.endpoint = tcp://XR_HOST:8559` | `--command-endpoint 'tcp://*:8559'` |
| Feedback | `UpperBodyOmniHandZmqCtrl.feedback_endpoint = tcp://*:8571` | `--feedback-endpoint tcp://ROBOT_HOST:8571` |

The hardware preset uses the same posture rate limits as the pelvis IK simulation, **0.2 m/s** for height and
**1.0 rad/s** for waist yaw. With the robot supported and the work area clear, use `RB+Y` to enter `JOINT_DEFAULT`, wait for
interpolation, then use `RB+X` to enter `RL_DEFAULT`. Release both XR grips and enable XR tracking with
joystick `L`. For the first hardware check, use small crouch and twist movements with one grip, verify each
finger trigger independently, then verify grip release, XR `Y` HOME, and the joystick damping command `RB+B`.
Check tracking and feedback loss before trying larger motions.

## Validation and migration

Run focused controller/executor and bridge tests, then the two-process simulation suite:

```bash
python -m unittest discover -s tests -p test_install_xr_executor.py
python -m unittest discover -s tests -p 'test_xr*.py'
XROBOTOOLKIT_TEST_PYTHON=/path/to/robojudo-xr/bin/python \
  python -m unittest discover -s tests -p test_xrobotoolkit_integration.py
# Repeat with XROBOTOOLKIT_TEST_BACKEND=lerobot.
```

Physical acceptance remains required: verify provisional wrist mounts, joint ordering/signs, exact hand identities,
low-speed operation, clutch release, HOME finger hold, recording, and XR/bridge/adapter interruptions. Simulation
has no hand contacts and does not qualify grasp physics, hardware latency or 50 completed commands per second.
Do not increase freshness timeouts to hide stale hardware measurements.

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

The bridge and RoboJuDo print concise INFO summaries once per second, without extra flags:

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
