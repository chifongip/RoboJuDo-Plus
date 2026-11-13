# XRoboToolkit X2 arm teleoperation

The optional `third_party/xrobotoolkit_teleop` submodule runs XR input, arm inverse kinematics, and optional self-collision
checks in a separate process. RoboJuDo's existing X2 locomanipulation pipeline owns walking, runtime modes, recording,
and robot hardware. The default bridge observes all 31 X2 joints and commands the 14 arm joints. Optional OmniHand Pro presets add
24 active finger commands and 38 finger observations. Both use the existing loco-mimic runtime.

```text
Headset/controllers -> PC Service -> XR bridge (XR environment)
                                        | bounded arm targets (8559)
                                        v
Joystick/keyboard -> RoboJuDo locomanipulation -> MuJoCo OR X2/AimDK
                                        |
                                        +-> measured joints, command baseline,
                                            limits and control intent (8571) -> bridge
```

## Environment and installation

The setup script creates a **single teleop-only Conda environment named `robojudo-xr`**, using Python 3.11,
with both Placo and LeRobot installed. RoboJuDo itself stays in its existing environment.
Linux x86_64/aarch64, Conda, and a system `g++` compiler are required. CMake and Ninja are installed into the new environment.
On Ubuntu, install `build-essential` first if `g++` is missing.

From the RoboJuDo-Plus repository root, run:

```bash
python scripts/install_xrobotoolkit_teleop.py
conda activate robojudo-xr
```

The launcher works with Python 3.10+ and does not require PyYAML or activation of the target environment.
It uses Conda to create Python 3.11, installs the editable `[x2]` package and pinned native dependencies,
downloads checksum-verified Pinocchio 4.1.0 sources, and builds the symbolic bindings for this interpreter.
The bindings are registered permanently inside the environment; no `PYTHONPATH` export is needed, including in new terminals.
After successful setup, the installer saves `LD_PRELOAD` pointing to the environment's `lib/libstdc++.so.6`
when that library is present. This makes normal Python sessions use the same C++ runtime as the installer checks,
including on Jetson. This replaces any previously configured `LD_PRELOAD` value for this environment.
If the environment is already active, run `conda deactivate` and `conda activate robojudo-xr` to apply the setting.
The initial compilation can take several minutes. The default is two build jobs; use `--jobs 1` on memory-limited systems.

If `robojudo-xr` already exists, setup stops before changing it. Choose a different name or explicitly reuse/repair it:

```bash
python scripts/install_xrobotoolkit_teleop.py --name my-xr
python scripts/install_xrobotoolkit_teleop.py --reuse
```

Concurrent setup commands for the same environment name are rejected before checking or modifying the target.
Reuse requires a Python 3.11 Conda environment. It never removes the environment and refuses unmanaged binding overlays.
A healthy installer-owned overlay is verified and reused; otherwise a replacement is built and verified before registration.
The invoking environment and teleop submodule worktree are preserved. The optional-module entrypoint
`python submodule_install.py xrobotoolkit_teleop` delegates to this same setup (that dispatcher requires PyYAML).
The installer does not run upstream setup scripts, install RoboJuDo, or install other robot SDKs, Torch, camera packages,
or hand-retargeting packages. PC Service remains a separate application.

Dependency pins live in `scripts/requirements_xrobotoolkit_teleop.txt`. CasADi 3.6.7 comes from conda-forge;
do not replace it with a pip CasADi wheel. Build logs and an installation manifest are retained under
`$CONDA_PREFIX/share/robojudo-xr/`. A failure reports its stage and log directory; after resolving the cause,
rerun with `--reuse`. A successful setup also verifies operation after removing its temporary source/build files.

## Verify installation

Setup checks both native import orders, construction of both IK backends, an IPOPT solve, package consistency,
and both bridge CLI entrypoints,
without initializing XR or opening a viewer. You can repeat the checks after activation:

```bash
python -m pip check
python -c "import placo; import pinocchio.casadi; import mujoco; print('OK')"
python -c "import pinocchio.casadi; import placo; print('OK')"
python -m xrobotoolkit_teleop.integrations.robojudo --help
```

Both solvers run in this same environment. There is no separate LeRobot environment or manual native-overlay step.

## Run with RoboJuDo

Install and start [XRoboToolkit PC Service](https://github.com/XR-Robotics/XRoboToolkit-PC-Service), connect the
headset/controllers, and enable headset pose streaming. Use two terminals from the RoboJuDo-Plus root.

In your normal RoboJuDo environment, start simulation:

```bash
python scripts/run_pipeline.py -c x2_xrobotoolkit
```

In the XR environment, start the headless bridge:

```bash
conda activate robojudo-xr
python -m xrobotoolkit_teleop.integrations.robojudo --robot x2 --ik-backend placo
```

The bridge uses one startup flag for arm, finger and homing collision checking:
include `--collision-checking` to enable checks; omit it to disable them (the CLI
default). The flag takes no value. For example, to enable these checks with OmniHand:

```bash
python -m xrobotoolkit_teleop.integrations.robojudo --robot x2_omnihand --collision-checking
```

Disabling this option skips IK collision avoidance/recovery and collision path
checks, including partial arm/hand application checks. Joint position and velocity
limits, finite-value validation, freshness deadlines, and acknowledgements remain
enforced. The setting is logged at startup and stays fixed until bridge restart.
It does not change simulation contacts, robot watchdogs, or standalone demo behavior.
Programmatic `TeleopSettings()` retains its enabled default; the CLI explicitly
selects the setting based on whether the flag is present.

Complete the usual preparation sequence: simulation keyboard `i` / joystick `Y` selects `JOINT_DEFAULT`; after
interpolation completes, `j` / `X` selects `RL_DEFAULT`. Toggle arm tracking with `t` / joystick `L`. Release both XR grips
before engaging; hold the corresponding grip to move an arm. Release a grip to hold that arm's target. XR `Y` held
returns both arms toward their defaults. Walking and policy switching use the inherited loco-mimic controls:
`Back` selects locomotion and `Start` selects mimic. XR arm tracking is available in locomotion mode after policy
interpolation completes.

Hand targets are relative to the measured torso, so walking, turning, and waist motion carry them with the robot.
The headset's horizontal heading maps controller motion to torso axes at engagement; subsequent head motion does
not move the targets. No robot odometry or independent physics simulation is used by the bridge.

For real deployment, follow the [X2 AimDK setup](x2_sim2real.md), then use `-c x2_xrobotoolkit_real` in the robot
terminal. The bridge command is identical. The XR presets inherit the observer environment and model: simulation
compensation is `ACTIVE`, while real-robot compensation defaults to `OFF`. Real joystick preparation uses
`RB+Y` for `JOINT_DEFAULT` and `RB+X` for `RL_DEFAULT`. Real-robot acceptance is still required; automated validation
uses synthetic XR input and MuJoCo, not a physical X2.

## Defaults and interruptions

The `x2_xrobotoolkit` simulation displays a labelled world frame (fixed), the measured torso frame (IK reference),
and hand target poses. Axes are X red, Y green, and Z blue; left target origins are yellow and right origins cyan.
Targets show the requested hand-center poses, including requests rejected by IK or collision checks. They are
display-only and do not refresh command freshness. Missing or expired targets disappear. The simulation preset
enables this through `UpperBodyZmqCtrlCfg.visualize_targets`; the environment also needs `visualize_extras=True`.
Real-robot presets leave target visualization disabled.

| Parameter | Default |
|---|---:|
| Arm command speed cap | 8.0 rad/s |
| Return speed cap | 1.0 rad/s |
| Control frequency | 50 Hz |
| Translation scale | 1.0 |
| CLI collision checking | Disabled; enable with `--collision-checking` |
| Collision margin / activation distance | 0.02 / 0.08 m |
| XR tracking timeout | 0.25 s |
| Robot measurement timeout | 0.1 s |
| Grip activation | greater than 0.9 |

The standalone demo and retargeter share upstream numeric defaults. Per-joint URDF velocity limits and the intersection of
URDF/deployment position limits also apply. The bridge accepts explicit overrides such as `--arm-velocity-limit 2.0`;
its effective caps cannot exceed those advertised by RoboJuDo's controller configuration. The new presets disable
additional EMA smoothing and reject invalid proposals instead of modifying a collision-checked target.

- Tracking loss, stale robot feedback, or bridge loss holds the last commanded arm targets. Brief interruptions
  automatically rebase controller references on fresh data, discarding motion made during the hold. Startup,
  explicit mode/intent changes, and a restarted bridge still require grip release before engagement.
- Explicit teleop disable requests a bounded return to defaults, with collision checks when enabled. If the bridge
  is offline, the request remains pending and the arms hold. After reconnection the return uses fresh measurements
  and does not require XR.
- Re-enabling tracking cancels the pending return and requires grip engagement. Whole-robot mode changes and AimDK
  safety states take priority over arm tracking/return requests.
- Recording accepts fresh tracking proposals, excluding homing and outage holds.

IK starts from acknowledged arm targets with the measured body pose, so servo lag in an ungripped arm cannot conflict
with its command speed bounds. Placo's posture objective uses the fixed configured default, rather than the previous
command; its shared solver also applies the upstream manipulability objective. LeRobot uses its upstream joint-angle
and filtered-history regularization. These objectives can choose different redundant-joint postures for the same hand pose.
Collision checks cover sampled paths from measured and previous commanded positions.
With collision checking enabled, rejected full steps try a bounded set of shorter, independently checked steps before holding.
They do not predict future locomotion motion or external obstacles. The XR SDK exposes a shared stream timestamp;
it cannot distinguish one device's cached pose from a live stationary pose while other devices keep streaming.

## Transport and configuration

The existing `UpperBodyZmqCtrlCfg` and pipeline are reused. The presets opt in with `feedback_endpoint` and
`ema_alpha=0`. All upper-body presets hold the entire configured group on timeout and return to defaults on explicit
disable. The JSON `positions` envelope remains unchanged, but publishers must now send all configured joints in each
frame; partial updates are rejected. XR disable returns through the bridge's checked HOME path, pending bridge recovery
if it is offline.
The same `X2LocomanipulationLocoMimicPipeline` runs arm-only and OmniHand XR presets. Proposal validation,
application, feedback publication and recording composition share the existing upper-body pipeline; the controller
supplies hand measurements, limits and acknowledgement status. There is no XR-specific controller or pipeline class.

The arm-only presets use schema version 1. The OmniHand XR presets use schema version 2 and the existing
`UpperBodyOmniHandZmqCtrlCfg` with `feedback_endpoint` configured. Without feedback, that same controller
retains the dex-teleop synchronized-frame protocol.

The bridge binds a command PUB socket on `tcp://127.0.0.1:8559`; RoboJuDo connects its existing SUB socket there.
RoboJuDo binds feedback PUB on `tcp://127.0.0.1:8571`; the bridge connects there. Do not run another arm publisher
on the same command endpoint. Both ports are configurable. For separate hosts, configure RoboJuDo's command
endpoint to the XR host and feedback binding to a reachable interface, then set the bridge's `--command-endpoint`
binding and `--feedback-endpoint` connection accordingly. Loopback is the default.

Version 1 single-frame JSON messages are `upper_body_state` and `upper_body_proposal`. Feedback carries measured
joints and sample ages, last commanded arms, default pose, position/speed limits, period, runtime mode, and intent
(`TRACK`, `HOME`, `HOLD`). Proposals carry all arm positions, status (`tracking` or `homing`), bridge ID, and echoed
session/generation/state/command IDs. Robot-owned deadlines and command acknowledgements reject delayed, replayed,
or superseded proposals without comparing host clocks. The pipeline rechecks freshness immediately before applying
the target; command acknowledgement occurs after successful environment submission, not physical actuator confirmation.
Feedback publication uses the environment's post-step cached snapshot and its original measurement timestamps.
Proposals are polled again after policy inference, immediately before arm override, to avoid an extra cycle of queueing.
A tracking-command timeout revokes the generation even when robot feedback stays healthy, so old proposals cannot
resume motion. Feedback identifies this reset separately from mode/session changes, allowing the bridge to rebase
an already engaged controller using fresh state. Acknowledgement retries also rebase without revoking engagement.
The bridge expects the bundled X2 kinematics; custom geometry requires a matching
retargeting model.

## Automated validation

In the RoboJuDo environment:

```bash
python -m unittest discover -s tests -p test_upper_body_feedback.py
# Optional loopback test: use the absolute path to the dedicated XR Python.
XROBOTOOLKIT_TEST_PYTHON=/path/to/robojudo-xr/bin/python \
  python -m unittest discover -s tests -p test_xrobotoolkit_integration.py -v
```

The second command runs real MuJoCo physics and the ONNX locomotion policy alongside a separate bridge interpreter
with synthetic XR input. It checks activation, movement, tracking loss, process loss, and deferred return after
bridge restart. Without the environment variable, this optional test is skipped. No hardware SDK is initialized.
Upstream headless solver/demo tests run with `python -m unittest discover -s tests -v` from the submodule root.

The original standalone demonstration remains available through
`python third_party/xrobotoolkit_teleop/scripts/simulation/teleop_agibot_x2_mujoco.py`. Its `--with-omnihand` mode
remains a separate standalone simulation.

## Troubleshooting

To measure OmniHand receive limits independently of XR and policy inference, stop
other hand clients and run this read-only probe on the robot using the same local
serial configuration as `check_omnihand_hcan.py`:

```bash
python scripts/probe_omnihand_receive.py --config /path/to/omnihand_serials.json \
  --timeouts-ms 10 20 25 50 100 200 --duration 30 --rate-hz 100 \
  --output-prefix /tmp/omnihand-receive
```

Both hands are read concurrently in an isolated SDK process for each candidate.
Identity verification uses the configured timeout (200 ms by default), then the
probe applies the candidate timeout directly.
It sends no motion commands. A parent watchdog terminates stuck reads or startup.
JSON summaries and per-read CSV record full SDK read latency (p50/p95/p99/max),
invalid readings/exceptions, valid receive rate, gaps between valid readings, and
the previous measurement's age when a read returns. Read-start timestamps match
the runtime's conservative timestamp convention. `--feedback-budget-ms` only
compares results against a budget; it does not change control configuration.

Use failure counts and tail latency to compare candidates. A read exceeding its
configured frame timeout shows that the timeout does not bound the full API call.
Finite joint arrays alone cannot prove that firmware returned a new sample. This
probe measures reads only; it does not establish an end-to-end control deadline or
measure command writes. Exit status is nonzero for failed stages or invalid reads.

OmniHand control uses the configured `frame_recv_timeout_ms` unchanged in both
feedback and streaming modes (200 ms by default). Configure it independently of
`feedback_timeout_s`: the SDK receive wait and robot measurement freshness budget
serve different purposes. Reads and writes remain serialized per hand handle.
A slow or stuck SDK call can still make measurements stale and trigger a hold;
the feedback freshness checks remain enforced independently of the receive timeout.

Combined proposals are admitted to the hand runtime immediately before body submission.
Admission failure retains the previous arm target, and body submission failure revokes
pending hand writes. Hardware writes cannot be rolled back or made simultaneous across
the robot and two hand handles. Collision checks therefore sample independent progress
of the arms, left hand and right hand, including partial application, with a bounded
sample budget. A matching hardware application ID proves both hand writes finished
before their measurement deadline; delayed observation of that ID does not itself
invalidate the command. Fresh measurements and matching session/generation still apply.

- To measure SDK timestamp updates independently of IK and ZMQ, stop the bridge and other XR SDK clients, leave
  PC Service/headset streaming, and run in `robojudo-xr`:

  ```bash
  python -m xrobotoolkit_teleop.integrations.xr_receive_rate --duration 30 --csv /tmp/xr-rate.csv
  ```

  The probe samples at 200 Hz and reports cumulative observed update rate, recent update-gap p95, lifetime maximum
  update gap, maximum local polling gap, source timestamp gaps, and 0.25 s timeout episodes. This measures changes
  visible through the SDK, not every network packet; faster updates between polls can be missed.
  It does not compare headset and host clocks.
  Repeat with RoboJuDo simulation running while the bridge remains stopped to check the effect of system load.
  Large update gaps with small local polling gaps point to the headset/PC Service/SDK path. Large polling gaps mean
  local scheduling or blocking also needs investigation. An average rate alone cannot explain intermittent timeouts.
  The recent p95 uses the latest 4096 observed update gaps; history is bounded for long captures.
- For repeated XR timestamp timeouts, run the bridge with `--xr-diagnostics`. Every two seconds it reports the SDK's
  raw timestamp, seconds since its last observed change, observed update rate, grip values, and robot intent.
  The observed rate measures bridge reads, not network packet rate. In TRACK intent, check whether the timestamp
  advances and whether a squeezed grip exceeds `0.9`. A stationary pose alone does not indicate a stale stream.
- The same flag prints arm round-trip timings: feedback age at publication, policy inference, environment stepping
  (including simulation rendering/physics), retargeting, and the last observed acknowledgement wait, all in milliseconds.
  These are local durations, not cross-host clock comparisons. `measurement_age_lower_bound` excludes network transit
  and queueing; the robot's age check remains authoritative. For command expiry, RoboJuDo logs the validation stage,
  deadline overrun, publication age, and configured budget. The 100 ms measurement deadline remains enforced.
  Timing fields show the most recent observation of each stage, rather than one synchronized command trace.
- Run one XR SDK client at a time when comparing against the standalone demo. The locally supplied PC Service
  implementation stops an existing feedback client when a new one subscribes.
- The launcher can use Python 3.10, but the target environment must use Python 3.11. If reusing a different
  interpreter version, choose a new `--name`; setup will not change an existing environment's Python version.
- If the installer reports a missing `x2` extra, run `git submodule update --init third_party/xrobotoolkit_teleop`
  to select the pinned revision, then retry installation.
- If `pip check` reports unrelated ROS packages in an isolated environment, use a fresh terminal without a sourced
  ROS workspace; a ROS `PYTHONPATH` can expose packages outside the active environment.
- If real-robot feedback remains invalid, check all joint streams and rebuild `aimdk_cpp` if its robot snapshot lacks
  `joint_received_time_sec`. Missing or stale measurements must not be replaced by publication timestamps.
- Missing native build or graphics libraries must be installed separately if pip or the import check reports them.
  Upstream setup is tested on Ubuntu 22.04; live simulation requires a working graphical desktop.

## Robot profiles and selectable IK

The bridge now uses the shared upstream IK backends. Select LeRobot with
`python -m xrobotoolkit_teleop.integrations.robojudo --robot x2 --ik-backend lerobot`.
The existing `robojudo_x2` entrypoint and constructor remain compatible, and Placo remains the default.
Placo now uses the upstream manipulability objective; matching hand poses can produce different redundant-joint postures.
LeRobot uses its upstream posture objective and filter. Both preserve the same command bounds and honor the collision-checking flag.
Stateful solver history is committed only after feedback confirms the exact submitted command.

The installer builds and registers LeRobot's Pinocchio CasADi bindings alongside Placo in `robojudo-xr`.
Both methods use the same Python 3.11 interpreter and compatible Conda CasADi/IPOPT libraries.
See the [bridge architecture and robot extension guide](../third_party/xrobotoolkit_teleop/docs/robojudo_bridge.md)
and [native dependency guide](../third_party/xrobotoolkit_teleop/docs/lerobot_ik.md).

To exercise LeRobot in the two-process integration test, also set `XROBOTOOLKIT_TEST_BACKEND=lerobot` alongside
`XROBOTOOLKIT_TEST_PYTHON`. Tests always import this repository's bundled bridge source.
X2 is the first production robot profile. Other scalar-joint robots can register model resources, observed/controlled
joints, reference/end-effector frames and XR device mappings without modifying bridge transport or backend dispatch.
The `x2_omnihand` profile adds fingers through schema version 2; mobile-base commands remain outside this protocol.

## Setup validation

The clean installation was validated on Linux x86_64 with Python 3.11.18, NumPy 2.2.6, Placo 0.11.0,
Pinocchio 4.1.0 and Conda CasADi 3.6.7. Both native import orders, backend construction and IPOPT passed
without custom path exports, including after a new Conda activation and removal of the build trees.
The native suite passed 186 tests; both Placo and LeRobot passed the two-process RoboJuDo/MuJoCo integration tests.
The installer and optional-module dispatcher tests passed, including existing-target protection and verified reuse.
Native binding runtime paths point only to their own target environment. This clean-install validation did not cover aarch64.


## OmniHand Pro through robot-side HCAN

Simulation uses the existing 31-joint body policy and observer, with 38 additional articulated finger joints and
24 position actuators in the same MuJoCo world. Both IK backends support the `x2_omnihand` profile:

```bash
# RoboJuDo environment
python scripts/run_pipeline.py -c x2_xrobotoolkit_omnihand
# Separate terminal, XR environment
python -m xrobotoolkit_teleop.integrations.robojudo --robot x2_omnihand --ik-backend placo
# Alternatively: --ik-backend lerobot
```

Use the normal preparation sequence and tracking toggle. Squeeze either trigger to close its hand; release the
trigger to open. Triggers work independently of the grips and arm IK. Grips clutch only the arms; release both
grips before engaging arm tracking. Finger targets use a coordinated ramp capped at 2 rad/s by default; the bridge supports
`--hand-velocity-limit`, additionally bounded by the robot's configured cap and model limits. XR Y and explicit
tracking disable return the arms through bounded HOME while retaining finger targets. Outages hold targets.

Physical hands connect through HCAN adapters attached to the **robot**, with channel 0 and hand device ID 1 by
default. The XR host does not open adapters or install the vendor SDK. In the robot's RoboJuDo environment:

```bash
python submodule_install.py omnihand_sdk
sudo bash third_party/omnihand_sdk/linux/aarch64/setup_udev.sh
# Use linux/x64 on an x86_64 robot host; log out and back in after initial udev setup.
```

Create a robot-local JSON file, for example `.local/omnihand.json` (JSON files are ignored by this repository):

```json
{
  "left_hcan_serial": "LEFT_ADAPTER_HW_SER",
  "right_hcan_serial": "RIGHT_ADAPTER_HW_SER",
  "left_hand_serial": "EXACT_LEFT_VENDOR_SERIAL",
  "right_hand_serial": "EXACT_RIGHT_VENDOR_SERIAL"
}
```

Adapter selection uses `hcan_serial_number`, following the vendor HCAN demos. Hand identity uses **exact**
`product_seq_num` matching after removing trailing NUL padding only; suffixes are not accepted. Both hands
must verify product information, device ID, hand serial, and finite joint measurements before takeover. Missing
or incorrect serial configuration stops startup rather than falling back to adapter enumeration indices.
Both dex-teleop and XR controllers use the single `OmniHandRuntime` in `omnihand_runtime.py`, with the same
process isolation and cleanup. Existing deployments retain HCAN adapter indices, ZLGCAN and SocketCAN.
Serial selection requires HCAN; other transports reject serial fields rather than silently ignoring them.
Dex-teleop retains its latest-only `set_joint_commands` API; checked XR uses strict `submit` with acknowledgement.
The shared `startup_timeout_s` defaults to 15 s. No separate HCAN runtime implementation is maintained.

Stop other adapter clients, then run the read-only check before enabling robot motion:

```bash
python scripts/check_omnihand_hcan.py --config .local/omnihand.json
export ROBOJUDO_OMNIHAND_CONFIG="$PWD/.local/omnihand.json"
python scripts/run_pipeline.py -c x2_xrobotoolkit_omnihand_real
```

The bridge command is identical to simulation. Follow the normal X2 real-robot preparation procedure.
The check only verifies identity and reads joint angles; it does not send motion commands. One persistent spawned
SDK process owns both adapters for HCAN, ZLGCAN, and SocketCAN. It opens and verifies the left hand before opening
the right, retaining both connections in the same SDK context. This avoids competing per-process HCAN scans;
the parent never loads the SDK. After initialization each hand has its own I/O thread, so the two reads overlap
without making concurrent calls on the same hand handle. Each thread checks authorization and deadline before
its hand write and publishes the write result before starting another read. Measurement timestamps are retained
when publishing acknowledgements; a write does not make old measurements fresh. A command is acknowledged only
when both hands applied it under the current authorization. Startup and cleanup waits are bounded, and SDK receives
retain their timeout;
a worker fault cancels outstanding work. A native call that hangs makes measurements stale; shutdown escalates
to process termination if needed. Partial startup failure also releases the process and its adapters.
Default receive timeout is 200 ms and worker startup budget is 15 s. These do not relax the
100 ms robot feedback/proposal deadline. Repeated slow replies cause holds and need diagnosis on the robot.

After updating a Jetson deployment, stop all other OmniHand clients and validate USB reliability:

```bash
for attempt in $(seq 1 10); do
  python scripts/check_omnihand_hcan.py --config .local/omnihand.json || break
done
```

All ten runs must verify both hands and exit successfully. Then run this 60-second read-only check from the
repository root in the installed RoboJuDo environment. It uses the existing freshness limits and reports the
oldest observed sample age. It never enables takeover or sends motion commands. The file and main guard are
required for multiprocessing spawn. Repeat the normal check afterward to verify adapters can be reopened.

```bash
cat > /tmp/check_omnihand_duration.py <<'PY'
import json
import time
from pathlib import Path
from robojudo.controller.ctrl_cfgs import OmniHandCfg
from robojudo.controller.omnihand_runtime import OmniHandRuntime


def main():
    cfg = OmniHandCfg.model_validate(json.loads(Path(".local/omnihand.json").read_text()))
    runtime = OmniHandRuntime(cfg, read_only=True, require_serials=True)
    oldest_age = 0.0
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            state = runtime.get_data()
            if not state["joint_state_fresh"]:
                raise RuntimeError("OmniHand measurements became stale")
            oldest_age = max(oldest_age, time.monotonic() - min(state["sample_times"].values()))
            time.sleep(0.01)
        print(f"Both hands remained fresh for 60 s; oldest sample age: {oldest_age:.3f} s")
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
PY
python /tmp/check_omnihand_duration.py
python scripts/check_omnihand_hcan.py --config .local/omnihand.json
```

Keep the default safety thresholds during validation. Staleness under normal load blocks deployment and needs
diagnosis; do not compensate by increasing freshness timeouts. Retain the validation logs. The installer still
installs the Python wheel; first-time USB permissions remain a separate `setup_udev.sh` step.

Schema version 2 keeps the existing message types and session/generation/deadline fields. State `positions` and
`joint_age_s` cover 69 joints (31 body + 38 fingers); proposal `positions`, state `last_commanded_positions`,
`default_positions`, and `position_limits` cover 38 commanded joints (14 arms + 24 active fingers). State adds
`hand_velocity_limit`. Active hand names/order follow `OMNIHAND_JOINT_NAMES`; passive observations use the seven
SDK passive joints per hand. Both endpoints default to the existing loopback ports. A version/profile mismatch
is rejected. Run only one command publisher.

The robot revalidates proposals before application, submits arms through its environment, and then dispatches
fingers. Acknowledgement requires successful arm submission and both hand SDK writes for the same proposal.
The vendor Python angle setter exposes no per-write status: acknowledgement confirms that both SDK calls
returned without exceptions, not firmware acceptance, physical arrival, or simultaneous actuation.
Partial application cannot be rolled back; it
invalidates the generation and cancels pending work. New feedback exposes the resulting baseline. Recording
includes the 14 arm and 24 active finger observations/commands only on acknowledged fresh tracking frames.

When enabled, collision checks use measured fingers, nonlinear passive coupling, and sampled combined command paths. When
finger motion fails, the bridge may send an independently checked arm escape with fingers held. Internal same-hand
collision pairs remain excluded as in the standalone model. Hand contacts are disabled in simulation; this is
not an object-grasp physics or tactile-control implementation. Wrist mounts remain provisional until checked
against the physical adapters. Before qualifying deployment, verify mounts, exact identities, SDK joint ordering
and signs, low-speed one-hand operation, grip release, HOME finger hold, and XR/bridge/adapter interruptions.

Rebuild the floating hand model with `python scripts/build_x2_xr_omnihand_assets.py`. It reuses the submodule's
hand meshes, source finger dynamics/couplings, short-wrist data, and their existing license/provenance metadata.
Automated two-process integration tests cover both arm-only and OmniHand presets with synthetic XR input:

```bash
XROBOTOOLKIT_TEST_PYTHON=/path/to/robojudo-xr/bin/python \
  python -m unittest discover -s tests -p test_xrobotoolkit_integration.py
# Repeat with XROBOTOOLKIT_TEST_BACKEND=lerobot.
python -m unittest discover -s tests -p test_xr_omnihand.py
```
