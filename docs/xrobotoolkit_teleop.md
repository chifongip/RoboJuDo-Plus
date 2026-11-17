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
limits, finite-value validation, freshness deadlines, and robot-side command validation remain
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

IK starts from robot-reported completed arm targets with the measured body pose, so servo lag in an ungripped arm cannot conflict
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
an already engaged controller using fresh state. Missing completion reports do not trigger bridge retries or rebases.
The bridge expects the bundled X2 kinematics; custom geometry requires a matching
retargeting model.

### Continuous publishing and robot-side selection

Every bridge profile, including arm-only control, samples XR and publishes complete checked targets at
`--control-frequency` (50 Hz by default), independently of command completion. There is no acknowledgement wait
or compatibility switch. During valid idle tracking, complete holding targets keep the stream alive after grip-release
prerequisites are met. HOLD, invalid XR, expired feedback, and rejected safety checks suppress command publication;
HOME can run without XR. Feedback snapshots may be reused only while their measurements remain fresh.

RoboJuDo keeps the newest valid waiting frame and discards intermediate frames. Arm-only targets are selected at
robot control ticks. Combined arm-and-hand control finishes the current hand writes before selecting another frame;
new arrivals never interrupt a write. Selection does not make physical arm and finger motion simultaneous. Freshness,
intent, limits, velocity bounds, and the exact command baseline are rechecked before submission. A frame made obsolete
by completion of an earlier command is discarded; the bridge continuously supplies replacements using updated feedback.
Consequently, robot execution frequency may be lower than bridge publishing frequency.

Feedback schemas 1 and 2 include nullable `applied_bridge_id` and `applied_frame_id`, identifying the last completed
proposal in the current generation. Arm-only completion means successful environment submission. Combined completion
also requires both hand writes; neither waits for joints to reach their targets. `last_commanded_positions` stays at a
coherent completed baseline while a combined command is in flight. Resetting the generation clears completion identity.
Update RoboJuDo and the bridge together. Solver history is committed only for an explicitly matching completed frame;
uncompleted proposals remain bounded and cannot advance the solver's accepted history.

With `--xr-diagnostics`, `XR` reports observed SDK timestamp updates, and `publish` reports sent command frames per
second. `completion_latency` reports asynchronous send-to-observed-completion latency; it does not pace publishing.
The additional `Completion p95 (ms)` line breaks down matched completions:

- `robot_admission`: robot receipt/validation to hand submission, including waiting behind an in-flight command.
- `left_hand_queue` / `right_hand_queue`: runtime enqueue to hardware write start, including IPC, worker scheduling,
  and an ongoing measurement read. Commands wake the workers independently of measurement deadlines.
- `left_hand_write` / `right_hand_write`: duration of each SDK write call.
- `left_hand_read` / `right_hand_read`: latest completed SDK measurement read captured with the frame's completion.
  These describe read cost, not necessarily the particular read that delayed that command.
- `left_receipt_to_write` / `right_receipt_to_write`: robot receipt/validation to that hand's SDK write completion.
- `arm_submission`: duration of the environment's arm submission call, not physical motion completion.
- `submission_skew`: span between arm submission start and the two hand write starts. This measures matching-frame
  software submission skew, not physical synchronization.
- `completion_observation`: last hand write completion to the robot observing both writes and acknowledging.
- `worker_to_parent`: worker observation of matching dual-hand completion to the parent's first observation of it.
- `parent_to_ack`: first parent observation of that frame's completion to pipeline acknowledgement.
- `robot_ack_wait`: acknowledgement to the first successfully enqueued acknowledgement snapshot's construction.
- `feedback_snapshot_lag`: construction-time gap between the first acknowledgement snapshot and the snapshot the
  bridge actually processed. Present only when the robot supplies first-publication metadata.
- `robot_total`: robot receipt/validation to first acknowledgement snapshot construction.
- `transport_and_poll`: bridge round-trip latency minus `robot_total`, covering command/feedback transport, socket
  queues, feedback encoding/send, and polling at either end. It cannot distinguish network delay from polling waits.

Raw robot-host monotonic timestamps are included in `feedback_diagnostics.command_timestamps` and belong to the
feedback's `applied_bridge_id` / `applied_frame_id`. They clear on generation reset. Robot workers share that clock;
the bridge compares durations only, so robot and bridge hosts do not need synchronized clocks. Hand branches overlap,
and each p95 is calculated independently, so adding the displayed percentiles does not reconstruct total p95.
`first_feedback_built` is frozen only after a successful local ZeroMQ send; a failed nonblocking send retries with
another candidate timestamp. Successful enqueue does not guarantee subscriber delivery. With older robots,
`robot_ack_wait` and `robot_total` fall back to the processed snapshot's `feedback_built` timestamp.
`Completion observations: idle=..., periodic=..., before_policy=..., after_policy=...` reports where matching frames were acknowledged in the reporting
window. Older peers without these optional timestamps continue operating without a breakdown.

Hand commands now wake each SDK-owning thread instead of waiting for its next measurement tick. Reads stay at
`joint_state_fps`, with missed deadlines skipped rather than replayed. An SDK read already in progress must finish
before the same handle can write. No SDK timeout, velocity limit, or measurement rate is changed.

The standard `scripts/run_pipeline.py` runner services combined completion notifications during its post-step idle
wait, on the same thread that owns control and feedback. It reuses body measurements with their original timestamps;
it does not run policy inference, advance simulation, or issue another command. The supervisor checks completion at
its existing 5 ms interval. Control-step completion checks remain a fallback for other runners and full-speed mode.
The bridge also receives feedback between publication deadlines, without running extra IK or publishing early.
Arms and both hands retain one proposal identity and one combined frame in flight; partial completion cannot advance
the baseline. Physical writes remain separate and cannot be rolled back if a later component fails.

The existing low-latency configuration enables worker notifications and completion checkpoints. Use the same
verified hand identities and robot setup with:

```bash
python scripts/run_pipeline.py -c x2_xrobotoolkit_omnihand_real_direct_low_latency
```

The bridge command is unchanged:

```bash
python -m xrobotoolkit_teleop.integrations.robojudo --robot x2_omnihand --ik-backend lerobot --xr-diagnostics
```

This configuration changes only `OmniHandCfg.direct_completion_notifications` to `True`; the ordinary
`x2_xrobotoolkit_omnihand_real_direct` configuration keeps it `False`. Startup logs identify the selected mode.
The worker signals the existing nonblocking readiness socket only after publishing matching dual-hand completion.
If the snapshot slot is busy, the dispatcher retries publication without waiting for another SDK read. Hints never
authorize a command or replace generation/freshness checks. The supervisor still owns health and reconnects;
periodic completion checks and supervisor hints remain available. The pipeline's existing idle/post-step service
points remain in place; this configuration also enables guarded checks before and after policy inference. Control
frequency, submission order, and the combined-frame barrier are unchanged. No new native calls release
the GIL, no additional thread issues hardware commands, and no SDK timeout or CPU/thread setting changes.
No additional profile or opt-in flag is needed for these checkpoints: selecting
`x2_xrobotoolkit_omnihand_real_direct_low_latency` enables them through the existing direct-notification setting.
The first check runs after intent/safety handling and immediately before inference; the second runs after inference
and before refreshing proposals/selecting the next command. Only a pending frame with successful arm submission,
matching intent/session/generation, and fresh matching dual-hand completion can be acknowledged. Each checkpoint
reuses the existing completion routine and cached body feedback, and publishes nothing when no completed frame is
ready. No new acknowledgement states, hardware writer, or control step is introduced. Dry-run and active recording
keep the existing completion path; recorder bookkeeping is unchanged.

`Processing p95 (ms)` additionally reports `checkpoint_before` and `checkpoint_after`, including eligible no-op
checks. These costs are excluded from `policy`. Acknowledgement source labels distinguish these checkpoints from
idle and periodic post-step servicing. Optional fields remain compatible with older peers.

Select the original direct configuration to roll back the notification and checkpoint optimizations.

For phase 2, compare three 60-second Jetson runs per mode using the same updated timing metadata and workload.
Targets are completion-observation p95 below 8 ms and round-trip p95 below 35 ms, with hand-queue performance retained.
Require zero mixed frames, duplicate acknowledgements, stale replay, or new watchdog/shutdown failures. Control-loop
p99 and overrun rate must remain within baseline run-to-run variation. Keep the path opt-in if targets are unmet.
Hardware measurements are separate from fake-SDK tests; an existing collision-enabled synthetic finger-motion
threshold failure reproduces on the committed baseline and must not be bypassed by disabling collision checks in
real deployment.

For a hardware comparison, collect three 60-second `--xr-diagnostics` runs before and after with identical rates,
collision settings, and workload. Compare hand queue, SDK reads, completion observation, submission skew, total
latency, completions/s, stale-feedback faults, and control overruns. Evaluation targets are queue and observation
p95 below 10 ms, completion p95 below 40 ms, and at least 25 completions/s under continuous publication. These are
measurement targets, not hardware guarantees; blocking SDK reads may limit the improvement.

Robot `received`, `superseded`, `rejected`, and `completed` counters are cumulative diagnostics for command selection.
The former `ack_wait` diagnostic has been removed. Expired solver transactions are discarded separately from the
bounded completion metadata, so a late confirmation still contributes to completion counts and latency without
committing expired solver history. Completion metadata retains the newest 512 sent frames and clears on rebases/resets.

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

OmniHand skips empty, incorrectly sized (expected 19 angles), nonnumeric, or nonfinite
measurements and read exceptions, keeping the previous valid positions and their
original timestamps. Both hands must have fresh feedback before
either hand can receive another command. The worker enforces the stricter of the
hand measurement timeout and feedback measurement budget, even if the main loop is
delayed. Recoverable outages hold arm targets and skip recording stale samples.

SDK joint limits constrain outgoing commands, not measurement acceptance. Finite
measured angles outside those limits are retained unchanged and do not stop startup
or reconnection. Calibration offsets, sensor noise, or mechanical compliance can put
measurements outside the command range. Only the initial float32 command baseline is
clamped to that range; creating this baseline sends no motion command. Previously
issued targets remain retained across reconnection, as described below.

For serial-verified HCAN configurations, `omnihand.auto_reconnect` defaults to `true`.
After `reconnect_timeout_s` (default 1 second) without valid measurements, a background
supervisor stops the old SDK process and opens both adapters in a new process. It
rechecks both hand identities without motor initialization. Failed attempts wait
`reconnect_interval_s` (default 1 second), doubling up to 5 seconds; each attempt uses
the existing `startup_timeout_s`. Reconnection does not block the control loop.
Index-selected adapters do not reconnect automatically because their physical identity
cannot be guaranteed. Set `auto_reconnect: false` to require a restart after persistent
communication failures. Identity mismatches and command-write failures always require
operator investigation and a restart.

After feedback recovers, previously requested tracking resumes automatically using new
commands and the existing XR reference rebasing and speed checks. Motion made during
the hold is discarded. Pending commands and acknowledgements from before the outage
are invalidated; turning tracking off during an outage keeps it off after recovery.
The runtime retains the last reported hardware command targets across reconnection;
measured drift does not replace them. Reconnection itself sends no motion commands.
The runtime snapshot adds `connection_status`, `connection_generation`,
`stream_generation`, `last_error`, and per-hand `hand_health` diagnostics with measurement
age, consecutive read failures, and the last read error. No feedback wire schema changes
are required.

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
  observed update rate, milliseconds since its last timestamp change, robot intent, and grip/trigger values in
  left/right order. Unknown values are shown as `?`; the raw timestamp is omitted.
  The observed rate measures bridge reads, not network packet rate. In TRACK intent, check whether the timestamp
  advances and whether a squeezed grip exceeds `0.9`. A stationary pose alone does not indicate a stale stream.
- The same flag separates measurement ages from execution durations. `feedback_age_at_publish` remains the age of
  the oldest measurement, not publication execution time. `feedback_diagnostics` identifies that joint and the
  maximum ages for `body`, `left_hand`, and `right_hand`. Body timestamps are local AimDK callback receipt times;
  hand timestamps are captured before the SDK read. They are preserved when copied or acknowledged.
  Policy/environment processing, body/hand snapshot copies, feedback assembly, hand IPC handoff, and retargeting
  are timed separately. The `previous_publication` record contains build and JSON/send durations, state ID,
  generation, and send status for the preceding publication, because current send time is not yet known when
  encoding its frame. These optional diagnostic fields do not change the wire schema version; older feedback
  messages without diagnostics are still accepted.
  Every two seconds, the console shows a short feedback summary: latest measurement age, the oldest source
  (body/left hand/right hand), p95 age, and sent/acknowledged command counts. A second line shows p95 processing
  times for policy inference, `env_step`, retargeting and successful acknowledgement waits. Missing stages are omitted;
  unavailable feedback is labelled explicitly. Raw timing dictionaries and snapshot IDs are not printed.
  Detailed fields remain available in the feedback message. Idle intervals do not report a historical
  acknowledgement as a new round trip.
  These are local durations, not cross-host clock comparisons. `measurement_age_lower_bound` excludes network transit
  and queueing; the robot's age check remains authoritative. For command expiry, RoboJuDo logs the validation stage,
  deadline overrun, publication age, and configured budget. The 100 ms measurement deadline remains enforced.
  Timing fields show the most recent observation of each stage, rather than one synchronized command trace.

Feedback publication stays on the 50 Hz control loop. Immediately before publication, AimDK feedback copies the
latest native joint cache without rerunning policy inference, observer updates, or physics. This copy is separate
from the earlier policy snapshot. Simulation and other environments provide their latest cached measurements.
Both hand threads publish to a bounded shared-memory latest-snapshot slot; the main thread and recovery supervisor
can read it without consuming it or waiting for a supervisor polling tick. Sequence numbers and connection
generations identify each snapshot. Reads use nonblocking slot lock acquisition: contention retains the previous
timestamped snapshot, whose normal freshness checks still apply. Worker restarts allocate fresh IPC so a killed
writer cannot leave the replacement blocked on an abandoned lock. A single hand snapshot is reused for
acknowledgement decisions and feedback assembly; a generation change during retrieval cancels the old decision.
Worker spawning runs outside the parent snapshot lock. Slot contention can skip a publication or read; it does
not wait for the other process, and the next successful publication replaces the older snapshot.
Retained measurements during reconnection do not make feedback valid until the hand runtime is ready, even when
their original timestamps are still within the freshness budget.

For before/after measurements, run `--xr-diagnostics` with identical sampling/control settings during idle and
active arm/finger motion. Compare per-source measurement age, hand handoff, execution-time distributions and
timeouts. The top-level XR OmniHand presets set `omnihand.joint_state_fps=50.0`; the generic `OmniHandCfg` default
remains 100 Hz. A sub-millisecond SDK read does not imply a sub-millisecond age at an arbitrarily phased 50 Hz
publication. This change does not alter receive timeouts or the feedback freshness budget.

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

### Direct finger target experiment

Use `python scripts/run_pipeline.py -c x2_xrobotoolkit_omnihand_real_direct`
with the updated XR bridge and the same local hand serial configuration used by
`x2_xrobotoolkit_omnihand_real`. This real-hardware preset sends the final
trigger-proportional grasp target without the bridge's finger velocity ramp.
The robot accepts those larger finger steps only when both sides explicitly
agree on `hand_command_mode=direct`. Arm limits and collision checks remain active.
The original preset uses `filtered` finger targets.

The experiment preserves the device's existing control mode and motor speed
settings. It does not establish a firmware speed cap. SDK read/write scheduling
and repeated target submission are unchanged. Revoking further commands does
not cancel movement toward a target already accepted by the hand.

For an A/B comparison, use identical device settings, arm posture, and slow
partial/full trigger presses with each preset. Record commanded and measured
finger positions and command intervals; compare pauses, total movement time,
and visible jitter. SDK read/write durations require additional instrumentation
if not available in the local diagnostic setup. Do not run a separate SDK probe
while the pipeline owns the adapters. Hardware observations are required to
conclude whether jitter improves.

### Shared XR target filtering

Updated XR peers negotiate `robojudo_filter_v1`. The bridge sends desired targets
and predicts the exact float32 targets produced by RoboJuDo's EMA/velocity filter.
It collision-checks those issued targets; RoboJuDo reproduces them before admission.
Update both peers together: proposals without the contract or with inconsistent
predictions are rejected. XR presets keep `ema_alpha=0.0`; the general configuration
default remains 0.95. The supported finger command modes are `filtered` and `direct`.
Filtered fingers use the shared EMA/velocity filter; direct fingers retain device-managed motion.

Bridge settings and per-joint model velocity limits first bound desired targets; the
robot-owned filter then applies its configured EMA and velocity cap. Direct finger
targets bypass both velocity filters.

Filtering advances from the acknowledged issued baseline, not from unsent bridge
proposals. It never autonomously advances during missing proposals or pending hand
writes. Existing completion timings and command counters remain available; proposals
include `desired_positions` separately from issued `positions` for tracing. SDK I/O
and motor settings are unchanged. Command delivery gaps can still cause pauses.

### Robot-authoritative combined target executor (opt in)

`x2_xrobotoolkit_omnihand_real_direct_executor` retains direct fingers and the
low-latency completion notifications, but constructs executable targets in RoboJuDo.
The existing presets and schemas 1/2 keep their previous behavior.

Install numeric support **using the RoboJuDo environment's interpreter**:

```bash
python scripts/install_xr_executor.py
# Equivalent submodule installer entry:
python submodule_install.py xr_executor
```

The installer adds pip `pin==4.1.0` and `coal==3.0.3`, installs the shared XR
package without pulling its SDK/IK dependencies into RoboJuDo, and loads the
articulated model and queries its collision geometry. The same native pins are
available through RoboJuDo's `xr-executor` extra; the shared XR package and its
matching assets are also required. The bridge continues using its separate XR
environment. No source build or native-library search-path override is introduced.

Start the bridge with `--robot x2_omnihand --ik-backend lerobot
--authoritative-executor`. For hardware, select the new executor preset and the
existing local hand serial configuration. The bridge and RoboJuDo must both use
the updated submodule and matching model assets.

Collision checking remains explicit. The new preset defaults to disabled, matching
the bridge CLI default. To enable it, set the OmniHand controller's
`executor_collision_checking=True` and start the bridge with `--collision-checking`.
The robot's `executor_collision_margin` and `executor_collision_trigger` must match
the bridge settings (defaults 0.02 and 0.08 m). Enabled initialization/query failures
block motion; they never fall back to disabled checks. Model identity and collision
settings must match even when checking is disabled.

Schema 3 uses `upper_body_state` feedback and `upper_body_desired` commands.
Command `positions` contains the complete **desired** 14-arm/24-finger target;
it is not an executable or collision-certified transition. Commands include:

- Existing bridge/frame, robot session/generation, feedback state, command ID,
  intent, status, and hand-mode fields. Command ID is diagnostic, rather than an
  exact-baseline admission requirement.
- `executor`: `contract=robojudo_executor_v1`, model fingerprint, collision enablement,
  clearance margin, and trigger. The fingerprint covers portable URDF/mesh content,
  collision pairs, joint ordering, limits, velocities, and passive coupling.
- `velocity_limits`: positive `arm`, `return`, and `hand` bridge settings. RoboJuDo
  intersects applicable caps with its own configuration and model limits.
- `active_joint_names`: joints authorized to move by this sample. Inactive joints
  hold the robot's local completed baseline, preventing delayed feedback from
  moving an arm after clutch release. HOME always holds fingers.

RoboJuDo keeps one newest waiting desired frame and one combined execution in
flight. At a control opportunity it consumes the desired frame once, filters from
the current completed baseline, constructs bounded float32 targets, and validates
the actual transition. Arm velocity budgets use one configured control period;
pauses do not accumulate extra movement allowance. Robot-side `ema_alpha` remains
0.0 by default and can be configured for this locally validated schema. Direct tracking fingers bypass
software velocity ramps and retain device-managed motion.

Collision validation shares the bridge's transition implementation: measured and
commanded starting configurations, nonlinear passive coupling, independent arm and
hand application paths, recovery criteria, and sampling limits. Rejected candidates
hold; this executor does not search shorter steps. Session, generation, ownership,
intent, measurement freshness, and robot-owned deadlines still govern submission.
An expired or consumed sample cannot authorize repeated steps.

Completion still requires successful arm submission and both hand SDK writes.
Only combined completion advances the coherent command baseline. Partial failures
retain the existing generation invalidation and recovery behavior. A waiting desired
frame can survive completion of its predecessor while its measurement lease remains
valid. Feedback asynchronously reports actual issued positions and completion identity;
it does not gate bridge target construction. LeRobot commits successful desired IK
history independently; its internal filter is separate from final executable filtering.

Validation commands (no physical hardware):

```bash
python -m unittest discover -s tests -p test_xr_target_executor.py
XROBOTOOLKIT_TEST_PYTHON=/path/to/robojudo-xr/bin/python \
  XROBOTOOLKIT_TEST_BACKEND=lerobot \
  python -m unittest discover -s tests -p test_xrobotoolkit_integration.py -k Executor
```

For a Jetson A/B comparison retain identical collision settings, workload, SDK
settings and freshness thresholds. Existing admission, worker, first-acknowledgement
and completion timings remain available; `executor_validation` reports the added
numeric validation duration. Compare completion rate, command intervals, queue age,
and feedback turnaround. The executor removes completion-baseline invalidation;
control servicing and collision cost can still limit throughput. Simulation tests
do not establish a hardware latency improvement or 50 completed frames/s. Recording
behavior is outside this executor experiment's validation scope.

A [local A/B benchmark](xr_executor_local_latency.md) found higher completion
throughput and steadier intervals, but no lower per-frame latency. It includes
measured results, limitations, and a reproducible simulation-only benchmark command.

For the remaining acknowledgement turnaround, see the
[bounded feedback-latency trace](xr_feedback_latency_trace.md). It separates
first-packet receipt, diagnostic observation and actual hand-write completion,
and documents the local stopping result and Jetson capture procedure.
