# XRoboToolkit X2 arm teleoperation

The optional `third_party/xrobotoolkit_teleop` submodule runs XR input, arm inverse kinematics, and self-collision
checks in a separate process. RoboJuDo's existing X2 locomanipulation pipeline owns walking, runtime modes, recording,
and robot hardware. The bridge observes all 31 X2 joints and commands only the 14 arm joints. OmniHand control and
loco-mimic integration are deferred.

```text
Headset/controllers -> PC Service -> XR bridge (XR environment)
                                        | checked arm targets (8559)
                                        v
Joystick/keyboard -> RoboJuDo locomanipulation -> MuJoCo OR X2/AimDK
                                        |
                                        +-> measured joints, command baseline,
                                            limits and control intent (8571) -> bridge
```

## Environment and installation

The RoboJuDo installer supports Python 3.11+ on Linux x86_64/aarch64. A dedicated environment is recommended for
dependency isolation, but an existing compatible environment also works. With Conda installed, run these commands
from the RoboJuDo-Plus repository root:

```bash
conda create -n robojudo-xr python=3.11 pip -y
conda activate robojudo-xr

# The optional-module installer reads submodule_cfg.yaml with PyYAML.
python -m pip install pyyaml
python submodule_install.py xrobotoolkit_teleop
```

To reuse an environment, activate it and run the final two commands, skipping environment creation. The installer
preserves the submodule worktree and installs its `x2` extra in editable mode using the invoking Python interpreter.
It neither creates nor switches environments and does not run upstream setup scripts or install the full RoboJuDo
runtime. When returning in a new terminal, activate `robojudo-xr` before using the sample.

The `x2` extra installs NumPy, Meshcat, Placo, XRoboToolkit SDK, MuJoCo, Tyro, and PyZMQ, plus their transitive dependencies.
It supports the arm-only and simulated OmniHand Pro modes. It excludes other robot SDKs, Torch, camera packages,
and hand-retargeting packages. The upstream `[all]` extra remains available for other robot examples.

## Verify installation

Installation checks dependency and controller imports without initializing XR or opening a viewer. You can repeat
the checks without a headset or PC Service:

```bash
python -m pip check
python -c "from xrobotoolkit_teleop.simulation.agibot_x2_teleop_controller import AgibotX2TeleopController; print('OK')"
python -m xrobotoolkit_teleop.integrations.robojudo_x2 --help
```

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
python -m xrobotoolkit_teleop.integrations.robojudo_x2
```

Complete the usual preparation sequence: simulation keyboard `i` / joystick `Y` selects `JOINT_DEFAULT`; after
interpolation completes, `j` / `X` selects `RL_DEFAULT`. Toggle arm tracking with `t` / `Start`. Release both XR grips
before engaging; hold the corresponding grip to move an arm. Release a grip to hold that arm's target. XR `Y` held
returns both arms toward their defaults. Walking and runtime/recording commands remain on the existing controls.

Hand targets are relative to the measured torso, so walking, turning, and waist motion carry them with the robot.
The headset's horizontal heading maps controller motion to torso axes at engagement; subsequent head motion does
not move the targets. No robot odometry or independent physics simulation is used by the bridge.

For real deployment, follow the [X2 AimDK setup](x2_sim2real.md), then use `-c x2_xrobotoolkit_real` in the robot
terminal. The bridge command is identical. Real-robot acceptance is still required; automated validation uses
synthetic XR input and MuJoCo, not a physical X2.

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
| Collision margin / activation distance | 0.02 / 0.08 m |
| XR tracking timeout | 0.25 s |
| Robot measurement timeout | 0.1 s |
| Grip activation | greater than 0.9 |

The standalone demo and retargeter share upstream defaults. Per-joint URDF velocity limits and the intersection of
URDF/deployment position limits also apply. The bridge accepts explicit overrides such as `--arm-velocity-limit 2.0`;
its effective caps cannot exceed those advertised by RoboJuDo's controller configuration. The new presets disable
additional EMA smoothing and reject invalid proposals instead of modifying a collision-checked target.

- Tracking loss, stale robot feedback, or bridge loss holds the last commanded arm targets. Brief interruptions
  automatically rebase controller references on fresh data, discarding motion made during the hold. Startup,
  explicit mode/intent changes, and a restarted bridge still require grip release before engagement.
- Explicit teleop disable requests a collision-checked return to defaults. If the bridge is offline, the request
  remains pending and the arms hold. After reconnection the return uses fresh measurements and does not require XR.
- Re-enabling tracking cancels the pending return and requires grip engagement. Whole-robot mode changes and AimDK
  safety states take priority over arm tracking/return requests.
- Recording accepts fresh tracking proposals, excluding homing and outage holds.

IK starts from acknowledged arm targets with the measured body pose, so servo lag in an ungripped arm cannot conflict
with its command speed bounds. Its posture objective uses the fixed configured default, rather than the previous
command, to discourage redundant-joint drift and preserve the preferred elbow branch through repeated poses.
Collision checks cover sampled paths from measured and previous commanded positions.
Rejected full steps try a bounded set of shorter, independently checked steps before holding.
They do not predict future locomotion motion or external obstacles. The XR SDK exposes a shared stream timestamp;
it cannot distinguish one device's cached pose from a live stationary pose while other devices keep streaming.

## Transport and configuration

The existing `UpperBodyZmqCtrlCfg` and pipeline are reused. The presets opt in with `feedback_endpoint` and
`ema_alpha=0`. All upper-body presets hold the entire configured group on timeout and return to defaults on explicit
disable. The JSON `positions` envelope remains unchanged, but publishers must now send all configured joints in each
frame; partial updates are rejected. XR disable returns through the bridge's checked HOME path, pending bridge recovery
if it is offline.
Feedback mode currently supports arm-only control, not `UpperBodyOmniHandZmqCtrlCfg`.

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
simulates hands independently of the deferred RoboJuDo hand integration.

## Troubleshooting

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
- If Python 3.10 is selected, activate the Python 3.11+ environment and check `python --version` before rerunning.
- If the installer reports a missing `x2` extra, run `git submodule update --init third_party/xrobotoolkit_teleop`
  to select the pinned revision, then retry installation.
- If `pip check` reports unrelated ROS packages in an isolated environment, use a fresh terminal without a sourced
  ROS workspace; a ROS `PYTHONPATH` can expose packages outside the active environment.
- If real-robot feedback remains invalid, check all joint streams and rebuild `aimdk_cpp` if its robot snapshot lacks
  `joint_received_time_sec`. Missing or stale measurements must not be replaced by publication timestamps.
- Missing native build or graphics libraries must be installed separately if pip or the import check reports them.
  Upstream setup is tested on Ubuntu 22.04; live simulation requires a working graphical desktop.
