<div align="center">
<h1>RoboJuDo-Plus 🤖</h1>

<em>Deployment infrastructure for Unitree G1 and AgiBot X2 humanoid robots.</em>

<p>
  <img src="https://img.shields.io/badge/platform-Ubuntu%20%7C%20macOS%20%7C%20Windows-green" alt="platform"/>
  <img src="https://img.shields.io/badge/robot-Unitree%20G1%20%7C%20AgiBot%20X2-orange" alt="supported robots"/>
  <a href="https://creativecommons.org/licenses/by-nc/4.0/">
    <img src="https://img.shields.io/badge/License-CC--BY--NC--4.0-lightgrey.svg" alt="license"/>
  </a>
</p>

</div>

RoboJuDo-Plus is a modular Python-first framework for deploying learned policies in simulation and on humanoid robots. It
separates the controller, environment, policy, and pipeline so that a policy can be adapted to a different robot or
runtime with minimal configuration changes.

> This guide updates the original project documentation for the codebase starting at `b43869b`. The original README is
> preserved in [`README.original.md`](README.original.md) for attribution and historical reference.

## Contents

- [What changed after `b43869b`](#what-changed-after-b43869b)
- [TODO](#todo)
- [Architecture](#architecture)
- [Setup](#setup)
- [Quick start: simulation](#quick-start-simulation)
- [Unitree G1 deployment](#unitree-g1-deployment)
- [AgiBot X2 deployment](#agibot-x2-deployment)
- [Loco-mimic pipeline](#loco-mimic-pipeline)
- [ZMQ Control](#zmq-control)
- [Safety and troubleshooting](#safety-and-troubleshooting)
- [Development and documentation](#development-and-documentation)

## What changed after `b43869b`

The post-`b43869b` codebase adds or expands:

- AgiBot X2 simulation and AimDK/ROS 2 sim-to-real deployment.
- G1 and X2 locomanipulation pipelines, including partial-joint policy mapping.
- Loco-mimic pipelines with interpolated locomotion/mimic transitions.
- Four-mode activation and safety state machines for the locomanipulation deployments.
- AMP fall-recovery policies for G1 and X2.
- Optional named upper-body control over ZMQ.
- X2 BeyondMimic policies and G1/X2 cross-robot BeyondMimic support.
- G1 odometry and state-estimator integration updates.
- Operator-triggered AMP recovery support.

## TODO

- [ ] Confirm the real X2 odometry source. On PC1, `/aima/mc/leg_odometry` stops publishing after the native motion-control module is stopped with `aima em stop-app mc`; find a robust odometry solution.
- [ ] Validate IMU quaternions in real-robot environments, retain the last valid sample through a short invalid-packet grace window, and force damping only after persistent invalid or stale state; keep pipeline finite checks as a final safety fallback.
- [x] Add ZMQ velocity control under [ZMQ Control](#zmq-control).

## Architecture

RoboJuDo-Plus composes four runtime pieces:

- **Controller** reads a joystick, keyboard, motion stream, or external command source and produces `ctrl_data`.
- **Environment** reads sensors and applies policy targets in MuJoCo or on a real robot.
- **Policy** converts environment and controller data into actions or PD targets.
- **Pipeline** coordinates policies, interpolation, safety checks, and environment commands.

Configuration classes are registered by name, so most runs use the same entry point:

```bash
python scripts/run_pipeline.py -c <config-name>
```

The symbols `🖥️` and `🤖` used in older documentation mean simulation and real-robot support respectively. A real-robot
configuration is never a substitute for validating a policy in simulation first.

## Setup

### 1. Create an environment

RoboJuDo-Plus requires Python 3.11 or newer.

```bash
git clone https://github.com/chifongip/RoboJuDo-Plus.git
cd RoboJuDo-Plus

conda create -n robojudo-plus python=3.11 -y
conda activate robojudo-plus

pip install -e .
```

For development tools, use:

```bash
pip install -e ".[dev]"
```

If a CPU-only PyTorch installation is appropriate for your machine, install it before RoboJuDo-Plus:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

### 2. Install optional modules

Edit [`submodule_cfg.yaml`](submodule_cfg.yaml) to select optional modules, then run:

```bash
python submodule_install.py
```

The default installation enables the MuJoCo viewer. Install individual modules when needed:

```bash
# Unitree G1 real-robot backend; install the official unitree_sdk2 first.
python submodule_install.py unitree_cpp

# AgiBot X2 AimDK backend; source ROS 2 Humble first.
python submodule_install.py aimdk

# ROS 2 Xbox/PS5 joystick controller; source ROS 2 Humble first.
python submodule_install.py ros2_joy_cpp

# XR/dexterous-hand teleoperation support.
python submodule_install.py dex_teleop

# OmniHand Python SDK; selects the active CPython ABI and x86_64/aarch64 wheel.
python submodule_install.py omnihand_sdk

# CASIA Hand-M SDK; builds its bundled native extension for the active Linux architecture.
python submodule_install.py casiahand_sdk

# PHC motion controller support.
python submodule_install.py phc
```

The optional installers may require an active Conda environment and external SDKs. See:

- [Unitree setup](docs/unitree_setup.md)
- [X2 sim-to-real setup](docs/x2_sim2real.md)
- [policy documentation](docs/policy.md)

### Update submodules after pulling

After pulling RoboJuDo-Plus changes, synchronize every tracked submodule to the revisions recorded by the pulled
commit:

```bash
git pull
git submodule update --init --recursive
```

The submodule command updates source checkouts only; it does not rebuild or reinstall optional packages. Rerun the
installer for each optional module whose source changed or whose native artifacts are needed on the current machine:

```bash
python submodule_install.py <module>
```

For example, use `python submodule_install.py aimdk` after an AimDK update. The installer preserves an existing
submodule worktree, so always run `git submodule update --init --recursive` first when the pulled commit changes a
submodule revision. Use `--clean` only when you intentionally want to discard local changes in a module.

### 3. Verify the base installation

The base package does not require every optional SDK or submodule. Verify the core installation first:

```bash
python -c "import robojudo; print('RoboJuDo-Plus available')"
```

After installing the optional modules needed by your selected configurations, run the test suite before connecting a
robot. The suite checks the modules available in the current environment; it does not install missing dependencies:

```bash
python -m unittest discover -s tests
```

For strict X2 native-backend validation after AimDK installation:

```bash
ROBOJUDO_REQUIRE_AIMDK=1 python -m unittest discover -s tests
```

## Quick start: simulation

Start with the default Unitree G1 MuJoCo pipeline. An Xbox-compatible joystick is expected by the default controller:

```bash
python scripts/run_pipeline.py
```

Common G1 simulation examples:

```bash
# BeyondMimic
python scripts/run_pipeline.py -c g1_beyondmimic

# G1 locomanipulation
python scripts/run_pipeline.py -c g1_23_locomanipulation_stiff

# G1 locomanipulation loco-mimic with recovery
python scripts/run_pipeline.py -c g1_23_locomanipulation_locomimic
```

Common X2 simulation examples:

```bash
# X2 deployment policy
python scripts/run_pipeline.py -c x2

# X2 locomanipulation
python scripts/run_pipeline.py -c x2_locomanipulation

# X2 locomanipulation loco-mimic with recovery
python scripts/run_pipeline.py -c x2_locomimic

# X2 locomanipulation loco-mimic with BeyondMimic policies
python scripts/run_pipeline.py -c x2_locomimic_beyondmimic
```

For the standard joystick, the left stick controls planar movement and the right stick controls yaw. Policy-specific
command mappings are listed in the sections below and in [`docs/controller.md`](docs/controller.md).

## Unitree G1 deployment

RoboJuDo-Plus supports two Unitree environments:

- `UnitreeEnv`, based on the official `unitree_sdk2py` package and suitable for G1/H1 deployments.
- `UnitreeCppEnv`, based on the `unitree_cpp` binding and preferred for the resource-constrained G1 onboard computer.

Follow [`docs/unitree_setup.md`](docs/unitree_setup.md) for official SDK installation, onboard versus workstation
deployment, and network setup.

### Install the G1 backend

Install the official Unitree C++ SDK first, then install the RoboJuDo-Plus binding:

```bash
python submodule_install.py unitree_cpp
python -c "from robojudo.environment import UnitreeCppEnv; print('UnitreeCppEnv available')"
```

If using the Python SDK instead, install `unitree_sdk2py` according to its official instructions and verify:

```bash
python -c "from robojudo.environment import UnitreeEnv; print('UnitreeEnv available')"
```

### Configure the network interface

Edit [`robojudo/config/g1/g1_cfg.py`](robojudo/config/g1/g1_cfg.py), or derive a local configuration from `g1_real`.
Set the interface connected to the robot, commonly `eth0` on the onboard computer:

```python
env: G1RealEnvCfg = G1RealEnvCfg(
    env_type="UnitreeCppEnv",
    unitree=G1UnitreeCfg(
        net_if="eth0",
    ),
)
```

Use `env_type="UnitreeEnv"` only when the Python SDK path is intended. Confirm the interface and robot startup state
using the [official Unitree deployment guide](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/deploy/deploy_real/README.md).

### Run a G1 policy

Place the robot in a safe supported position, keep the emergency stop available, and start with a policy already
validated in simulation:

```bash
python scripts/run_pipeline.py -c g1_real
```

Other G1 real-robot configurations include:

```bash
# BeyondMimic
python scripts/run_pipeline.py -c g1_23_beyondmimic_real

# Locomanipulation
python scripts/run_pipeline.py -c g1_23_locomanipulation_stiff_real
python scripts/run_pipeline.py -c g1_29_locomanipulation_stiff_real

# Loco-mimic
python scripts/run_pipeline.py -c g1_23_locomanipulation_locomimic_real
python scripts/run_pipeline.py -c g1_29_locomanipulation_locomimic_real
```

The `g1_23_*` configurations use a logical 23-DOF policy while retaining the standard Unitree motor transport. The
`g1_29_*` configurations use the native 29-DOF policy layout.

## AgiBot X2 deployment

X2 real-robot environments use the AimDK native backend and ROS 2. The real configuration maps the 29 policy joints to
the 31-joint robot interface by name; the two head joints remain at their configured default positions.

### Install AimDK

The managed installer expects ROS 2 Humble to be available and an active Conda environment:

```bash
source /opt/ros/humble/setup.bash
conda activate robojudo-plus
python submodule_install.py aimdk
source third_party/aimdk/install/setup.bash
```

Source the generated AimDK setup file in every shell used for real X2 deployment. The installer initializes missing
AimDK and `aimdk_cpp` submodules, preserves existing worktrees, builds the required message package, and installs the
native Python extension.

Verify the native backend explicitly:

```bash
ROBOJUDO_REQUIRE_AIMDK=1 python -m unittest discover -s tests
```

Before enabling commands, confirm that the four joint-state topics and `/aima/hal/imu/torso/state` are updating.
The standard X2 real presets use `odometry_type="NONE"`; the BeyondMimic loco-mimic preset uses `"DUMMY"`. Neither
requires an odometry topic for activation; stale joint or IMU data still prevents activation and triggers damping.
Configurations that enable `AIMDK` odometry require
`/aima/mc/leg_odometry`; the SuperOdom loco-mimic preset requires `/laser_odometry`.

For the complete topic, frame, timeout, and preflight description, see [`docs/x2_sim2real.md`](docs/x2_sim2real.md).

### ROS 2 DDS latency workaround

The X2 robot's ROS 2 DDS configuration can cause data latency. When deploying on X2, refer to
[ros2-domain-bridge](https://github.com/chifongip/ros2-domain-bridge.git) for a temporary workaround that relays the
affected data and mitigates the latency issue.

### Run an X2 policy

Run the real configuration only with the robot supported and an operator holding the emergency stop:

```bash
# Standard X2 policy
python scripts/run_pipeline.py -c x2_real

# Locomanipulation
python scripts/run_pipeline.py -c x2_locomanipulation_real

# Loco-mimic with the supplied X2 deployment policy
python scripts/run_pipeline.py -c x2_locomimic_real

# Loco-mimic with X2 BeyondMimic policies
python scripts/run_pipeline.py -c x2_locomimic_beyondmimic_real
```

The X2 policy loop runs at 50 Hz. The native backend republishes the active mode at 500 Hz and latches damping when
position commands stop arriving for the configured timeout.

## Loco-mimic pipeline

Loco-mimic combines a locomotion policy with one or more motion-mimic policies. Policy changes are interpolated so that
the robot can move between policy targets instead of switching abruptly. The feature is available in general G1
loco-mimic presets and in the four-mode G1/X2 locomanipulation loco-mimic presets.

### Four operating modes

The four-mode locomanipulation pipelines start in `PASSIVE_DEFAULT` and use this activation sequence:

| Mode | Purpose | Typical simulation joystick | Typical simulation keyboard |
| --- | --- | --- | --- |
| `PASSIVE_DEFAULT` | Zero stiffness and damping; use only while supported. | `A` | `K` |
| `DAMPING_DEFAULT` | Zero stiffness with configured damping, normally `5.0`. | `B` | `L` |
| `JOINT_DEFAULT` | Interpolate every environment joint to the configured default pose. | `Y` | `I` |
| `RL_DEFAULT` | Arm position control and enable policy inference. | `X` | `J` |

`RL_DEFAULT` is rejected until `JOINT_DEFAULT` interpolation completes. Returning to passive or damping invalidates the
preparation step, so run `JOINT_DEFAULT` again before restarting RL. Policy inference and interpolation remain stopped
outside the active RL/recovery modes.

### Three policy controls

These are the three policy-selection controls exposed by recovery-enabled loco-mimic presets:

| Command | Meaning | Simulation joystick | Simulation keyboard | G1/X2 real controller |
| --- | --- | --- | --- | --- |
| `[POLICY_LOCO]` | Select locomotion and interpolate back to it. | `Back` | `]` | `Select` or `Back` |
| `[POLICY_MIMIC]` | Select the active mimic policy. | `Start` | `[` | `Start` |
| `[POLICY_RECOVERY]` | Activate AMP recovery after a fall. | right-stick click `R` | `R` / `r` | right-stick trigger `R2` / `R` |

Mimic-policy selection is separate from these three controls:

- Simulation: `RB` or `;` selects the next mimic policy; `LB` or `'` selects the previous one.
- G1/X2 real controller: `R1`/`RB` selects next; `L1`/`LB` selects previous.

The next/previous selection is accepted while the locomotion policy is active and interpolation is idle. Entering
mimic, recovery, passive, damping, or a shutdown path disables incompatible auxiliary controls.

### Recovery flow

Recovery is available in the recovery-enabled G1/X2 locomanipulation loco-mimic configurations. After a fall:

1. Select `JOINT_DEFAULT` and wait for interpolation to finish.
2. Request `[POLICY_RECOVERY]` while the robot tilt exceeds the recovery threshold.
3. After the recovery policy raises the robot and tilt is below the threshold, select `[POLICY_LOCO]` to return through
   the locomotion policy.

Recovery cannot be entered while the robot is upright, before joint-default interpolation completes, or from an
unsupported mode. Passive, damping, and shutdown remain immediate safety exits.

### Example configurations

G1:

```bash
python scripts/run_pipeline.py -c g1_23_locomanipulation_locomimic
python scripts/run_pipeline.py -c g1_23_locomanipulation_default_locomimic
python scripts/run_pipeline.py -c g1_29_locomanipulation_locomimic
```

The broader G1 policy families are also available through:

```bash
python scripts/run_pipeline.py -c g1_locomimic_beyondmimic
python scripts/run_pipeline.py -c g1_locomimic_asap
```

X2:

```bash
python scripts/run_pipeline.py -c x2_locomimic
python scripts/run_pipeline.py -c x2_locomimic_beyondmimic
```

The X2 locomanipulation loco-mimic simulation includes an elastic band attached to the torso. Keyboard `7`, `8`, and
`9` lower, lift, and toggle the band. The band is simulation-only.

## ZMQ Control

### Upper Body Control

The current [`UpperBodyZmqCtrl`](robojudo/controller/upper_body_zmq_ctrl.py) interface receives named upper-body joint
positions in radians over `tcp://127.0.0.1:8559`:

```json
{
  "positions": {
    "left_shoulder_pitch_joint": 0.35,
    "right_elbow_joint": -0.87
  }
}
```

Messages may contain any non-empty subset of configured joints. Partial updates are merged, invalid messages are
rejected, targets are clamped and EMA-filtered, and stale streams return smoothly to the configured defaults after
`0.25` seconds. Control starts disabled and is available only when the active policy does not own those joints.

In locomanipulation loco-mimic simulation presets, toggle it with joystick `L` or keyboard `T`; other bindings depend on
the selected robot configuration. Test the interface with:

```bash
python scripts/test_upper_body_zmq.py
python scripts/test_upper_body_zmq.py --robot g1-23  # or g1-29 / x2
```

X2 accepts its configured arm joints; G1 locomanipulation presets expose the matching upper-body subset for their 23-DOF
or 29-DOF layout. See [`docs/x2_sim2real.md`](docs/x2_sim2real.md) and [`docs/policy.md`](docs/policy.md) for details.

### Velocity Control

`VelocityZmqCtrl` receives body-frame velocity commands using the JSON shape and SI units of ROS
`geometry_msgs/Twist`, without requiring ROS:

```json
{
  "linear": {"x": 0.5, "y": 0.0, "z": 0.0},
  "angular": {"x": 0.0, "y": 0.0, "z": 0.3}
}
```

The controller subscribes to `tcp://127.0.0.1:8558` by default. Positive `x` is forward, positive `y` is left, and
positive angular `z` is counter-clockwise yaw. Locomotion policies use those three planar components, clamp them to
their trained ranges, and ignore the remaining validated Twist components.

Messages older than `0.25` seconds are stale. When multiple velocity sources are configured, each must have a unique
`velocity_priority`; larger values win. The opt-in `g1_zmq` and `g1_real_zmq` configurations assign joystick or
Unitree priority `300` and ZMQ priority `100`, so manual stick activity takes control immediately. Centering the
stick holds an authoritative zero command for the configured `0.5` second lease, after which a fresh ZMQ command
resumes. Continuous neutral stick samples do not
renew the lease. Buttons and safety triggers remain active regardless of which source currently owns velocity.
A stale stream with no active fallback produces zero planar velocity.

Test in MuJoCo before using a real robot:

```bash
python scripts/run_pipeline.py -c g1_zmq
# In another terminal:
python scripts/test_velocity_zmq.py
```

For a custom configuration, assign every velocity source a unique priority; controller-list order is irrelevant. For
example, use `JoystickCtrlCfg(velocity_priority=300)` with
`VelocityZmqCtrlCfg(velocity_priority=100, endpoint="tcp://127.0.0.1:9000", timeout_s=0.5)`. The interface is supported
by Unitree, Unitree-without-gait, Smooth, AMO, ASAP Loco, and G1/X2 Locomanipulation policies; motion mimic, tracking,
recovery, H2H, X2 deploy, and TWIST motion-stream policies do not consume velocity commands.

## Safety and troubleshooting

Real-robot deployment can cause violent motion, falls, hardware damage, or injury. The policies and assets are provided
for research. Use appropriate supervision, physical support, an accessible emergency stop, and a safe test area.

Before enabling a real policy:

- Validate the exact policy and robot model in simulation.
- Put the robot in a supported and mechanically safe position.
- Confirm the emergency damping command works before attempting unsupported motion.
- Verify the configured network interface, robot SDK, policy assets, and model joint mapping.
- For X2, confirm joint, IMU, and leg-odometry topics are fresh before enabling commands.
- Stop if the logs report stale state, non-finite targets, repeated target clamping, excessive tilt, or control-loop frame
  drops.

Useful checks:

```bash
# Confirm registered modules import.
python -m unittest discover -s tests

# Require the X2 native backend rather than allowing the tests to skip it.
ROBOJUDO_REQUIRE_AIMDK=1 python -m unittest discover -s tests

# Inspect available publisher options.
python scripts/test_upper_body_zmq.py --help
```

For Unitree network and startup troubleshooting, see [`docs/unitree_setup.md`](docs/unitree_setup.md). For X2 topics,
odometry frames, command timeouts, and preflight checks, see [`docs/x2_sim2real.md`](docs/x2_sim2real.md).

## Development and documentation

The main package is organized into:

```text
robojudo/
├── controller/    # joystick, keyboard, Unitree, motion, and external controls
├── environment/   # MuJoCo, Unitree, AimDK, and shared environment interfaces
├── policy/        # policy wrappers and robot-specific policies
├── pipeline/      # policy execution, switching, interpolation, and safety state machines
├── config/        # registered robot, environment, policy, and pipeline configurations
└── tools/         # shared adapters and utilities
```

Read the component guides before adding a module:

- [`docs/controller.md`](docs/controller.md)
- [`docs/environment.md`](docs/environment.md)
- [`docs/policy.md`](docs/policy.md)
- [`docs/system_schematic.md`](docs/system_schematic.md)

To add a policy or controller, create its configuration, register it with the relevant registry, and add an import-only
test under `tests/`. Keep real-robot configuration changes explicit and avoid committing credentials, private model
artifacts, or machine-specific network settings.

## Citation and related projects

If you use this work, please cite the project and the original author’s work as described in the
[original README](README.original.md).

External projects used by RoboJuDo-Plus:

- [Unitree SDK2 Python](https://github.com/unitreerobotics/unitree_sdk2_python)
- [Unitree SDK2](https://github.com/unitreerobotics/unitree_sdk2)
- [UnitreeCpp](https://github.com/chifongip/unitree_cpp)
- [AimDK](https://github.com/chifongip/aimdk)
- [AimDK C++/Python binding](https://github.com/chifongip/aimdk_cpp)
- [ros2_joy_cpp](https://github.com/chifongip/ros2_joy_cpp)
- [ZED Proxy](https://github.com/HansZ8/ZED-Proxy)
- [DEX Teleop](https://github.com/wrfbreeze/dex_teleop)
- [MuJoCo Python Viewer](https://github.com/rohanpsingh/mujoco-python-viewer)
- [PHC](https://github.com/ZhengyiLuo/PHC)
- [ProtoMotions](https://github.com/NVlabs/ProtoMotions)

Bundled packages and third-party components:

- [`packages/aimdk_cpp`](packages/aimdk_cpp) — RoboJuDo’s X2 AimDK native binding; see its [build configuration](packages/aimdk_cpp/pyproject.toml).
- [`packages/ros2_joy_cpp`](packages/ros2_joy_cpp) — native ROS 2 Joy subscriber package.
- [`packages/unitree_cpp`](packages/unitree_cpp) — Unitree G1 native binding; see its [package README](packages/unitree_cpp/README.md).
- [`packages/zed_proxy`](packages/zed_proxy) — optional ZED camera odometry submodule.
- [`third_party/aimdk`](third_party/aimdk) — pinned AimDK SDK used by X2 deployment.
- [`third_party/dex_teleop`](third_party/dex_teleop) — optional XR arm and dexterous-hand teleoperation module; see its [README](third_party/dex_teleop/README.md).
- [`third_party/mujoco_viewer`](third_party/mujoco_viewer) — optional MuJoCo visualization module; see its [README](third_party/mujoco_viewer/README.md).
- [`third_party/phc`](third_party/phc) — optional PHC motion controller submodule.
- [`third_party/patches`](third_party/patches) — RoboJuDo patches and robot-specific add-ons for optional dependencies.
