# Policy

**Policy** is the component that controls the robot. It receives the `env_data` from the environment, `ctrl_data` from the controller, organize the observation and infer the action fo robot.

## [Policy](#policy)

`Policy` is the base class for all policies. It defines the interface for the policy, as in [base_policy.py](../robojudo/policy/base_policy.py)

---

We provide the following policies:
- [UnitreePolicy](#policy--unitreepolicy)
- [AMOPolicy](#policy--amopolicy)
- [H2HStudentPolicy](#policy--h2hstudentpolicy)
- [HugWBCPolicy](#policy--hugwbcpolicy)
- [AmpRecoveryPolicy](#policy--amprecoverypolicy)
- [BeyondMimicPolicy](#policy--beyondmimicpolicy)
- [ASAPPolicy](#policy--asappolicy)
- [KungfuBotGeneralPolicy](#policy--kungfubotgeneralpolicy)
- [TwistPolicy](#policy--twistpolicy)
- [LocomanipulationPolicy](#policy--locomanipulationpolicy)
- [ProtoMotionsTrackerPolicy](#policy--protomotionstrackerpolicy)

## [Policy](#policy) > [UnitreePolicy](#policy--unitreepolicy)

`UnitreePolicy` is the policy that controls the robot using the [Unitree official policy](https://github.com/unitreerobotics/unitree_rl_gym).

script: [unitree_policy.py](../robojudo/policy/unitree_policy.py)

To control the robot using `UnitreePolicy`, you can refer `_get_commands()`:

`commands`:
- `commands[0]`, [-1, 1], control the robot to walk forward and backward
- `commands[1]`, [-1, 1], control the robot to walk left and right
- `commands[2]`, [-1, 1], control the robot to turn left and right

for instance, use `JoystickCtrl` to control:

```python
def _get_commands(self, ctrl_data: dict) -> list[float]:
    commands = np.zeros(3)
    for key in ctrl_data.keys():
        if key in ["JoystickCtrl", "UnitreeCtrl"]:
            axes = ctrl_data[key]["axes"]
            lx, ly, rx, ry = axes["LeftX"], axes["LeftY"], axes["RightX"], axes["RightY"]

            commands[0] = command_remap(ly, self.commands_map[0])
            commands[1] = command_remap(lx, self.commands_map[1])
            commands[2] = command_remap(rx, self.commands_map[2])
            break
    return commands
```

### [UnitreeWoGaitPolicy](#policy--unitreewogaitpolicy)

For Unitree G1, we also provide `UnitreeWoGaitPolicy`, which supports the new `Unitree-G1-29dof-Velocity` conig from [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab).

The difference is that `UnitreeWoGaitPolicy` does not include gait in the observation, so the robot will not keep stepping when standing.

script: [unitree_policy.py](../robojudo/policy/unitree_policy.py)

## [Policy](#policy) > [AMOPolicy](#policy--amopolicy)

`AMOPolicy` is the policy that controls the robot using the [AMO](https://github.com/OpenTeleVision/AMO).

script: [amo_policy.py](../robojudo/policy/amo_policy.py)

To control the robot using `AMOPolicy`, you can refer `_get_commands()`:

`commands`:
- `commands[0]`, [-1, 1], control the robot to walk forward and backward
- `commands[1]`, [-1, 1], control the robot to turn left and right
- `commands[2]`, [-1, 1], control the robot to walk left and right
- `commands[3]`, [-0.5, 0.8], control the robot torso height
- `commands[4]`, [-1.57, 1.57], control the robot torso yaw
- `commands[5]`, [-0.52, 1.57], control the robot torso pitch
- `commands[6]`, [-0.7, 0.7], control the robot torso roll

You can apply your own controller to control the robot using `AMOPolicy`. Just set the `commands` in `_get_comands()`


## [Policy](#policy) > [H2HStudentPolicy](#policy--h2hstudentpolicy)

`H2HStudentPolicy` is the policy that controls the robot using the [human2humanoid](https://github.com/LeCAR-Lab/human2humanoid).

> PHC Submodule is needed for motionlib control. check [README](../README.md#setup).

script: [h2hstudent_policy.py](../robojudo/policy/h2hstudent_policy.py)

`H2HStudentPolicy` is controlled by `MotionH2HCtrl`. check code [motion_h2h_ctrl.py](../robojudo/controller/motion_h2h_ctrl.py).

For motion source:
- `Unitree H1`: Simply use the motion retargeting pipeline from [human2humanoid](https://github.com/LeCAR-Lab/human2humanoid).
- `Unitree G1`: As not officially supported, we use the PHC pipeline. Our submodule patch enables 29dof G1. Check [unitree_g1_29dof_fitting.yaml](../third_party/phc/phc/data/cfg/robot/unitree_g1_29dof_fitting.yaml).

You can refer to `g1_h2h` config in [g1_cfg.py](../robojudo/config/g1/g1_cfg.py) for more details.


## [Policy](#policy) > [HugWBCPolicy](#policy--hugwbcpolicy)

`HugWBCPolicy` is the policy that controls the robot using the [HugWBC](https://github.com/apexrl/HugWBC).

script: [hugwbc_policy.py](../robojudo/policy/hugwbc_policy.py)

🥺Will release soon.


## [Policy](#policy) > [AmpRecoveryPolicy](#policy--amprecoverypolicy)

`AmpRecoveryPolicy` deploys the standalone fall-recovery actors trained by AMP_mjlab. The policy always receives a
zero twist command and uses four frames of term-major observation history, matching the original mjlab task. The
bundled models control the full trained joint set and use the default pose, PD gains, and per-joint action scales
recorded by each training run.

Three simulation presets are available:

```bash
python scripts/run_pipeline.py -c g1_amp_recovery
python scripts/run_pipeline.py -c g1_23_amp_recovery
python scripts/run_pipeline.py -c x2_amp_recovery
```

The G1 presets control 29 and 23 joints respectively. The X2 model controls 29 joints while its two head joints remain
at their environment defaults. All presets run at 50 Hz with a 5 ms simulation step and four simulation steps per
policy action. Use MuJoCo mouse perturbations to place the robot in front, back, or side fallen states when evaluating
recovery. Press the configured reset shortcut to respawn the simulation.

The bundled policies are the final `model_20000` exports from
`/home/ubuntu/AMP_mjlab/logs/fall_recovery`. Deployment uses the explicit observation layout, default poses, PD gains,
and action scales recorded in this repository, and requires no AMP_mjlab runtime installation. These presets are
simulation-only; they do not enable real-robot recovery or automatic switching from another locomotion policy.


## [Policy](#policy) > [BeyondMimicPolicy](#policy--beyondmimicpolicy)

`BeyondMimicPolicyBase` is the shared ONNX runtime for BeyondMimic-style motion tracking exports. Robot-specific
entry points are implemented by `G1BeyondMimicPolicy` and `X2BeyondMimicPolicy`, following the same split as the
locomanipulation policies. Joint order, default position, PD gains, action scale, and anchor body are loaded from ONNX
metadata. The fixed BeyondMimic observation layout is validated against the ONNX metadata before inference.

The bundled motion can be read directly from the ONNX model. G1 also supports an external `BeyondMimicCtrl` with an
NPZ motion by setting `use_motion_from_model=False`.

Model locations are selected by `policy_name`:

- G1 and G1-23DoF: `assets/models/g1/beyondmimic/<policy_name>.onnx`
- X2: `assets/models/x2/beyondmimic/<policy_name>.onnx`

[`BeyondMimicPolicyCfg`](../robojudo/policy/policy_cfgs.py) provides the shared options. Robot examples are in
[g1_beyondmimic_policy_cfg.py](../robojudo/config/g1/policy/g1_beyondmimic_policy_cfg.py) and
[x2_beyondmimic_policy_cfg.py](../robojudo/config/x2/policy/x2_beyondmimic_policy_cfg.py):

- `policy_name`: model filename without `.onnx`.
- `without_state_estimator`: must match `observation_names` in the ONNX metadata.
- `use_modelmeta_config`: use joint parameters embedded by the mjlab exporter.
- `use_motion_from_model`: use the reference motion embedded in the ONNX model.

Run the robot-specific presets after placing the exported models at the paths above:

```bash
python scripts/run_pipeline.py -c g1_beyondmimic
python scripts/run_pipeline.py -c g1_23_beyondmimic
python scripts/run_pipeline.py -c x2_beyondmimic
```

The X2 presets require a `No-State-Estimation` export. The policy controls 29 joints; the two head joints remain at
the X2 environment defaults. On real X2 hardware, FK supplies the `torso_quat` used for motion-anchor orientation;
the no-state model does not consume a base linear-velocity observation. Real presets are
`g1_23_beyondmimic_real` and `x2_beyondmimic_real`.

## [Policy](#policy) > [AsapPolicy](#policy--asappolicy)

`AsapPolicy` is the policy that controls the robot using the [ASAP](https://github.com/LeCAR-Lab/ASAP).

Also, [KungfuBot](https://github.com/TeleHuman/PBHC) is supported by `AsapPolicy`.

RoboJuDo support both `deepmimic` and `decoupled_locomotion` of the official repo, implemeted as `AsapPolicy` and `AsapLocoPolicy`.

We fully reproduced the original repository, including keyboard and joystick mapping:
- `i` to make the robot the initial position
- `o` to emergence stop the robot

for locomotion policy:
- `=` to switch between tapping and walking for the locomotion policy
- `w/a/s/d` to control the linear velocity
- `q/e` to control the angular velocity
- `z` to set all commands to zero

for policy switch:
- `[` to switch to MotionMimic
- `]` to switch to LocoMotion
- `;` toggle next mimic policy
- `'` toggle prev mimic policy

or with joystick:
- `Left` to switch between tapping and walking for the locomotion policy
- `Up/Down` to control the height
- `left axes` to control the linear velocity
- `right axes` to control the angular velocity
- `Select/Back` to switch to LocoMotion
- `Start` to switch to MotionMimic
- `R1/RB` to toggle next mimic policy
- `L1/LB` to toggle prev mimic policy

script: [asap_policy.py](../robojudo/policy/asap_policy.py)

> For your convenience, `CR7_level1` checkpoint is included, you can ran sim2sim with `g1_asap` config in [g1_asap_cfg.py](../robojudo/config/g1/g1_asap_cfg.py).

You can add more models to `assets/models/g1/asap/mimic`. Any model in the official repo [ASAP-sim2real](https://github.com/LeCAR-Lab/ASAP/tree/main/sim2real/models), [PBHC](https://github.com/TeleHuman/PBHC/tree/main/example) and [RoboMimic_Deploy](https://github.com/ccrpRepo/RoboMimic_Deploy) should work.

This example highlights the advantages of RoboJudo:
- Modular code & config with easy implementation and strong readability
- Flexible policy switching, with interpolation support.
- Convenient external controller processing

You can refer to `g1_asap` and `g1_asap_loco` config in [g1_asap_cfg.py](../robojudo/config/g1/g1_asap_cfg.py) for test and details.

## [Policy](#policy) > [KungfuBotGeneralPolicy](#policy--kungfubotgeneralpolicy)

`KungfuBotGeneralPolicy` is the policy that controls the robot using the [PBHC](https://github.com/TeleHuman/PBHC)-KungfuBot2.

To be noted, this is for **KungfuBot2** general model, for KungfuBot, please use [AsapPolicy](#policy--asappolicy).

> PHC Submodule is needed for motionlib control. check [README](../README.md#setup).

script: [kungfubot_policy.py](../robojudo/policy/kungfubot_policy.py)

- `KungfuBotGeneralPolicy` is controlled by `MotionKungfuBotCtrl`
    - check code [motion_kungfubot_ctrl.py](../robojudo/controller/motion_kungfubot_ctrl.py).
    - motions from PBHC pipeline are supported. Put your motion files in `assets/motions/g1/phc/kungfubot/`.

You can refer to `g1_kungfubot2` config in [g1_cfg.py](../robojudo/config/g1/g1_cfg.py) for test and details.

## [Policy](#policy) > [TwistPolicy](#policy--twistpolicy)

`TwistPolicy` is the policy that controls the robot using the [TWIST](https://github.com/YanjieZe/TWIST).

script: [twist_policy.py](../robojudo/policy/twist_policy.py)

For TwistPolicy, we implement two motion source controllers:

- `TwistRedisCtrl` at [twist_redis_ctrl.py](../robojudo/controller/twist_redis_ctrl.py): 
    - get motion from redis server, which is used in the original repo.
    - it works with the motion server like [server_high_level_motion_lib.py](https://github.com/YanjieZe/TWIST/blob/42d8c134739eee51f28d7cc0ff72a86728afb8dc/deploy_real/server_high_level_motion_lib.py)

- `MotionTwistCtrl` at [motion_twist_ctrl.py](../robojudo/controller/motion_twist_ctrl.py): 
    - get motion from local .pkl files. 
    - this is base on the PHC MotionLib, and **uses the same motion format as PHC**. 
        - PHC Submodule is needed. check [README](../README.md#setup).
    - motions from [PBHC](https://github.com/TeleHuman/PBHC) pipeline is supported,  put your motion files in `assets/motions/g1/phc/`.

You can refer to `g1_twist` config in [g1_cfg.py](../robojudo/config/g1/g1_cfg.py) for test and details.

## [Policy](#policy) > [LocomanipulationPolicy](#policy--locomanipulationpolicy)

`LocomanipulationPolicy` deploys locomotion policies whose learned actions control the lower body while an optional
named-joint ZMQ stream controls the remaining upper-body joints. G1 simulation presets are provided for the supplied
23-DOF and 29-DOF exports:

```bash
python scripts/run_pipeline.py -c g1_23_locomanipulation_default
python scripts/run_pipeline.py -c g1_23_locomanipulation_stiff
python scripts/run_pipeline.py -c g1_29_locomanipulation_stiff
```

Each preset starts in `PASSIVE_DEFAULT` and uses the same guarded four-mode sequence as X2:

| Mode | Joystick | Behavior |
| --- | --- | --- |
| `PASSIVE_DEFAULT` | `A` | Apply zero torque. Use only while the simulated robot is supported. |
| `DAMPING_DEFAULT` | `B` | Apply damping `5.0` to every joint. |
| `JOINT_DEFAULT` | `Y` | Interpolate all joints to the checkpoint's recorded pose over 1.5 seconds. |
| `RL_DEFAULT` | `X` | Restore the recorded policy gains, reset policy state, and run inference. |

`RL_DEFAULT` is rejected until `JOINT_DEFAULT` completes. Entering passive or damping invalidates preparation, so the
joint-default interpolation must complete again before RL can restart. Upper-body streaming is available only in
`RL_DEFAULT` and is disabled whenever that mode is left.

These presets also start with a simulation-only elastic band attached to `torso_link`. It uses the X2 defaults: a
`[0, 0, 3]` world anchor, `200 N/m` stiffness, `100 Ns/m` damping, and zero rest length. Keyboard controls are `7` to
lower the robot, `8` to lift it, and `9` to release or reactivate the band. Control-mode switching does not change the
band state; respawning the simulation resets it to active with zero rest length.

The 23-DOF policy produces 13 learned actions and exposes 10 arm joints to the upper-body stream. The 29-DOF policy
produces 15 learned actions, including the two additional waist joints, and exposes 14 arm joints. Each preset uses the
PD gains and per-joint action scales recorded in its ONNX export; do not mix a checkpoint with a different preset.

Joystick controls:

- Left stick: forward/backward and lateral velocity
- Right stick X: yaw velocity
- D-pad Up/Down: body height
- D-pad Left/Right: waist yaw command
- Back/Select: reset locomotion commands
- Start: enable or disable the upper-body stream
- LB+RB+A: stop the simulation
- LB+RB+Y: respawn the simulated robot

Keyboard controls use `w/s`, `a/d`, and `q/e` for velocity, `r/f` for height, `z/c` for waist yaw, and `x` to reset
commands. Press `t` to toggle the upper-body stream, `o` to stop, and `i` to respawn.

The upper-body controller subscribes to `tcp://127.0.0.1:8559` by default and accepts partial named-joint updates:

```json
{"positions": {"left_shoulder_pitch_joint": 0.8}}
```

Unknown joints and non-finite values are rejected. Targets are clamped to joint limits, smoothed, and returned toward
the default pose if messages become stale. The stream starts disabled and affects only joints outside the policy action
set.

Native Unitree real-robot variants are also registered:

```bash
python scripts/run_pipeline.py -c g1_23_locomanipulation_default_real
python scripts/run_pipeline.py -c g1_23_locomanipulation_stiff_real
python scripts/run_pipeline.py -c g1_29_locomanipulation_stiff_real
```

Use `python scripts/test_upper_body_zmq.py --robot g1-23` or `--robot g1-29` to publish the matching predefined
upper-body test poses.

The 23-DOF variants keep a logical 23-joint policy/environment layout while using the standard 29-slot Unitree motor
transport. Logical joints map to motor indices `[0..12, 15..19, 22..26]`; targets and gains for the six omitted slots
are zero, and feedback is selected through the same mapping. The 29-DOF variants use the transport directly.
Real variants use the wireless remote, with `A/B/Y/X` selecting the four modes, `Start` toggling upper-body streaming,
and `L1+R1+A` shutting down. They enable joint-limit clipping, low-state freshness checks, and a 100 ms C++ command
watchdog that enters damping if position commands stop. Existing G1 real configurations retain their previous timeout
behavior unless these Unitree timeout options are explicitly enabled.
Rebuild the optional binding with `python submodule_install.py unitree_cpp` before using these real configurations.

G1 Locomanipulation can also act as the locomotion half of the four-mode loco-mimic pipeline. The included test
presets pair it with two 29-DOF BeyondMimic exports that do not require a state estimator: `Jump_wose` and
`Dance_wose`.

```bash
# MuJoCo
python scripts/run_pipeline.py -c g1_23_locomanipulation_default_locomimic
python scripts/run_pipeline.py -c g1_23_locomanipulation_locomimic
python scripts/run_pipeline.py -c g1_29_locomanipulation_locomimic

# Real G1
python scripts/run_pipeline.py -c g1_23_locomanipulation_default_locomimic_real
python scripts/run_pipeline.py -c g1_23_locomanipulation_locomimic_real
python scripts/run_pipeline.py -c g1_29_locomanipulation_locomimic_real
```

The logical 23-DOF presets explicitly enable the missing-DOF adapter. The six 29-DOF-only joints use the mimic
model's default positions and zero velocities in observations, and their output actions are discarded. Other policy
and environment combinations remain strict unless `pad_missing_dofs=True` is set on that policy config.

In simulation, `Back`/`Start` select loco/mimic, `LB`/`RB` select the previous/next mimic, and `L` toggles upper-body
ZMQ while idle in loco. Keyboard equivalents are `]`/`[`, `'`/`;`, and `t`. On the Unitree remote, use
`Select`/`Start`, `L1`/`R1`, and `L2`, respectively. Upper-body ZMQ is disabled during interpolation and mimic, then
resynchronized to the current loco target before it becomes available again. When the mimic reaches its configured
maximum timestep, the pipeline automatically interpolates back to loco. Only the upper non-locomotion joints are
interpolated or overridden during policy switching; the lower body remains under Locomanipulation until mimic becomes
active.

These presets also provide operator-triggered AMP fall recovery. After a fall, select `JOINT_DEFAULT`, then press the
right-stick button (`R`) in simulation, `r` on the keyboard, or `R2` on the Unitree remote to enter recovery.
Recovery is accepted only while the robot is both in `JOINT_DEFAULT` and fallen. Once it is upright
(tilt below `1.0 rad`), select loco to start the normal smooth transition back to `RL_DEFAULT`; `JOINT_DEFAULT` is not
required for this return. Passive, damping, and shutdown remain immediate exits throughout recovery. The recovery
policy's explicit PD gains are applied on entry and the loco gains are restored during the return transition.
As with the normal `JOINT_DEFAULT` to `RL_DEFAULT` transition, the configured joint-default interpolation must finish
before the recovery command is accepted.
If the robot starts or respawns fallen, it retains its initial `PASSIVE_DEFAULT` mode; startup tilt alone does not
force damping. An explicit operator `PASSIVE_DEFAULT` command is likewise retained despite the tilt. Selecting another
mode clears the passive override. An explicitly selected `JOINT_DEFAULT` is also retained while fallen so recovery can
be requested; passive and damping remain available at all times.

Both G1 loco-mimic variants refresh their born-place position and heading frame when a transition activates the loco or
mimic policy. This alignment-only refresh intentionally preserves the elastic band's active state and rest length.
BeyondMimic startup also checks `observation_names` metadata against `without_state_estimator`: exports without the
state estimator omit `motion_anchor_pos_b` and `base_lin_vel` but continue to use IMU orientation and angular velocity.

script: [locomanipulation_policy.py](../robojudo/policy/locomanipulation_policy.py)

## [Policy](#policy) > [ProtoMotionsTrackerPolicy](#policy--protomotionstrackerpolicy)

> Thanks to [NVLabs](https://github.com/NVlabs) for ProtoMotions, and to [Chen Tessler](https://github.com/tesslerc) and [Yifeng Jiang](https://github.com/jyf588) for contributing this RoboJuDo integration.

`ProtoMotionsTrackerPolicy` deploys a [ProtoMotions](https://github.com/NVlabs/ProtoMotions) tracker exported as a unified ONNX pipeline.


For the full deployment walkthrough, please read the official ProtoMotions docs first:

- [ProtoMotions G1 deployment workflow](https://nvlabs.github.io/ProtoMotions/tutorials/workflows/g1_deployment.html#step-5-deploy-via-robojudo-simulation)
- [NVLabs / ProtoMotions](https://github.com/NVlabs/ProtoMotions)

> **Important:** You must clone the `ProtoMotions` repository locally as a sibling directory named `protomotions`. RoboJuDo imports `deployment.motion_utils` and `deployment.state_utils` from that checkout at runtime.


### Core files

- policy implementation: [protomotions_tracker_policy.py](../robojudo/policy/protomotions_tracker_policy.py)
- policy config: [g1_protomotions_tracker_cfg.py](../robojudo/config/g1/policy/g1_protomotions_tracker_cfg.py)
- pipeline configs: [g1_cfg.py](../robojudo/config/g1/g1_cfg.py)
- launcher: [run_tracker_pipeline.py](../scripts/run_tracker_pipeline.py)
- built-in tracker assets:
  - `assets/models/g1/protomotions_tracker/unified_pipeline.onnx`
  - `assets/models/g1/protomotions_tracker/unified_pipeline.yaml`
  - `assets/motions/g1/g1_bones_seed_mini.pt`

### How to run

This repository uses the following built-in configs:

- `g1_protomotions_tracker` for MuJoCo simulation
- `g1_protomotions_tracker_real` for real G1 deployment

Simulation:

```bash
python scripts/run_tracker_pipeline.py -c g1_protomotions_tracker \
 --motion-path assets/motions/g1/g1_bones_seed_mini.pt \
 --motion-index 0
```

Real robot:

```bash
python scripts/run_tracker_pipeline.py -c g1_protomotions_tracker_real \
  --motion-path assets/motions/g1/g1_bones_seed_mini.pt \
  --motion-index 0
```

To test your own exported tracker, add:

```bash
--onnx-path /path/to/unified_pipeline.onnx
```
