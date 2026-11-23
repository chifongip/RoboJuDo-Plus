# G1 arm disturbance observer

G1 uses the [shared observer configuration and runtime](joint_observer.md).
This guide covers the stiff 23-DOF presets, simulation validation, and the
Unitree interface for 23- and 29-DOF G1 configurations.

## Stiff 23-DOF simulation

Run the opt-in preset with the existing `policy_23dof_stiff` checkpoint, whose
arm joints are position-controlled by the pipeline. The observer holds its
additive command for each 20 ms interval. The validation table below used
a 2 Nm absolute correction cap. Parameters come from unloaded MuJoCo standing
and walking trajectories with arm excitation. The fit reduced held-out centered
velocity prediction RMS from 0.0417 to 0.0362 rad/s. Gravity and joint coupling
can also contribute to the residual and produce correction during unloaded motion.

```bash
python scripts/run_pipeline.py -c g1_23_locomanipulation_locomimic_observer
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 python scripts/evaluate_g1_observer.py \
  --fit --output logs/g1_observer/stiff_fit
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 python scripts/evaluate_g1_observer.py \
  --correction-cap 2.0 --output logs/g1_observer/stiff
```

With a 10 N downward load per hand, the baseline and ACTIVE trials produced
the following results. Release peaks are measured against each mode's matched
unloaded trial and include the pose at the instant of release.

| Motion | Carrying arm RMS, OFF → ACTIVE | Release hand peak, OFF → ACTIVE |
| --- | ---: | ---: |
| Standing | 0.837° → 0.00045° | 4.56 → 3.10 mm |
| Walking at 0.25 m/s | 0.865° → 0.346° | 5.93 → 3.98 mm |

SHADOW reproduced the OFF trajectories exactly and issued zero torque.
The selected settings produced no actuator saturation, and the longest ACTIVE
loop took 3.73 ms in the headless run. Unloaded peak standing tilt rose from
1.87° to 2.12°.
Omit `--correction-cap` to use the current preset's fraction-only bounds. That
run produced 1.98° of unloaded peak standing tilt against the same 1.87° baseline.

## Real-robot deployment

The real observer preset uses the existing stiff G1 checkpoint and Unitree
controller settings. The three-argument native `step` takes motor-order
positions, arm-order effort, and the expected safety reset generation, as on
AimDK. The environment derives `UnitreeConfig.arm_motor_indices` from the
observer joint names and the logical-to-motor mapping. The controller uses
this order for effort limits, commands, and receipts. It accepts motor counts
up to the HG packet capacity. G1 joint layout validation lives in the Python
environment configuration.

`G1_23ObserverRealEnvCfg` maps ten arm joints into the 29-slot motor command.
`G1ObserverRealEnvCfg` accepts a model for all fourteen arm joints on the full
29-DOF G1, including wrist pitch and yaw. Supply the 29-DOF model when creating
that configuration.

Unitree timestamps the whole accepted LowState packet. Feedback timestamps
and position-command receipts share a native steady clock. The receipts
record host acceptance and drive the observer's command history.
[Unitree G1 SDK command writer](https://github.com/unitreerobotics/unitree_sdk2/blob/main/example/g1/low_level/g1_dual_arm_example.cpp#L1452-L1468)

Rebuild `unitree_cpp` with `python submodule_install.py unitree_cpp` before
SHADOW. The native in-process regression test is enabled by configuring the
package with `-DUNITREE_CPP_BUILD_TESTS=ON` and running CTest.

On a supported G1, enable `debug.log_obs` in the local real observer
configuration. Collect a position-only baseline with `env.observer.mode`
set to `OFF`, then set it to `SHADOW` and repeat the same standing and
walking commands.

```bash
python scripts/run_pipeline.py -c g1_23_locomanipulation_locomimic_real_observer
```

The debug trace includes `extras.observer` and `extras.control_step_ms`. Confirm
zero arm effort in SHADOW receipts and low-command packets, validate LowState
tick progress and feedback age, and track full-loop overruns against the
20 ms period on the deployment computer.

The real preset defaults to OFF. Selecting ACTIVE uses the shared observer
actuator limits, correction fraction, and optional absolute cap. The
registered actuator bounds come from the G1 DoF configuration. Firmware
behavior and continuous hardware torque limits remain unverified.
