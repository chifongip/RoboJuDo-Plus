# G1 arm disturbance observer

G1 uses the [shared observer configuration and runtime](joint_observer.md).
This guide covers the stiff 23-DOF presets, simulation validation, and the
Unitree interface for 23- and 29-DOF G1 configurations.

## Stiff 23-DOF simulation

The opt-in preset uses the `policy_23dof_stiff` checkpoint and position-controlled
arms. The observer holds each additive torque command for 20 ms. Gravity and
joint coupling can contribute to correction during unloaded motion.

```bash
python scripts/run_pipeline.py -c g1_23_locomanipulation_locomimic_observer
OPENBLAS_NUM_THREADS=1 python -m scripts.evaluate_joint_observer \
  --robot g1 --output logs/g1_observer/stiff_validation
```

The closed-loop evaluation uses the preset's correction limits and a 10 N
downward load per hand, the default for both robots. Peak hand displacement
includes the instant of release and is measured against each mode's matched
unloaded trial.

| Motion | Carrying arm RMS, OFF → ACTIVE | Release hand peak, OFF → ACTIVE |
| --- | ---: | ---: |
| Standing | 0.837° → 0.00050° | 4.56 → 3.93 mm |
| Walking at 0.25 m/s | 0.865° → 0.322° | 5.93 → 6.43 mm |

SHADOW matched OFF and issued zero torque. No actuator saturation occurred.

## Additive-torque calibration

```bash
OPENBLAS_NUM_THREADS=1 python -m scripts.evaluate_joint_observer \
  --robot g1 --fit --output logs/g1_observer/standing_greybox_fit
```

G1 uses the [shared torque-probe procedure](x2_upper_body_integral.md#reproduce-the-x2-coefficients).
Both robots fit coefficients from unloaded default and grasp trials and a
carry trial with a 20 N downward load per hand.
In the pelvis frame, G1 shifts both X2 hand centers of mass 10 cm backward
along x and 15 cm down along z. Inverse kinematics matches those targets
and minimizes hand orientation error within G1's joint limits. Calibration and
the ZMQ publisher share the resulting targets in `robojudo/config/g1/carry_pose.py`.
Run `python scripts/test_upper_body_zmq.py --robot g1-23` and press `7`
(`custom_arms`) to select this pose.
The fit seeds friction at 0.10 N·m and fixes wrist-roll inertia at
0.01020 kg·m², the rounded joint-space inertia at the nominal default pose.
The search-bound check applies to the free parameters.

All ten joints passed standing validation across 33 acquisitions with the
updated pose. With both models ACTIVE in the 10 N-per-hand check, the walking
release peak rose from 6.38 mm with the previous coefficients to 6.43 mm with
the current fit. Neither model saturated. The traces and comparison are in
`logs/g1_observer/updated_carry_20261125/`.

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
