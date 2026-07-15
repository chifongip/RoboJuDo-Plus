# X2 Sim2Real Deployment

The `x2_real` configuration runs the 29-joint X2 ONNX policy through the 31-joint AimDK robot interface.
RoboJuDo reorders joint state by name before constructing the `1x151` observation, then maps the `1x29` policy output
back to the robot-native order. The two head joints are not policy-driven; position control actively holds them at their
configured defaults.

## Prerequisites

Initialize and build the pinned AimDK SDK, then install the ROS 2 extension with the managed installer:

```bash
source /opt/ros/humble/setup.bash
conda activate robojudo_test
python submodule_install.py aimdk
source third_party/aimdk/install/setup.bash
```

The installer builds only `aimdk_msgs`, installs the Python/CMake build tools into the active environment, and installs
`aimdk_cpp` without PEP 517 build isolation. Source the generated AimDK setup file in every shell used for real X2
deployment. Existing changes inside the AimDK submodule are preserved; use `python submodule_install.py --clean aimdk`
only when you intentionally want to discard them.

Verify the native backend explicitly after installation:

```bash
ROBOJUDO_REQUIRE_AIMDK=1 python -m unittest discover -s tests
```

This strict test mode fails instead of skipping when `aimdk_cpp` or one of its ROS 2 libraries cannot be loaded.

Confirm that all four joint-state topics and `/aima/hal/imu/torso/state` are updating before enabling commands. Missing or stale state prevents activation and triggers damping.

## Control Modes

The X2 pipeline starts in `PASSIVE_DEFAULT` and uses the same four operating modes as the standalone controller:

| Mode | Joystick | Behavior |
| --- | --- | --- |
| `PASSIVE_DEFAULT` | `A` | Zero stiffness and damping on all 31 joints. Use only while the robot is supported. |
| `DAMPING_DEFAULT` | `B` | Zero stiffness and damping `5.0` on all 31 joints. |
| `JOINT_DEFAULT` | `Y` | Interpolate all joints to the standalone preparation pose over 1.5 seconds. |
| `RL_DEFAULT` | `X` | Reset and warm up ONNX, then run the 29-joint policy from timestep 1. |

`RL_DEFAULT` is rejected until `JOINT_DEFAULT` completes. Entering passive or damping invalidates preparation, so joint preparation must run again before restarting RL. `LB+RB+A` requests damping followed by process shutdown. In simulation, `LB+RB+Y` resets the model and returns to passive mode.

## Simulation ElasticBand

The `x2` MuJoCo configuration starts with a virtual tension band attached to `torso_link`. Its default world anchor is `[0, 0, 3]`, stiffness is `200 N/m`, damping is `100 Ns/m`, and rest length is `0 m`. The band force is applied before every physics substep in all four control modes. The viewer renders the band as an orange capsule with a sphere at the world anchor; both are hidden when the band is released.

- `7`: lower the robot by increasing rest length by `0.1 m`.
- `8`: lift the robot by decreasing rest length by `0.1 m`, clamped at zero.
- `9`: release or reactivate the band.

The settings are defined by `elastic_band` in `X2MujocoEnvCfg`. Set `visualize=False` to hide the geometry while retaining the force, or adjust `visual_radius`, `visual_rgba`, and `anchor_radius` to change its appearance. The band is simulation-only; `x2_real` does not register these keyboard controls or apply an external force.

## Startup Sequence

Run the real configuration only with the robot supported and an operator holding the emergency stop:

```bash
python scripts/run_pipeline.py -c x2_real
```

With the robot supported, select `JOINT_DEFAULT` and wait for the completion log. Then select `RL_DEFAULT`. ONNX time does not advance in passive, damping, or joint-preparation modes.

The policy runs at 50 Hz. `aimdk_cpp` republishes the active mode at 500 Hz, matching the standalone X2 controller. If position commands stop for 100 ms, the backend latches damping. Position control cannot resume until a new mode transition explicitly re-arms it. Stale state, a non-finite target, excessive tilt, and shutdown also force damping.

## Preflight Checks

- Verify the startup log contains all 29 policy-to-environment joint mappings.
- Keep the robot supported for the first default-pose and low-amplitude policy trials.
- Confirm joystick `B` causes damping before attempting unsupported operation.
- Do not deploy if targets are repeatedly clamped or control-loop frame drops are reported.
