# X2 BeyondMimic real-state safety replay

This workflow checks a state-estimator BeyondMimic model using real X2 sensor
data without commanding the robot. It has three stages:

1. Validate a stable synchronized capture and audit its measured joint limits.
   A direct BeyondMimic launch is retained as a non-gating diagnostic.
2. Ground the captured joints and IMU orientation from the X2 foot collision
   geometry, then run the production preparation sequence: 75 `JOINT_DEFAULT`
   steps and the 101-step locomotion-to-mimic transition.
3. Remove simulated preparation support and compare unsupported active-policy
   rollouts using ideal 10 Hz odometry and the captured delivery timing with
   detrended pose/yaw residuals.

The capture utility only creates ROS subscriptions. It does not instantiate
`AgiBotCppEnv`, AimDK command transport, or any ROS publisher. Keep the robot
mechanically supported and in damping mode while capturing.

## Capture 10 seconds

The recorder must use the ROS Humble system Python because the installed ROS
message bindings target that interpreter:

```bash
source /opt/ros/humble/setup.bash
source /home/ubuntu/aimdk/install/setup.bash
/usr/bin/python3 scripts/capture_x2_beyondmimic_state.py \
  --duration 10 \
  --output captures/x2_supported_damping.msgpack
```

It records:

- `/laser_odometry`
- `/aima/hal/imu/torso/state`
- `/aima/hal/joint/leg/state`
- `/aima/hal/joint/waist/state`
- `/aima/hal/joint/arm/state`
- `/aima/hal/joint/head/state`

Capture files are ignored by Git because they can be large and machine-specific.

## Run the offline safety gate

Use the normal RoboJuDo Python environment:

```bash
python scripts/replay_x2_beyondmimic_state.py \
  --capture captures/x2_supported_damping.msgpack \
  --policy-name Solo_dance \
  --steps 100
```

To watch both seeded MuJoCo rollouts:

```bash
python scripts/replay_x2_beyondmimic_state.py \
  --capture captures/x2_supported_damping.msgpack \
  --policy-name Solo_dance \
  --steps 100 \
  --visualize
```

Reports are written to `benchmark_results/x2_real_state_replay/results.json`
and `summary.md`. A pass means this capture satisfied the configured offline
gates; it is not authorization to remove physical support or skip normal
real-robot deployment checks.

The capture used for this workflow may itself be mechanically supported. The
MuJoCo elastic band therefore remains active only through the preparation
sequence and is disabled before the active BeyondMimic steps. Position targets
are clamped in this replay to match `AgiBotCppEnv`; MuJoCo already clamps
actuator torque. Target clipping and torque saturation are reported as
non-gating diagnostics.

## Capture validation

The replay rejects the capture before policy execution if:

- odometry is not `map -> lidar_chest_front`;
- an IMU or odometry quaternion is invalid;
- odometry reports degeneration or has a delivery gap over 0.3 seconds;
- joint/IMU data is older than 0.1 seconds;
- any of the 31 X2 joints is absent; or
- no stable two-second window is found.

After initialization, the moving simulation generates its own virtual sensor
poses. Recorded absolute LiDAR poses are never imposed on the simulated robot.
Only the captured sample schedule and detrended short-term residuals are used
for the captured-profile comparison.

SuperOdom initializes LiDAR translation at zero rather than publishing a
ground-referenced robot height. Both the real X2 environment and this replay
therefore subtract the first converted pelvis position from later converted
positions. This changes only the translation origin: quaternion, displacement,
and body-frame velocity are preserved. Policy-switch alignment remains a
separate operation and is still performed at BeyondMimic activation.
