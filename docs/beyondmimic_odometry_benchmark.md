# BeyondMimic low-rate odometry benchmark

This benchmark tests the X2 `Solo_dance` BeyondMimic policy with pose odometry sampled more slowly than the
50 Hz policy loop. It is intended as a simulation safety gate before using 10 Hz LiDAR-inertial odometry on the
real robot.

Only the odometry-derived root/torso position and body-frame linear velocity are sampled at the selected rate.
Pelvis IMU orientation, angular velocity, joint state, control, and policy inference remain at 50 Hz. The simulator
constructs a virtual chest LiDAR pose and passes it through the same sensor-to-torso-to-pelvis transform used by the
real X2 environment. Velocity comes from timestamped world-position differences and is rotated by the physical,
un-aligned root quaternion.

The benchmark is headless, deterministic by seed, and disables the elastic suspension band so it cannot conceal a
fall. It checks 0, 90, 180, and 270 degree initial headings by default.

## Quick validation

Run the short smoke suite:

```bash
python scripts/benchmark_beyondmimic_odometry.py --smoke
```

This compares 50 Hz and 10 Hz at 0 and 180 degrees and verifies that a 0.4 second odometry outage triggers a stale
data abort.

## Visualize a 10 Hz run

Run one deterministic 10 Hz case in the MuJoCo viewer:

```bash
python scripts/benchmark_beyondmimic_odometry.py \
  --rates 10 \
  --headings 180 \
  --seeds 0 \
  --scenarios nominal \
  --steps 6747 \
  --visualize \
  --output /tmp/beyondmimic_10hz_visual
```

Visualization mode runs at real-time 50 Hz while the position and linear-velocity estimator updates at 10 Hz. The
red arrow is the reference anchor, green is the odometry-derived robot anchor, cyan is their position error, and
yellow is the FK torso pose. The suspension band remains disabled.

## Full safety gate

```bash
python scripts/benchmark_beyondmimic_odometry.py
```

The default matrix contains:

- nominal odometry at 50, 20, 10, and 5 Hz;
- headings of 0, 90, 180, and 270 degrees;
- five deterministic seeds;
- 10 Hz latency/jitter, noise, dropout, combined-degradation, and timeout cases;
- all 6747 frames of the `Solo_dance` motion.

Results are written to `benchmark_results/beyondmimic_odometry/` as JSON, CSV, and Markdown. A non-zero exit code
means the safety gate failed.

For a smaller targeted run:

```bash
python scripts/benchmark_beyondmimic_odometry.py \
  --rates 50 10 \
  --headings 0 180 \
  --seeds 0 1 \
  --scenarios nominal combined timeout \
  --output /tmp/beyondmimic_odometry
```

## Gate interpretation

The 10 Hz nominal cases must not fall or raise an error. Their anchor tracking is compared with the matched 50 Hz
baseline, odometry position RMSE must remain at or below 0.08 m, and body-frame velocity RMSE at or below 0.35 m/s.
Non-timeout stress cases must complete without a fall or stale-data exception. The intentional timeout case must
abort after its odometry age exceeds 0.3 seconds.

Passing simulation is necessary but not sufficient for deployment. The real LiDAR pipeline must also publish the
validated sensor frame and timestamps, maintain comparable latency/noise, and enforce the same freshness timeout.
