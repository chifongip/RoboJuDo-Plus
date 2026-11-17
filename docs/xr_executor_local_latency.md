# Local executor latency comparison

Local verification on 2026-11-16 found higher completion throughput and steadier
completion intervals, but **did not demonstrate lower per-frame latency**.

The comparison ran on the available x86_64 host, using two Python processes,
LeRobot IK, headless MuJoCo, 50 Hz synthetic XR sampling, both grips engaged,
and continuously varying arm and finger targets. The hand runtime simulated a
1 ms combined write delay. Both versions used direct fingers and enabled the
existing policy-boundary completion checkpoints. One combined frame remained in
flight. Collision settings were matched; freshness thresholds were unchanged.
Each condition ran twice for six measured seconds, with baseline/executor order
reversed for the second repetition.

| Collision checks | Version | Completed frames/s | Mean bridge-observed acknowledgement RTT | Mean XR sample to robot acknowledgement |
| --- | --- | ---: | ---: | ---: |
| Off | Existing low-latency behavior (schema 2) | 39.1 | 21.9 ms | 15.5 ms |
| Off | Authoritative executor (schema 3) | 48.7 | 24.7 ms | 15.7 ms |
| On | Existing low-latency behavior (schema 2) | 27.0 | 24.3 ms | 25.6 ms |
| On | Authoritative executor (schema 3) | 48.9 | 33.7 ms | 28.6 ms |

Completion-gap p95 was 40.7–41.1 ms for the existing behavior and 20.9–21.1 ms
for the executor across these trials. The executor rejected zero commands in the
measured windows; the existing behavior rejected 58–59 per trial with collision
checks off and 130–132 with checks on. Rejection counters include both receive-
and application-stage rejection; they do not independently classify the cause.

Frame rates are pooled completions divided by elapsed time. Latency means are
weighted by the corresponding sample counts. These are short measurements,
without confidence intervals. Lower completion gaps describe improved command
cadence, not lower individual-command latency. Per-frame latency statistics only
include acknowledged frames; the versions complete different subsets of input
frames. They therefore must be interpreted alongside throughput and rejections.

The bridge RTT begins after target construction/publication and ends when the
bridge observes completion. Moving collision work from the bridge into RoboJuDo
changes where work falls relative to that boundary. The benchmark additionally
measures XR sampling to robot acknowledgement using the common host monotonic
clock; this measure also showed no latency improvement. The benchmark-only
sampling timestamp is diagnostic and never authorizes motion.

This host does not reproduce the Jetson's observed 41–44 ms completion round
trip and 21–22 completed frames/s. The simulated runtime has no HCAN, native SDK
reads/writes, independent hand-worker scheduling, or idle completion-notification
socket. Completion means submission and write completion, not physical joint
arrival. These results support improved local throughput, not a claim that the
Jetson latency problem has been resolved.

Reproduce with the RoboJuDo interpreter and the existing XR environment:

```bash
python tests/benchmark_xr_executor.py \
  --bridge-python /path/to/robojudo-xr/bin/python \
  --seconds 6 --trials 2 --hand-delay-ms 1 \
  --output /tmp/xr-executor-motion-benchmark.json
```

The script always uses simulation and synthetic hands. It writes per-trial
latency distributions, completion gaps, counts and host information to JSON.
Increasing trial duration/repetitions is supported. For hardware verification,
repeat the same metrics with the two real presets on the Jetson under matched
collision settings, XR workload and SDK configuration; the local script does not
issue hardware commands.
