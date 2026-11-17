# Nonlinear observer performance

The optimized observer retains the four integration substeps, 20 bounded
bisections, best-residual candidate selection, timestamp alignment, accepted
command replay, filtering, torque limits, slew limits, and anti-windup behavior.
It accelerates numerical execution without reducing solver accuracy or changing
the 50 Hz control target. Deployment observer mode remains OFF by default.

## Enable acceleration

Install into the environment that runs the pipeline:

```bash
python -m pip install -e '.[observer-fast]'
```

`JointObserverCfg.backend` supports:

| Value | Behavior |
| --- | --- |
| `auto` (default) | Use Numba when installed; otherwise use NumPy. |
| `numpy` | Explicit reference implementation; no Numba import or compilation. |
| `numba` | Require acceleration; fail at initialization if unavailable. |

The compiled backend applies to the nonlinear observer. Explicit `numba` with
`algorithm="linearized"` is rejected. The selected backend is logged at
initialization and exposed as `JointObserver.observer.backend`.

Compilation/cache loading occurs during observer construction, before control
starts. For AgiBotCppEnv, observer construction precedes native AimDK controller
creation. The kernel receives one normalized contiguous float64/bool signature,
independent of joint count and history length; strided and read-only callers are
normalized at the Python boundary. No parallel workers or fast-math are used.
Installed-but-broken compiler dependencies or compilation failures stop startup;
there is no silent switch to a slower backend after a control-loop failure.

Numba is an optional dependency, tested here at 0.65.1 with NumPy 2.4.6 and
Python 3.11.15 on Linux x86_64. Installing it on the development host does not
install it on a robot computer. Confirm the selected backend and timing there.

## Investigation and implementation

A fresh nonlinear update previously made one nominal prediction plus 23
candidate predictions: lower bound, upper bound, previous estimate, and 20
bisections. Each prediction integrates four nominal substeps. The synchronous
simulation uses one previous command; deployment replays multiple accepted
commands over per-joint receive intervals. Consequently a synchronous benchmark
alone understates the deployment path's work.

The initial synthetic timestamped profile spent about 73% of cumulative workload
time in `predict_state()`: 68,449 integration calls over 1,100 updates including
warmup. Timestamp bookkeeping and `np.isclose()` coverage checks were also
repeated for every candidate. These measurements identify this offline
workload's bottleneck, not the exact bottleneck of the reported robot run.

The implementation has two layers:

1. `runtime.py` prepares timestamp intersections, active segment masks, coverage,
   and elapsed feedback time once per proposal. The NumPy backend benefits from
   this too. This first optimization reduced the measured timestamped median
   from 4.857 ms to 4.312 ms (about 11%).
2. `accelerated.py` fuses candidate replay, integration, and all bisections in a
   compiled loop. Per-segment `h`, inertia/h, and elastic coefficients are
   computed once. Per-joint state and sensitivity remain scalar local variables
   rather than repeatedly allocating small arrays and dispatching NumPy calls.

The nominal prediction still uses the reference callback. Filtering, freshness,
output limiting, acknowledgement, and reset semantics stay in Python. Custom
predictors and overridden dynamics use their callbacks rather than being
silently replaced by the built-in compiled dynamics.

Candidate saturation and friction branches remain candidate-dependent.
Nonfinite intermediate calculations, including branches NumPy evaluates but
does not select, are rejected. Boundary shape checks prevent unchecked compiled
array access. Prepared callbacks avoid per-update reference cycles, limiting
avoidable cyclic-GC work in the control loop.

## Reproduce timing

```bash
PYTHONPATH=. python scripts/benchmark_observer.py \
  --backend numpy --steps 2000 --warmup 100 --output /tmp/observer-numpy.json
PYTHONPATH=. python scripts/benchmark_observer.py \
  --backend numba --steps 2000 --warmup 100 --output /tmp/observer-numba.json \
  --profile /tmp/observer.prof
python -m pstats /tmp/observer.prof
```

The benchmark uses the X2 simulation observer parameters and gains. It measures
proposal plus acknowledgement, excluding input generation, physics, policy
inference, native transport, and status logging. Both synchronous and timestamped
workloads include changing commands and clipped receipts. Timestamped feedback
includes per-joint delays and repeated samples, retaining up to four commands.
This is synthetic numerical replay, not recorded robot data or a complete
locomotion pipeline.

Construction time is reported separately, including import/compilation/cache
loading. Warmup samples are excluded from latency statistics. cProfile runs
separately after timing so its instrumentation does not affect the percentiles.
Repeat measurements without concurrent workloads on the deployment computer.

## Validation

```bash
python -m unittest discover -s tests -p '*observer*.py'
```

Run once with the optional extra installed and once without it to exercise both
acceleration and automatic fallback. The compiled tests skip when Numba is absent.

Coverage includes:

- Exact committed pre-refactor output fixtures for ACTIVE and SHADOW replay.
- Differential asynchronous replay with resets, repeated/expired timestamps,
  partial coverage, multiple command segments, and clipped receipts.
- Randomized inversion across joint counts, saturation/friction transitions,
  masks, and segmented history, compared exactly with NumPy.
- Closed-loop load/release/reversal and gain changes, compared exactly with NumPy.
- Paired headless X2 MuJoCo simulation over 120 steps of load, release, and
  reversal, comparing qpos, qvel, issued effort, and observer diagnostics exactly.
- Custom predictor preservation, invalid/nonfinite input rejection, no deferred
  compilation for strided/read-only inputs, and prompt callback reclamation.
- Stale feedback skips predictor preparation, and missing compiler dependencies
  retain their original diagnostic errors.

Local observer-suite results and final timing measurements are recorded below.
The broader repository run reached 413 tests and reported four errors outside
the observer: missing `robojudo_recorder`, and missing velocity priorities in
three GR00T configurations. Its ZeroMQ tests required running outside the
socket-restricted sandbox.

## Final local measurements

Three sequential runs per backend, alternating backend order, each with 100
warmup and 2,000 measured updates per path. Timing ran without concurrent tests.
The NumPy comparison includes the earlier prepared-history optimization.
Percentiles below are the median of the three per-run percentiles; maximum is
the largest observed update across all three runs, not a worst-case bound.

| Path | Backend | p50 | p95 | p99 | Observed maximum |
| --- | --- | ---: | ---: | ---: | ---: |
| synchronous | numpy | 1.628 ms | 1.678 ms | 1.710 ms | 3.346 ms |
| synchronous | numba | 0.162 ms | 0.172 ms | 0.181 ms | 0.301 ms |
| timestamped | numpy | 4.451 ms | 5.904 ms | 6.062 ms | 7.437 ms |
| timestamped | numba | 0.354 ms | 0.436 ms | 0.452 ms | 0.554 ms |

The timestamped median improved about 12.6x (92% lower); synchronous median
improved about 10x. All measured observer updates were below 20 ms. Both paths
exclude the rest of the pipeline, so this does not demonstrate the robot's
end-to-end deadline. The first compiled construction in this cached benchmark
process took 95 ms; later constructions took under 0.2 ms. First-ever compilation
is separate from these cached startup measurements.

After review and cleanup, the observer suite ran all 90 tests successfully with
acceleration installed. Without the optional extra, it passed with eight
compiled-only checks skipped. Missing-dependency fallback and diagnostic tests
run in both environments.

## Remaining deployment checks

The local measurements do not establish that the reported 22–29 ms full robot
pipeline now meets 20 ms. Measure feedback acquisition, policy inference,
observer proposal, native submission/acknowledgement, and diagnostics separately
on the deployment computer; aggregate p50/p95/p99/max and deadline misses.
Correlate observer latency with active replay segments and fresh joint count.

Do not infer worst-case execution time from these samples. Large accepted-command
histories and host scheduling can affect latency. No robot was commanded during
this investigation. Slower/asynchronous estimation or fewer bisections would be
separate control changes and are not part of this implementation.
