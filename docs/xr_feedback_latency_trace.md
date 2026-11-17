# Bounded feedback-latency investigation

This change adds diagnostic evidence and earlier completion observation. It does
not change target generation, submission, collision validation, freshness limits,
SDK settings, or the one-combined-frame execution barrier.

The bridge now attributes each matching completion to the receive timestamp of
its first valid snapshot, rather than the final selected snapshot. Validation and
bounded batch handling still precede diagnostic bookkeeping; its actual execution
time is traced separately from the attributed timestamp. It continues constructing commands
from the newest validated feedback and accepting solver history only through the
existing newest-state path. Duplicate completion reports count once. The
`--no-observe-all-completions` bridge flag restores the previous latest-only
observation for comparison. Diagnostic observation never authorizes motion.

The `completed` bridge diagnostic can now include completions previously skipped
while draining a batch. A higher diagnostic count alone is not a throughput gain;
use the robot's completion-counter change or correlated robot acknowledgements.

## Capture a bounded trace

Tracing is disabled by default. Enable it independently in each process, with
different output paths. Each process holds at most 8,192 records in memory and
exports at orderly shutdown, outside the measured control path. No per-frame file
I/O or joint-position recording is added. Overflow and incomplete windows are
explicitly marked and are ineligible for response comparisons.

Simulation example, using the existing RoboJuDo and XR interpreters respectively:

```bash
python scripts/run_pipeline.py -c x2_xrobotoolkit_omnihand \
  --upper-body-latency-trace /tmp/robot-feedback-trace.json \
  --latency-trace-delay-s 10 --latency-trace-duration-s 30

python -m xrobotoolkit_teleop.integrations.robojudo \
  --robot x2_omnihand --ik-backend lerobot \
  --latency-trace-path /tmp/bridge-feedback-trace.json \
  --latency-trace-delay-s 10 --latency-trace-duration-s 30
```

The delay allows startup and engagement before collection. Windows start relative
to each process's initialization; the analyzer uses their overlapping interval.
Wait until both windows finish before stopping the processes. Abrupt process
termination cannot export an in-memory trace. To inspect hardware, retain the
user-started pipeline's existing configuration, collision settings, and SDK setup,
and append these trace options. Add `--authoritative-executor` to the bridge only
when using the matching schema-3 executor preset. Do not change presets just to
collect a trace.

Configuration equivalents on the feedback-enabled upper-body controller are
`latency_trace_path`, `latency_trace_delay_s`, and `latency_trace_duration_s`.
The pipeline CLI requires exactly one such controller to prevent ambiguous output.

Analyze the exported files with any standard Python interpreter:

```bash
python scripts/analyze_xr_latency_trace.py \
  /tmp/robot-feedback-trace.json /tmp/bridge-feedback-trace.json \
  --output /tmp/feedback-trace-analysis.json
```

## What the evidence means

Frames are correlated by session, generation, bridge ID and frame ID. Feedback
packets additionally use state ID. The trace includes:

- XR sampling start and command publication in the bridge;
- robot command receipt, arm submission, both hand-write completions, and robot acknowledgement;
- feedback send attempts and every application-level bridge packet receipt,
  including validation failures;
- first diagnostic observation of each matching completion.

Existing worker timestamps are copied when the owner acknowledges completion;
worker processes gain no new tracing work. `first_ack_state_id` is an optional
feedback diagnostic that persists across repeated snapshots for a completion.
It identifies the first PUB send attempt accepted locally, not confirmed delivery.
A later snapshot retaining an absent first-state ID is useful evidence, but cannot
by itself prove that PUB's queue dropped the packet. The analyzer distinguishes
an absent first packet, a received-but-invalid packet, and a valid earlier snapshot
skipped during completion observation.

Cross-process intervals require identical nonempty boot ID and Linux time-namespace
identity for the monotonic clock. Otherwise sample-to-write and interprocess
turnaround intervals are omitted. Missing hand-write stamps are not replaced by
acknowledgement timestamps. The primary response metric ends at the later of the
two hand-write completions; the report separately includes the later of those
writes and successful arm submission. Neither measures physical joint arrival.
Acknowledgement RTT and receive-to-observation delay are separate metrics.

## Local paired verification

The simulation benchmark now follows the production bridge's feedback-aware wait
loop. It adds simulated per-hand write timestamps and exports the same bounded
traces used for hardware diagnosis:

```bash
python tests/benchmark_xr_executor.py --feedback-comparison \
  --bridge-python /path/to/robojudo-xr/bin/python \
  --seconds 4 --trials 2 \
  --output /tmp/feedback-latency-comparison.json
```

This compares latest-only and earliest observation on schema 3. Holding and moving
workloads, with collision checks off and on, produce 16 alternating-order trials.
Use `--workload holding|moving` and `--collision-checks off|on` for a smaller run.
Traces are saved beside the result, or in `--trace-directory`. The original
schema-2/schema-3 comparison remains available without `--feedback-comparison`.
Because the wait loop now matches production, its timings should not be treated
as identical experimental conditions to older sleep-based benchmark results.

On the available x86_64 host on 2026-11-17, all 16 four-second traces were eligible,
with 3,058 matched completed frames, no missing first acknowledgements, no received
invalid first acknowledgements, and no generation resets. Paired changes in
sample-to-hand-write p95 ranged from a 0.79 ms regression to a 0.84 ms improvement.
There was no consistent response improvement. Receive-to-metric-timestamp
deferral p95 was approximately 0.39–0.42 ms for the old path; the new
path attributes the metric to the first packet receipt, so that attribution gap is
zero by construction. Actual diagnostic-code execution is timestamped separately
from the attributed receipt timestamp in new traces.
A follow-up trace check with separate execution timestamps measured about 0.51 ms
p95 from receipt to actual diagnostic processing in both paths. This is a reporting
correction, not evidence of faster execution.

The required 5 ms consistent response gain was not met. The traces also did not
justify the conditional feedback-buffer experiment, so feedback PUB's queue
limit remains one; no new buffer setting or transport was introduced.

For Jetson validation, collect three paired 30-second runs in holding and slow
combined arm/finger motion, alternating comparison order and keeping collision
settings, 50 Hz publication, SDK settings and freshness limits fixed. Recommend an
execution-path change only for a consistent sample-to-write p95 improvement of at
least 5 ms, with no more than 5% throughput loss and no additional rejection or
generation-reset failures. If a trace overflows, shorten the capture and report
the changed duration; never compare an overflowed trace as a complete run.
If only diagnostic RTT improves, retain the execution path and stop this optimization.
The local run cannot reproduce native HCAN workers, idle completion notifications,
or the Jetson's reported 40 ms delay.
