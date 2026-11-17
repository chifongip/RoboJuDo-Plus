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
justify changing the default feedback PUB queue limit of one. The later Jetson
experiment below can now be configured without editing source code.

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


## ZMQ queue settings after the Jetson experiment

The supplied Jetson traces showed 685/821 first acknowledgements absent at send
HWM=1 and 0/1009 absent at HWM=8. Mean acknowledgement RTT fell from 34.9 to
21.7 ms. Sample-to-write p95 increased from 26.2 to 37.1 ms, however, and the
HWM=8 run had one generation reset. This supports testing feedback delivery with
8, but does not establish a response improvement or justify making 8 the default.

Queue settings are now explicit and logged at startup:

| Socket | Setting | Default | Allowed values |
| --- | --- | --- | --- |
| RoboJuDo command SUB | `command_receive_hwm` | 8 for XR presets; 100 for other presets | 1–100 |
| RoboJuDo feedback PUB | `feedback_send_hwm` | 1 | 1–8 |
| Bridge feedback SUB | `feedback_receive_hwm` | 8 | 1–16 |
| Bridge command PUB | unchanged | 1 | fixed |

XR command receive queues previously used 100. They now use 8 to limit the
number of waiting commands held in the ZMQ pipe when servicing falls behind.
This is a queueing change requiring Jetson verification, not a measured latency
improvement. The 100-message robot drain budget, 32-message bridge receive budget,
backlog rejection, freshness deadlines, newest-valid selection, and combined-frame
barrier are unchanged. Other presets retain their previous command receive limit.
HWM values are message-count settings, not strict bounds on end-to-end latency;
transport buffering and socket drop/backpressure behavior still matter.

Append these options to the existing RoboJuDo launch command to test feedback
send HWM=8 while using the new XR command receive limit:

```bash
--upper-body-feedback-send-hwm 8 --upper-body-command-receive-hwm 8
```

For the baseline, change only `--upper-body-feedback-send-hwm` to `1`. To isolate
the command receive limit from earlier runs, use `--upper-body-command-receive-hwm
100` consistently for both feedback comparisons, then compare 100 versus 8
separately. Do not change both queue sizes between members of a pair.

The bridge receive option is `--feedback-receive-hwm 8`; it is already the default.
Keep it fixed for the initial feedback send comparison. For example, a
simulation-only queue configuration check uses:

```bash
python scripts/run_pipeline.py -c x2_xrobotoolkit_omnihand \
  --upper-body-feedback-send-hwm 8 --upper-body-command-receive-hwm 8
python -m xrobotoolkit_teleop.integrations.robojudo \
  --robot x2_omnihand --ik-backend lerobot --feedback-receive-hwm 8
```

Retain the current hardware preset, matching executor flag and collision options
when testing on the Jetson. Remove manual socket edits so these settings take
effect from validated configuration. Restart both processes and retain their
startup queue logs with each trace.

The versioned XR sockets do not use conflation: it would discard intermediate
acknowledgements and does not preserve multipart framing. This leaves the existing
multipart rejection intact. `LINGER=0`, nonblocking publication, and packet-size
limits remain unchanged. See the official [ZeroMQ socket options](https://libzmq.readthedocs.io/en/latest/zmq_setsockopt.html)
and [PUB socket behavior](https://zeromq.org/socket-api/) for these semantics.

Earlier local benchmark figures in this document predate the smaller XR command
receive queue. The configuration and protocol tests do not establish its hardware
latency benefit; use paired Jetson traces, including reset reasons and write
completion latency, to decide whether to retain it.
