"""Local schema-2/schema-3 A/B benchmark using synthetic XR and simulated hands.

Run with the RoboJuDo interpreter. No physical robot or hand SDK is initialized.
Completion timing includes owner-thread observation of an injected hand delay;
this does not reproduce the Jetson's native SDK, notifications, or CPU scheduling.
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import test_xrobotoolkit_integration as integration  # noqa: E402
from analyze_xr_latency_trace import analyze  # noqa: E402

from robojudo.utils.latency_trace import LatencyTrace  # noqa: E402


def distribution(values):
    values = list(values)
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }


class SimulatedHands:
    """Single combined pending write, observed at existing owner-thread checkpoints."""

    takeover_requested = True

    def __init__(self, env, delay):
        self.env, self.delay = env, delay
        self.pending = self.applied = None
        self.stamps = {}

    def get_data(self):
        if self.pending is not None:
            positions, identity, ready_at, deadline = self.pending
            if time.monotonic() >= ready_at and ready_at <= deadline:
                self.env.submit_hand_targets(positions)
                self.applied, self.pending = identity, None
                self.stamps["parent_snapshot_received"] = time.monotonic()
        snapshot = self.env.hand_snapshot()
        return dict(
            snapshot,
            joint_state_fresh=snapshot["fresh"],
            applied=self.applied,
            command_timestamps=dict(self.stamps) if self.applied is not None else {},
        )

    def submit(self, positions, identity, deadline):
        if self.pending is not None:
            raise AssertionError("combined frame overwritten while in flight")
        self.applied = None
        now = time.monotonic()
        self.pending = positions.copy(), identity, now + self.delay, deadline
        self.stamps = {"worker_completion_ready": now + self.delay}
        for side in ("left", "right"):
            self.stamps.update(
                {f"{side}_submitted": now, f"{side}_write_started": now, f"{side}_write_completed": now + self.delay}
            )

    def acknowledge(self):
        pass

    def set_takeover_enabled(self, enabled):
        self.takeover_requested = enabled
        if not enabled:
            self.reset()

    def invalidate_commands(self):
        self.reset()

    def reset(self):
        self.pending = self.applied = None
        self.stamps = {}

    def close(self):
        self.reset()


class Harness(integration.XrobotoolkitIntegrationTests):
    with_omnihand = True

    def setUp(self):
        super().setUp()
        for cfg in (self.pipeline._upper_body_cfg, self.receiver.cfg_ctrl):
            cfg.hand_command_mode = "direct"
            cfg.omnihand.direct_completion_notifications = True


def run_trial(
    executor,
    collision,
    duration,
    hand_delay,
    *,
    observe_all_completions=True,
    workload="moving",
    trace_directory=None,
    trial=1,
):
    harness = Harness(methodName="runTest")
    harness.authoritative_executor = executor
    harness.collision_checking = collision
    completed = []
    prefix = (
        f"schema{3 if executor else 2}-{workload}-collision{int(collision)}-"
        f"observe{int(observe_all_completions)}-{trial}"
    )
    bridge_path = trace_directory / f"{prefix}-bridge.json" if trace_directory else None
    environment = {
        "XROBOTOOLKIT_OBSERVE_ALL": str(observe_all_completions),
        "XROBOTOOLKIT_LATENCY_TRACE": str(bridge_path) if bridge_path else "",
        "XROBOTOOLKIT_TRACE_DURATION": str(duration),
    }
    saved_environment = {k: os.environ.get(k) for k in environment}
    os.environ.update(environment)
    try:
        harness.setUp()
        harness.receiver._hand_runtime = SimulatedHands(harness.pipeline.env, hand_delay)
        feedback = harness.receiver.feedback
        acknowledge = feedback.acknowledge

        def observe(proposal=None):
            now = time.monotonic()
            stamps = proposal.get("_command_timestamps", {}) if proposal else {}
            sampled = proposal.get("_benchmark_xr_sampled_at", now) if proposal else now
            completed.append(
                {
                    "at": now,
                    "robot_ms": 1000 * (now - stamps.get("received", now)),
                    "sample_to_ack_ms": 1000 * (now - sampled),
                }
            )
            return acknowledge(proposal)

        feedback.acknowledge = observe
        harness.pipeline._set_upper_body_enabled(True)
        child = harness.start_bridge()
        harness.wait_until(lambda: feedback.command_id > 0, timeout=10)
        harness.cycles(0.2)
        harness.control(child, grip=1.0, right_grip=1.0, trigger=0.05, right_trigger=0.05)
        harness.cycles(0.3)
        if workload == "moving":
            harness.control(child, benchmark_motion=True)
        harness.cycles(0.5)
        harness.control(child, reset_metrics=True)
        harness.cycles(0.06)
        completed.clear()
        robot_path = trace_directory / f"{prefix}-robot.json" if trace_directory else None
        if robot_path:
            feedback.trace = LatencyTrace(str(robot_path), duration_s=duration)
        before = feedback.command_counts.copy()
        started = time.monotonic()
        harness.cycles(duration)
        elapsed = time.monotonic() - started
        samples = list(completed)
        counts = {k: v - before[k] for k, v in feedback.command_counts.items()}
        report_path = Path(harness.directory) / "bridge-metrics.json"
        harness.control(child, report_path=str(report_path))
        harness.wait_until(report_path.is_file)
        bridge = json.loads(report_path.read_text())
        correlation = None
        if robot_path:
            feedback.trace.export()
            bridge_path.write_text(json.dumps(bridge["trace"]))
            correlation = analyze(feedback.trace.snapshot(), bridge["trace"])
        return {
            "workload": workload,
            "observe_all_completions": observe_all_completions,
            "correlation": correlation,
            "executor": executor,
            "collision_checking": collision,
            "elapsed_s": elapsed,
            "completed_hz": len(samples) / elapsed,
            "completion_gap_ms": distribution(1000 * np.diff([v["at"] for v in samples])),
            "robot_receipt_to_ack_ms": distribution(v["robot_ms"] for v in samples),
            "xr_sample_to_ack_ms": distribution(v["sample_to_ack_ms"] for v in samples),
            "bridge_publish_hz": bridge["published"] / bridge["elapsed_s"],
            "bridge_timing_ms": {k: distribution(v) for k, v in bridge["timing_ms"].items()},
            "command_counts": counts,
        }
    except Exception:
        print("\n".join(p.read_text() for p in Path(harness.directory).glob("*.log")), file=sys.stderr)
        raise
    finally:
        harness.doCleanups()
        for key, value in saved_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def feedback_decision(rows):
    pairs = {}
    for row in rows:
        pairs.setdefault((row["workload"], row["collision_checking"], row["trial"]), {})[
            row["observe_all_completions"]
        ] = row
    gains = []
    justified = []
    for pair in pairs.values():
        if set(pair) != {False, True} or not all(r["correlation"]["eligible_for_comparison"] for r in pair.values()):
            return {"adoption_recommended": False, "reason": "incomplete or invalid paired traces"}
        before, after = pair[False], pair[True]
        gains.append(
            before["correlation"]["metrics"]["sample_to_write_completion_ms"]["p95"]
            - after["correlation"]["metrics"]["sample_to_write_completion_ms"]["p95"]
        )
        justified.append(before["correlation"]["evidence"]["first_ack_absent"] > 0)
    guards = all(
        pair[True]["completed_hz"] >= 0.95 * pair[False]["completed_hz"]
        and pair[True]["command_counts"]["rejected"] <= pair[False]["command_counts"]["rejected"]
        and pair[True]["correlation"]["evidence"]["generation_resets"]
        <= pair[False]["correlation"]["evidence"]["generation_resets"]
        for pair in pairs.values()
    )
    passed = bool(gains) and min(gains) >= 5.0 and guards
    return {
        "adoption_recommended": passed,
        "paired_p95_response_gains_ms": gains,
        "buffer_experiment_justified": bool(justified) and all(justified),
        "reason": "response criterion met locally; Jetson validation still required"
        if passed
        else "no consistent >=5 ms response gain; keep execution and buffering defaults",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-python", required=True)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--hand-delay-ms", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("/tmp/xr-executor-benchmark.json"))
    parser.add_argument(
        "--feedback-comparison",
        action="store_true",
        help="Compare latest-only and earliest feedback observation on schema 3",
    )
    parser.add_argument("--workload", choices=("holding", "moving", "both"), default="both")
    parser.add_argument("--collision-checks", choices=("off", "on", "both"), default="both")
    parser.add_argument("--trace-directory", type=Path)
    args = parser.parse_args()
    if args.seconds <= 0 or args.trials <= 0 or args.hand_delay_ms < 0:
        parser.error("seconds and trials must be positive; hand delay must be nonnegative")
    os.environ["XROBOTOOLKIT_TEST_PYTHON"] = args.bridge_python
    os.environ["XROBOTOOLKIT_TEST_BACKEND"] = "lerobot"
    # Match the production wait loop, including feedback service between publishing ticks.
    integration.BRIDGE_PROCESS = integration.BRIDGE_PROCESS.replace(
        "time.sleep(max(0, deadline - time.monotonic()))", "bridge.wait_until_publish(deadline)"
    ).replace(
        'authoritative_executor=sys.argv[6] == "True")',
        """authoritative_executor=sys.argv[6] == "True",
                       observe_all_completions=os.environ.get("XROBOTOOLKIT_OBSERVE_ALL", "True") == "True",
                       latency_trace_path=os.environ.get("XROBOTOOLKIT_LATENCY_TRACE") or None,
                       latency_trace_duration_s=float(os.environ.get("XROBOTOOLKIT_TRACE_DURATION", "30")))""",
    )
    # Export bounded bridge diagnostics once per trial, outside the measured loop.
    integration.BRIDGE_PROCESS = integration.BRIDGE_PROCESS.replace("import json", "import json\nimport math", 1)
    integration.BRIDGE_PROCESS = integration.BRIDGE_PROCESS.replace(
        "    def get_timestamp_ns(self):", "    def get_timestamp_ns(self):\n        self.sampled_at = time.monotonic()"
    ).replace(
        "try:\n    deadline = time.monotonic()",
        """real_commands = bridge.commands
class TimestampedCommands:
    def send_json(self, message, **kwargs):
        message = dict(message)
        message["_benchmark_xr_sampled_at"] = getattr(xr, "sampled_at", time.monotonic())
        return real_commands.send_json(message, **kwargs)
bridge.commands = TimestampedCommands()
try:
    deadline = time.monotonic()""",
    )
    integration.BRIDGE_PROCESS = integration.BRIDGE_PROCESS.replace("+ 0.05", "+ 0.02").replace(
        "        bridge.poll()",
        """        if getattr(xr, "reset_metrics", False):
            bridge.trace.reset()
            bridge._timing_samples.clear()
            bridge._proposal_count = bridge._completion_count = 0
            benchmark_started = time.monotonic()
            xr.reset_metrics = False
        if getattr(xr, "benchmark_motion", False):
            if not hasattr(xr, "motion_started"):
                xr.motion_started = time.monotonic()
            phase = math.sin(math.pi * (time.monotonic() - xr.motion_started))
            xr.z = -0.005 + 0.003 * phase
            xr.trigger = 0.05 + 0.01 * phase
            xr.right_trigger = 0.05 - 0.01 * phase
        bridge.poll()
        if getattr(xr, "report_path", None):
            temporary = xr.report_path + ".tmp"
            with open(temporary, "w") as report:
                json.dump({"elapsed_s": time.monotonic() - benchmark_started,
                           "published": bridge._proposal_count,
                           "timing_ms": {k: list(v) for k, v in bridge._timing_samples.items()},
                           "trace": bridge.trace.snapshot() if bridge.trace.enabled else None}, report)
            os.replace(temporary, xr.report_path)
            xr.report_path = None""",
    )
    result = {
        "host": platform.platform(),
        "machine": platform.machine(),
        "hand_delay_ms": args.hand_delay_ms,
        "seconds_per_trial": args.seconds,
        "backend": "lerobot",
        "workload": "50 Hz XR sampling, both grips engaged, 0.5 Hz small arm/finger sinusoid",
        "trials": [],
    }
    if args.feedback_comparison:
        directory = args.trace_directory or args.output.with_suffix("")
        directory.mkdir(parents=True, exist_ok=True)
        result["comparison"] = "latest-only vs earliest received feedback observation; schema 3; send HWM remains 1"
        workloads = ("holding", "moving") if args.workload == "both" else (args.workload,)
        collisions = (False, True) if args.collision_checks == "both" else (args.collision_checks == "on",)
        for workload in workloads:
            for collision in collisions:
                for trial in range(args.trials):
                    for observe in (False, True) if trial % 2 == 0 else (True, False):
                        row = run_trial(
                            True,
                            collision,
                            args.seconds,
                            args.hand_delay_ms / 1000,
                            observe_all_completions=observe,
                            workload=workload,
                            trace_directory=directory,
                            trial=trial + 1,
                        )
                        row["trial"] = trial + 1
                        result["trials"].append(row)
                        args.output.write_text(json.dumps(result, indent=2))
                        print(
                            json.dumps(
                                {
                                    "workload": workload,
                                    "collision": collision,
                                    "earliest": observe,
                                    "trial": trial + 1,
                                    "correlation": row["correlation"],
                                }
                            ),
                            flush=True,
                        )
        result["decision"] = feedback_decision(result["trials"])
        args.output.write_text(json.dumps(result, indent=2))
        print(f"Results: {args.output}", flush=True)
        return
    for collision in (False, True):
        for trial in range(args.trials):
            # Alternate ordering to reduce warmup/order bias.
            for executor in (False, True) if trial % 2 == 0 else (True, False):
                row = run_trial(executor, collision, args.seconds, args.hand_delay_ms / 1000)
                row["trial"] = trial + 1
                result["trials"].append(row)
                args.output.write_text(json.dumps(result, indent=2))
                print(
                    json.dumps(
                        {
                            k: row[k]
                            for k in (
                                "executor",
                                "collision_checking",
                                "trial",
                                "completed_hz",
                                "robot_receipt_to_ack_ms",
                                "completion_gap_ms",
                            )
                        }
                    ),
                    flush=True,
                )
    print(f"Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
