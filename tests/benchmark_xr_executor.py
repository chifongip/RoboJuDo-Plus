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

import test_xrobotoolkit_integration as integration  # noqa: E402


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

    def get_data(self):
        if self.pending is not None:
            positions, identity, ready_at, deadline = self.pending
            if time.monotonic() >= ready_at and ready_at <= deadline:
                self.env.submit_hand_targets(positions)
                self.applied, self.pending = identity, None
        snapshot = self.env.hand_snapshot()
        return dict(snapshot, joint_state_fresh=snapshot["fresh"], applied=self.applied)

    def submit(self, positions, identity, deadline):
        if self.pending is not None:
            raise AssertionError("combined frame overwritten while in flight")
        self.applied = None
        self.pending = positions.copy(), identity, time.monotonic() + self.delay, deadline

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

    def close(self):
        self.reset()


class Harness(integration.XrobotoolkitIntegrationTests):
    with_omnihand = True

    def setUp(self):
        super().setUp()
        for cfg in (self.pipeline._upper_body_cfg, self.receiver.cfg_ctrl):
            cfg.hand_command_mode = "direct"
            cfg.omnihand.direct_completion_notifications = True


def run_trial(executor, collision, duration, hand_delay):
    harness = Harness(methodName="runTest")
    harness.authoritative_executor = executor
    harness.collision_checking = collision
    completed = []
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
        harness.control(child, benchmark_motion=True)
        harness.cycles(0.5)
        harness.control(child, reset_metrics=True)
        harness.cycles(0.06)
        completed.clear()
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
        return {
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-python", required=True)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--hand-delay-ms", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("/tmp/xr-executor-benchmark.json"))
    args = parser.parse_args()
    if args.seconds <= 0 or args.trials <= 0 or args.hand_delay_ms < 0:
        parser.error("seconds and trials must be positive; hand delay must be nonnegative")
    os.environ["XROBOTOOLKIT_TEST_PYTHON"] = args.bridge_python
    os.environ["XROBOTOOLKIT_TEST_BACKEND"] = "lerobot"
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
                           "timing_ms": {k: list(v) for k, v in bridge._timing_samples.items()}}, report)
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
