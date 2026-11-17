#!/usr/bin/env python3
"""Join bounded robot/bridge traces without using telemetry for authorization."""

import argparse
import json
from pathlib import Path


def distribution(values):
    values = sorted(values)
    if not values:
        return {"n": 0}

    def percentile(fraction):
        index = (len(values) - 1) * fraction
        low = int(index)
        return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (index - low)

    return {"n": len(values), "mean": sum(values) / len(values), "p50": percentile(0.5), "p95": percentile(0.95)}


def frame_key(record):
    return tuple(record.get(k) for k in ("session_id", "generation", "bridge_id", "frame_id"))


def analyze(robot, bridge):
    reasons = []
    for label, trace in (("robot", robot), ("bridge", bridge)):
        if trace.get("version") != 1:
            raise ValueError(f"unsupported {label} trace version")
        if trace.get("overflow"):
            reasons.append(f"{label} trace overflow")
        if not trace.get("window_complete"):
            reasons.append(f"{label} trace window incomplete")
    domain = robot.get("clock_domain")
    same_clock = bool(
        domain
        and domain.get("boot_id")
        and domain.get("time_namespace")
        and domain.get("clock") == "monotonic"
        and domain == bridge.get("clock_domain")
    )
    if not same_clock:
        reasons.append("same-host monotonic clock domain not verified; cross-process intervals omitted")
    starts = [t.get("started_at") for t in (robot, bridge)]
    if any(v is None for v in starts):
        reasons.append("trace disabled")
        start = end = 0
    else:
        start = max(starts)
        end = min(t["started_at"] + t["duration_s"] for t in (robot, bridge))
    if end <= start:
        reasons.append("no overlapping measurement window")
    pubs = {
        frame_key(r): r for r in bridge["records"] if r["event"] == "command_published" and None not in frame_key(r)
    }
    acks = {
        frame_key(r): r for r in robot["records"] if r["event"] == "robot_acknowledged" and None not in frame_key(r)
    }
    received = {}
    observed = {}
    packets = {}
    for r in bridge["records"]:
        if r["event"] == "feedback_received":
            packets[(r.get("session_id"), r.get("generation"), r.get("state_id"))] = r
            if r.get("valid"):
                key = frame_key(r)
                if key not in received or r["at"] < received[key]["at"]:
                    received[key] = r
        elif r["event"] == "completion_observed":
            observed.setdefault(frame_key(r), r)
    sends = {}
    for r in robot["records"]:
        if r["event"] == "feedback_send" and r.get("accepted"):
            sends.setdefault(frame_key(r), r)
    metrics = {
        k: []
        for k in (
            "sample_to_write_completion_ms",
            "sample_to_combined_completion_ms",
            "robot_receipt_to_write_completion_ms",
            "write_completion_to_robot_ack_ms",
            "robot_ack_to_first_received_ms",
            "first_send_to_first_received_ms",
            "first_receive_to_observation_ms",
            "first_receive_to_metric_timestamp_ms",
            "acknowledgement_rtt_ms",
        )
    }
    evidence = dict(
        first_ack_packets=0,
        first_ack_absent=0,
        first_ack_received_invalid=0,
        earlier_valid_snapshot_skipped=0,
        missing_write_stamps=0,
        matched_frames=0,
    )
    for key, pub in pubs.items():
        ack, receipt = acks.get(key), received.get(key)
        if ack is None or receipt is None:
            continue
        if same_clock and not (start <= pub.get("sampled_at", pub["at"]) <= ack["at"] <= receipt["at"] <= end):
            continue
        evidence["matched_frames"] += 1
        metrics["acknowledgement_rtt_ms"].append(1000 * (receipt["at"] - pub["at"]))
        obs = observed.get(key)
        if obs:
            # Older traces used the event timestamp for metric attribution only.
            # Never present that value as the time the diagnostic code executed.
            metric_at = obs.get("metric_at", obs["at"])
            metrics["first_receive_to_metric_timestamp_ms"].append(1000 * (metric_at - receipt["at"]))
            if "metric_at" in obs:
                metrics["first_receive_to_observation_ms"].append(1000 * (obs["at"] - receipt["at"]))
            if obs.get("state_id") != receipt.get("state_id"):
                evidence["earlier_valid_snapshot_skipped"] += 1
        first = sends.get(key)
        if first:
            evidence["first_ack_packets"] += 1
            packet = packets.get((first.get("session_id"), first.get("generation"), first.get("state_id")))
            if packet is None:
                evidence["first_ack_absent"] += 1
            elif not packet.get("valid"):
                evidence["first_ack_received_invalid"] += 1
        stamps = ack.get("stamps", {})
        required = ("left_write_completed", "right_write_completed", "arm_submission_completed", "received")
        if not all(k in stamps for k in required):
            evidence["missing_write_stamps"] += 1
            continue
        completed = max(stamps["left_write_completed"], stamps["right_write_completed"])
        combined_completed = max(completed, stamps["arm_submission_completed"])
        intervals = {
            "robot_receipt_to_write_completion_ms": completed - stamps["received"],
            "write_completion_to_robot_ack_ms": ack["at"] - completed,
        }
        if same_clock:
            intervals.update(
                sample_to_write_completion_ms=completed - pub["sampled_at"],
                sample_to_combined_completion_ms=combined_completed - pub["sampled_at"],
                robot_ack_to_first_received_ms=receipt["at"] - ack["at"],
            )
            if first:
                intervals["first_send_to_first_received_ms"] = receipt["at"] - first["at"]
        for name, duration in intervals.items():
            if duration < 0:
                reasons.append("negative correlated interval; clock or trace ordering invalid")
            else:
                metrics[name].append(1000 * duration)
    generations = {
        (r.get("session_id"), r.get("generation"))
        for r in robot["records"]
        if r["event"] == "command_received" and start <= r["at"] <= end
    }
    evidence["generation_resets"] = max(0, len(generations) - 1)
    if not metrics["sample_to_write_completion_ms"]:
        reasons.append("no fully matched sample-to-write-completion measurements")
    return {
        "eligible_for_comparison": not reasons,
        "reasons": sorted(set(reasons)),
        "same_clock_verified": same_clock,
        "evidence": evidence,
        "metrics": {k: distribution(v) for k, v in metrics.items()},
        "note": "Absent packets are evidence for investigation, not proof of PUB queue loss.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot", type=Path)
    parser.add_argument("bridge", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(json.loads(args.robot.read_text()), json.loads(args.bridge.read_text()))
    output = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
