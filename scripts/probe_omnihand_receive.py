"""Sweep OmniHand SDK receive timeouts using read-only, serial-verified HCAN handles.

Stop other OmniHand clients before running. Each timeout runs in a separate SDK
process with concurrent left/right readers and a parent-enforced wall-clock limit.
No initialization or motion command is sent to either hand.
"""

import argparse
import csv
import json
import multiprocessing as mp
import threading
import time
from pathlib import Path

import numpy as np

from robojudo.controller.ctrl_cfgs import OmniHandCfg
from robojudo.controller.omnihand_runtime import connect_verified_hand


def read_hand(hand, side, duration, rate_hz, budget_ms):
    """Measure complete SDK reads, retaining the runtime's read-start timestamps."""
    rows = []
    started = time.monotonic()
    last_valid_start = last_valid_end = None
    while time.monotonic() - started < duration:
        before = time.monotonic()
        error = ""
        try:
            angles = np.asarray(hand.get_all_joint_angles(), dtype=float)
            if angles.shape != (19,) or not np.isfinite(angles).all():
                raise ValueError(f"expected 19 finite angles, received shape={angles.shape}")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        after = time.monotonic()
        valid = not error
        # At receipt of a new sample, the previous valid measurement was still
        # the latest available until this instant. Include failures and pacing.
        peak_age = None if last_valid_start is None else (after - last_valid_start) * 1000
        valid_gap = None if not valid or last_valid_end is None else (after - last_valid_end) * 1000
        rows.append(
            {
                "side": side,
                "elapsed_s": before - started,
                "read_ms": (after - before) * 1000,
                "valid": valid,
                "valid_gap_ms": valid_gap,
                "previous_measurement_age_ms": peak_age,
                "error": error,
            }
        )
        if valid:
            last_valid_start, last_valid_end = before, after
        time.sleep(max(0, 1 / rate_hz - (time.monotonic() - before)))
    elapsed = time.monotonic() - started
    good = [row for row in rows if row["valid"]]

    def percentiles(values):
        return (
            dict(zip(("p50", "p95", "p99", "max"), np.percentile(values, [50, 95, 99, 100]).tolist(), strict=True))
            if values
            else None
        )

    return {
        "side": side,
        "attempts": len(rows),
        "valid": len(good),
        "failures": len(rows) - len(good),
        "valid_hz": len(good) / elapsed,
        "read_ms": percentiles([row["read_ms"] for row in rows]),
        "valid_read_ms": percentiles([row["read_ms"] for row in good]),
        "valid_gap_ms": percentiles([row["valid_gap_ms"] for row in good if row["valid_gap_ms"] is not None]),
        "previous_measurement_age_ms": percentiles(
            [row["previous_measurement_age_ms"] for row in rows if row["previous_measurement_age_ms"] is not None]
        ),
        "reads_over_feedback_budget": sum(row["read_ms"] > budget_ms for row in rows),
        "measurement_age_over_feedback_budget": sum(
            row["previous_measurement_age_ms"] is not None and row["previous_measurement_age_ms"] > budget_ms
            for row in rows
        ),
        "rows": rows,
    }


def probe_worker(connection, config, timeout_ms, duration, rate_hz, budget_ms):
    try:
        cfg = OmniHandCfg.model_validate(config)
        # Verify identity at the original configured timeout, independently of
        # the candidate measurement timeout. Keep both handles alive together.
        hands = {side: connect_verified_hand(cfg, side) for side in ("left", "right")}
        for hand in hands.values():
            hand.set_frame_recv_timeout(timeout_ms)
        connection.send({"phase": "ready"})
        results = {}

        def reader(side):
            results[side] = read_hand(hands[side], side, duration, rate_hz, budget_ms)
            results[side]["reads_over_receive_timeout"] = sum(
                row["read_ms"] > timeout_ms for row in results[side]["rows"]
            )

        threads = [threading.Thread(target=reader, args=(side,)) for side in hands]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        connection.send({"status": "ok", "hands": [results[side] for side in hands]})
    except Exception as exc:
        connection.send({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        connection.close()


def run_probe(config, timeout_ms, duration, rate_hz, budget_ms, startup_timeout, grace):
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=probe_worker, args=(sender, config, timeout_ms, duration, rate_hz, budget_ms), daemon=True
    )
    process.start()
    sender.close()
    deadline = time.monotonic() + startup_timeout
    phase = "startup"
    try:
        while time.monotonic() < deadline:
            if receiver.poll(0.1):
                try:
                    message = receiver.recv()
                except EOFError:
                    break
                if message.get("phase") == "ready":
                    phase = "reads"
                    deadline = time.monotonic() + duration + grace
                else:
                    return {"timeout_ms": timeout_ms, **message}
        return {
            "timeout_ms": timeout_ms,
            "status": "error",
            "error": f"{phase} did not complete; worker exit code={process.exitcode}",
        }
    finally:
        receiver.close()
        process.join(timeout=0.5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join(timeout=2)
        process.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Same local serial JSON as check_omnihand_hcan.py")
    parser.add_argument("--timeouts-ms", type=int, nargs="+", default=[10, 20, 25, 50, 100, 200])
    parser.add_argument("--duration", type=float, default=10, help="Seconds per timeout, excluding connection")
    parser.add_argument("--rate-hz", type=float, default=100, help="Requested read rate per hand")
    parser.add_argument(
        "--feedback-budget-ms", type=float, default=100, help="Comparison budget; does not change control"
    )
    parser.add_argument("--startup-timeout", type=float, default=20)
    parser.add_argument(
        "--hang-grace", type=float, default=3, help="Additional seconds before terminating a stuck worker"
    )
    parser.add_argument("--output-prefix", type=Path, default=Path("/tmp/omnihand-receive"))
    args = parser.parse_args()
    for name in ("duration", "rate_hz", "feedback_budget_ms", "startup_timeout", "hang_grace"):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if any(not 10 <= value <= 1000 for value in args.timeouts_ms):
        parser.error("--timeouts-ms values must be between 10 and 1000")
    cfg = OmniHandCfg.model_validate(json.loads(args.config.read_text()))
    if cfg.transport != "hcan" or not all(
        getattr(cfg, f"{side}_{kind}_serial") for side in ("left", "right") for kind in ("hcan", "hand")
    ):
        parser.error("HCAN adapter and hand serials are required for both sides")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = Path(f"{args.output_prefix}.csv")
    json_path = Path(f"{args.output_prefix}.json")
    summaries = []
    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "timeout_ms",
                "side",
                "elapsed_s",
                "read_ms",
                "valid",
                "valid_gap_ms",
                "previous_measurement_age_ms",
                "error",
            ],
        )
        writer.writeheader()
        for timeout_ms in args.timeouts_ms:
            print(f"Testing {timeout_ms} ms receive timeout for {args.duration:g} s ...", flush=True)
            result = run_probe(
                cfg.model_dump(),
                timeout_ms,
                args.duration,
                args.rate_hz,
                args.feedback_budget_ms,
                args.startup_timeout,
                args.hang_grace,
            )
            for hand in result.get("hands", []):
                writer.writerows({"timeout_ms": timeout_ms, **row} for row in hand.pop("rows"))
            output.flush()
            summaries.append(result)
            json_path.write_text(
                json.dumps(
                    {
                        "duration_s": args.duration,
                        "requested_hz": args.rate_hz,
                        "feedback_budget_ms": args.feedback_budget_ms,
                        "results": summaries,
                    },
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            )
            print(json.dumps(result, allow_nan=False), flush=True)
    print(f"Saved {csv_path} and {json_path}")
    return int(
        any(
            result["status"] != "ok" or any(hand["failures"] or not hand["valid"] for hand in result["hands"])
            for result in summaries
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
