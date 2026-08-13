"""Monitor the AimDK streams that participate in X2 state freshness checks."""

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from robojudo.config.config_manager import ConfigManager
from robojudo.environment.agibot_cpp_env import format_state_freshness_report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", default="x2_real", help="Real X2 RoboJuDo config (default: x2_real)")
    parser.add_argument(
        "--output", type=Path, help="JSONL output path (default: timestamped file in the current directory)"
    )
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--append", action="store_true", help="append a new run to an existing output file")
    output_mode.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    parser.add_argument("--poll-rate", type=float, default=100.0, help="Freshness checks per second (default: 100)")
    parser.add_argument(
        "--summary-interval", type=float, default=1.0, help="Periodic summary interval in seconds (default: 1)"
    )
    args = parser.parse_args()
    if args.poll_rate <= 0.0:
        parser.error("--poll-rate must be positive")
    if args.summary_interval <= 0.0:
        parser.error("--summary-interval must be positive")
    return args


def report_to_dict(report) -> dict:
    return {
        "required_streams_fresh": report.required_streams_fresh,
        "reasons": list(report.reasons),
        "imu": {
            "received": report.imu_received,
            "age_sec": report.imu_age_sec,
        },
        "joints": {
            "missing": list(report.missing_joint_names),
            "stale": list(report.stale_joint_names),
            "age_sec": dict(report.joint_age_sec),
        },
        "odometry": {
            "required": report.odometry_required,
            "received": report.odometry_received,
            "valid": report.odometry_valid,
            "degenerate": report.odometry_degenerate,
            "age_sec": report.odometry_age_sec,
            "last_rejection_reason": report.last_odometry_rejection_reason or None,
            "last_rejection_age_sec": report.last_odometry_rejection_age_sec,
        },
        "stream_telemetry": {
            stream_name: stream_telemetry_to_dict(telemetry)
            for stream_name, telemetry in report.stream_telemetry.items()
        },
    }


def timestamp_to_dict(sec, nanosec) -> dict | None:
    if sec is None or nanosec is None:
        return None
    return {"sec": sec, "nanosec": nanosec}


def stream_telemetry_to_dict(telemetry) -> dict:
    return {
        "topic": telemetry.topic,
        "received_count": telemetry.received_count,
        "last_receive_age_sec": telemetry.last_receive_age_sec,
        "receive_rate_hz": telemetry.receive_rate_hz,
        "last_inter_arrival_sec": telemetry.last_inter_arrival_sec,
        "max_inter_arrival_sec": telemetry.max_inter_arrival_sec,
        "sequence_gap_count": telemetry.sequence_gap_count,
        "sequence_nonmonotonic_count": telemetry.sequence_nonmonotonic_count,
        "last_sequence": telemetry.last_sequence,
        "header_stamp": timestamp_to_dict(
            telemetry.last_header_stamp_sec, telemetry.last_header_stamp_nanosec
        ),
        "measurement_stamp": timestamp_to_dict(
            telemetry.last_measurement_stamp_sec, telemetry.last_measurement_stamp_nanosec
        ),
        "last_joint_names": list(telemetry.last_joint_names),
    }


def format_stream_telemetry(report) -> str:
    streams = []
    for stream_name, telemetry in report.stream_telemetry.items():
        rate = "n/a" if telemetry.receive_rate_hz is None else f"{telemetry.receive_rate_hz:.1f} Hz"
        age = "never" if telemetry.last_receive_age_sec is None else f"{telemetry.last_receive_age_sec:.3f}s"
        max_gap = "n/a" if telemetry.max_inter_arrival_sec is None else f"{telemetry.max_inter_arrival_sec:.3f}s"
        sequence = ""
        if telemetry.sequence_gap_count or telemetry.sequence_nonmonotonic_count:
            sequence = (
                f", sequence gaps={telemetry.sequence_gap_count}"
                f", nonmonotonic={telemetry.sequence_nonmonotonic_count}"
            )
        streams.append(f"{stream_name}: {rate}, age={age}, max_gap={max_gap}{sequence}")
    return " | ".join(streams)


def report_fingerprint(report) -> tuple:
    return (
        report.required_streams_fresh,
        tuple(report.reasons),
        tuple(report.missing_joint_names),
        tuple(report.stale_joint_names),
        report.odometry_valid,
        report.odometry_degenerate,
        report.last_odometry_rejection_reason,
    )


def build_controller_config(config_name: str) -> tuple[dict, object]:
    cfg = ConfigManager(config_name=config_name).get_cfg()
    cfg_env = cfg.env
    if cfg_env.env_type != "AgiBotCppEnv" or cfg.robot != "x2":
        raise ValueError(f"{config_name!r} is not a real X2 AimDK configuration")

    controller_cfg = cfg_env.aimdk.to_dict()
    controller_cfg.update(
        {
            "act": False,
            "node_name": f"robojudo_x2_state_monitor_{os.getpid()}",
            "enable_odometry": cfg_env.odometry_type in ("AIMDK", "SUPERODOM"),
            "joint_names": cfg_env.dof.joint_names,
            "leg_joint_names": cfg_env.leg_joint_names,
            "waist_joint_names": cfg_env.waist_joint_names,
            "arm_joint_names": cfg_env.arm_joint_names,
            "head_joint_names": cfg_env.head_joint_names,
            "stiffness": cfg_env.dof.stiffness,
            "damping": cfg_env.dof.damping,
        }
    )
    return controller_cfg, cfg_env


def write_event(output, event: str, **fields):
    record = {"timestamp": datetime.now(UTC).isoformat(), "event": event, **fields}
    output.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
    output.flush()


def default_output_path() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path(f"x2_state_monitor_{timestamp}.jsonl")


def open_output_file(path: Path, *, append: bool = False, overwrite: bool = False):
    mode = "a" if append else "w" if overwrite else "x"
    try:
        return path.open(mode, encoding="utf-8")
    except FileExistsError as exc:
        raise SystemExit(f"Output file already exists: {path}. Use --append or --overwrite explicitly.") from exc


def main():
    args = parse_args()
    output_path = args.output or default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from aimdk_cpp import AimdkController
    except ImportError as exc:
        raise SystemExit(
            "aimdk_cpp is unavailable; source ROS 2 and third_party/aimdk/install/setup.bash, "
            "then reinstall packages/aimdk_cpp"
        ) from exc

    try:
        controller_cfg, cfg_env = build_controller_config(args.config)
    except (AttributeError, KeyError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    state_timeout = float(cfg_env.aimdk.state_timeout)
    topics = {
        "leg": cfg_env.aimdk.leg_state_topic,
        "waist": cfg_env.aimdk.waist_state_topic,
        "arm": cfg_env.aimdk.arm_state_topic,
        "head": cfg_env.aimdk.head_state_topic,
        "imu": cfg_env.aimdk.base_imu_topic,
        "odometry": cfg_env.aimdk.odometry_topic if controller_cfg["enable_odometry"] else None,
    }

    with open_output_file(output_path, append=args.append, overwrite=args.overwrite) as output:
        controller = AimdkController(controller_cfg)
        poll_period = 1.0 / args.poll_rate
        previous_fingerprint = None
        next_summary = time.monotonic()
        print(f"Monitoring {args.config} AimDK state; writing {output_path}")
        print("The monitor is passive (act=False) and has no command publishers or publishing thread.")

        try:
            write_event(
                output,
                "startup",
                config=args.config,
                node_name=controller_cfg["node_name"],
                state_timeout_sec=state_timeout,
                odometry_timeout_sec=float(cfg_env.aimdk.odometry_timeout),
                telemetry_window_sec=float(controller_cfg.get("telemetry_window_sec", 1.0)),
                topics=topics,
                expected_joint_names=list(cfg_env.dof.joint_names),
            )
            while True:
                loop_start = time.monotonic()
                report = controller.get_state_freshness_report(state_timeout)
                fingerprint = report_fingerprint(report)
                if fingerprint != previous_fingerprint:
                    detail = format_state_freshness_report(report)
                    print(f"[{datetime.now().astimezone().isoformat(timespec='milliseconds')}] {detail}")
                    write_event(output, "state_change", detail=detail, report=report_to_dict(report))
                    previous_fingerprint = fingerprint

                if loop_start >= next_summary:
                    telemetry_summary = format_stream_telemetry(report)
                    print(f"[{datetime.now().astimezone().isoformat(timespec='milliseconds')}] {telemetry_summary}")
                    write_event(
                        output,
                        "snapshot",
                        telemetry_summary=telemetry_summary,
                        report=report_to_dict(report),
                    )
                    next_summary = loop_start + args.summary_interval

                remaining = poll_period - (time.monotonic() - loop_start)
                if remaining > 0.0:
                    time.sleep(remaining)
        except KeyboardInterrupt:
            print("Stopping monitor.")
            write_event(output, "shutdown", reason="keyboard_interrupt")
        finally:
            controller.shutdown()


if __name__ == "__main__":
    main()
