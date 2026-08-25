"""Passively stress X2 state delivery with multiple independent ROS 2 subscribers."""

import argparse
import json
import multiprocessing
import queue
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

DEFAULT_TOPICS = {
    "leg": "/aima/hal/joint/leg/state",
    "waist": "/aima/hal/joint/waist/state",
    "arm": "/aima/hal/joint/arm/state",
    "head": "/aima/hal/joint/head/state",
    "imu": "/aima/hal/imu/torso/state",
}


@dataclass
class StreamMetrics:
    received_count: int = 0
    receive_gap_count: int = 0
    header_gap_count: int = 0
    nonmonotonic_header_count: int = 0
    first_receive_ns: int | None = None
    last_receive_ns: int | None = None
    last_header_ns: int | None = None
    max_receive_gap_sec: float = 0.0
    max_header_gap_sec: float = 0.0

    def observe(self, receive_ns: int, header_ns: int, gap_threshold_sec: float) -> dict | None:
        gap_event = None
        if self.last_receive_ns is not None and self.last_header_ns is not None:
            receive_gap_sec = (receive_ns - self.last_receive_ns) / 1e9
            header_gap_sec = (header_ns - self.last_header_ns) / 1e9
            self.max_receive_gap_sec = max(self.max_receive_gap_sec, receive_gap_sec)
            self.max_header_gap_sec = max(self.max_header_gap_sec, header_gap_sec)
            if receive_gap_sec >= gap_threshold_sec:
                self.receive_gap_count += 1
                gap_event = {
                    "receive_gap_sec": receive_gap_sec,
                    "header_gap_sec": header_gap_sec,
                }
            if header_gap_sec >= gap_threshold_sec:
                self.header_gap_count += 1
            if header_gap_sec <= 0.0:
                self.nonmonotonic_header_count += 1

        if self.first_receive_ns is None:
            self.first_receive_ns = receive_ns
        self.last_receive_ns = receive_ns
        self.last_header_ns = header_ns
        self.received_count += 1
        return gap_event

    def summary(self) -> dict:
        elapsed_sec = 0.0
        if self.first_receive_ns is not None and self.last_receive_ns is not None:
            elapsed_sec = (self.last_receive_ns - self.first_receive_ns) / 1e9
        receive_rate_hz = None
        if elapsed_sec > 0.0 and self.received_count > 1:
            receive_rate_hz = (self.received_count - 1) / elapsed_sec
        return {**asdict(self), "receive_rate_hz": receive_rate_hz}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processes",
        type=int,
        default=4,
        help="independent subscriber processes/DDS participants (default: 4)",
    )
    parser.add_argument(
        "--copies-per-topic",
        type=int,
        default=1,
        help="subscriptions to each topic in every process (default: 1)",
    )
    parser.add_argument("--duration", type=float, default=60.0, help="measured duration in seconds (default: 60)")
    parser.add_argument("--warmup", type=float, default=3.0, help="discovery warmup in seconds (default: 3)")
    parser.add_argument(
        "--gap-threshold",
        type=float,
        default=0.1,
        help="receive/header gap event threshold in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--qos-depth",
        type=int,
        default=1,
        help="KEEP_LAST history depth (default: 1; use 10000 only to study backlog replay)",
    )
    parser.add_argument("--executor-threads", type=int, default=3, help="executor threads per process (default: 3)")
    parser.add_argument("--output", type=Path, help="optional JSONL event and summary output")
    args = parser.parse_args(argv)
    for name in ("processes", "copies_per_topic", "qos_depth", "executor_threads"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("duration", "gap_threshold"):
        if getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0.0:
        parser.error("--warmup cannot be negative")
    return args


def header_nanoseconds(message) -> int:
    return message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec


def subscriber_worker(worker_id: int, settings: dict, event_queue) -> None:
    try:
        import rclpy
        from aimdk_msgs.msg import JointStateArray
        from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Imu

        rclpy.init(args=None)
        node = Node(f"x2_state_subscriber_stress_{worker_id}_{multiprocessing.current_process().pid}")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=settings["qos_depth"],
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        metrics = {
            f"{stream}_{copy}": StreamMetrics()
            for stream in DEFAULT_TOPICS
            for copy in range(settings["copies_per_topic"])
        }
        callback_groups = {stream: MutuallyExclusiveCallbackGroup() for stream in DEFAULT_TOPICS}
        subscriptions = []
        measuring = False

        def callback(message, stream_key):
            if not measuring:
                return
            gap = metrics[stream_key].observe(
                time.monotonic_ns(), header_nanoseconds(message), settings["gap_threshold"]
            )
            if gap is not None:
                event_queue.put(
                    {
                        "event": "gap",
                        "worker": worker_id,
                        "stream": stream_key,
                        **gap,
                    }
                )

        for stream, topic in DEFAULT_TOPICS.items():
            message_type = Imu if stream == "imu" else JointStateArray
            for copy in range(settings["copies_per_topic"]):
                stream_key = f"{stream}_{copy}"
                subscriptions.append(
                    node.create_subscription(
                        message_type,
                        topic,
                        lambda message, key=stream_key: callback(message, key),
                        qos,
                        callback_group=callback_groups[stream],
                    )
                )

        executor = MultiThreadedExecutor(num_threads=settings["executor_threads"])
        executor.add_node(node)
        event_queue.put({"event": "ready", "worker": worker_id, "node": node.get_name()})
        warmup_deadline = time.monotonic() + settings["warmup"]
        while rclpy.ok() and time.monotonic() < warmup_deadline:
            executor.spin_once(timeout_sec=0.05)

        measuring = True
        deadline = time.monotonic() + settings["duration"]
        while rclpy.ok() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.05)

        event_queue.put(
            {
                "event": "summary",
                "worker": worker_id,
                "streams": {name: value.summary() for name, value in metrics.items()},
            }
        )
        executor.shutdown()
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()
    except BaseException as exc:
        event_queue.put({"event": "error", "worker": worker_id, "error": f"{type(exc).__name__}: {exc}"})
        raise


def timestamped(record: dict) -> dict:
    return {"timestamp": datetime.now().astimezone().isoformat(), **record}


def main(argv=None) -> int:
    args = parse_args(argv)
    settings = {
        "copies_per_topic": args.copies_per_topic,
        "duration": args.duration,
        "executor_threads": args.executor_threads,
        "gap_threshold": args.gap_threshold,
        "qos_depth": args.qos_depth,
        "warmup": args.warmup,
    }
    context = multiprocessing.get_context("spawn")
    event_queue = context.Queue()
    processes = [
        context.Process(target=subscriber_worker, args=(worker_id, settings, event_queue))
        for worker_id in range(args.processes)
    ]
    output = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output = args.output.open("x", encoding="utf-8")

    def emit(record):
        stamped = timestamped(record)
        if output is not None:
            output.write(json.dumps(stamped, separators=(",", ":"), sort_keys=True) + "\n")
            output.flush()
        if record["event"] == "gap":
            print(
                f"worker={record['worker']} stream={record['stream']} "
                f"receive_gap={record['receive_gap_sec']:.6f}s "
                f"header_gap={record['header_gap_sec']:.6f}s",
                flush=True,
            )
        elif record["event"] in {"ready", "error"}:
            print(json.dumps(record, sort_keys=True), flush=True)

    print(
        f"Starting {args.processes} passive processes x {args.copies_per_topic} copies x "
        f"{len(DEFAULT_TOPICS)} topics; QoS depth={args.qos_depth}. No publishers are created.",
        flush=True,
    )
    for process in processes:
        process.start()

    summaries = []
    try:
        while any(process.is_alive() for process in processes):
            try:
                record = event_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            emit(record)
            if record["event"] == "summary":
                summaries.append(record)
    except KeyboardInterrupt:
        print("Stopping subscriber processes.", flush=True)
        for process in processes:
            process.terminate()
    finally:
        for process in processes:
            process.join()
        while True:
            try:
                record = event_queue.get_nowait()
            except queue.Empty:
                break
            emit(record)
            if record["event"] == "summary":
                summaries.append(record)
        if output is not None:
            output.close()

    failed = [process for process in processes if process.exitcode != 0]
    observed = [
        (summary["worker"], stream, values)
        for summary in summaries
        for stream, values in summary["streams"].items()
        if values["receive_gap_count"]
    ]
    all_streams = [values for summary in summaries for values in summary["streams"].values()]
    received_streams = [values for values in all_streams if values["received_count"]]
    max_receive_gap_sec = max((values["max_receive_gap_sec"] for values in received_streams), default=0.0)
    max_header_gap_sec = max((values["max_header_gap_sec"] for values in received_streams), default=0.0)
    min_receive_rate_hz = min(
        (values["receive_rate_hz"] for values in received_streams if values["receive_rate_hz"] is not None),
        default=None,
    )
    print(
        f"Completed workers={len(summaries)}/{args.processes}; streams_with_receive_gaps={len(observed)}; "
        f"worker_failures={len(failed)}; streams_receiving={len(received_streams)}/{len(all_streams)}; "
        f"min_rate={'n/a' if min_receive_rate_hz is None else f'{min_receive_rate_hz:.1f}Hz'}; "
        f"max_receive_gap={max_receive_gap_sec:.6f}s; max_header_gap={max_header_gap_sec:.6f}s",
        flush=True,
    )
    for worker, stream, values in observed:
        print(
            f"worker={worker} stream={stream} rate={values['receive_rate_hz']:.1f}Hz "
            f"gaps={values['receive_gap_count']} max_receive_gap={values['max_receive_gap_sec']:.6f}s "
            f"max_header_gap={values['max_header_gap_sec']:.6f}s",
            flush=True,
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
