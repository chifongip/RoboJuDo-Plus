#!/usr/bin/env python3
"""Passively capture X2 state for an offline BeyondMimic safety replay.

This node creates subscriptions only. It never opens AimDK, creates a command
publisher, or sends a command to the robot.
"""

import argparse
import signal
import threading
import time
from pathlib import Path

import msgpack
import rclpy
from aimdk_msgs.msg import JointStateArray
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

TOPICS = {
    "/laser_odometry": ("odometry", Odometry),
    "/aima/hal/imu/torso/state": ("imu", Imu),
    "/aima/hal/joint/leg/state": ("joint", JointStateArray),
    "/aima/hal/joint/waist/state": ("joint", JointStateArray),
    "/aima/hal/joint/arm/state": ("joint", JointStateArray),
    "/aima/hal/joint/head/state": ("joint", JointStateArray),
}


def _stamp(message) -> float:
    stamp = message.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _vector3(vector) -> list[float]:
    return [float(vector.x), float(vector.y), float(vector.z)]


def _quaternion(quaternion) -> list[float]:
    return [float(quaternion.x), float(quaternion.y), float(quaternion.z), float(quaternion.w)]


class PassiveCapture(Node):
    def __init__(self, output: Path, duration: float):
        super().__init__("x2_beyondmimic_passive_capture")
        self.output = output
        self.duration = duration
        self.start_monotonic = time.monotonic()
        self.counts = {topic: 0 for topic in TOPICS}
        self._lock = threading.Lock()
        self._stream = output.open("wb")
        self._packer = msgpack.Packer(use_bin_type=True)
        self._closed = False
        self._write(
            {
                "kind": "metadata",
                "schema_version": 1,
                "started_unix": time.time(),
                "duration_requested": duration,
                "topics": {topic: kind for topic, (kind, _) in TOPICS.items()},
                "safety": "subscriber-only; no command publishers are created",
            }
        )
        for topic, (kind, message_type) in TOPICS.items():
            callback = self._callback(kind, topic)
            self.create_subscription(message_type, topic, callback, qos_profile_sensor_data)

    def _write(self, record: dict):
        with self._lock:
            self._stream.write(self._packer.pack(record))

    def _callback(self, kind: str, topic: str):
        def callback(message):
            receipt_time = time.monotonic() - self.start_monotonic
            common = {
                "kind": kind,
                "topic": topic,
                "receipt_time": receipt_time,
                "stamp": _stamp(message),
                "frame_id": message.header.frame_id,
            }
            if kind == "odometry":
                record = {
                    **common,
                    "child_frame_id": message.child_frame_id,
                    "position": _vector3(message.pose.pose.position),
                    "quaternion": _quaternion(message.pose.pose.orientation),
                    "pose_covariance": [float(value) for value in message.pose.covariance],
                    "linear_velocity": _vector3(message.twist.twist.linear),
                    "angular_velocity": _vector3(message.twist.twist.angular),
                    "twist_covariance": [float(value) for value in message.twist.covariance],
                }
            elif kind == "imu":
                record = {
                    **common,
                    "quaternion": _quaternion(message.orientation),
                    "angular_velocity": _vector3(message.angular_velocity),
                    "linear_acceleration": _vector3(message.linear_acceleration),
                    "orientation_covariance": [float(value) for value in message.orientation_covariance],
                }
            else:
                record = {
                    **common,
                    "measurement_stamp": (
                        float(message.header.meas_stamp.sec) + float(message.header.meas_stamp.nanosec) * 1e-9
                    ),
                    "joints": [
                        {
                            "name": joint.name,
                            "position": float(joint.position),
                            "velocity": float(joint.velocity),
                            "effort": float(joint.effort),
                            "error_code": int(joint.error_code),
                        }
                        for joint in message.joints
                    ],
                }
            self.counts[topic] += 1
            self._write(record)

        return callback

    def close(self):
        if self._closed:
            return
        self._closed = True
        elapsed = time.monotonic() - self.start_monotonic
        self._write({"kind": "summary", "duration": elapsed, "counts": self.counts})
        with self._lock:
            self._stream.flush()
            self._stream.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=10.0, help="capture duration in seconds")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("captures/x2_beyondmimic_state.msgpack"),
        help="portable msgpack output path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.duration <= 0.0:
        raise ValueError("--duration must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = PassiveCapture(args.output, args.duration)
    signal.signal(signal.SIGTERM, lambda *_: rclpy.shutdown())
    print("Passive capture started: this process creates subscriptions only.")
    print(f"Writing {args.duration:g} seconds to {args.output}")
    try:
        deadline = node.start_monotonic + args.duration
        while rclpy.ok():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            # Check the wall-clock deadline after every dispatched callback.
            # A ROS timer can be starved by the X2's several 500 Hz streams.
            rclpy.spin_once(node, timeout_sec=min(0.05, remaining))
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    print(f"Capture complete: {node.counts}")


if __name__ == "__main__":
    main()
