import argparse
import json
import time

import rclpy
import zmq
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage


def parse_args():
    parser = argparse.ArgumentParser(description="Forward a ROS 2 CompressedImage topic to RoboJuDo recorder")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--node-name", required=True)
    parser.add_argument("--qos-reliability", choices=("best_effort", "reliable"), required=True)
    parser.add_argument("--qos-depth", type=int, required=True)
    return parser.parse_args()


class CompressedImageBridge(Node):
    def __init__(self, args):
        super().__init__(args.node_name)
        self._sequence = 0
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.PUSH)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SNDHWM, 2)
        self._socket.connect(args.endpoint)
        reliability = (
            ReliabilityPolicy.BEST_EFFORT
            if args.qos_reliability == "best_effort"
            else ReliabilityPolicy.RELIABLE
        )
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=args.qos_depth,
            reliability=reliability,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._subscription = self.create_subscription(CompressedImage, args.topic, self._on_image, qos)
        self.get_logger().info(f"Forwarding CompressedImage topic {args.topic} to {args.endpoint}")

    def _on_image(self, message):
        self._sequence += 1
        header = json.dumps(
            {"sequence": self._sequence, "timestamp_ns": time.monotonic_ns()},
            separators=(",", ":"),
        ).encode()
        try:
            self._socket.send_multipart([header, bytes(message.data)], flags=zmq.NOBLOCK)
        except zmq.Again:
            pass

    def destroy_node(self):
        self._socket.close(linger=0)
        return super().destroy_node()


def main():
    args = parse_args()
    rclpy.init(args=[])
    node = CompressedImageBridge(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
