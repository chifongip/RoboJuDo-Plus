import argparse
import json
import time

import rclpy
import zmq
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image


def parse_args():
    parser = argparse.ArgumentParser(description="Forward a ROS 2 image topic to RoboJuDo recorder")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--message-type", choices=("compressed", "raw"), default="compressed")
    parser.add_argument("--node-name", required=True)
    parser.add_argument("--qos-reliability", choices=("best_effort", "reliable"), required=True)
    parser.add_argument("--qos-depth", type=int, required=True)
    return parser.parse_args()


class ImageBridge(Node):
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
        ros_message_type = CompressedImage if args.message_type == "compressed" else Image
        self._message_type = args.message_type
        self._subscription = self.create_subscription(ros_message_type, args.topic, self._on_image, qos)
        self.get_logger().info(f"Forwarding {ros_message_type.__name__} topic {args.topic} to {args.endpoint}")

    def _on_image(self, message):
        self._sequence += 1
        metadata = {
            "message_type": self._message_type,
            "sequence": self._sequence,
            "timestamp_ns": time.monotonic_ns(),
        }
        if self._message_type == "raw":
            metadata.update(
                height=message.height,
                width=message.width,
                encoding=message.encoding,
                is_bigendian=message.is_bigendian,
                step=message.step,
            )
        header = json.dumps(metadata, separators=(",", ":")).encode()
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
    node = ImageBridge(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
