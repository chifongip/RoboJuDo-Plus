#!/usr/bin/env python3
"""Publish predefined X2 arm poses for testing UpperBodyZmqCtrl.

Start ``x2_locomanipulation`` or ``x2_locomanipulation_real``, enter
``RL_DEFAULT``, and enable the upper-body override before selecting poses here.
Targets are published continuously so the controller's freshness timeout stays
active.
"""

import argparse
import math
import select
import sys
import termios
import time
import tty

import zmq

from robojudo.config.x2.env.x2_env_cfg import X2_ARM_JOINT_NAMES, X2_POSITION_LIMITS_BY_NAME
from robojudo.config.x2.policy.x2_locomanipulation_policy_cfg import X2_LOCOMANIPULATION_DEFAULT_POS

DEFAULT_ARM_POSE = dict(zip(X2_ARM_JOINT_NAMES, X2_LOCOMANIPULATION_DEFAULT_POS[15:29], strict=True))


def pose(**overrides: float) -> dict[str, float]:
    result = DEFAULT_ARM_POSE.copy()
    result.update(overrides)
    return result


POSES = {
    "default": DEFAULT_ARM_POSE,
    "forward": pose(
        left_shoulder_pitch_joint=0.9,
        left_shoulder_roll_joint=0.15,
        left_elbow_joint=-1.0,
        right_shoulder_pitch_joint=0.9,
        right_shoulder_roll_joint=-0.15,
        right_elbow_joint=-1.0,
    ),
    "raised": pose(
        left_shoulder_pitch_joint=0.0,
        left_shoulder_roll_joint=1.2,
        left_elbow_joint=-0.8,
        right_shoulder_pitch_joint=0.0,
        right_shoulder_roll_joint=-1.2,
        right_elbow_joint=-0.8,
    ),
    "wide": pose(
        left_shoulder_pitch_joint=0.0,
        left_shoulder_roll_joint=0.8,
        left_elbow_joint=-0.2,
        right_shoulder_pitch_joint=0.0,
        right_shoulder_roll_joint=-0.8,
        right_elbow_joint=-0.2,
    ),
    "carry": pose(
        left_shoulder_pitch_joint=0.6,
        left_shoulder_roll_joint=0.2,
        left_elbow_joint=-1.5,
        left_wrist_pitch_joint=0.2,
        right_shoulder_pitch_joint=0.6,
        right_shoulder_roll_joint=-0.2,
        right_elbow_joint=-1.5,
        right_wrist_pitch_joint=0.2,
    ),
    "wave_left": pose(
        left_shoulder_pitch_joint=0.1,
        left_shoulder_roll_joint=1.2,
        left_elbow_joint=-1.5,
        left_wrist_roll_joint=0.5,
    ),
}

KEY_TO_POSE = {
    "0": "default",
    "1": "forward",
    "2": "raised",
    "3": "wide",
    "4": "carry",
    "5": "wave_left",
}


def validate_poses():
    expected_names = set(X2_ARM_JOINT_NAMES)
    for pose_name, positions in POSES.items():
        if set(positions) != expected_names:
            raise ValueError(f"Pose {pose_name!r} does not define all X2 arm joints")
        for joint_name, value in positions.items():
            if not math.isfinite(value):
                raise ValueError(f"Pose {pose_name!r} contains a non-finite value for {joint_name}")
            lower, upper = X2_POSITION_LIMITS_BY_NAME[joint_name]
            if not lower <= value <= upper:
                raise ValueError(
                    f"Pose {pose_name!r} value {joint_name}={value} is outside [{lower}, {upper}]"
                )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bind",
        default="tcp://*:8559",
        help="ZMQ PUB endpoint to bind (default: tcp://*:8559)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=50.0,
        help="Publish frequency in Hz (default: 50)",
    )
    parser.add_argument(
        "--pose",
        choices=POSES,
        default="default",
        help="Initial predefined pose (default: default)",
    )
    args = parser.parse_args()
    if args.rate <= 0.0:
        parser.error("--rate must be positive")
    if not sys.stdin.isatty():
        parser.error("an interactive terminal is required")
    return args


def print_controls():
    print("\nPredefined X2 arm poses:")
    for key, pose_name in KEY_TO_POSE.items():
        print(f"  {key} - {pose_name}")
    print("  q - stop publishing and quit")
    print("\nThe robot returns toward its default arm pose after publishing stops.\n")


def main():
    args = parse_args()
    validate_poses()

    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.LINGER, 0)
    try:
        publisher.bind(args.bind)
    except zmq.ZMQError as exc:
        publisher.close(linger=0)
        context.term()
        raise SystemExit(f"Failed to bind {args.bind}: {exc}") from None

    print(f"Upper-body test publisher bound to {args.bind}")
    print("Waiting for the RoboJuDo subscriber to connect...")
    time.sleep(0.5)
    print_controls()

    current_pose_name = args.pose
    print(f"Pose: {current_pose_name}")
    terminal_fd = sys.stdin.fileno()
    old_terminal_settings = termios.tcgetattr(terminal_fd)
    period = 1.0 / args.rate
    next_publish_at = time.monotonic()

    try:
        tty.setcbreak(terminal_fd)
        while True:
            if select.select([sys.stdin], [], [], 0.0)[0]:
                key = sys.stdin.read(1).lower()
                if key == "q":
                    break
                if key in KEY_TO_POSE:
                    current_pose_name = KEY_TO_POSE[key]
                    print(f"\nPose: {current_pose_name}")

            publisher.send_json(
                {
                    "positions": POSES[current_pose_name],
                    "source": "scripts/test_upper_body_zmq.py",
                    "pose": current_pose_name,
                }
            )
            next_publish_at += period
            sleep_s = next_publish_at - time.monotonic()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            else:
                next_publish_at = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(terminal_fd, termios.TCSADRAIN, old_terminal_settings)
        publisher.close(linger=0)
        context.term()
        print("\nStopped upper-body test publisher.")


if __name__ == "__main__":
    main()
