#!/usr/bin/env python3
"""Publish predefined X2 or G1 arm poses for testing UpperBodyZmqCtrl.

Start the matching Locomanipulation configuration, enter ``RL_DEFAULT``, and
enable the upper-body override before selecting poses here. Targets are
published continuously so the controller's freshness timeout stays active.
"""

import argparse
import math
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass

import zmq

from robojudo.config.g1.policy.g1_locomanipulation_policy_cfg import (
    G1Locomanipulation23ObsDoF,
    G1Locomanipulation29ObsDoF,
)
from robojudo.config.x2.env.x2_env_cfg import X2_ARM_JOINT_NAMES, X2_POSITION_LIMITS_BY_NAME
from robojudo.config.x2.policy.x2_locomanipulation_policy_cfg import X2_LOCOMANIPULATION_DEFAULT_POS


@dataclass(frozen=True)
class TestProfile:
    label: str
    joint_names: tuple[str, ...]
    position_limits: dict[str, tuple[float, float]]
    poses: dict[str, dict[str, float]]


def make_poses(default_pose: dict[str, float], elbow_sign: float) -> dict[str, dict[str, float]]:
    def pose(**overrides: float) -> dict[str, float]:
        result = default_pose.copy()
        result.update(overrides)
        return result

    carry_overrides = {
        "left_shoulder_pitch_joint": 0.6,
        "left_shoulder_roll_joint": 0.2,
        "left_elbow_joint": elbow_sign * 1.5,
        "right_shoulder_pitch_joint": 0.6,
        "right_shoulder_roll_joint": -0.2,
        "right_elbow_joint": elbow_sign * 1.5,
    }
    for joint_name in ("left_wrist_pitch_joint", "right_wrist_pitch_joint"):
        if joint_name in default_pose:
            carry_overrides[joint_name] = 0.2

    return {
        "default": default_pose,
        "forward": pose(
            left_shoulder_pitch_joint=0.9,
            left_shoulder_roll_joint=0.15,
            left_elbow_joint=elbow_sign * 1.0,
            right_shoulder_pitch_joint=0.9,
            right_shoulder_roll_joint=-0.15,
            right_elbow_joint=elbow_sign * 1.0,
        ),
        "raised": pose(
            left_shoulder_pitch_joint=0.0,
            left_shoulder_roll_joint=1.2,
            left_elbow_joint=elbow_sign * 0.8,
            right_shoulder_pitch_joint=0.0,
            right_shoulder_roll_joint=-1.2,
            right_elbow_joint=elbow_sign * 0.8,
        ),
        "wide": pose(
            left_shoulder_pitch_joint=0.0,
            left_shoulder_roll_joint=0.8,
            left_elbow_joint=elbow_sign * 0.2,
            right_shoulder_pitch_joint=0.0,
            right_shoulder_roll_joint=-0.8,
            right_elbow_joint=elbow_sign * 0.2,
        ),
        "carry": pose(**carry_overrides),
        "wave_left": pose(
            left_shoulder_pitch_joint=0.1,
            left_shoulder_roll_joint=1.2,
            left_elbow_joint=elbow_sign * 1.5,
            left_wrist_roll_joint=0.5,
        ),
    }


def make_profile(label: str, joint_names: list[str], default_pos: list[float], position_limits) -> TestProfile:
    names = tuple(joint_names)
    default_pose = dict(zip(names, default_pos, strict=True))
    limits = {
        name: (float(limit[0]), float(limit[1]))
        for name, limit in zip(names, position_limits, strict=True)
    }
    elbow_sign = -1.0 if label == "X2" else 1.0
    poses = make_poses(default_pose, elbow_sign)
    if label == "X2":
        poses["mirrored_arms"] = {
            **default_pose,
            "left_shoulder_pitch_joint": 0.3792,
            "left_shoulder_roll_joint": -0.0129,
            "left_shoulder_yaw_joint": -0.5437,
            "left_elbow_joint": -1.7308,
            "left_wrist_pitch_joint": -0.4580,
            "left_wrist_roll_joint": -0.4,
            "left_wrist_yaw_joint": 1.6873,
            "right_shoulder_pitch_joint": 0.3792,
            "right_shoulder_roll_joint": 0.0129,
            "right_shoulder_yaw_joint": 0.5437,
            "right_elbow_joint": -1.7308,
            "right_wrist_pitch_joint": -0.4580,
            "right_wrist_roll_joint": 0.4,
            "right_wrist_yaw_joint": -1.6873,
        }
    return TestProfile(
        label=label,
        joint_names=names,
        position_limits=limits,
        poses=poses,
    )


def make_g1_profile(label: str, dof) -> TestProfile:
    upper_start = dof.joint_names.index("left_shoulder_pitch_joint")
    return make_profile(
        label,
        dof.joint_names[upper_start:],
        dof.default_pos[upper_start:],
        dof.position_limits[upper_start:],
    )


PROFILES = {
    "x2": make_profile(
        "X2",
        X2_ARM_JOINT_NAMES,
        X2_LOCOMANIPULATION_DEFAULT_POS[15:29],
        [X2_POSITION_LIMITS_BY_NAME[name] for name in X2_ARM_JOINT_NAMES],
    ),
    "g1-23": make_g1_profile("G1 23-DOF", G1Locomanipulation23ObsDoF()),
    "g1-29": make_g1_profile("G1 29-DOF", G1Locomanipulation29ObsDoF()),
}

KEY_TO_POSE = {
    "0": "default",
    "1": "forward",
    "2": "raised",
    "3": "wide",
    "4": "carry",
    "5": "wave_left",
    "6": "mirrored_arms",
}


def validate_profile(profile: TestProfile):
    expected_names = set(profile.joint_names)
    for pose_name, positions in profile.poses.items():
        if set(positions) != expected_names:
            raise ValueError(f"Pose {pose_name!r} does not define all {profile.label} arm joints")
        for joint_name, value in positions.items():
            if not math.isfinite(value):
                raise ValueError(f"Pose {pose_name!r} contains a non-finite value for {joint_name}")
            lower, upper = profile.position_limits[joint_name]
            if not lower <= value <= upper:
                raise ValueError(
                    f"Pose {pose_name!r} value {joint_name}={value} is outside [{lower}, {upper}]"
                )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot",
        choices=PROFILES,
        default="x2",
        help="Joint layout to publish for (default: x2)",
    )
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
        default="default",
        help="Initial predefined pose (default: default)",
    )
    args = parser.parse_args()
    if args.rate <= 0.0:
        parser.error("--rate must be positive")
    if not sys.stdin.isatty():
        parser.error("an interactive terminal is required")
    return args


def print_controls(profile: TestProfile):
    print(f"\nPredefined {profile.label} arm poses:")
    for key, pose_name in KEY_TO_POSE.items():
        if pose_name in profile.poses:
            print(f"  {key} - {pose_name}")
    print("  q - stop publishing and quit")
    print("\nThe robot returns toward its default arm pose after publishing stops.\n")


def main():
    args = parse_args()
    profile = PROFILES[args.robot]
    if args.pose not in profile.poses:
        raise SystemExit(f"Pose {args.pose!r} is not available for {profile.label}")
    validate_profile(profile)

    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.LINGER, 0)
    try:
        publisher.bind(args.bind)
    except zmq.ZMQError as exc:
        publisher.close(linger=0)
        context.term()
        raise SystemExit(f"Failed to bind {args.bind}: {exc}") from None

    print(f"Upper-body test publisher for {profile.label} bound to {args.bind}")
    print("Waiting for the RoboJuDo subscriber to connect...")
    time.sleep(0.5)
    print_controls(profile)

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
                if key in KEY_TO_POSE and KEY_TO_POSE[key] in profile.poses:
                    current_pose_name = KEY_TO_POSE[key]
                    print(f"\nPose: {current_pose_name}")

            publisher.send_json(
                {
                    "positions": profile.poses[current_pose_name],
                    "source": "scripts/test_upper_body_zmq.py",
                    "robot": args.robot,
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
