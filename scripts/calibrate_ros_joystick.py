#!/usr/bin/env python3
"""Interactively identify the raw joy_node layout of a game controller."""

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

COMMON_CONTROLS = [
    ("face_south", "south face button ({south})"),
    ("face_east", "east face button ({east})"),
    ("face_west", "west face button ({west})"),
    ("face_north", "north face button ({north})"),
    ("left_shoulder", "left shoulder button ({left_shoulder})"),
    ("right_shoulder", "right shoulder button ({right_shoulder})"),
    ("back", "back/create button ({back})"),
    ("start", "start/options button ({start})"),
    ("guide", "guide/system button ({guide})"),
    ("left_stick_click", "left stick click ({left_stick_click})"),
    ("right_stick_click", "right stick click ({right_stick_click})"),
    ("misc_button_1", "share/touchpad or other extra button"),
    ("misc_button_2", "mute or second extra button"),
    ("dpad_up", "D-pad Up"),
    ("dpad_down", "D-pad Down"),
    ("dpad_left", "D-pad Left"),
    ("dpad_right", "D-pad Right"),
    ("left_trigger", "left trigger fully ({left_trigger})"),
    ("right_trigger", "right trigger fully ({right_trigger})"),
    ("left_stick_right", "left stick fully Right"),
    ("left_stick_up", "left stick fully Up"),
    ("right_stick_right", "right stick fully Right"),
    ("right_stick_up", "right stick fully Up"),
]

LABELS = {
    "xbox": {
        "south": "A",
        "east": "B",
        "west": "X",
        "north": "Y",
        "left_shoulder": "LB",
        "right_shoulder": "RB",
        "back": "View",
        "start": "Menu",
        "guide": "Xbox",
        "left_stick_click": "L3",
        "right_stick_click": "R3",
        "left_trigger": "LT",
        "right_trigger": "RT",
    },
    "ps5": {
        "south": "Cross",
        "east": "Circle",
        "west": "Square",
        "north": "Triangle",
        "left_shoulder": "L1",
        "right_shoulder": "R1",
        "back": "Create",
        "start": "Options",
        "guide": "PS",
        "left_stick_click": "L3",
        "right_stick_click": "R3",
        "left_trigger": "L2",
        "right_trigger": "R2",
    },
}


@dataclass
class AxisObservation:
    index: int
    neutral: float
    observed_min: float
    observed_max: float
    peak_value: float
    peak_delta: float


@dataclass
class ButtonObservation:
    index: int
    neutral: int
    observed_values: list[int]


def controls_for_profile(profile: str) -> list[tuple[str, str]]:
    labels = LABELS[profile]
    return [(name, prompt.format(**labels)) for name, prompt in COMMON_CONTROLS]


def analyze_samples(neutral_axes, neutral_buttons, samples, axis_threshold: float) -> dict:
    """Return every raw axis/button that moved away from the neutral sample."""
    max_axes = max([len(neutral_axes), *(len(sample.axes) for sample in samples)], default=0)
    max_buttons = max([len(neutral_buttons), *(len(sample.buttons) for sample in samples)], default=0)

    axes = []
    for index in range(max_axes):
        neutral = float(neutral_axes[index]) if index < len(neutral_axes) else 0.0
        values = [float(sample.axes[index]) for sample in samples if index < len(sample.axes)]
        if not values:
            continue
        observed_min = min(values)
        observed_max = max(values)
        peak_value = max(values, key=lambda value: abs(value - neutral))
        peak_delta = peak_value - neutral
        if abs(peak_delta) >= axis_threshold:
            axes.append(
                asdict(
                    AxisObservation(
                        index=index,
                        neutral=neutral,
                        observed_min=observed_min,
                        observed_max=observed_max,
                        peak_value=peak_value,
                        peak_delta=peak_delta,
                    )
                )
            )

    buttons = []
    for index in range(max_buttons):
        neutral = int(neutral_buttons[index]) if index < len(neutral_buttons) else 0
        values = {int(sample.buttons[index]) for sample in samples if index < len(sample.buttons)}
        changed_values = sorted(value for value in values if value != neutral)
        if changed_values:
            buttons.append(
                asdict(ButtonObservation(index=index, neutral=neutral, observed_values=changed_values))
            )

    return {"buttons": buttons, "axes": axes}


def has_activity(neutral_axes, neutral_buttons, sample, axis_threshold: float) -> bool:
    observation = analyze_samples(neutral_axes, neutral_buttons, [sample], axis_threshold)
    return bool(observation["buttons"] or observation["axes"])


def poll_samples(subscriber):
    result = subscriber.poll()
    return list(result.samples)


def wait_for_sample(subscriber, timeout_s: float):
    deadline = time.monotonic() + timeout_s
    latest = None
    while time.monotonic() < deadline:
        samples = poll_samples(subscriber)
        if samples:
            latest = samples[-1]
            break
        time.sleep(0.02)
    return latest


def capture_control(
    subscriber,
    neutral_axes,
    neutral_buttons,
    timeout_s: float,
    capture_window_s: float,
    axis_threshold: float,
):
    deadline = time.monotonic() + timeout_s
    samples = []
    capture_deadline = None
    while time.monotonic() < deadline:
        new_samples = poll_samples(subscriber)
        if new_samples:
            samples.extend(new_samples)
            if capture_deadline is None and any(
                has_activity(neutral_axes, neutral_buttons, sample, axis_threshold) for sample in new_samples
            ):
                capture_deadline = time.monotonic() + capture_window_s
        if capture_deadline is not None and time.monotonic() >= capture_deadline:
            break
        time.sleep(0.01)
    return analyze_samples(neutral_axes, neutral_buttons, samples, axis_threshold)


def print_observation(observation: dict):
    if not observation["buttons"] and not observation["axes"]:
        print("  No input change detected.")
        return
    for button in observation["buttons"]:
        print(
            f"  buttons[{button['index']}]: neutral={button['neutral']} "
            f"observed={button['observed_values']}"
        )
    for axis in observation["axes"]:
        print(
            f"  axes[{axis['index']}]: neutral={axis['neutral']:+.3f} "
            f"peak={axis['peak_value']:+.3f} delta={axis['peak_delta']:+.3f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(LABELS), required=True, help="Physical controller family")
    parser.add_argument("--connection", choices=("usb", "bluetooth", "other"), default="usb")
    parser.add_argument("--topic", default="/joy")
    parser.add_argument("--output", type=Path, help="Output JSON path")
    parser.add_argument("--timeout", type=float, default=8.0, help="Seconds to wait for each input")
    parser.add_argument("--message-timeout", type=float, default=10.0, help="Seconds to wait for the first Joy message")
    parser.add_argument("--capture-window", type=float, default=0.6, help="Seconds to capture after activity begins")
    parser.add_argument("--axis-threshold", type=float, default=0.35, help="Minimum axis delta to record")
    args = parser.parse_args()
    if args.timeout <= 0 or args.message_timeout <= 0 or args.capture_window <= 0:
        parser.error("timeout values must be positive")
    if not 0 < args.axis_threshold <= 1:
        parser.error("--axis-threshold must be in (0, 1]")
    return args


def main():
    args = parse_args()
    try:
        from ros2_joy_cpp import JoySubscriber
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "The native ros2_joy_cpp package is unavailable. Source ROS 2 and run "
            "`python submodule_install.py ros2_joy_cpp`."
        ) from exc

    output = args.output or Path(f"ros_joy_{args.profile}_{args.connection}_calibration.json")
    subscriber = JoySubscriber(topic=args.topic, queue_capacity=1024)
    results = {}
    try:
        print(f"Listening on {args.topic} for a {args.profile} controller ({args.connection}).")
        print("Release every controller input, then press Enter to capture the neutral state.")
        input()
        neutral = wait_for_sample(subscriber, args.message_timeout)
        if neutral is None:
            raise RuntimeError(
                f"No Joy messages arrived on {args.topic}. Confirm that `ros2 run joy joy_node` is running."
            )

        neutral_axes = [float(value) for value in neutral.axes]
        neutral_buttons = [int(value) for value in neutral.buttons]
        print(f"Neutral sample: {len(neutral_axes)} axes, {len(neutral_buttons)} buttons")
        print("For each prompt: release all controls, press Enter, then operate only the requested control.")
        print("Type s to skip a control or q to save the partial calibration and finish.\n")

        for name, prompt in controls_for_profile(args.profile):
            response = input(f"Prepare {prompt}. Press Enter to capture [s=skip, q=finish]: ").strip().lower()
            if response == "q":
                break
            if response == "s":
                results[name] = {"skipped": True, "buttons": [], "axes": []}
                continue
            poll_samples(subscriber)  # discard samples collected while preparing
            print(f"  Now operate {prompt}...")
            observation = capture_control(
                subscriber,
                neutral_axes,
                neutral_buttons,
                args.timeout,
                args.capture_window,
                args.axis_threshold,
            )
            results[name] = observation
            print_observation(observation)
    except KeyboardInterrupt:
        print("\nInterrupted; saving the completed controls.")
    finally:
        subscriber.close()

    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "connection": args.connection,
        "topic": args.topic,
        "ros_distro": os.environ.get("ROS_DISTRO"),
        "platform": platform.platform(),
        "python": sys.version,
        "neutral": {
            "axes": neutral_axes if "neutral_axes" in locals() else [],
            "buttons": neutral_buttons if "neutral_buttons" in locals() else [],
        },
        "controls": results,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Calibration saved to {output.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
