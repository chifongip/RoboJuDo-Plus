"""Interactive publisher for RoboJuDo's Locomanipulation posture ZMQ control."""

import argparse
import select
import sys
import termios
import time
import tty

import zmq

ROBOT_DEFAULTS = {
    "g1": {"height": 0.76, "height_limits": (0.5, 0.78)},
    "x2": {"height": 0.64, "height_limits": (0.3, 0.64)},
}
WAIST_YAW_LIMITS = (-1.5708, 1.5708)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="tcp://*:8557", help="ZMQ PUB endpoint to bind (default: tcp://*:8557)")
    parser.add_argument(
        "--robot",
        choices=sorted(ROBOT_DEFAULTS),
        default="g1",
        help="Set the initial height and limits",
    )
    parser.add_argument("--rate", type=float, default=20.0, help="Publish frequency in Hz (default: 20)")
    parser.add_argument("--height-step", type=float, default=0.02, help="Height increment in m")
    parser.add_argument("--waist-yaw-step", type=float, default=0.1, help="Waist-yaw increment in rad")
    args = parser.parse_args()
    if args.rate <= 0.0 or args.height_step <= 0.0 or args.waist_yaw_step <= 0.0:
        parser.error("rate and increments must be positive")
    if not sys.stdin.isatty():
        parser.error("an interactive terminal is required")
    return args


def posture(height: float, waist_yaw: float) -> dict:
    return {"height": height, "waist_yaw": waist_yaw}


def main():
    args = parse_args()
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.LINGER, 0)
    try:
        publisher.bind(args.bind)
    except zmq.ZMQError as exc:
        publisher.close(linger=0)
        context.term()
        raise SystemExit(f"Failed to bind {args.bind}: {exc}") from None

    defaults = ROBOT_DEFAULTS[args.robot]
    print(f"Posture publisher bound to {args.bind}; waiting for subscribers...")
    print("R/F: height up/down, Z/C: waist yaw left/right, X: reset, Q: quit")
    time.sleep(0.5)
    command = [defaults["height"], 0.0]
    terminal_fd = sys.stdin.fileno()
    old_terminal_settings = termios.tcgetattr(terminal_fd)
    period = 1.0 / args.rate

    try:
        tty.setcbreak(terminal_fd)
        while True:
            if select.select([sys.stdin], [], [], 0.0)[0]:
                key = sys.stdin.read(1).lower()
                if key == "q":
                    break
                if key == "r":
                    command[0] += args.height_step
                elif key == "f":
                    command[0] -= args.height_step
                elif key == "z":
                    command[1] += args.waist_yaw_step
                elif key == "c":
                    command[1] -= args.waist_yaw_step
                elif key == "x":
                    command[:] = [defaults["height"], 0.0]
                command[0] = min(max(command[0], defaults["height_limits"][0]), defaults["height_limits"][1])
                command[1] = min(max(command[1], WAIST_YAW_LIMITS[0]), WAIST_YAW_LIMITS[1])
                print(f"\rheight={command[0]:.3f} waist_yaw={command[1]:+.2f}", end="", flush=True)
            publisher.send_json(posture(*command))
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(terminal_fd, termios.TCSADRAIN, old_terminal_settings)
        publisher.close(linger=0)
        context.term()
        print("\nStopped posture publisher.")


if __name__ == "__main__":
    main()
