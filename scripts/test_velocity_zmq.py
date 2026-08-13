"""Interactive publisher for RoboJuDo's ROS Twist-shaped ZMQ velocity control."""

import argparse
import select
import sys
import termios
import time
import tty

import zmq


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="tcp://*:8558", help="ZMQ PUB endpoint to bind (default: tcp://*:8558)")
    parser.add_argument("--rate", type=float, default=20.0, help="Publish frequency in Hz (default: 20)")
    parser.add_argument("--linear-step", type=float, default=0.1, help="Linear velocity increment in m/s")
    parser.add_argument("--angular-step", type=float, default=0.1, help="Yaw velocity increment in rad/s")
    args = parser.parse_args()
    if args.rate <= 0.0 or args.linear_step <= 0.0 or args.angular_step <= 0.0:
        parser.error("rate and velocity increments must be positive")
    if not sys.stdin.isatty():
        parser.error("an interactive terminal is required")
    return args


def twist(linear_x: float, linear_y: float, angular_z: float) -> dict:
    return {
        "linear": {"x": linear_x, "y": linear_y, "z": 0.0},
        "angular": {"x": 0.0, "y": 0.0, "z": angular_z},
    }


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

    print(f"Velocity publisher bound to {args.bind}; waiting for subscribers...")
    print("W/S: forward/back, A/D: left/right, Q/E: yaw left/right, Space: zero, X: quit")
    time.sleep(0.5)
    command = [0.0, 0.0, 0.0]
    terminal_fd = sys.stdin.fileno()
    old_terminal_settings = termios.tcgetattr(terminal_fd)
    period = 1.0 / args.rate

    try:
        tty.setcbreak(terminal_fd)
        while True:
            if select.select([sys.stdin], [], [], 0.0)[0]:
                key = sys.stdin.read(1).lower()
                if key == "x":
                    break
                if key == "w":
                    command[0] += args.linear_step
                elif key == "s":
                    command[0] -= args.linear_step
                elif key == "a":
                    command[1] += args.linear_step
                elif key == "d":
                    command[1] -= args.linear_step
                elif key == "q":
                    command[2] += args.angular_step
                elif key == "e":
                    command[2] -= args.angular_step
                elif key == " ":
                    command[:] = [0.0, 0.0, 0.0]
                print(f"\rcommand x={command[0]:+.2f} y={command[1]:+.2f} yaw={command[2]:+.2f}", end="", flush=True)
            publisher.send_json(twist(*command))
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        publisher.send_json(twist(0.0, 0.0, 0.0))
        termios.tcsetattr(terminal_fd, termios.TCSADRAIN, old_terminal_settings)
        publisher.close(linger=0)
        context.term()
        print("\nStopped velocity publisher after sending a zero command.")


if __name__ == "__main__":
    main()
