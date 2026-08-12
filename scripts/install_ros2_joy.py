#!/usr/bin/env python3
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
ROS2_JOY_CPP_DIR = ROOT_DIR / "packages" / "ros2_joy_cpp"


def run(command: list[str], *, cwd: Path = ROOT_DIR):
    print(f"Running: {shlex.join(command)}")
    subprocess.run(command, cwd=cwd, check=True)


def validate_environment():
    if not os.environ.get("ROS_DISTRO"):
        raise RuntimeError("ROS 2 is not sourced. Run `source /opt/ros/<distro>/setup.bash` first.")
    if not (ROS2_JOY_CPP_DIR / "pyproject.toml").is_file():
        raise RuntimeError(f"ROS 2 Joy C++ package is incomplete: {ROS2_JOY_CPP_DIR}")


def main():
    validate_environment()
    run([sys.executable, "-m", "pip", "install", "scikit-build-core", "pybind11"])
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-build-isolation",
            ROS2_JOY_CPP_DIR.as_posix(),
        ]
    )
    run([sys.executable, "-c", "import ros2_joy_cpp; print(ros2_joy_cpp.__file__)"])
    print("ROS 2 Joy C++ installation complete.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
