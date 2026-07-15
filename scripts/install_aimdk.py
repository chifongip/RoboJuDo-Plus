#!/usr/bin/env python3
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
AIMDK_DIR = ROOT_DIR / "third_party" / "aimdk"
AIMDK_SETUP = AIMDK_DIR / "install" / "setup.bash"
AIMDK_CPP_DIR = ROOT_DIR / "packages" / "aimdk_cpp"


def run(command: list[str], *, cwd: Path = ROOT_DIR, env: dict[str, str] | None = None):
    print(f"Running: {shlex.join(command)}")
    subprocess.run(command, cwd=cwd, env=env, check=True)


def sourced_environment(setup_file: Path) -> dict[str, str]:
    command = f"source {shlex.quote(setup_file.as_posix())} && env -0"
    result = subprocess.run(
        ["bash", "-c", command],
        cwd=ROOT_DIR,
        check=True,
        stdout=subprocess.PIPE,
    )
    environment = os.environ.copy()
    for entry in result.stdout.split(b"\0"):
        if not entry or b"=" not in entry:
            continue
        key, value = entry.split(b"=", 1)
        environment[key.decode()] = value.decode()
    return environment


def validate_environment():
    if not os.environ.get("ROS_DISTRO"):
        raise RuntimeError("ROS 2 is not sourced. Run `source /opt/ros/<distro>/setup.bash` first.")
    if not os.environ.get("CONDA_PREFIX"):
        raise RuntimeError("No Conda environment is active. Activate the RoboJuDo environment first.")
    if shutil.which("colcon") is None:
        raise RuntimeError("colcon was not found. Install colcon before building AimDK.")


def main():
    validate_environment()
    run(["git", "submodule", "update", "--init", "third_party/aimdk"])
    if not (AIMDK_DIR / "src" / "aimdk_msgs" / "package.xml").is_file():
        raise RuntimeError(f"AimDK submodule is incomplete: {AIMDK_DIR}")

    run(
        [
            "colcon",
            "--log-base",
            "log",
            "build",
            "--base-paths",
            "src",
            "--build-base",
            "build",
            "--install-base",
            "install",
            "--packages-select",
            "aimdk_msgs",
            "--cmake-args",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        cwd=AIMDK_DIR,
    )

    run([sys.executable, "-m", "pip", "install", "scikit-build-core", "pybind11"])
    aimdk_environment = sourced_environment(AIMDK_SETUP)
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-build-isolation",
            AIMDK_CPP_DIR.as_posix(),
        ],
        env=aimdk_environment,
    )
    run(
        [sys.executable, "-c", "import aimdk_cpp; print(aimdk_cpp.__file__)"],
        env=aimdk_environment,
    )
    print(f"AimDK installation complete. Source {AIMDK_SETUP.relative_to(ROOT_DIR)} before X2 real deployment.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
