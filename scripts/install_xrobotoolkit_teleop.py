#!/usr/bin/env python3
"""Install the optional XRoboToolkit X2 sample into the current Python environment."""

import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
XROBOTOOLKIT_DIR = ROOT_DIR / "third_party" / "xrobotoolkit_teleop"
REQUIRED_FILES = (
    "pyproject.toml",
    "xrobotoolkit_teleop/simulation/agibot_x2_teleop_controller.py",
    "xrobotoolkit_teleop/integrations/robojudo_x2.py",
    "xrobotoolkit_teleop/common/x2_arm_retargeter.py",
    "scripts/simulation/teleop_agibot_x2_mujoco.py",
    "assets/agibot/x2/scene.xml",
    "assets/agibot/x2/x2_fixed.urdf",
    "assets/agibot/x2/x2_ultra.urdf",
    "assets/agibot/x2/collisions.json",
)
IMPORT_CHECK = "\n".join(
    (
        "import xrobotoolkit_sdk",
        "import placo",
        "import mujoco",
        "import tyro",
        "from xrobotoolkit_teleop.simulation.agibot_x2_teleop_controller import AgibotX2TeleopController",
        "from xrobotoolkit_teleop.integrations.robojudo_x2 import RoboJuDoX2Bridge",
        "print('XRoboToolkit X2 dependencies and controller imports passed.')",
    )
)


def run(command: list[str], *, cwd: Path = ROOT_DIR) -> None:
    print(f"Running: {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def validate_environment() -> None:
    if sys.platform != "linux":
        raise RuntimeError("The XRoboToolkit installer currently supports Linux only.")
    if sys.version_info < (3, 11):
        raise RuntimeError("Activate a Python 3.11+ environment before installing XRoboToolkit.")
    if platform.machine().lower() not in {"x86_64", "amd64", "aarch64", "arm64"}:
        raise RuntimeError("The XRoboToolkit SDK requires Linux x86_64 or aarch64.")
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix and sys.prefix == sys.base_prefix and Path(conda_prefix).resolve() != Path(sys.prefix).resolve():
        raise RuntimeError("The active Conda environment does not match this Python interpreter. Check PATH.")
    missing = [name for name in REQUIRED_FILES if not (XROBOTOOLKIT_DIR / name).is_file()]
    if missing:
        raise RuntimeError(
            f"XRoboToolkit submodule is incomplete: {', '.join(missing)}. "
            "Run `git submodule update --init third_party/xrobotoolkit_teleop` first."
        )
    # Import after the version check so Python 3.10 still gets an actionable error.
    import tomllib

    with (XROBOTOOLKIT_DIR / "pyproject.toml").open("rb") as metadata_file:
        metadata = tomllib.load(metadata_file)
    if "x2" not in metadata.get("project", {}).get("optional-dependencies", {}):
        raise RuntimeError("This XRoboToolkit checkout does not provide the 'x2' extra. Update the submodule revision.")


def main() -> None:
    validate_environment()
    print(f"Installing XRoboToolkit with {sys.executable} into {sys.prefix}", flush=True)
    run([sys.executable, "-m", "pip", "install", "-e", f"{XROBOTOOLKIT_DIR.as_posix()}[x2]"])
    # Import only: do not initialize the SDK, connect to a headset, or open a viewer.
    run([sys.executable, "-c", IMPORT_CHECK])
    print("XRoboToolkit X2 installation complete. Run the bridge in this environment alongside RoboJuDo.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
