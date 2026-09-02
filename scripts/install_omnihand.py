#!/usr/bin/env python3
"""Install the bundled OmniHand wheel for the active Python and CPU architecture."""

import platform
import shlex
import subprocess
import sys
from importlib import metadata
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SDK_DIR = ROOT_DIR / "third_party" / "omnihand_sdk"
ARCHITECTURE_DIRS = {
    "amd64": "x64",
    "arm64": "aarch64",
    "x86_64": "x64",
    "aarch64": "aarch64",
}


def run(command: list[str]):
    print(f"Running: {shlex.join(command)}")
    subprocess.run(command, cwd=ROOT_DIR, check=True)


def select_wheel() -> Path:
    if platform.system() != "Linux":
        raise RuntimeError(f"OmniHand deployment supports Linux, got {platform.system()}")
    if platform.python_implementation() != "CPython":
        raise RuntimeError(
            f"OmniHand wheels require CPython, got {platform.python_implementation()}"
        )

    machine = platform.machine().lower()
    architecture_dir = ARCHITECTURE_DIRS.get(machine)
    if architecture_dir is None:
        raise RuntimeError(
            f"Unsupported OmniHand architecture {machine!r}; expected x86_64/amd64 or aarch64/arm64"
        )

    python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    wheel_dir = SDK_DIR / "linux" / architecture_dir / "python"
    matches = sorted(wheel_dir.glob(f"omnihand-*-{python_tag}-{python_tag}-linux_*.whl"))
    if len(matches) != 1:
        available = ", ".join(path.name for path in sorted(wheel_dir.glob("*.whl"))) or "none"
        raise RuntimeError(
            f"Expected exactly one OmniHand wheel for {python_tag}/{machine} in {wheel_dir}, "
            f"found {len(matches)}; available: {available}"
        )
    return matches[0]


def main():
    if not SDK_DIR.is_dir():
        raise RuntimeError(
            f"OmniHand SDK submodule is missing: {SDK_DIR}. "
            "Run `git submodule update --init third_party/omnihand_sdk`."
        )
    wheel = select_wheel()
    run([sys.executable, "-m", "pip", "install", "--force-reinstall", wheel.as_posix()])
    run(
        [
            sys.executable,
            "-c",
            (
                "from importlib.metadata import version; "
                "from omnihand import OmniHandPro2025; "
                "print('omnihand', version('omnihand'), OmniHandPro2025.__name__)"
            ),
        ]
    )
    print(
        "OmniHand Python SDK installation complete. For first-time USB-CAN setup, run "
        f"`sudo bash {wheel.parents[1] / 'setup_udev.sh'}` and then log out and back in."
    )


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError, metadata.PackageNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
