#!/usr/bin/env python3
"""Install RoboJuDo Recorder's RealSense backend, including Jetson support."""

import argparse
import os
import platform
import shlex
import shutil
import site
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
RECORDER_DIR = ROOT_DIR / "packages" / "robojudo_recorder"
LIBREALSENSE_VERSION = "2.56.4"
LIBREALSENSE_REPOSITORY = "https://github.com/realsenseai/librealsense.git"
APT_PACKAGES = (
    "build-essential",
    "cmake",
    "git",
    "libssl-dev",
    "libusb-1.0-0-dev",
    "libudev-dev",
    "pkg-config",
)


def run(command: list[str], *, cwd: Path = ROOT_DIR, env: dict[str, str] | None = None):
    print(f"Running: {shlex.join(command)}")
    subprocess.run(command, cwd=cwd, env=env, check=True)


def require_linux():
    if platform.system() != "Linux":
        raise RuntimeError("RealSense installation is supported by this script only on Linux.")
    if sys.version_info < (3, 11):
        raise RuntimeError("RoboJuDo Recorder requires Python 3.11 or newer.")


def install_recorder(*, with_pypi_binding: bool):
    package = f"{RECORDER_DIR}[realsense]" if with_pypi_binding else str(RECORDER_DIR)
    run([sys.executable, "-m", "pip", "install", "-e", package])


def install_system_dependencies():
    sudo = [] if os.geteuid() == 0 else ["sudo"]
    run([*sudo, "apt-get", "update"])
    run([*sudo, "apt-get", "install", "-y", *APT_PACKAGES])


def source_root(cache_dir: Path) -> Path:
    return cache_dir / f"librealsense-{LIBREALSENSE_VERSION}"


def prepare_source(cache_dir: Path) -> Path:
    source_dir = source_root(cache_dir)
    if source_dir.exists():
        if not (source_dir / ".git").is_dir():
            raise RuntimeError(
                f"Refusing to replace non-git path {source_dir}. Remove it or select another --cache-dir."
            )
        print(f"Using existing librealsense source: {source_dir}")
        return source_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            "git",
            "clone",
            "--branch",
            f"v{LIBREALSENSE_VERSION}",
            "--depth",
            "1",
            LIBREALSENSE_REPOSITORY,
            source_dir.as_posix(),
        ]
    )
    return source_dir


def configure_and_build(source_dir: Path, jobs: int):
    build_dir = source_dir / "build-robojudo"
    run(
        [
            "cmake",
            "-S",
            source_dir.as_posix(),
            "-B",
            build_dir.as_posix(),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DBUILD_PYTHON_BINDINGS=ON",
            "-DBUILD_SHARED_LIBS=OFF",
            "-DBUILD_EXAMPLES=OFF",
            "-DBUILD_GRAPHICAL_EXAMPLES=OFF",
            "-DBUILD_TOOLS=OFF",
            "-DFORCE_RSUSB_BACKEND=ON",
        ]
    )
    run(["cmake", "--build", build_dir.as_posix(), "--parallel", str(jobs)])
    install_python_binding(source_dir, build_dir)


def install_python_binding(source_dir: Path, build_dir: Path):
    bindings = sorted((build_dir / "wrappers" / "python").glob("pyrealsense2*.so"))
    if len(bindings) != 1:
        found = ", ".join(path.name for path in bindings) or "none"
        raise RuntimeError(f"Expected one built pyrealsense2 extension, found: {found}")

    init_file = source_dir / "wrappers" / "python" / "pyrealsense2" / "__init__.py"
    if not init_file.is_file():
        raise RuntimeError(f"pyrealsense2 package initializer was not found: {init_file}")

    python_package_dir = Path(site.getsitepackages()[0]).resolve() / "pyrealsense2"
    python_package_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(init_file, python_package_dir / "__init__.py")
    shutil.copy2(bindings[0], python_package_dir / bindings[0].name)
    print(f"Installed pyrealsense2 binding in {python_package_dir}")


def install_udev_rules(source_dir: Path):
    script = source_dir / "scripts" / "setup_udev_rules.sh"
    if not script.is_file():
        raise RuntimeError(f"librealsense udev installer was not found: {script}")
    sudo = [] if os.geteuid() == 0 else ["sudo"]
    run([*sudo, "bash", script.as_posix()], cwd=source_dir)


def verify_installation():
    run(
        [
            sys.executable,
            "-c",
            "import pyrealsense2 as rs; print('pyrealsense2', rs.__version__)",
        ]
    )


def default_cache_dir() -> Path:
    configured = os.environ.get("XDG_CACHE_HOME")
    cache_root = Path(configured).expanduser() if configured else Path.home() / ".cache"
    return cache_root / "robojudo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument(
        "--skip-system-deps",
        action="store_true",
        help="Do not install apt build dependencies (use when they are already present).",
    )
    parser.add_argument(
        "--skip-udev-rules",
        action="store_true",
        help="Do not install librealsense USB permission rules.",
    )
    parser.add_argument(
        "--force-source",
        action="store_true",
        help="Build librealsense from source even when a PyPI wheel is expected.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    require_linux()
    build_from_source = platform.machine().lower() in {"aarch64", "arm64"} or args.force_source

    if not build_from_source:
        install_recorder(with_pypi_binding=True)
        verify_installation()
        print("RoboJuDo Recorder RealSense installation complete.")
        return

    print(
        f"No compatible pyrealsense2 wheel is assumed for {platform.machine()} / Python "
        f"{sys.version_info.major}.{sys.version_info.minor}; building librealsense {LIBREALSENSE_VERSION} "
        "with the user-space RSUSB backend."
    )
    if not args.skip_system_deps:
        install_system_dependencies()
    for executable in ("cmake", "git"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"{executable} was not found on PATH.")

    install_recorder(with_pypi_binding=False)
    source_dir = prepare_source(args.cache_dir.expanduser().resolve())
    configure_and_build(source_dir, max(1, args.jobs))
    if not args.skip_udev_rules:
        install_udev_rules(source_dir)
    verify_installation()
    print("RoboJuDo Recorder RealSense installation complete. Reconnect the camera before recording.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
