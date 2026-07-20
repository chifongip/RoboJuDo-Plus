#!/usr/bin/env python3
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEX_TELEOP_DIR = ROOT_DIR / "third_party" / "dex_teleop"
REQUIREMENTS_FILE = DEX_TELEOP_DIR / "requirements.txt"


def run(
    command: list[str],
    *,
    cwd: Path = ROOT_DIR,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    print(f"Running: {shlex.join(command)}")
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture_output else None,
    )


def require_executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(f"{name} was not found on PATH.")
    return executable


def conda_environment_name(prefix: Path) -> str:
    configured_name = os.environ.get("CONDA_DEFAULT_ENV")
    if configured_name:
        return configured_name
    if prefix.parent.name == "envs":
        return prefix.name
    return "base"


def detect_current_conda_environment(conda: str) -> tuple[str, Path]:
    current_prefix = Path(sys.prefix).resolve()
    active_conda_prefix = os.environ.get("CONDA_PREFIX")

    if active_conda_prefix:
        conda_prefix = Path(active_conda_prefix).resolve()
        if current_prefix != conda_prefix:
            raise RuntimeError(
                f"The current Python environment ({current_prefix}) does not match the active Conda environment "
                f"({conda_prefix}). Deactivate the nested virtual environment or activate the intended "
                "Conda environment."
            )
        return conda_environment_name(conda_prefix), conda_prefix

    result = run([conda, "env", "list", "--json"], capture_output=True)
    try:
        conda_prefixes = [Path(prefix).resolve() for prefix in json.loads(result.stdout).get("envs", [])]
    except (json.JSONDecodeError, AttributeError, TypeError) as exc:
        raise RuntimeError("Could not read the Conda environment list.") from exc

    if current_prefix in conda_prefixes:
        return conda_environment_name(current_prefix), current_prefix

    environment_kind = "virtual environment" if sys.prefix != sys.base_prefix else "Python environment"
    raise RuntimeError(
        f"The current {environment_kind} ({current_prefix}) is not a Conda environment. "
        "Conda cannot install packages into a standard venv; activate the intended Conda environment first."
    )


def main():
    if not (DEX_TELEOP_DIR / "pyproject.toml").is_file():
        raise RuntimeError(f"dex_teleop submodule is incomplete: {DEX_TELEOP_DIR}")
    if not REQUIREMENTS_FILE.is_file():
        raise RuntimeError(f"dex_teleop requirements file was not found: {REQUIREMENTS_FILE}")

    conda = require_executable("conda")
    environment_name, environment_prefix = detect_current_conda_environment(conda)
    print(f"Using current Conda environment: {environment_name} ({environment_prefix})")

    run(
        [
            conda,
            "install",
            "--prefix",
            environment_prefix.as_posix(),
            "--yes",
            "pinocchio=3.1.0",
            "numpy=1.26.4",
            "--channel",
            "conda-forge",
        ],
        cwd=DEX_TELEOP_DIR,
    )
    run(
        [sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_FILE.name],
        cwd=DEX_TELEOP_DIR,
    )
    run(
        [
            sys.executable,
            "-c",
            (
                "import numpy, pinocchio; "
                "assert numpy.__version__ == '1.26.4', numpy.__version__; "
                "assert pinocchio.__version__ == '3.1.0', pinocchio.__version__; "
                "print(f'numpy={numpy.__version__}, pinocchio={pinocchio.__version__}')"
            ),
        ],
        cwd=DEX_TELEOP_DIR,
    )
    print(f"dex_teleop installation complete in Conda environment: {environment_name}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
