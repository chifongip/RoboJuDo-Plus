#!/usr/bin/env python3
"""Create a teleop-only Conda environment with Placo and LeRobot IK."""

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import urllib.request
from contextlib import contextmanager
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
XROBOTOOLKIT_DIR = ROOT_DIR / "third_party" / "xrobotoolkit_teleop"
REQUIREMENTS = ROOT_DIR / "scripts" / "requirements_xrobotoolkit_teleop.txt"
SOURCE_URL = "https://github.com/stack-of-tasks/pinocchio/archive/refs/tags/v4.1.0.tar.gz"
SOURCE_SHA256 = "6088e6954883af233489cbabc391bc70b884f9a7a57cd79f11dd28bcad62e08d"
OWNER = "robojudo-xrobotoolkit-setup-v1"
REQUIRED_FILES = (
    "pyproject.toml",
    "scripts/misc/build_lerobot_bindings.py",
    "xrobotoolkit_teleop/simulation/agibot_x2_teleop_controller.py",
    "xrobotoolkit_teleop/integrations/robojudo_x2.py",
    "xrobotoolkit_teleop/common/x2_arm_retargeter.py",
    "xrobotoolkit_teleop/integrations/robojudo.py",
    "xrobotoolkit_teleop/integrations/retargeter.py",
    "xrobotoolkit_teleop/integrations/robot_profiles/x2.py",
    "xrobotoolkit_teleop/integrations/robot_profiles/__init__.py",
    "xrobotoolkit_teleop/common/teleop_settings.py",
    "xrobotoolkit_teleop/ik/registry.py",
    "scripts/simulation/teleop_agibot_x2_mujoco.py",
    "assets/agibot/x2/scene.xml",
    "assets/agibot/x2/x2_fixed.urdf",
    "assets/agibot/x2/x2_ultra.urdf",
    "assets/agibot/x2/collisions.json",
)

# Remove foreign Python/native search paths from the invoking shell (e.g. ROS).
FOREIGN_PATHS = (
    "PYTHONPATH",
    "PYTHONHOME",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "CMAKE_PREFIX_PATH",
    "PKG_CONFIG_PATH",
    "CASADIPATH",
    "CPATH",
    "CPLUS_INCLUDE_PATH",
    "LIBRARY_PATH",
    "CMAKE_TOOLCHAIN_FILE",
    "CFLAGS",
    "CXXFLAGS",
    "LDFLAGS",
)


def clean_environment():
    env = {key: value for key, value in os.environ.items() if key not in FOREIGN_PATHS}

    # Internal installation runs under the target Conda interpreter. Load its
    # C++ runtime before Placo or other native extensions (needed on Jetson).
    if "--install-prefix" in sys.argv:
        runtime = Path(sys.prefix) / "lib" / "libstdc++.so.6"
        if runtime.is_file():
            env["LD_PRELOAD"] = str(runtime)

    return env


def run(command, *, capture=False, log=None):
    display = list(map(str, command))
    if "-c" in display:
        display[display.index("-c") + 1] = "<Python setup helper>"
    print(f"Running: {shlex.join(display)}", flush=True)
    kwargs = dict(cwd=ROOT_DIR, check=True, text=True, env=clean_environment())
    if log is not None:
        print(f"  log: {log}", flush=True)
        with Path(log).open("w") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, **kwargs)
        return None
    return subprocess.run(command, capture_output=capture, **kwargs)


def validate_source():
    if sys.platform != "linux":
        raise RuntimeError("The XRoboToolkit installer supports Linux only.")
    if platform.machine().lower() not in {"x86_64", "amd64", "aarch64", "arm64"}:
        raise RuntimeError("The XRoboToolkit SDK requires Linux x86_64 or aarch64.")
    missing = [name for name in REQUIRED_FILES if not (XROBOTOOLKIT_DIR / name).is_file()]
    if missing:
        raise RuntimeError(
            f"XRoboToolkit submodule is incomplete: {', '.join(missing)}. "
            "Run `git submodule update --init third_party/xrobotoolkit_teleop` first."
        )
    if not REQUIREMENTS.is_file():
        raise RuntimeError(f"Missing installer requirements: {REQUIREMENTS}")
    if shutil.which("g++") is None:
        raise RuntimeError("g++ is required to build LeRobot bindings. Install your system C++ build tools first.")


def find_conda():
    configured = os.environ.get("CONDA_EXE")
    candidate = shutil.which(configured) if configured else None
    candidate = candidate or shutil.which("conda")
    if candidate is None:
        raise RuntimeError("Conda was not found. Install Miniconda/Anaconda or set CONDA_EXE.")
    return candidate


def environment_prefix(conda, name):
    result = run([conda, "info", "--envs", "--json"], capture=True)
    data = json.loads(result.stdout)
    matches = {Path(p).resolve() for p in data["envs"] if Path(p).name == name}
    # Conda may omit an interrupted or unmanaged directory from its env list.
    # Protect those directories too, rather than creating over their contents.
    matches.update(
        (Path(directory) / name).resolve()
        for directory in data.get("envs_dirs", [])
        if (Path(directory) / name).exists()
    )
    if len(matches) > 1:
        raise RuntimeError(f"Multiple Conda environments are named {name!r}; choose another --name.")
    return next(iter(matches), None)


@contextmanager
def installation_lock(name):
    """Serialize setup before checking or creating the named environment.

    Keep the lock file after release: unlinking it could let another process
    acquire a different inode while an existing waiter still uses the old one.
    The internal conda-run child is covered by the launching parent's lock.
    """
    import fcntl

    path = Path(tempfile.gettempdir()) / f"robojudo-xr-setup-{os.getuid()}-{name}.lock"
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another setup is running for {name!r}; wait for it to finish.") from exc
        yield


def create_environment(args):
    conda = find_conda()
    prefix = environment_prefix(conda, args.name)
    if prefix is not None and not args.reuse:
        raise RuntimeError(f"Environment {args.name!r} already exists. Use --reuse explicitly, or choose --name NAME.")
    if prefix is not None:
        run(
            [
                conda,
                "run",
                "--no-capture-output",
                "--prefix",
                str(prefix),
                "python",
                "-I",
                str(Path(__file__).resolve()),
                "--check-prefix",
                str(prefix),
            ]
        )
        run(
            [
                conda,
                "install",
                "--yes",
                "--prefix",
                str(prefix),
                "--override-channels",
                "--channel",
                "conda-forge",
                "python=3.11",
                "numpy=2.2.6",
                "casadi=3.6.7",
            ]
        )
    if prefix is None:
        run(
            [
                conda,
                "create",
                "--yes",
                "--name",
                args.name,
                "--override-channels",
                "--channel",
                "conda-forge",
                "python=3.11",
                "pip",
                "numpy=2.2.6",
                "casadi=3.6.7",
            ]
        )
        prefix = environment_prefix(conda, args.name)
        if prefix is None:
            raise RuntimeError("Conda did not report the newly created environment.")
    # The internal command validates the actual target interpreter before mutation.
    run(
        [
            conda,
            "run",
            "--no-capture-output",
            "--prefix",
            str(prefix),
            "python",
            "-I",
            str(Path(__file__).resolve()),
            "--install-prefix",
            str(prefix),
            "--jobs",
            str(args.jobs),
        ]
    )
    runtime = prefix / "lib" / "libstdc++.so.6"
    if runtime.is_file():
        # Keep normal Python sessions consistent with installer verification.
        run([conda, "env", "config", "vars", "set", "--prefix", str(prefix), f"LD_PRELOAD={runtime}"])
        print("Saved the Conda C++ runtime preload. Reactivate the environment if it is already active.")
    print(f"\nSetup complete. Activate with: conda activate {args.name}")
    print("Run: python -m xrobotoolkit_teleop.integrations.robojudo --robot x2 --ik-backend placo")
    print("Or select: --ik-backend lerobot")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_manifest(path):
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or data.get("owner") != OWNER:
        raise RuntimeError(f"Unmanaged installer manifest: {path}")
    return data


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def download_source(directory):
    archive = directory / "pinocchio-4.1.0.tar.gz"
    if not archive.exists() or digest(archive) != SOURCE_SHA256:
        temporary = archive.with_suffix(".download")
        with urllib.request.urlopen(SOURCE_URL, timeout=120) as source, temporary.open("wb") as output:
            shutil.copyfileobj(source, output)
        if digest(temporary) != SOURCE_SHA256:
            raise RuntimeError(f"Pinocchio source SHA-256 mismatch; downloaded file retained at {temporary}")
        temporary.replace(archive)
    return archive


def extract_source(archive, destination):
    with tarfile.open(archive) as source:
        source.extractall(destination, filter="data")
    return destination / "pinocchio-4.1.0"


def registration_text():
    return "import os, sys; sys.path.insert(0, os.path.join(sys.prefix, 'lerobot-python'))\n"


def promote_overlay(staged, overlay, registration, check=None):
    """Keep the previous working registration if promotion fails."""
    backup = overlay.with_name("lerobot-python.previous")
    if backup.exists():
        raise RuntimeError(f"Previous overlay backup exists; inspect it before retrying: {backup}")
    existed = overlay.exists()
    previous_registration = registration.read_bytes() if registration.exists() else None
    if existed:
        overlay.rename(backup)
    try:
        staged.rename(overlay)
        temporary = registration.with_suffix(".tmp")
        temporary.write_text(registration_text())
        temporary.replace(registration)
        if check is not None:
            check()
    except Exception:
        if overlay.exists():
            overlay.rename(staged)
        if existed:
            backup.rename(overlay)
        if previous_registration is None:
            registration.unlink(missing_ok=True)
        elif not registration.exists() or registration.read_bytes() != previous_registration:
            registration.write_bytes(previous_registration)
        raise
    if existed:
        shutil.rmtree(backup)


def verify(prefix, log):
    smoke = r"""
import sys
from pathlib import Path
assert Path(sys.prefix).resolve() == Path(sys.argv[1]).resolve(), 'Wrong environment'
expected = Path(sys.prefix) / 'lerobot-python'
assert Path(pinocchio.__file__).resolve().is_relative_to(expected.resolve()), 'Wrong Pinocchio overlay'
import casadi as ca
import mujoco, xrobotoolkit_sdk, tyro, zmq
from xrobotoolkit_teleop.integrations.robojudo import RoboJuDoBridge
from xrobotoolkit_teleop.integrations.robojudo_x2 import RoboJuDoX2Bridge
from xrobotoolkit_teleop.integrations.retargeter import ExternalStateRetargeter

def forbid_xr(*args, **kwargs):
    raise RuntimeError('Verification must not initialize XR')

xrobotoolkit_sdk.init = forbid_xr
for backend in ('placo', 'lerobot'):
    retargeter = ExternalStateRetargeter(ik_backend=backend)
    retargeter.close()
assert ca.has_nlpsol('ipopt'), 'IPOPT unavailable'
opti = ca.Opti()
x = opti.variable()
opti.minimize((x - 1) ** 2)
opti.solver('ipopt', {'ipopt': {'print_level': 0, 'sb': 'yes'}, 'print_time': False})
assert abs(opti.solve().value(x) - 1) < 1e-6
"""
    for index, imports in enumerate(
        (
            "import placo; import pinocchio; import pinocchio.casadi\n",
            "import pinocchio; import pinocchio.casadi; import placo\n",
        )
    ):
        run([sys.executable, "-I", "-c", imports + smoke, str(prefix)], log=log / f"imports-{index}.log")
    run([sys.executable, "-I", "-m", "pip", "check"], log=log / "pip-check.log")
    for module in ("robojudo", "robojudo_x2"):
        run(
            [sys.executable, "-I", "-m", f"xrobotoolkit_teleop.integrations.{module}", "--help"],
            log=log / f"{module}-help.log",
        )


def validate_target(prefix):
    if sys.version_info[:2] != (3, 11) or Path(sys.prefix).resolve() != prefix.resolve():
        raise RuntimeError("The target must be a Python 3.11 Conda environment matching --install-prefix.")
    if not (prefix / "conda-meta").is_dir() or Path(os.environ.get("CONDA_PREFIX", "")).resolve() != prefix.resolve():
        raise RuntimeError("Run the target installation through conda run.")
    import tomllib

    with (XROBOTOOLKIT_DIR / "pyproject.toml").open("rb") as stream:
        if "x2" not in tomllib.load(stream).get("project", {}).get("optional-dependencies", {}):
            raise RuntimeError("This checkout has no 'x2' extra; update the submodule revision.")
    state = prefix / "share" / "robojudo-xr"
    overlay = prefix / "lerobot-python"
    registration = Path(sysconfig.get_paths()["purelib"]) / "00_lerobot_pinocchio.pth"
    manifest_path = state / "manifest.json"
    previous = read_manifest(manifest_path)
    if previous and previous.get("prefix") != str(prefix.resolve()):
        raise RuntimeError("Installer manifest belongs to a different prefix; use a fresh environment.")
    if (overlay.exists() or registration.exists()) and not previous:
        raise RuntimeError("Unmanaged LeRobot overlay/registration exists; choose a fresh environment.")
    if overlay.with_name("lerobot-python.previous").exists():
        raise RuntimeError("An interrupted overlay promotion left lerobot-python.previous; inspect it before retrying.")
    if registration.exists() and registration.read_text() != registration_text():
        raise RuntimeError(f"Unmanaged registration contents: {registration}")
    return state, overlay, registration, manifest_path, previous


def expected_versions():
    versions = {"casadi": "3.6.7"}
    for line in REQUIREMENTS.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, version = line.split("==")
            versions[name] = version
    return versions


def install_current(prefix, jobs):
    state, overlay, registration, manifest_path, previous = validate_target(prefix)
    state.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="setup-", dir=state))
    builder = XROBOTOOLKIT_DIR / "scripts/misc/build_lerobot_bindings.py"
    recipe = {"requirements": digest(REQUIREMENTS), "builder": digest(builder), "source": SOURCE_SHA256}
    manifest = {
        "owner": OWNER,
        "prefix": str(prefix.resolve()),
        "python": sys.version,
        "recipe": recipe,
        "complete": False,
        "overlay": str(overlay),
        "log_directory": str(work),
    }
    write_json(manifest_path, manifest)
    stage = "dependencies"
    source = None
    try:
        print(f"Installing into {prefix}; logs: {work}", flush=True)
        run(
            [sys.executable, "-I", "-m", "pip", "install", "-r", str(REQUIREMENTS), "-e", f"{XROBOTOOLKIT_DIR}[x2]"],
            log=work / "dependencies.log",
        )
        from importlib.metadata import version

        pins = expected_versions()
        versions = {name: version(name) for name in pins}
        if versions != pins:
            raise RuntimeError(f"Unexpected native dependency versions: {versions}")
        if not list((prefix / "conda-meta").glob("casadi-3.6.7-*.json")):
            raise RuntimeError("CasADi 3.6.7 must be installed by Conda, not pip.")
        stage = "bindings"
        reusable = (
            previous.get("complete")
            and previous.get("recipe") == recipe
            and previous.get("versions") == versions
            and overlay.is_dir()
            and registration.is_file()
        )
        if reusable:
            try:
                check_logs = work / "reuse-check"
                check_logs.mkdir()
                verify(prefix, check_logs)
            except subprocess.CalledProcessError:
                reusable = False
        if not reusable:
            archive = download_source(state)
            source = extract_source(archive, work)
            # Ignore an older registered overlay while building its replacement.
            bootstrap = (
                "import sys, runpy; from pathlib import Path; "
                "sys.path = [p for p in sys.path if Path(p).resolve() != Path(sys.argv[1]).resolve()]; "
                "sys.argv = sys.argv[2:]; runpy.run_path(sys.argv[0], run_name='__main__')"
            )
            run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    bootstrap,
                    str(overlay),
                    str(builder),
                    "--source",
                    str(source),
                    "--build",
                    str(work / "build"),
                    "--output",
                    str(work / "overlay"),
                    "--casadi-prefix",
                    str(prefix),
                    "--jobs",
                    str(jobs),
                ],
                log=work / "bindings.log",
            )
            # The installed extensions must not depend on their build tree.
            for path in (source, work / "build"):
                if path.exists():
                    shutil.rmtree(path)
            stage = "registration and verification"
            promote_overlay(work / "overlay", overlay, registration, check=lambda: verify(prefix, work))
        manifest.update(
            complete=True,
            versions=versions,
            runtime_versions={
                name: version(name)
                for name in ("xrobotoolkit-teleop", "xrobotoolkit-sdk", "mujoco", "meshcat", "tyro", "pyzmq")
            },
        )
        write_json(manifest_path, manifest)
        print(f"Both IK backends verified. Manifest: {manifest_path}", flush=True)
    except Exception as exc:
        manifest.update(complete=False, failed_stage=stage, error=str(exc))
        write_json(manifest_path, manifest)
        raise RuntimeError(f"XR setup failed during {stage}. Logs retained in {work}: {exc}") from exc


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="robojudo-xr", help="Conda environment name (default: robojudo-xr)")
    parser.add_argument("--reuse", action="store_true", help="Explicitly install/repair an existing target environment")
    parser.add_argument("--jobs", type=int, default=2, help="Parallel native build jobs (default: 2)")
    parser.add_argument("--install-prefix", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--check-prefix", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.name) or args.name in {"base", "root"}:
        parser.error("--name must be a non-base Conda environment name, not a path")
    return args


def main(argv=None):
    args = parse_args(argv)
    validate_source()
    if args.check_prefix is not None:
        validate_target(args.check_prefix)
    elif args.install_prefix is not None:
        install_current(args.install_prefix, args.jobs)
    else:
        with installation_lock(args.name):
            create_environment(args)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
