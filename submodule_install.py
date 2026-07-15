#!/usr/bin/env python3
import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

CONFIG_FILE = "submodule_cfg.yaml"


def run(cmd, cwd=None, required=False):
    print(f"Running: {cmd}")
    try:
        subprocess.run(cmd, shell=True, cwd=cwd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")
        if required:
            raise


def apply_patch(path: Path, patch: str):
    patch_path = (path / patch).resolve()
    if not patch_path.exists():
        print(f"Patch {patch} not found, skipping.")
        return

    check = ["git", "-C", path.as_posix(), "apply", "--check", patch_path.as_posix()]
    reverse_check = ["git", "-C", path.as_posix(), "apply", "--reverse", "--check", patch_path.as_posix()]
    if subprocess.run(check, check=False).returncode == 0:
        subprocess.run(check[:4] + [patch_path.as_posix()], check=True)
    elif subprocess.run(reverse_check, check=False).returncode == 0:
        print(f"Patch {patch} is already applied, preserving the existing worktree.")
    else:
        raise RuntimeError(f"Patch {patch} cannot be applied cleanly to {path}.")


def load_config():
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)


def install_submodules(selected=None, clean=False):
    config = load_config()
    for name, info in config.items():
        print(f"\n----- Installing submodule: {name} -----")
        install = info.get("install", False)
        if (selected is None and not install) or (selected is not None and name not in selected):
            print(f"Skipping submodule: {name}")
            continue

        path = Path(info["path"])
        patches = info.get("patches", [])
        addons = info.get("addons", [])

        print(f"Initializing submodule '{name}'...")
        if clean and (path / ".git").exists():
            print(f"Cleaning existing submodule '{name}'...")
            run(f"git -C {shlex.quote(path.as_posix())} reset --hard", required=True)
            run(f"git -C {shlex.quote(path.as_posix())} clean -fd", required=True)
        elif (path / ".git").exists():
            print(f"Preserving existing worktree for submodule '{name}'.")
        run(f"git submodule update --init {shlex.quote(path.as_posix())}", required=True)

        if not path.exists():
            print(f"Path {path} does not exist. Skipping {name}.")
            continue

        for patch in patches:
            print(f"Applying patch {patch} for '{name}'...")
            apply_patch(path, patch)

        for addon in addons:
            addon_path = path / addon
            if addon_path.exists():
                print(f"Adding addon '{addon_path}' to '{name}'...")
                shutil.copytree(addon_path, path, dirs_exist_ok=True)
            else:
                print(f"Addon path {addon} does not exist, skipping.")

        if installer := info.get("installer"):
            run(f"{shlex.quote(sys.executable)} {shlex.quote(installer)}", required=True)
            continue

        # install Python package
        packages = [path]  # main package
        if (extra_packages := info.get("extra_packages", None)) is not None:
            packages += extra_packages
        for pkg in packages:
            run(f"pip install -e {shlex.quote(pkg.as_posix())}", required=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Initialize and install optional RoboJuDo modules.")
    parser.add_argument("--clean", action="store_true", help="Discard submodule changes before installation.")
    parser.add_argument("modules", nargs="*", help="Optional module names; defaults to entries with install: true.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected_modules = args.modules or None
    # selected_modules will override install cfg if provided
    install_submodules(selected_modules, clean=args.clean)

    # Usage example:
    # python submodule_install.py mujoco_viewer unitree_cpp
