import importlib.util
import json
import os
import subprocess
import unittest
from pathlib import Path
from unittest.mock import call, patch


INSTALLER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "install_dex_teleop.py"
INSTALLER_SPEC = importlib.util.spec_from_file_location("install_dex_teleop", INSTALLER_PATH)
assert INSTALLER_SPEC is not None and INSTALLER_SPEC.loader is not None
install_dex_teleop = importlib.util.module_from_spec(INSTALLER_SPEC)
INSTALLER_SPEC.loader.exec_module(install_dex_teleop)


class TestInstallDexTeleop(unittest.TestCase):
    def test_detects_active_conda_environment(self):
        prefix = "/opt/conda/envs/robot"
        with patch.object(install_dex_teleop.sys, "prefix", prefix):
            with patch.dict(
                os.environ,
                {"CONDA_PREFIX": prefix, "CONDA_DEFAULT_ENV": "robot"},
                clear=True,
            ):
                self.assertEqual(
                    install_dex_teleop.detect_current_conda_environment("conda"),
                    ("robot", Path(prefix)),
                )

    def test_detects_conda_environment_from_current_python_prefix(self):
        prefix = "/opt/conda/envs/robot"
        completed = subprocess.CompletedProcess([], 0, stdout=json.dumps({"envs": [prefix]}))
        with patch.object(install_dex_teleop.sys, "prefix", prefix):
            with patch.dict(os.environ, {}, clear=True):
                with patch.object(install_dex_teleop, "run", return_value=completed):
                    self.assertEqual(
                        install_dex_teleop.detect_current_conda_environment("conda"),
                        ("robot", Path(prefix)),
                    )

    def test_rejects_standard_virtual_environment(self):
        prefix = "/workspace/.venv"
        completed = subprocess.CompletedProcess([], 0, stdout=json.dumps({"envs": ["/opt/conda"]}))
        with patch.object(install_dex_teleop.sys, "prefix", prefix):
            with patch.object(install_dex_teleop.sys, "base_prefix", "/usr"):
                with patch.dict(os.environ, {}, clear=True):
                    with patch.object(install_dex_teleop, "run", return_value=completed):
                        with self.assertRaisesRegex(RuntimeError, "not a Conda environment"):
                            install_dex_teleop.detect_current_conda_environment("conda")

    def test_main_installs_with_conda_then_pip(self):
        prefix = Path("/opt/conda/envs/robot")
        python = (prefix / "bin/python").as_posix()
        completed = subprocess.CompletedProcess([], 0)

        with patch.object(install_dex_teleop, "require_executable", return_value="/usr/bin/conda"):
            with patch.object(
                install_dex_teleop,
                "detect_current_conda_environment",
                return_value=("robot", prefix),
            ):
                with patch.object(install_dex_teleop.sys, "executable", python):
                    with patch.object(install_dex_teleop, "run", return_value=completed) as run:
                        install_dex_teleop.main()

        self.assertEqual(
            run.call_args_list[0],
            call(
                [
                    "/usr/bin/conda",
                    "install",
                    "--prefix",
                    prefix.as_posix(),
                    "--yes",
                    "pinocchio=3.1.0",
                    "--channel",
                    "conda-forge",
                ],
                cwd=install_dex_teleop.DEX_TELEOP_DIR,
            ),
        )
        self.assertEqual(len(run.call_args_list), 2)
        self.assertEqual(
            run.call_args_list[1],
            call(
                [python, "-m", "pip", "install", "-r", "requirements.txt"],
                cwd=install_dex_teleop.DEX_TELEOP_DIR,
            ),
        )


if __name__ == "__main__":
    unittest.main()
