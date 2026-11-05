import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

INSTALLER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "install_xrobotoolkit_teleop.py"
SPEC = importlib.util.spec_from_file_location("install_xrobotoolkit_teleop", INSTALLER_PATH)
assert SPEC is not None and SPEC.loader is not None
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


class TestInstallXrobotoolkitTeleop(unittest.TestCase):
    def setUp(self):
        self.source = Path(self.enterContext(tempfile.TemporaryDirectory())) / "XRoboToolkit source"
        for name in installer.REQUIRED_FILES:
            target = self.source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch()
        (self.source / "pyproject.toml").write_text("[project.optional-dependencies]\nx2 = []\n")
        for owner, attribute, replacement in (
            (installer, "XROBOTOOLKIT_DIR", self.source),
            (installer.sys, "platform", "linux"),
            (installer.sys, "version_info", (3, 11)),
        ):
            self.enterContext(patch.object(owner, attribute, replacement))
        self.enterContext(patch.object(installer.platform, "machine", return_value="x86_64"))
        self.enterContext(patch.dict(os.environ, {}, clear=True))

    def test_accepts_complete_submodule_without_conda(self):
        installer.validate_environment()

    def test_rejects_old_python_before_installing(self):
        with (
            patch.object(installer.sys, "version_info", (3, 10)),
            patch.object(installer, "run") as run,
            self.assertRaisesRegex(RuntimeError, "Python 3.11"),
        ):
            installer.main()
        run.assert_not_called()

    def test_rejects_unsupported_platform_and_architecture(self):
        with patch.object(installer.sys, "platform", "win32"):
            with self.assertRaisesRegex(RuntimeError, "Linux only"):
                installer.validate_environment()
        with patch.object(installer.platform, "machine", return_value="riscv64"):
            with self.assertRaisesRegex(RuntimeError, "x86_64 or aarch64"):
                installer.validate_environment()

    def test_accepts_aarch64(self):
        with patch.object(installer.platform, "machine", return_value="aarch64"):
            installer.validate_environment()

    def test_missing_asset_reports_submodule_initialization_command(self):
        (self.source / "assets/agibot/x2/scene.xml").unlink()
        with self.assertRaisesRegex(RuntimeError, "git submodule update --init"):
            installer.validate_environment()

    def test_rejects_revision_without_x2_extra_before_installing(self):
        (self.source / "pyproject.toml").write_text('[project]\nname = "old-version"\n')
        with patch.object(installer, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "Update the submodule revision"):
                installer.main()
        run.assert_not_called()

    def test_rejects_malformed_metadata_before_installing(self):
        (self.source / "pyproject.toml").write_text("[invalid metadata")
        with patch.object(installer, "run") as run, self.assertRaises(ValueError):
            installer.main()
        run.assert_not_called()

    def test_rejects_mismatched_conda_interpreter(self):
        with (
            patch.dict(os.environ, {"CONDA_PREFIX": "/different/conda/environment"}),
            patch.object(installer.sys, "prefix", "/current/python"),
            patch.object(installer.sys, "base_prefix", "/current/python"),
            self.assertRaisesRegex(RuntimeError, "does not match"),
        ):
            installer.validate_environment()

    def test_accepts_venv_inside_conda(self):
        with (
            patch.dict(os.environ, {"CONDA_PREFIX": "/parent/conda"}),
            patch.object(installer.sys, "prefix", "/temporary/venv"),
            patch.object(installer.sys, "base_prefix", "/parent/conda"),
        ):
            installer.validate_environment()

    def test_installs_and_verifies_with_invoking_interpreter(self):
        python = "/active environment/bin/python"
        with patch.object(installer.sys, "executable", python), patch.object(installer, "run") as run:
            installer.main()
        self.assertEqual(
            run.call_args_list,
            [
                call([python, "-m", "pip", "install", "-e", f"{self.source.as_posix()}[x2]"]),
                call([python, "-c", installer.IMPORT_CHECK]),
            ],
        )
        self.assertNotIn(".init()", installer.IMPORT_CHECK)
        self.assertNotIn("AgibotX2TeleopController(", installer.IMPORT_CHECK)

    def test_stops_after_install_or_import_failure(self):
        for failing_step in range(2):
            with self.subTest(failing_step=failing_step):
                results = [None] * failing_step + [subprocess.CalledProcessError(1, "pip or import")]
                with patch.object(installer, "run", side_effect=results) as run:
                    with self.assertRaises(subprocess.CalledProcessError):
                        installer.main()
                self.assertEqual(run.call_count, failing_step + 1)

    def test_run_uses_checked_argument_list(self):
        command = ["/python with spaces", "-m", "pip", "install", "package"]
        with patch.object(installer.subprocess, "run") as run:
            installer.run(command)
        run.assert_called_once_with(command, cwd=installer.ROOT_DIR, check=True)


if __name__ == "__main__":
    unittest.main()
