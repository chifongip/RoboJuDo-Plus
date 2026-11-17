"""Installer orchestration and native artifact lifecycle without Conda or compilation."""

import hashlib
import importlib.util
import io
import json
import os
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

INSTALLER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "install_xrobotoolkit_teleop.py"
SPEC = importlib.util.spec_from_file_location("install_xrobotoolkit_teleop", INSTALLER_PATH)
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


class InstallerFixture(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.source = self.directory / "XRoboToolkit source"
        for name in installer.REQUIRED_FILES:
            target = self.source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch()
        (self.source / "pyproject.toml").write_text("[project.optional-dependencies]\nx2 = []\n")
        self.enterContext(patch.object(installer, "XROBOTOOLKIT_DIR", self.source))
        self.enterContext(patch.object(installer.sys, "platform", "linux"))
        self.enterContext(patch.object(installer.platform, "machine", return_value="x86_64"))
        self.enterContext(patch.dict(os.environ, {}, clear=True))


class InstallerTests(InstallerFixture):
    def test_bootstrap_accepts_python310_and_checks_compiler(self):
        with (
            patch.object(installer.sys, "version_info", (3, 10)),
            patch.object(installer.shutil, "which", return_value="g++"),
        ):
            installer.validate_source()
        with (
            patch.object(installer.shutil, "which", return_value=None),
            self.assertRaisesRegex(RuntimeError, "g\\+\\+"),
        ):
            installer.validate_source()

    def test_rejects_platform_architecture_and_incomplete_source(self):
        with patch.object(installer.sys, "platform", "win32"), self.assertRaisesRegex(RuntimeError, "Linux only"):
            installer.validate_source()
        with patch.object(installer.platform, "machine", return_value="riscv64"):
            with self.assertRaisesRegex(RuntimeError, "x86_64 or aarch64"):
                installer.validate_source()
        (self.source / "assets/agibot/x2/scene.xml").unlink()
        with self.assertRaisesRegex(RuntimeError, "git submodule update"):
            installer.validate_source()

    def test_default_arguments_and_invalid_names(self):
        args = installer.parse_args([])
        self.assertEqual((args.name, args.jobs, args.reuse), ("robojudo-xr", 2, False))
        for value in ("base", "root", "../env", "/tmp/env", "bad name"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                installer.parse_args(["--name", value])
        with self.assertRaises(SystemExit):
            installer.parse_args(["--jobs", "0"])

    def test_conda_discovery_uses_configured_executable(self):
        with patch.dict(os.environ, {"CONDA_EXE": "/custom/conda"}):
            with patch.object(installer.shutil, "which", side_effect=lambda name: name) as which:
                self.assertEqual(installer.find_conda(), "/custom/conda")
                which.assert_called_once_with("/custom/conda")
        with patch.object(installer.shutil, "which", return_value=None), self.assertRaisesRegex(RuntimeError, "Conda"):
            installer.find_conda()

    def test_environment_lookup_rejects_ambiguous_names(self):
        result = SimpleNamespace(stdout=json.dumps({"envs": ["/a/envs/xr", "/b/envs/xr"]}))
        with patch.object(installer, "run", return_value=result), self.assertRaisesRegex(RuntimeError, "Multiple"):
            installer.environment_prefix("conda", "xr")

    def test_unregistered_target_directory_is_not_treated_as_new(self):
        target = self.directory / "envs/xr"
        target.mkdir(parents=True)
        (target / "unrelated-file").write_text("preserve")
        result = SimpleNamespace(stdout=json.dumps({"envs": [], "envs_dirs": [str(target.parent)]}))
        with patch.object(installer, "run", return_value=result):
            self.assertEqual(installer.environment_prefix("conda", "xr"), target)
        self.assertEqual((target / "unrelated-file").read_text(), "preserve")

    def test_existing_environment_is_untouched_without_reuse(self):
        with (
            patch.object(installer, "find_conda", return_value="conda"),
            patch.object(installer, "environment_prefix", return_value=self.directory),
            patch.object(installer, "run") as run,
            self.assertRaisesRegex(RuntimeError, "already exists"),
        ):
            installer.create_environment(installer.parse_args([]))
        run.assert_not_called()

    def test_concurrent_setup_is_rejected_and_lock_is_reusable(self):
        with patch.object(installer.tempfile, "gettempdir", return_value=str(self.directory)):
            with installer.installation_lock("test-xr"):
                with self.assertRaisesRegex(RuntimeError, "Another setup"):
                    with installer.installation_lock("test-xr"):
                        self.fail("Concurrent setup acquired the same lock")
                # Independent environment names remain usable.
                with installer.installation_lock("another-xr"):
                    pass
            with installer.installation_lock("test-xr"):
                pass

    def test_launcher_keeps_lock_during_setup_and_releases_it_on_failure(self):
        def fail_setup(args):
            with self.assertRaisesRegex(RuntimeError, "Another setup"):
                with installer.installation_lock(args.name):
                    self.fail("Launcher released its lock before setup finished")
            raise RuntimeError("simulated setup failure")

        with (
            patch.object(installer.tempfile, "gettempdir", return_value=str(self.directory)),
            patch.object(installer, "create_environment", side_effect=fail_setup),
            self.assertRaisesRegex(RuntimeError, "simulated setup failure"),
        ):
            installer.main(["--name", "test-xr"])
        with patch.object(installer.tempfile, "gettempdir", return_value=str(self.directory)):
            with installer.installation_lock("test-xr"):
                pass

    def test_creates_target_and_routes_install_to_target_python(self):
        prefix = self.directory / "new-env"
        with (
            patch.object(installer, "find_conda", return_value="conda"),
            patch.object(installer, "environment_prefix", side_effect=[None, prefix]),
            patch.object(installer, "run") as run,
        ):
            installer.create_environment(installer.parse_args(["--name", "test-xr", "--jobs", "3"]))
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(commands[0][:2], ["conda", "create"])
        self.assertIn("python=3.11", commands[0])
        self.assertIn("casadi=3.6.7", commands[0])
        self.assertIn("numpy=2.2.6", commands[0])
        self.assertEqual(commands[1][:2], ["conda", "run"])
        self.assertIn(str(prefix), commands[1])
        self.assertEqual(commands[1][-4:], ["--install-prefix", str(prefix), "--jobs", "3"])
        self.assertNotIn("remove", str(commands))

    def test_successful_setup_persists_target_runtime_for_activation(self):
        prefix = self.directory / "new-env"
        runtime = prefix / "lib/libstdc++.so.6"
        runtime.parent.mkdir(parents=True)
        runtime.touch()
        with (
            patch.object(installer, "find_conda", return_value="conda"),
            patch.object(installer, "environment_prefix", side_effect=[None, prefix]),
            patch.object(installer, "run") as run,
        ):
            installer.create_environment(installer.parse_args(["--name", "test-xr"]))
        commands = [call.args[0] for call in run.call_args_list]
        self.assertIn("--install-prefix", commands[-2])
        self.assertEqual(
            commands[-1],
            ["conda", "env", "config", "vars", "set", "--prefix", str(prefix), f"LD_PRELOAD={runtime}"],
        )

    def test_reuse_validates_target_before_installing_conda_dependencies(self):
        with (
            patch.object(installer, "find_conda", return_value="conda"),
            patch.object(installer, "environment_prefix", return_value=self.directory),
            patch.object(installer, "run") as run,
        ):
            installer.create_environment(installer.parse_args(["--reuse"]))
        commands = [call.args[0] for call in run.call_args_list]
        self.assertIn("--check-prefix", commands[0])
        self.assertEqual(commands[1][:2], ["conda", "install"])
        self.assertIn("--install-prefix", commands[2])
        with (
            patch.object(installer, "find_conda", return_value="conda"),
            patch.object(installer, "environment_prefix", return_value=self.directory),
            patch.object(installer, "run", side_effect=subprocess.CalledProcessError(1, "preflight")) as run,
            self.assertRaises(subprocess.CalledProcessError),
        ):
            installer.create_environment(installer.parse_args(["--reuse"]))
        self.assertEqual(run.call_count, 1)

    def test_download_verifies_checksum_and_reuses_valid_archive(self):
        data = b"archive contents"
        with (
            patch.object(installer, "SOURCE_SHA256", hashlib.sha256(data).hexdigest()),
            patch.object(installer.urllib.request, "urlopen", return_value=io.BytesIO(data)) as download,
        ):
            archive = installer.download_source(self.directory)
            self.assertEqual(archive.read_bytes(), data)
            self.assertEqual(installer.download_source(self.directory), archive)
            download.assert_called_once()

    def test_bad_checksum_never_promotes_download(self):
        with (
            patch.object(installer.urllib.request, "urlopen", return_value=io.BytesIO(b"corrupt")),
            self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"),
        ):
            installer.download_source(self.directory)
        self.assertFalse((self.directory / "pinocchio-4.1.0.tar.gz").exists())

    def test_archive_cannot_escape_destination(self):
        archive = self.directory / "bad.tar"
        with tarfile.open(archive, "w") as output:
            info = tarfile.TarInfo("../escape")
            info.size = 1
            output.addfile(info, io.BytesIO(b"x"))
        with self.assertRaises(tarfile.FilterError):
            installer.extract_source(archive, self.directory / "destination")
        self.assertFalse((self.directory / "escape").exists())

    def test_foreign_search_paths_are_not_passed_to_subprocesses(self):
        with patch.dict(
            os.environ, {"PYTHONPATH": "/foreign", "CASADIPATH": "/other", "LD_PRELOAD": "/foreign.so", "KEEP": "value"}
        ):
            with patch.object(installer.subprocess, "run") as run:
                installer.run(["python", "-I", "--version"])
        env = run.call_args.kwargs["env"]
        self.assertNotIn("PYTHONPATH", env)
        self.assertNotIn("CASADIPATH", env)
        self.assertNotIn("LD_PRELOAD", env)
        self.assertEqual(env["KEEP"], "value")
        self.assertTrue(run.call_args.kwargs["check"])


class TargetInstallerTests(InstallerFixture):
    def setUp(self):
        super().setUp()
        self.prefix = self.directory / "env"
        (self.prefix / "conda-meta").mkdir(parents=True)
        (self.prefix / "conda-meta/casadi-3.6.7-test.json").touch()
        self.purelib = self.prefix / "lib/python3.11/site-packages"
        self.purelib.mkdir(parents=True)
        self.enterContext(patch.object(installer.sys, "prefix", str(self.prefix)))
        self.enterContext(patch.object(installer.sys, "version_info", (3, 11)))
        self.enterContext(patch.object(installer.sysconfig, "get_paths", return_value={"purelib": str(self.purelib)}))
        self.enterContext(patch.dict(os.environ, {"CONDA_PREFIX": str(self.prefix)}))

    def test_wrong_interpreter_or_prefix_fails_before_mutation(self):
        with patch.object(installer.sys, "version_info", (3, 10)), self.assertRaisesRegex(RuntimeError, "Python 3.11"):
            installer.validate_target(self.prefix)
        with self.assertRaisesRegex(RuntimeError, "matching"):
            installer.validate_target(self.directory)
        with patch.dict(os.environ, {"CONDA_PREFIX": "/other"}), self.assertRaisesRegex(RuntimeError, "conda run"):
            installer.validate_target(self.prefix)

    def test_missing_x2_extra_fails_before_install(self):
        (self.source / "pyproject.toml").write_text('[project]\nname = "old"\n')
        with self.assertRaisesRegex(RuntimeError, "x2"):
            installer.validate_target(self.prefix)

    def test_unmanaged_artifacts_are_preserved(self):
        overlay = self.prefix / "lerobot-python"
        overlay.mkdir()
        with self.assertRaisesRegex(RuntimeError, "Unmanaged"):
            installer.validate_target(self.prefix)
        self.assertTrue(overlay.exists())
        self.assertFalse((self.prefix / "share").exists())

    def test_overlay_promotion_rolls_back_if_registration_fails(self):
        overlay = self.prefix / "lerobot-python"
        staged = self.prefix / "staged"
        overlay.mkdir()
        staged.mkdir()
        (overlay / "old").touch()
        (staged / "new").touch()
        registration = self.purelib / "00_lerobot_pinocchio.pth"
        registration.write_text("original")
        with patch.object(Path, "write_text", side_effect=OSError("disk full")), self.assertRaises(OSError):
            installer.promote_overlay(staged, overlay, registration)
        self.assertTrue((overlay / "old").exists())
        self.assertTrue((staged / "new").exists())
        self.assertEqual(registration.read_text(), "original")

    def test_overlay_promotion_rolls_back_failed_native_verification(self):
        overlay = self.prefix / "lerobot-python"
        staged = self.prefix / "staged"
        overlay.mkdir()
        staged.mkdir()
        (overlay / "working").touch()
        (staged / "replacement").touch()
        registration = self.purelib / "00_lerobot_pinocchio.pth"
        registration.write_text("old registration")

        def fail():
            raise RuntimeError("native verification failed")

        with self.assertRaisesRegex(RuntimeError, "native verification failed"):
            installer.promote_overlay(staged, overlay, registration, check=fail)
        self.assertTrue((overlay / "working").exists())
        self.assertTrue((staged / "replacement").exists())
        self.assertEqual(registration.read_text(), "old registration")

    def test_complete_install_and_verified_reuse(self):
        state = self.prefix / "share/robojudo-xr"
        builds = []

        def fake_run(command, **kwargs):
            if "--output" in command:
                staged = Path(command[command.index("--output") + 1])
                staged.mkdir()
                (staged / "pinocchio").mkdir()
                builds.append(command)

        def extract(archive, work):
            source = work / "source"
            source.mkdir()
            return source

        def verify(prefix, work):
            self.assertTrue((prefix / "lerobot-python/pinocchio").exists())
            self.assertEqual((self.purelib / "00_lerobot_pinocchio.pth").read_text(), installer.registration_text())

        with (
            patch.object(installer, "run", side_effect=fake_run) as run,
            patch.object(installer, "download_source", return_value=self.directory / "archive"),
            patch.object(installer, "extract_source", side_effect=extract),
            patch.object(installer, "verify", side_effect=verify),
            patch(
                "importlib.metadata.version", side_effect=lambda name: installer.expected_versions().get(name, "1.0")
            ),
        ):
            installer.install_current(self.prefix, 2)
            self.assertEqual(len(builds), 1)
            manifest = json.loads((state / "manifest.json").read_text())
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["versions"]["casadi"], "3.6.7")
            self.assertFalse((Path(manifest["log_directory"]) / "source").exists())
            installer.install_current(self.prefix, 2)
            self.assertEqual(len(builds), 1)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(commands[0][:4], [installer.sys.executable, "-I", "-m", "pip"])
        self.assertIn(f"{self.source}[x2]", commands[0])
        self.assertNotIn(str(installer.ROOT_DIR) + "[", str(commands))

    def test_failed_dependency_stage_stops_and_records_failure(self):
        with (
            patch.object(installer, "run", side_effect=subprocess.CalledProcessError(1, "pip")) as run,
            self.assertRaisesRegex(RuntimeError, "dependencies"),
        ):
            installer.install_current(self.prefix, 2)
        self.assertEqual(run.call_count, 1)
        manifest = json.loads((self.prefix / "share/robojudo-xr/manifest.json").read_text())
        self.assertFalse(manifest["complete"])
        self.assertEqual(manifest["failed_stage"], "dependencies")


if __name__ == "__main__":
    unittest.main()
