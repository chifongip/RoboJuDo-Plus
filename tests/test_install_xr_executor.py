"""Executor installation resolves valid metadata without pulling in XR/IK dependencies."""

import importlib.metadata
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party/xrobotoolkit_teleop"
SPEC = importlib.util.spec_from_file_location("install_xr_executor", ROOT / "scripts/install_xr_executor.py")
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


class ExecutorInstallerTests(unittest.TestCase):
    def setUp(self):
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.source = directory / "XR executor source"
        module = self.source / "xrobotoolkit_teleop/common/executor_model.py"
        module.parent.mkdir(parents=True)
        module.touch()
        self.enterContext(patch.object(installer, "SOURCE", self.source))

    def test_resolves_executor_extra_before_verifying_profiles(self):
        with patch.object(installer.subprocess, "run") as run:
            installer.main()
        install, verify = run.call_args_list
        self.assertEqual(install.args[0], [sys.executable, "-m", "pip", "install", "-e", f"{self.source}[executor]"])
        self.assertTrue(install.kwargs["check"])
        self.assertEqual(verify.args[0][:2], [sys.executable, "-c"])
        self.assertIn("('x2', 'x2_omnihand')", verify.args[0][2])
        self.assertTrue(verify.kwargs["check"])

    def test_missing_submodule_fails_before_installing(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(installer, "SOURCE", Path(directory)),
                patch.object(installer.subprocess, "run") as run,
                self.assertRaisesRegex(RuntimeError, "submodule"),
            ):
                installer.main()
        run.assert_not_called()

    def test_install_failure_prevents_verification(self):
        failure = subprocess.CalledProcessError(1, "pip")
        with (
            patch.object(installer.subprocess, "run", side_effect=failure) as run,
            self.assertRaises(subprocess.CalledProcessError),
        ):
            installer.main()
        self.assertEqual(run.call_count, 1)


@unittest.skipUnless((SOURCE / "pyproject.toml").is_file(), "initialize the optional XR submodule for metadata tests")
class ExecutorPackageMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        directory = Path(cls.directory.name)
        # Build real distribution metadata in a temporary source tree, avoiding
        # network access, native compilation, and artifacts in the submodule.
        (directory / "pyproject.toml").write_bytes((SOURCE / "pyproject.toml").read_bytes())
        (directory / "README.md").write_text("Metadata fixture\n")
        (directory / "xrobotoolkit_teleop").mkdir()
        (directory / "xrobotoolkit_teleop/__init__.py").touch()
        metadata = directory / "metadata"
        metadata.mkdir()
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from setuptools.build_meta import prepare_metadata_for_build_wheel; "
                "print(prepare_metadata_for_build_wheel('metadata'))",
            ],
            cwd=directory,
            text=True,
            capture_output=True,
            timeout=60,
        )
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        (dist_info,) = metadata.glob("*.dist-info")
        cls.requirements = [Requirement(value) for value in importlib.metadata.PathDistribution(dist_info).requires]

    def dependencies(self, extra):
        return {
            requirement.name: str(requirement.specifier)
            for requirement in self.requirements
            if requirement.marker is None or requirement.marker.evaluate({"extra": extra})
        }

    def test_base_and_executor_metadata_exclude_sdk_and_ik(self):
        self.assertEqual(self.dependencies(""), {"numpy": ""})
        self.assertEqual(self.dependencies("executor"), {"numpy": "", "pin": "==4.1.0", "coal": "==3.0.3"})

    def test_teleop_workflows_keep_sdk_and_ik_dependencies(self):
        self.assertEqual(
            self.dependencies("teleop"),
            {"numpy": "", "meshcat": "", "placo": ">=0.11.0", "xrobotoolkit-sdk": ">=1.1.0"},
        )
        for extra in ("x2", "all"):
            with self.subTest(extra=extra):
                (teleop,) = (
                    requirement
                    for requirement in self.requirements
                    if requirement.name == "xrobotoolkit_teleop" and requirement.marker.evaluate({"extra": extra})
                )
                self.assertEqual(teleop.extras, {"teleop"})


if __name__ == "__main__":
    unittest.main()
