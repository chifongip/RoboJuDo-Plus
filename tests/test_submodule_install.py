import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import submodule_install


class TestSubmoduleInstall(unittest.TestCase):
    def _run_install(self, clean):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "module"
            path.mkdir()
            (path / ".git").touch()
            config = {"example": {"install": False, "path": path.as_posix(), "installer": "installer.py"}}
            commands = []

            def capture(command, cwd=None, required=False):
                del cwd, required
                commands.append(command)

            with patch.object(submodule_install, "load_config", return_value=config):
                with patch.object(submodule_install, "run", side_effect=capture):
                    submodule_install.install_submodules(["example"], clean=clean)
            return commands

    def test_existing_submodule_is_preserved_by_default(self):
        commands = self._run_install(clean=False)
        self.assertFalse(any("reset --hard" in command for command in commands))
        self.assertFalse(any("clean -fd" in command for command in commands))
        self.assertFalse(any("submodule update --init" in command for command in commands))

    def test_missing_submodule_is_initialized(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "module"
            config = {"example": {"install": False, "path": path.as_posix(), "installer": "installer.py"}}
            commands = []

            with patch.object(submodule_install, "load_config", return_value=config):
                with patch.object(submodule_install, "run", side_effect=lambda command, **_: commands.append(command)):
                    submodule_install.install_submodules(["example"])

        self.assertTrue(any("submodule update --init" in command for command in commands))

    def test_clean_requires_explicit_option(self):
        commands = self._run_install(clean=True)
        self.assertTrue(any("reset --hard" in command for command in commands))
        self.assertTrue(any("clean -fd" in command for command in commands))
        self.assertTrue(any("submodule update --init" in command for command in commands))

    def test_local_package_skips_git_submodule_initialization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "module"
            path.mkdir()
            config = {
                "example": {
                    "install": False,
                    "local": True,
                    "path": path.as_posix(),
                    "installer": "installer.py",
                }
            }
            commands = []

            with patch.object(submodule_install, "load_config", return_value=config):
                with patch.object(submodule_install, "run", side_effect=lambda command, **_: commands.append(command)):
                    submodule_install.install_submodules(["example"])

        self.assertFalse(any("git submodule" in command for command in commands))
        self.assertTrue(any("installer.py" in command for command in commands))


if __name__ == "__main__":
    unittest.main()
