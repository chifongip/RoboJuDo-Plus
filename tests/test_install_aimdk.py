import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

INSTALLER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "install_aimdk.py"
INSTALLER_SPEC = importlib.util.spec_from_file_location("install_aimdk", INSTALLER_PATH)
assert INSTALLER_SPEC is not None and INSTALLER_SPEC.loader is not None
install_aimdk = importlib.util.module_from_spec(INSTALLER_SPEC)
INSTALLER_SPEC.loader.exec_module(install_aimdk)


class TestInstallAimdk(unittest.TestCase):
    def test_main_initializes_sdk_and_backend_submodules(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            aimdk_dir = temp_path / "third_party" / "aimdk"
            aimdk_cpp_dir = temp_path / "packages" / "aimdk_cpp"
            (aimdk_dir / "src" / "aimdk_msgs").mkdir(parents=True)
            (aimdk_dir / "src" / "aimdk_msgs" / "package.xml").touch()
            (aimdk_dir / "install").mkdir()
            (aimdk_dir / "install" / "setup.bash").touch()
            aimdk_cpp_dir.mkdir(parents=True)
            (aimdk_cpp_dir / "pyproject.toml").touch()

            with patch.object(install_aimdk, "validate_environment"):
                with patch.object(install_aimdk, "run") as run:
                    with patch.object(install_aimdk, "sourced_environment", return_value={"PATH": "/bin"}):
                        with patch.object(install_aimdk, "ROOT_DIR", temp_path):
                            with patch.object(install_aimdk, "AIMDK_DIR", aimdk_dir):
                                with patch.object(install_aimdk, "AIMDK_SETUP", aimdk_dir / "install" / "setup.bash"):
                                    with patch.object(install_aimdk, "AIMDK_CPP_DIR", aimdk_cpp_dir):
                                        install_aimdk.main()

        self.assertEqual(
            run.call_args_list[0],
            call(["git", "submodule", "update", "--init", "third_party/aimdk", "packages/aimdk_cpp"]),
        )
        self.assertEqual(
            run.call_args_list[-1],
            call(
                [install_aimdk.sys.executable, "-c", "import aimdk_cpp; print(aimdk_cpp.__file__)"],
                env={"PATH": "/bin"},
            ),
        )

    def test_main_preserves_initialized_submodules(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            aimdk_dir = temp_path / "third_party" / "aimdk"
            aimdk_cpp_dir = temp_path / "packages" / "aimdk_cpp"
            (aimdk_dir / "src" / "aimdk_msgs").mkdir(parents=True)
            (aimdk_dir / "src" / "aimdk_msgs" / "package.xml").touch()
            (aimdk_dir / "install").mkdir()
            (aimdk_dir / "install" / "setup.bash").touch()
            (aimdk_dir / ".git").touch()
            aimdk_cpp_dir.mkdir(parents=True)
            (aimdk_cpp_dir / "pyproject.toml").touch()
            (aimdk_cpp_dir / ".git").touch()

            with patch.object(install_aimdk, "validate_environment"):
                with patch.object(install_aimdk, "run") as run:
                    with patch.object(install_aimdk, "sourced_environment", return_value={"PATH": "/bin"}):
                        with patch.object(install_aimdk, "ROOT_DIR", temp_path):
                            with patch.object(install_aimdk, "AIMDK_DIR", aimdk_dir):
                                with patch.object(install_aimdk, "AIMDK_SETUP", aimdk_dir / "install" / "setup.bash"):
                                    with patch.object(install_aimdk, "AIMDK_CPP_DIR", aimdk_cpp_dir):
                                        install_aimdk.main()

        git_update = ["git", "submodule", "update"]
        self.assertFalse(any(call_args.args[0][:3] == git_update for call_args in run.call_args_list))


if __name__ == "__main__":
    unittest.main()
