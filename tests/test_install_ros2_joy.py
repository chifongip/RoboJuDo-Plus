import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

INSTALLER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "install_ros2_joy.py"
INSTALLER_SPEC = importlib.util.spec_from_file_location("install_ros2_joy", INSTALLER_PATH)
assert INSTALLER_SPEC is not None and INSTALLER_SPEC.loader is not None
install_ros2_joy = importlib.util.module_from_spec(INSTALLER_SPEC)
INSTALLER_SPEC.loader.exec_module(install_ros2_joy)


class TestInstallRos2Joy(unittest.TestCase):
    def test_main_installs_package_for_active_interpreter(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = Path(temp_dir) / "ros2_joy_cpp"
            package_dir.mkdir()
            (package_dir / "pyproject.toml").touch()

            with patch.object(install_ros2_joy, "validate_environment"):
                with patch.object(install_ros2_joy, "run") as run:
                    with patch.object(install_ros2_joy, "ROS2_JOY_CPP_DIR", package_dir):
                        install_ros2_joy.main()

        self.assertEqual(
            run.call_args_list,
            [
                call(
                    [
                        install_ros2_joy.sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "scikit-build-core",
                        "pybind11",
                    ]
                ),
                call(
                    [
                        install_ros2_joy.sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--force-reinstall",
                        "--no-build-isolation",
                        package_dir.as_posix(),
                    ]
                ),
                call(
                    [
                        install_ros2_joy.sys.executable,
                        "-c",
                        "import ros2_joy_cpp; print(ros2_joy_cpp.__file__)",
                    ]
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
