import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "install_omnihand.py"
SPEC = importlib.util.spec_from_file_location("robojudo_install_omnihand", SCRIPT_PATH)
INSTALLER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSTALLER)


class TestInstallOmniHand(unittest.TestCase):
    def test_selects_wheel_for_active_linux_python_and_architecture(self):
        python_tag = f"cp{INSTALLER.sys.version_info.major}{INSTALLER.sys.version_info.minor}"
        with tempfile.TemporaryDirectory() as temporary:
            sdk_dir = Path(temporary)
            for machine, directory in (("x86_64", "x64"), ("aarch64", "aarch64")):
                with self.subTest(machine=machine):
                    wheel_dir = sdk_dir / "linux" / directory / "python"
                    wheel_dir.mkdir(parents=True)
                    expected = wheel_dir / f"omnihand-1.0-{python_tag}-{python_tag}-linux_{machine}.whl"
                    expected.touch()
                    (wheel_dir / f"omnihand-1.0-cp39-cp39-linux_{machine}.whl").touch()
                    with (
                        patch.object(INSTALLER, "SDK_DIR", sdk_dir),
                        patch.object(INSTALLER.platform, "system", return_value="Linux"),
                        patch.object(INSTALLER.platform, "python_implementation", return_value="CPython"),
                        patch.object(INSTALLER.platform, "machine", return_value=machine),
                    ):
                        self.assertEqual(INSTALLER.select_wheel(), expected)

    def test_rejects_unsupported_architecture(self):
        with patch.object(INSTALLER.platform, "machine", return_value="riscv64"):
            with self.assertRaisesRegex(RuntimeError, "Unsupported OmniHand architecture"):
                INSTALLER.select_wheel()


if __name__ == "__main__":
    unittest.main()
