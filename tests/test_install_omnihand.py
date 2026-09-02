import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "install_omnihand.py"
SPEC = importlib.util.spec_from_file_location("robojudo_install_omnihand", SCRIPT_PATH)
INSTALLER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSTALLER)


class TestInstallOmniHand(unittest.TestCase):
    def test_selects_wheel_for_active_linux_python_and_architecture(self):
        wheel = INSTALLER.select_wheel()

        self.assertTrue(wheel.is_file())
        self.assertIn(f"cp{INSTALLER.sys.version_info.major}{INSTALLER.sys.version_info.minor}", wheel.name)
        expected_arch = INSTALLER.ARCHITECTURE_DIRS[INSTALLER.platform.machine().lower()]
        self.assertEqual(wheel.parents[1].name, expected_arch)

    def test_rejects_unsupported_architecture(self):
        with patch.object(INSTALLER.platform, "machine", return_value="riscv64"):
            with self.assertRaisesRegex(RuntimeError, "Unsupported OmniHand architecture"):
                INSTALLER.select_wheel()


if __name__ == "__main__":
    unittest.main()
