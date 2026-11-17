#!/usr/bin/env python3
"""Install numeric XR executor support into the invoking RoboJuDo interpreter."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party/xrobotoolkit_teleop"


def main():
    if not (SOURCE / "xrobotoolkit_teleop/common/executor_model.py").is_file():
        raise RuntimeError("Initialize/update the xrobotoolkit_teleop submodule first")
    subprocess.run([sys.executable, "-m", "pip", "install", "pin==4.1.0", "coal==3.0.3"], check=True)
    # Runtime only imports numeric modules; do not install SDK, Placo or IK dependencies here.
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(SOURCE)], check=True)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from xrobotoolkit_teleop.common.executor_model import ExecutorModel\n"
            "from xrobotoolkit_teleop.common.teleop_settings import TeleopSettings\n"
            "from xrobotoolkit_teleop.integrations.robot_profiles import load_profile\n"
            "settings = TeleopSettings()\n"
            "for name in ('x2', 'x2_omnihand'):\n"
            "    with load_profile(name, settings) as profile:\n"
            "        profile.validate()\n"
            "        model = ExecutorModel(profile, settings)\n"
            "        print(name, 'geometry pairs:', model.collision_checker.pair_count)\n"
            "        print('Model fingerprint:', model.fingerprint)\n",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
