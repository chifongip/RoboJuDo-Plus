"""Verify both serial-selected HCAN hands and read angles without sending motion commands."""

import argparse
import json
from pathlib import Path

from robojudo.controller.ctrl_cfgs import OmniHandCfg
from robojudo.controller.omnihand_runtime import OmniHandRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Robot-local JSON serial configuration")
    args = parser.parse_args()
    cfg = OmniHandCfg.model_validate(json.loads(args.config.read_text()))
    runtime = OmniHandRuntime(cfg, read_only=True, require_serials=True)
    try:
        state = runtime.get_data()
        if not state["joint_state_fresh"]:
            raise RuntimeError("hand measurements are stale")
        print(json.dumps({"verified": True, "joint_angles_rad": state["positions"]}, indent=2))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
