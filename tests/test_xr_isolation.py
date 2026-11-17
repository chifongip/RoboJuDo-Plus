"""Ordinary pipeline execution stays independent of optional XR/native imports."""

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ISOLATION_CHECK = r"""
import importlib.abc
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

installed = sys.argv[1] == "installed"
blocked = {"xrobotoolkit_sdk", "placo", "meshcat", "pinocchio", "coal"}
if not installed:
    blocked.add("xrobotoolkit_teleop")

class NoOptionalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            raise AssertionError(f"Unexpected optional dependency import: {fullname}")

sys.meta_path.insert(0, NoOptionalImports())
if installed:
    # Preload the installed shared model package as another application might.
    # Collision-disabled numeric models must not initialize native/IK/SDK code.
    from xrobotoolkit_teleop.common.executor_model import ExecutorModel
    from xrobotoolkit_teleop.common.teleop_settings import TeleopSettings
    from xrobotoolkit_teleop.integrations.robot_profiles import load_profile
    for name in ("x2", "x2_omnihand"):
        settings = TeleopSettings(collision_checking=False)
        with load_profile(name, settings) as profile:
            profile.validate()
            model = ExecutorModel(profile, settings)
            assert model.robot is None and model.collision_checker is None

from robojudo.config.g1.g1_cfg import g1_23_locomanipulation_default
from robojudo.config.x2 import x2_locomanipulation, x2_locomimic
from robojudo.controller.ctrl_cfgs import UpperBodyZmqCtrlCfg
from robojudo.controller.upper_body_casia_hand_zmq_ctrl import UpperBodyCasiaHandZmqCtrl
from robojudo.controller.upper_body_omnihand_zmq_ctrl import UpperBodyOmniHandZmqCtrl
from robojudo.pipeline.four_mode_pipeline import ControlMode
from robojudo.pipeline.g1_locomanipulation_pipeline import G1LocomanipulationPipeline
from robojudo.pipeline.x2_locomanipulation_loco_mimic_pipeline import X2LocomanipulationLocoMimicPipeline
from robojudo.pipeline.x2_locomanipulation_pipeline import X2LocomanipulationPipeline

configs = [x2_locomanipulation(), x2_locomimic(), g1_23_locomanipulation_default()]
for cfg in configs:
    for ctrl in cfg.ctrl:
        if isinstance(ctrl, UpperBodyZmqCtrlCfg):
            assert ctrl.feedback_endpoint is None

results = {}
for cfg, pipeline_class in (
    (configs[0], X2LocomanipulationPipeline),
    (configs[2], G1LocomanipulationPipeline),
):
    pipeline = pipeline_class.__new__(pipeline_class)
    pipeline.cfg = cfg
    pipeline.dt = 1 / cfg.policy.freq
    pipeline.mode = ControlMode.RL_DEFAULT
    pipeline._upper_body_cfg = next(c for c in cfg.ctrl if isinstance(c, UpperBodyZmqCtrlCfg))
    pipeline._upper_body_enabled = True
    pipeline._upper_body_stream_was_fresh = False
    pipeline._upper_body_generation = None
    default = np.asarray(cfg.env.dof.default_pos, dtype=np.float32)
    pipeline.env = SimpleNamespace(
        joint_names=cfg.env.dof.joint_names,
        dof_pos=default.copy(),
        default_pos=default.copy(),
        position_limits=np.asarray(cfg.env.dof.position_limits),
        step=Mock(),
    )
    pipeline._configure_upper_body_override()
    policy_target = default.copy()
    policy_target[0] += 0.01
    pipeline.policy = SimpleNamespace(
        get_observation=lambda env_data, ctrl_data: ({}, {}),
        get_pd_target=lambda obs: policy_target.copy(),
    )
    commands = {
        name: float(default[index] + 0.1)
        for name, index in zip(pipeline._upper_body_cfg.joint_names, pipeline._upper_body_indices, strict=True)
    }
    data = {"UpperBodyZmqCtrl": {"fresh": True, "joint_positions": commands}}
    target, _ = pipeline._step_rl_policy({}, data, dry_run=False)
    pipeline.env.step.assert_called_once()
    np.testing.assert_array_equal(pipeline.env.step.call_args.args[0], target)
    arms = pipeline._upper_body_indices
    assert np.any(target[arms] != default[arms])
    assert np.max(np.abs(target[arms] - default[arms])) <= (
        pipeline._upper_body_cfg.max_joint_velocity_rad_s * pipeline.dt + 1e-6
    )
    other = np.setdiff1d(np.arange(len(default)), arms)
    np.testing.assert_array_equal(target[other], policy_target[other])
    # A stale stream holds the issued arms while preserving policy-owned joints.
    data["UpperBodyZmqCtrl"]["fresh"] = False
    held, _ = pipeline._step_rl_policy({}, data, dry_run=False)
    np.testing.assert_array_equal(held, target)
    assert pipeline._feedback_receiver() is None
    results[cfg.robot] = target.tolist()

assert not any(name.split(".")[0] in blocked for name in sys.modules)
print("XR_ISOLATION_RESULT=" + json.dumps(results, sort_keys=True))
"""


class XrIsolationTests(unittest.TestCase):
    def check_pipeline(self, mode):
        result = subprocess.run(
            [sys.executable, "-c", ISOLATION_CHECK, mode],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        line = next(line for line in result.stdout.splitlines() if line.startswith("XR_ISOLATION_RESULT="))
        return json.loads(line.split("=", 1)[1])

    def test_ordinary_pipelines_execute_without_xr_dependencies(self):
        self.assertEqual(set(self.check_pipeline("absent")), {"x2", "g1"})

    @unittest.skipUnless(importlib.util.find_spec("xrobotoolkit_teleop"), "install XR executor support")
    def test_installed_xr_models_do_not_change_ordinary_pipeline_targets(self):
        self.assertEqual(self.check_pipeline("absent"), self.check_pipeline("installed"))


if __name__ == "__main__":
    unittest.main()
