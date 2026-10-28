"""Robot-independent physics coverage using a small model with arbitrary names."""

import tempfile
import unittest
from itertools import product
from pathlib import Path

import numpy as np

from robojudo.config.observer_cfg import JointObserverCfg
from robojudo.environment import env_registry
from robojudo.environment.env_cfgs import ObserverMujocoEnvCfg
from robojudo.tools.tool_cfgs import DoFConfig

MODEL = """
<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="10" contype="0" conaffinity="0"/>
      <body name="a">
        <joint name="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.3" size="0.03" mass="1" contype="0" conaffinity="0"/>
        <body name="b" pos="0 0 0.3">
          <joint name="middle" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.3" size="0.03" mass="1" contype="0" conaffinity="0"/>
          <body name="c" pos="0 0 0.3">
            <joint name="wrist" axis="0 1 0"/>
            <geom type="sphere" size="0.05" mass="1" contype="0" conaffinity="0"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="wrist"/>
    <motor joint="hinge"/>
    <motor joint="middle"/>
  </actuator>
</mujoco>
"""


class GenericObserverTests(unittest.TestCase):
    def config(self, names, mode="ACTIVE", algorithm="linearized"):
        return ObserverMujocoEnvCfg(
            xml="unused.xml",
            headless=True,
            visualize_extras=False,
            sim_dt=0.005,
            sim_decimation=4,
            dof=DoFConfig(
                joint_names=["hinge", "middle", "wrist"],
                default_pos=[0.0] * 3,
                stiffness=[10.0] * 3,
                damping=[1.0] * 3,
                torque_limits=[4.0] * 3,
                position_limits=[[-3.0, 3.0]] * 3,
            ),
            feedforward_effort_limits=dict.fromkeys(names, 0.5),
            observer=JointObserverCfg(
                algorithm=algorithm,
                mode=mode,
                joint_names=names,
                inertia=0.1,
                correction_limit_nm=0.5,
                slew_nm_s=1.0,
            ),
        )

    def plant(self, cfg):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        xml = Path(directory.name) / "robot.xml"
        xml.write_text(MODEL)
        cfg.xml = str(xml)
        env = env_registry.get("ObserverMujocoEnv")(cfg)
        self.addCleanup(env.shutdown)
        return env

    def test_arbitrary_joint_count_order_and_modes(self):
        for names in (["wrist", "hinge"], ["middle"]):
            for mode, algorithm in product(("ACTIVE", "SHADOW", "OFF"), ("linearized", "nonlinear")):
                with self.subTest(names=names, mode=mode, algorithm=algorithm):
                    env = self.plant(self.config(names, mode, algorithm))
                    indices = [env.joint_names.index(name) for name in names]
                    for name in names:
                        env.data.qfrc_applied[env.model.joint(name).dofadr[0]] = 0.3
                    peak = 0.0
                    previous = np.zeros(len(names))
                    for _ in range(30):
                        env.step(np.zeros(3))
                        issued = env.last_feedforward_effort[indices]
                        np.testing.assert_array_equal(np.delete(env.last_feedforward_effort, indices), 0)
                        self.assertTrue(np.all(abs(issued) <= 0.5))
                        self.assertTrue(np.all(abs(issued - previous) <= 0.02 + 1e-9))
                        self.assertTrue(np.all(abs(env.last_applied_torque) <= env.torque_limits))
                        peak = max(peak, float(np.max(abs(issued))))
                        previous = issued.copy()
                    if mode == "ACTIVE":
                        self.assertGreater(peak, 0)
                        self.assertEqual(env.get_observer_status()["joint_names"], names)
                        np.testing.assert_array_equal(env._observer.issued_nm, issued)
                        status = env.get_observer_status()
                        status["joint_names"].reverse()
                        status["joint_names"].append("unrelated_joint")
                        self.assertEqual(env.get_observer_status()["joint_names"], names)
                        np.testing.assert_array_equal(env._observer.issued_nm, issued)
                    else:
                        self.assertEqual(peak, 0)
                    if mode == "SHADOW":
                        self.assertGreater(np.max(abs(env._observer.proposed_nm)), 0)

    def test_configuration_rejects_incompatible_joints_and_parameters(self):
        cfg = self.config(["hinge"])
        for changes in (
            {"joint_names": ["unknown"]},
            {"joint_names": ["hinge", "hinge"]},
            {"inertia": [0.1, 0.2]},
            {"inertia": float("nan")},
            {"friction_viscosity": -1.0},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ObserverMujocoEnvCfg.model_validate(
                    {**cfg.model_dump(), "observer": {**cfg.observer.model_dump(), **changes}}
                )
        cfg.observer.correction_limit_fraction = 1.1
        with self.assertRaises(ValueError):
            self.plant(cfg)


if __name__ == "__main__":
    unittest.main()
