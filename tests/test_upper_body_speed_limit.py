import unittest
from types import SimpleNamespace

import numpy as np

from robojudo.controller.ctrl_cfgs import Gr00tZmqCtrlCfg, UpperBodyZmqCtrlCfg
from robojudo.pipeline.upper_body_zmq_pipeline import UpperBodyZmqPipelineMixin


class TestUpperBodySpeedLimit(unittest.TestCase):
    def make_pipeline(self, dt=0.02, alpha=0.0):
        pipeline = UpperBodyZmqPipelineMixin.__new__(UpperBodyZmqPipelineMixin)
        pipeline._upper_body_cfg = UpperBodyZmqCtrlCfg(joint_names=["left", "right"], ema_alpha=alpha)
        pipeline.dt = dt
        pipeline.env = SimpleNamespace(
            joint_names=["leg", "left", "right"],
            default_pos=np.zeros(3),
            dof_pos=np.asarray([0.0, 0.3, -0.3]),
            position_limits=np.asarray([[-2.0, 2.0]] * 3),
        )
        pipeline._upper_body_action_joint_names = lambda: ["leg"]
        pipeline._configure_upper_body_override()
        pipeline._upper_body_enabled = True
        pipeline._upper_body_stream_was_fresh = False
        return pipeline

    def step(self, pipeline, positions=None, fresh=True):
        previous = pipeline._upper_body_filtered.copy()
        result = pipeline._apply_upper_body_override(
            np.asarray([0.7, 0.0, 0.0]),
            {"UpperBodyZmqCtrl": {"fresh": fresh, "joint_positions": positions or {}}},
        )
        self.assertAlmostEqual(result[0], 0.7)
        self.assertTrue(np.all(np.abs(result[1:] - previous) <= pipeline.dt + 1e-7))
        np.testing.assert_array_equal(result[1:], pipeline._upper_body_filtered)
        return result[1:]

    def test_live_timeout_disable_and_reconnection(self):
        for dt in (0.01, 0.02, 0.04):
            for alpha in (0.0, 0.95):
                with self.subTest(dt=dt, alpha=alpha):
                    pipeline = self.make_pipeline(dt, alpha)
                    np.testing.assert_allclose(pipeline._upper_body_filtered, [0.3, -0.3])
                    self.step(pipeline, {"left": 2.0, "right": -2.0})
                    self.step(pipeline, fresh=False)
                    self.step(pipeline, {"left": -2.0, "right": 2.0})
                    pipeline._set_upper_body_enabled(False)
                    for _ in range(500):
                        target = self.step(pipeline, {"left": 2.0, "right": -2.0})
                        self.assertGreaterEqual(target[0], 0.0)
                        self.assertLessEqual(target[1], 0.0)
                    np.testing.assert_array_equal(target, [0.0, 0.0])
                    pipeline._set_upper_body_enabled(True)
                    self.step(pipeline, {"left": -2.0, "right": 2.0})

    def test_joint_limits_and_convergence(self):
        pipeline = self.make_pipeline()
        for _ in range(250):
            target = self.step(pipeline, {"left": 100.0, "right": -100.0})
            self.assertTrue(np.all(np.abs(target) <= 2.0))
        np.testing.assert_array_equal(target, [2.0, -2.0])

    def test_snap_cannot_bypass_limit(self):
        pipeline = self.make_pipeline(dt=0.0005, alpha=0.95)
        pipeline._upper_body_filtered[:] = 0.0
        target = self.step(pipeline, {"left": 0.0009, "right": -0.0009})
        np.testing.assert_allclose(target, [0.0005, -0.0005])

    def test_config_limits(self):
        self.assertEqual(UpperBodyZmqCtrlCfg(joint_names=["left"]).max_joint_velocity_rad_s, 1.0)
        self.assertEqual(Gr00tZmqCtrlCfg(joint_names=["left"]).max_joint_velocity_rad_s, 1.0)
        for cls in (UpperBodyZmqCtrlCfg, Gr00tZmqCtrlCfg):
            for value in (0.0, -1.0, float("nan"), float("inf"), -float("inf")):
                with self.subTest(cls=cls, value=value), self.assertRaises(ValueError):
                    cls(joint_names=["left"], max_joint_velocity_rad_s=value)
