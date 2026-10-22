"""Loopback-only PUB/SUB → CtrlManager → X2 filtering/observer → fake adapter."""

import json
import time
import unittest
from types import SimpleNamespace

import numpy as np
import zmq
from test_x2_arm_observer import FakeAdapter, config

from robojudo.config.x2 import x2_locomanipulation_real
from robojudo.controller.ctrl_manager import CtrlManager
from robojudo.pipeline.four_mode_pipeline import ControlMode
from robojudo.pipeline.x2_locomanipulation_pipeline import X2LocomanipulationPipeline
from robojudo.policy.x2_locomanipulation_policy import X2LocomanipulationPolicy
from robojudo.utils.x2_arm_observer import X2ArmObserver


class UpperBodyTransmissionTests(unittest.TestCase):
    def setUp(self):
        self.context = zmq.Context()
        self.addCleanup(self.context.term)
        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.setsockopt(zmq.LINGER, 0)
        self.addCleanup(lambda: self.publisher.close(linger=0))
        port = self.publisher.bind_to_random_port("tcp://127.0.0.1")
        self.endpoint = f"tcp://127.0.0.1:{port}"
        cfg = x2_locomanipulation_real()
        upper_cfg = cfg.ctrl[-1].model_copy(update={"endpoint": self.endpoint})
        dof = cfg.env.dof
        from robojudo.environment.agibot_cpp_env import AgiBotCppEnv

        env = AgiBotCppEnv.__new__(AgiBotCppEnv)
        env.enabled = True
        env.aimdk = FakeAdapter()
        env.cfg_env = cfg.env
        env.joint_names = list(dof.joint_names)
        env.num_dofs = len(env.joint_names)
        env.default_pos = np.asarray(dof.default_pos)
        env._dof_pos, env._dof_vel = env.default_pos.copy(), np.zeros(env.num_dofs)
        env.position_limits = np.asarray(dof.position_limits)
        env.stiffness, env.damping = np.asarray(dof.stiffness), np.asarray(dof.damping)
        env._control_joint_names = set(env.joint_names)
        env._last_clamp_log_time = 0.0
        env._arm_observer = X2ArmObserver(
            config(), env.joint_names, cfg.env.arm_joint_names, env.stiffness, env.damping, dof.torque_limits, 0.1
        )
        env._arm_observer_eligible = False
        env._observer_reset_generation = 0
        env._observer_feedback_time = time.monotonic()
        env._observer_sample_times = np.full(env.num_dofs, env._observer_feedback_time)
        pipeline = X2LocomanipulationPipeline.__new__(X2LocomanipulationPipeline)
        pipeline.env, pipeline.cfg, pipeline.dt = env, cfg, 0.02
        pipeline.mode = ControlMode.RL_DEFAULT
        pipeline._upper_body_cfg = upper_cfg
        pipeline._upper_body_enabled = False
        pipeline._upper_body_stream_was_fresh = False
        pipeline._upper_body_generation = None
        pipeline._configure_upper_body_override()
        pipeline.ctrl_manager = CtrlManager([upper_cfg], env=env)
        self.addCleanup(pipeline.ctrl_manager.close)
        self.receiver = pipeline.ctrl_manager.controllers.UpperBodyZmqCtrl.inst
        inner = X2LocomanipulationPolicy.__new__(X2LocomanipulationPolicy)
        self.policy_target = env.default_pos.astype(np.float32)
        self.policy_target[:15] += 0.005  # Track all policy-owned legs and waist.
        pipeline.policy = SimpleNamespace(
            policy=inner,
            get_observation=lambda env_data, ctrl_data: ({}, {}),
            get_pd_target=lambda obs: self.policy_target.copy(),
        )
        pipeline._set_upper_body_enabled(True)
        self.pipeline = pipeline
        self.arms = cfg.env.arm_joint_names

    def receive(self, positions):
        # PUB has no connection acknowledgement; retry snapshots until SUB sees
        # this exact one. This models the existing continuously publishing source.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            self.publisher.send_json({"positions": positions})
            data = self.pipeline.ctrl_manager.get_ctrl_data(None)
            if data.UpperBodyZmqCtrl.fresh and dict(data.UpperBodyZmqCtrl.joint_positions) == positions:
                return data
            time.sleep(0.005)
        self.fail("loopback subscriber did not receive the expected snapshot")

    def apply(self, data):
        self.pipeline.env._observer_feedback_time = time.monotonic()
        self.pipeline.env._observer_sample_times[:] = self.pipeline.env._observer_feedback_time
        target, _ = self.pipeline._step_rl_policy({}, data, dry_run=False)
        sent, effort = self.pipeline.env.aimdk.commands[-1]
        np.testing.assert_array_equal(sent, np.clip(target, *self.pipeline.env.position_limits.T))
        nonarm = [i for i, name in enumerate(self.pipeline.env.joint_names) if name not in self.arms]
        np.testing.assert_array_equal(target[nonarm], self.policy_target[nonarm])
        self.assertEqual(effort.shape, (14,))
        observer = self.pipeline.env._arm_observer
        np.testing.assert_array_equal(observer.observer.previous_target, sent[observer.indices])
        return target, effort

    def test_named_snapshots_reach_final_targets_and_atomic_observer_commands(self):
        # The real C++ publisher emits this same positions-only JSON envelope.
        positions = {
            name: float(self.pipeline.env.default_pos[self.pipeline.env.joint_names.index(name)] + 0.01 * (index + 1))
            for index, name in reversed(list(enumerate(self.arms)))
        }
        data = self.receive(positions)
        previous = self.pipeline._upper_body_filtered.copy()
        target, effort = self.apply(data)
        self.assertTrue(self.pipeline._upper_body_stream_was_fresh)
        for index, name in enumerate(self.arms):
            expected = 0.95 * previous[index] + 0.05 * positions[name]
            self.assertAlmostEqual(target[self.pipeline.env.joint_names.index(name)], expected, places=7)
        self.assertTrue(np.all(abs(target[self.pipeline._upper_body_indices] - previous) <= 0.0200001))
        np.testing.assert_array_equal(effort, 0)  # First observer sample initializes history.
        time.sleep(0.02)
        self.pipeline.env._dof_vel[self.pipeline.env._arm_observer.indices] = -0.1
        _, effort = self.apply(self.receive(positions))
        # The first receipt follows the initial snapshot; that partially unknown
        # interval initializes only. The next interval has complete command history.
        np.testing.assert_array_equal(effort, 0)
        time.sleep(0.02)
        _, effort = self.apply(self.receive(positions))
        self.assertGreater(np.max(abs(effort)), 0)
        np.testing.assert_array_equal(effort, self.pipeline.env._arm_observer.issued_nm)

    def test_malformed_and_multipart_packets_cannot_replace_a_valid_snapshot(self):
        positions = {name: 0.1 for name in self.arms}
        self.receive(positions)
        # Subscription setup can leave an earlier retried valid snapshot in
        # flight. Drain until quiet before measuring invalid-message freshness.
        while self.receiver._socket.poll(20, zmq.POLLIN):
            self.pipeline.ctrl_manager.get_ctrl_data(None)
        previous_time = self.receiver._last_received_at
        poison = json.dumps({"positions": {self.arms[0]: 1.0}}).encode()
        self.publisher.send_multipart([b"topic", poison])
        self.assertTrue(self.receiver._socket.poll(500, zmq.POLLIN))
        data = self.pipeline.ctrl_manager.get_ctrl_data(None)
        self.assertEqual(dict(data.UpperBodyZmqCtrl.joint_positions), positions)
        self.assertEqual(self.receiver._last_received_at, previous_time)
        self.assertGreater(self.receiver._last_invalid_log_at, float("-inf"))
        self.apply(data)

    def test_expired_or_previous_session_data_is_rechecked_after_inference(self):
        data = self.receive({name: 0.4 for name in self.arms})
        # Emulate data ageing during policy inference without changing the socket.
        for name in data.UpperBodyZmqCtrl.joint_received_at:
            data.UpperBodyZmqCtrl.joint_received_at[name] -= 1.0
        self.apply(data)
        self.assertFalse(self.pipeline._upper_body_stream_was_fresh)
        np.testing.assert_array_equal(self.pipeline._upper_body_filtered, self.pipeline._upper_body_default)
        old_data = self.receive({name: 0.3 for name in self.arms})
        self.pipeline._set_upper_body_enabled(False)
        self.pipeline._set_upper_body_enabled(True)
        self.apply(old_data)
        self.assertFalse(self.pipeline._upper_body_stream_was_fresh)
        self.apply(self.receive({name: 0.2 for name in self.arms}))
        self.assertTrue(self.pipeline._upper_body_stream_was_fresh)

    def test_timeout_then_publisher_reconnection_resumes_with_bounded_targets(self):
        self.apply(self.receive({name: 0.4 for name in self.arms}))
        self.publisher.close(linger=0)
        deadline = time.monotonic() + 0.3
        while time.monotonic() < deadline:
            data = self.pipeline.ctrl_manager.get_ctrl_data(None)
            previous = self.pipeline._upper_body_filtered.copy()
            target, _ = self.apply(data)
            self.assertTrue(np.all(abs(target[self.pipeline._upper_body_indices] - previous) <= 0.0200001))
            time.sleep(0.02)
        self.assertFalse(data.UpperBodyZmqCtrl.fresh)
        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.setsockopt(zmq.LINGER, 0)
        self.publisher.bind(self.endpoint)
        self.apply(self.receive({name: 0.2 for name in self.arms}))
        self.assertTrue(self.pipeline._upper_body_stream_was_fresh)

    def test_large_finite_positions_are_clipped_without_float32_overflow(self):
        data = self.receive({name: 1e300 for name in self.arms})
        with np.errstate(over="raise", invalid="raise"):
            target, _ = self.apply(data)
        self.assertTrue(np.isfinite(target).all())


if __name__ == "__main__":
    unittest.main()
