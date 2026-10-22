"""Payload mechanics and synchronization without a robot policy or hardware."""

import sys
import unittest
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from x2_compliant_payload import CompliantPayload
from x2_robustness_sim import X2CompliantPayloadEvaluationEnv


class TestPayload(unittest.TestCase):
    def equilibrium(self):
        model = CompliantPayload()
        model.initialize(np.zeros(3), np.zeros(3))
        model.position = model.mass * model.gravity / model.stiffness
        model.velocity[:] = 0
        model.initial_energy = model.energy()
        return model

    def test_equilibrium_and_supported_load_transfer(self):
        p = self.equilibrium()
        for fraction in (0, 0.1, 0.5, 1, 0.5, 0):
            force = p.evaluate(np.zeros(3), np.zeros(3), fraction)
            np.testing.assert_allclose(force, [0, 0, -20 * fraction], atol=1e-12)
            np.testing.assert_allclose(2 * force + fraction * p.force, 0, atol=1e-12)
            np.testing.assert_allclose(p.mass * p.gravity + fraction * p.force + p.support, 0, atol=1e-12)
            p.advance(0.005, np.zeros(3), np.zeros(3))
        self.assertAlmostEqual(p.telemetry()["payload_energy_residual_j"], 0, places=10)
        p.detach()
        np.testing.assert_array_equal(p.hand_force, np.zeros(3))

    def test_restoring_force_and_velocity_damping(self):
        p = self.equilibrium()
        p.position[0] += 0.01
        p.velocity[1] += 0.1
        hand = p.evaluate(np.zeros(3), np.zeros(3), 1)
        self.assertLess(p.force[0], 0)
        self.assertLess(p.force[1], 0)
        self.assertGreater(hand[0], 0)
        np.testing.assert_allclose(p.force, [-1, -0.1 * p.damping[1], 40])

    def test_fixed_anchor_energy_decays_and_step_refinement_converges(self):
        states = []
        for dt in (0.005, 0.0025, 0.00125):
            p = CompliantPayload()
            p.initialize(np.zeros(3), np.zeros(3))
            initial_energy = None
            equilibrium = p.mass * p.gravity / p.stiffness
            for k in range(round(8 / dt)):
                p.evaluate(np.zeros(3), np.zeros(3), 1)
                delta = p.position - equilibrium
                energy = 0.5 * p.mass * np.dot(p.velocity, p.velocity) + 0.5 * np.dot(p.stiffness * delta, delta)
                if initial_energy is None:
                    initial_energy = energy
                p.advance(dt, np.zeros(3), np.zeros(3))
                if k == round(1 / dt) - 1:
                    states.append(p.position.copy())
            self.assertLess(energy, initial_energy * 1e-6)
            self.assertGreater(p.dissipation, 0)
        self.assertLess(np.linalg.norm(states[1] - states[2]), np.linalg.norm(states[0] - states[2]))

    def test_signs_and_fixed_telemetry_schema(self):
        a, b = CompliantPayload(1), CompliantPayload(-1)
        keys = tuple(a.telemetry())
        for p in (a, b):
            p.initialize(np.ones(3), np.zeros(3))
            p.evaluate(np.ones(3), np.zeros(3), 1)
        equilibrium = np.ones(3) + a.mass * a.gravity / a.stiffness
        np.testing.assert_allclose(a.position + b.position, 2 * equilibrium)
        np.testing.assert_allclose(a.velocity, -b.velocity)
        a.detach()
        self.assertEqual(keys, tuple(a.telemetry()))
        self.assertTrue(np.isfinite(list(a.telemetry().values())).all())

    def test_duplicate_load_evaluation_does_not_advance_twice(self):
        env = X2CompliantPayloadEvaluationEnv.__new__(X2CompliantPayloadEvaluationEnv)
        env.model = mujoco.MjModel.from_xml_string("""<mujoco><option timestep="0.005" gravity="0 0 0"/>
          <worldbody><body name="left" pos="0 .2 1"><joint type="slide" axis="1 0 0"/>
          <geom size=".02" mass="1"/></body><body name="right" pos="0 -.2 1">
          <joint type="slide" axis="1 0 0"/><geom size=".02" mass="1"/></body></worldbody></mujoco>""")
        env.data = mujoco.MjData(env.model)
        env.data.time = 26
        env.measurement_start = 0
        env.ee_ids = [env.model.body(name).id for name in ("left", "right")]
        env.force = np.zeros(3)
        env.sim_dt = 0.005
        env.check_state = lambda: None
        env.configure_payload({})
        env.update_load()
        first = env.force.copy()
        env.update_load()
        np.testing.assert_array_equal(env.force, first)
        self.assertEqual(env.payload.steps, 0)
        mujoco.mj_step(env.model, env.data)
        env.evaluation_after_physics()
        env.evaluation_after_physics()
        self.assertEqual(env.payload.steps, 1)
        self.assertIsNone(env.payload_pending_at)
        markers = []

        class Viewer:
            def add_marker(self, **kwargs):
                markers.append(kwargs)

        env.viewer = Viewer()
        env._update_force_visualization()
        sphere = next(m for m in markers if m["id"] == 4200)
        np.testing.assert_array_equal(sphere["pos"], env.payload.position)
        self.assertEqual(sphere["rgba"][3], 1)
        env.payload.detach()
        markers.clear()
        env._update_force_visualization()
        self.assertEqual(next(m for m in markers if m["id"] == 4200)["rgba"][3], 0)


if __name__ == "__main__":
    unittest.main()
