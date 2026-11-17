"""Numerical replay against traces captured from commit f0ba949 before refactoring.

The fixture records proposed/limited/issued effort, estimate, prediction,
innovation and anti-windup over resets, repeated samples and clipped receipts.
"""

import unittest
from pathlib import Path

import numpy as np

from robojudo.config.observer_cfg import JointObserverCfg
from robojudo.config.x2.observer_parameters import JOINT_NAMES, selected_observer_parameters
from robojudo.utils.observers import JointObserver


def trace(kind):
    nonlinear = kind != "linearized"
    shadow = kind == "nonlinear_shadow"
    if nonlinear:
        names = ["leg", "waist", *reversed(JOINT_NAMES)]
        parameters = selected_observer_parameters()
        parameters.pop("method")
        cfg = JointObserverCfg(
            algorithm="nonlinear",
            joint_names=list(JOINT_NAMES),
            **parameters,
            mode="SHADOW" if shadow else "ACTIVE",
            actuator_limits_nm=[10.0] * 14,
            correction_limit_nm=1.0,
            slew_nm_s=2,
            release_slew_nm_s=16,
        )
        ctrl = JointObserver(cfg, names, [20.0] * 16, [2.0] * 16, actuator_limits=[10.0] * 16, max_dt=0.1)
    else:
        names = ["a", "other", "b"]
        cfg = JointObserverCfg(
            algorithm="linearized",
            mode="ACTIVE",
            joint_names=["b", "a"],
            inertia=[0.1, 0.15],
            friction_limit=0.3,
            friction_viscosity=40,
            bandwidth_rad_s=8,
            correction_limit_nm=0.8,
            slew_nm_s=2,
        )
        ctrl = JointObserver(cfg, names, [20.0] * 3, [2.0] * 3, actuator_limits=[10.0] * 3, max_dt=0.1)
    rng = np.random.default_rng(471)
    rows = []
    stamps = np.full(len(names), 100.0)
    for step in range(80):
        if step == 40:
            ctrl.reset()
        goal = np.full(len(names), 0.1 * np.sin(step * 0.11))
        q = rng.normal(0, 0.03, len(names))
        v = rng.normal(0, 0.15, len(names))
        if nonlinear:
            if step % 7:
                stamps[:] = 100 + step * 0.02
            proposed = ctrl.propose(goal, q, v, 1 + step * 0.02, sample_times=stamps)
        else:
            proposed = ctrl.propose(goal, q, v, 1 + step * 0.02)
        issued = np.asarray(proposed) * (0 if shadow else (0.5 if step % 9 == 0 else 1))
        if nonlinear:
            ctrl.acknowledge(issued, command_time=100 + step * 0.02)
        else:
            ctrl.acknowledge(issued)
        rows.append(
            np.concatenate(
                (
                    ctrl.proposed_nm,
                    ctrl.limited_nm,
                    ctrl.issued_nm,
                    ctrl.observer.estimated_disturbance_nm,
                    ctrl.observer.velocity_prediction,
                    ctrl.observer.velocity_innovation,
                    ctrl.observer.antiwindup.astype(float),
                )
            )
        )
    return np.asarray(rows)


class ObserverParityTests(unittest.TestCase):
    def test_committed_reference_traces(self):
        with np.load(Path(__file__).parent / "fixtures/observer_f0ba949.npz") as reference:
            for kind in reference.files:
                with self.subTest(kind=kind):
                    np.testing.assert_array_equal(trace(kind), reference[kind])


if __name__ == "__main__":
    unittest.main()
