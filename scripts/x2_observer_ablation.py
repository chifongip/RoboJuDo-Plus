"""Simulation-only observer factors; application presets never import this module.

All factorial variants share the nonlinear observer's zero-before-reversal output
policy. The original linearized observer is a separate reference because it can
cross zero within an update. Linearized inference deliberately leaves its raw
innovation/gain estimate unbounded until the existing filtered-estimate clamp,
matching the original estimator rather than adding an unrequested trust region.
"""

from itertools import product

import numpy as np

from robojudo.config.x2.observer_parameters import selected_observer_parameters
from robojudo.utils.observers import LinearizedDisturbanceObserver, NonlinearDisturbanceObserver

VARIANTS = {
    "original": {"estimator": "linearized", "release": 2.0, "unwinding": "frozen", "zero_before_reversal": False},
    **{
        f"{estimator}_{unwinding}_{rate:g}": {
            "estimator": estimator,
            "release": rate,
            "unwinding": unwinding,
            "zero_before_reversal": True,
        }
        for estimator, unwinding, rate in product(("linearized", "nonlinear"), ("frozen", "directional"), (2.0, 16.0))
    },
}


class AblationObserver(NonlinearDisturbanceObserver):
    def __init__(self, *args, estimator, unwinding, **kwargs):
        self.estimator, self.unwinding = estimator, unwinding
        super().__init__(*args, **kwargs)

    def infer_disturbance(self, velocity, predictor, fresh):
        if self.estimator == "nonlinear":
            return super().infer_disturbance(velocity, predictor, fresh)
        previous = self.estimated_disturbance_nm
        predicted, gain = predictor(previous)
        if np.any(fresh & (~np.isfinite(predicted) | ~np.isfinite(gain) | (gain <= 0))):
            raise ValueError("Invalid linearized ablation prediction")
        instantaneous = previous + (velocity - predicted) / gain
        self.inversion_residual_rad_s[:] = 0  # No nonlinear inversion is performed.
        return np.where(fresh, instantaneous, previous)

    def _allow_unwinding(self, previous, estimate, blocked_direction):
        if self.unwinding == "frozen":
            return np.where(blocked_direction != 0, previous, estimate)
        return super()._allow_unwinding(previous, estimate, blocked_direction)


def make_observer(variant, kp, kd, limits):
    factors = VARIANTS[variant]
    parameters = selected_observer_parameters()
    parameters.pop("method")
    if variant == "original":
        return LinearizedDisturbanceObserver(kp, kd, limits, **parameters)
    model = AblationObserver(
        kp, kd, limits, estimator=factors["estimator"], unwinding=factors["unwinding"], **parameters
    )
    model.release_slew = factors["release"]
    return model
