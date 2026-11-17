"""Optional compiled nonlinear replay; imported only when acceleration is requested.

No fast-math, parallel workers, or compilation in the control loop. Python keeps
ownership of timestamp selection, lifecycle, filtering and command limits.
"""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=False, error_model="numpy")
def _replay(joint, disturbance, position, velocity, parameters, segments, active):
    q, v = position[joint], velocity[joint]
    dq, dv = 0.0, 0.0
    kp, kd, limit, friction, viscosity = parameters[:, joint]
    for segment in range(segments.shape[0]):
        target, effort, h, mass_h, elastic_alpha = segments[segment, :, joint]
        old_q, old_v, old_dq, old_dv = q, v, dq, dv
        for _ in range(4):
            raw = kp * (target - q) - kd * v + effort
            saturated = abs(raw) > limit
            alpha = mass_h if saturated else elastic_alpha
            clipped = min(max(raw, -limit), limit)
            saturated_force = mass_h * v + clipped
            elastic_force = mass_h * v + kp * (target - q) + effort
            force = (saturated_force if saturated else elastic_force) + disturbance
            derivative_term = -kp * dq
            force_derivative = mass_h * dv + (0.0 if saturated else derivative_term) + 1
            denominator = alpha + viscosity
            trial = force / denominator
            friction_trial = viscosity * trial
            sliding = abs(friction_trial) > friction
            sign = 1.0 if force > 0 else (-1.0 if force < 0 else 0.0)
            sliding_v = (force - sign * friction) / alpha
            v = sliding_v if sliding else trial
            dv = force_derivative / (alpha if sliding else alpha + viscosity)
            q += h * v
            dq += h * dv
            # NumPy evaluates both branches of where, including inactive joints.
            # Retain fail-closed behavior rather than silently masking overflow.
            if not (
                np.isfinite(raw)
                and np.isfinite(saturated_force)
                and np.isfinite(elastic_force)
                and np.isfinite(denominator)
                and np.isfinite(derivative_term)
                and np.isfinite(force)
                and np.isfinite(friction_trial)
                and np.isfinite(sliding_v)
                and np.isfinite(q)
                and np.isfinite(v)
                and np.isfinite(dq)
                and np.isfinite(dv)
            ):
                raise ValueError("Nonfinite compiled observer prediction")
        if not active[segment, joint]:
            q, v, dq, dv = old_q, old_v, old_dq, old_dv
    return v, dv


@njit(cache=True, fastmath=False, error_model="numpy")
def _infer(position, velocity, parameters, segments, active, covered, measured, previous, limits, fresh):
    best = previous.copy()
    residuals = np.zeros_like(previous)
    for j in range(previous.size):
        lower, upper = -limits[j], limits[j]
        low_v, _ = _replay(j, lower, position, velocity, parameters, segments, active)
        high_v, _ = _replay(j, upper, position, velocity, parameters, segments, active)
        best_v, _ = _replay(j, best[j], position, velocity, parameters, segments, active)
        residual = abs(best_v - measured[j])
        if not np.isfinite(residual):
            raise ValueError("Nonfinite compiled observer residual")
        for candidate, predicted in ((lower, low_v), (upper, high_v)):
            error = abs(predicted - measured[j])
            if not np.isfinite(error):
                raise ValueError("Nonfinite compiled observer residual")
            if error < residual:
                best[j] = candidate
            residual = min(residual, error)
        for _ in range(20):
            midpoint = (lower + upper) / 2
            mid_v, gain = _replay(j, midpoint, position, velocity, parameters, segments, active)
            if not covered[j]:
                gain = 1.0
            if fresh[j] and not (
                np.isfinite(low_v) and np.isfinite(high_v) and np.isfinite(mid_v) and np.isfinite(gain) and gain > 0
            ):
                raise ValueError("Invalid nonlinear observer prediction")
            error = abs(mid_v - measured[j])
            if not np.isfinite(error):
                raise ValueError("Nonfinite compiled observer residual")
            if error < residual:
                best[j] = midpoint
            residual = min(residual, error)
            if mid_v < measured[j]:
                lower, low_v = midpoint, mid_v
            else:
                upper, high_v = midpoint, mid_v
        if fresh[j]:
            residuals[j] = residual
        else:
            best[j] = previous[j]
    return best, residuals


def warmup():
    """Compile the sole, contiguous float64/bool signature before control starts."""
    vector = np.zeros(1)
    parameters = np.ones((5, 1))
    segments = np.array([[[0.0], [0.0], [0.005], [200.0], [201.005]]])
    mask = np.ones(1, dtype=bool)
    _infer(vector, vector, parameters, segments, mask.reshape(1, 1), mask, vector, vector, np.ones(1), mask)


class CompiledPredictor:
    """Reference prediction callback plus a fused solver for the same replay."""

    def __init__(self, reference, model, position, velocity, segments, covered):
        self.reference = reference
        self.model = model
        self.position = np.array(position, dtype=np.float64, order="C", copy=True)
        self.velocity = np.array(velocity, dtype=np.float64, order="C", copy=True)
        self.parameters = np.ascontiguousarray(
            [
                model.kp,
                model.kd,
                model.torque_limits,
                model.friction_limit,
                model.friction_viscosity,
            ],
            dtype=np.float64,
        )
        n = self.position.size
        if (
            self.position.shape != model.kp.shape
            or self.velocity.shape != self.position.shape
            or self.parameters.shape != (5, n)
        ):
            raise ValueError("Compiled observer vectors must match model joints")
        self.segments = np.empty((len(segments), 5, n), dtype=np.float64)
        self.active = np.empty((len(segments), n), dtype=bool)
        for i, (active, target, effort, duration) in enumerate(segments):
            h = np.asarray(duration) / 4
            mass_h = model.inertia / h
            self.segments[i] = target, effort, np.broadcast_to(h, (n,)), mass_h, mass_h + model.kd + model.kp * h
            self.active[i] = active
        self.covered = np.array(covered, dtype=bool, order="C", copy=True)
        if self.covered.shape != self.position.shape:
            raise ValueError("Compiled observer coverage must match model joints")
        if not all(
            np.isfinite(value).all() for value in (self.position, self.velocity, self.parameters, self.segments)
        ):
            raise ValueError("Nonfinite compiled observer input")

    def __call__(self, disturbance):
        return self.reference(disturbance)

    def infer(self, measured, fresh):
        model = self.model
        arrays = [
            np.array(x, dtype=np.float64, order="C", copy=True)
            for x in (measured, model.estimated_disturbance_nm, model.limits)
        ]
        if any(x.shape != self.position.shape or not np.isfinite(x).all() for x in arrays):
            raise ValueError("Invalid compiled observer input")
        mask = np.array(fresh, dtype=bool, order="C", copy=True)
        if mask.shape != self.position.shape:
            raise ValueError("Invalid compiled observer freshness mask")
        return _infer(
            self.position, self.velocity, self.parameters, self.segments, self.active, self.covered, *arrays, mask
        )
