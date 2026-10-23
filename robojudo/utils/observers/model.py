"""Shared nominal joint dynamics and observer state initialization."""

import numpy as np


class JointDynamics:
    def __init__(
        self, stiffness, damping, torque_limits, *, inertia, friction_limit, friction_viscosity, bandwidth_rad_s=5.0
    ):
        self.kp = np.asarray(stiffness, dtype=float).copy()
        if self.kp.ndim != 1 or self.kp.size == 0:
            raise ValueError("Stiffness must be a nonempty joint vector")

        def vector(value):
            array = np.asarray(value, dtype=float)
            if array.ndim != 0 and array.shape != self.kp.shape:
                raise ValueError("Observer parameters must be scalar or matching joint vectors")
            return np.broadcast_to(array, self.kp.shape).copy()

        self.kd, self.torque_limits = vector(damping), vector(torque_limits)
        self.inertia, self.friction_limit = vector(inertia), vector(friction_limit)
        self.friction_viscosity = vector(friction_viscosity)
        values = (self.kp, self.kd, self.torque_limits, self.inertia, self.friction_limit, self.friction_viscosity)
        if any(not np.isfinite(x).all() or np.any(x < 0) for x in values):
            raise ValueError("Finite nonnegative observer model parameters required")
        if np.any(self.torque_limits <= 0) or np.any(self.inertia <= 0):
            raise ValueError("Positive inertia and torque limits required")
        if not np.isscalar(bandwidth_rad_s) or not np.isfinite(bandwidth_rad_s) or bandwidth_rad_s <= 0:
            raise ValueError("Positive finite scalar observer bandwidth required")
        self.bandwidth = float(bandwidth_rad_s)
        self.limits = np.minimum(8.0, 0.8 * self.torque_limits)
        self.slew = 2.0
        self.reset()

    def reset(self):
        self.effort = np.zeros_like(self.kp)
        self.estimated_disturbance_nm = np.zeros_like(self.kp)
        self.velocity_prediction = np.zeros_like(self.kp)
        self.velocity_innovation = np.zeros_like(self.kp)
        self.antiwindup = np.zeros_like(self.kp, dtype=bool)
        self._estimate_before_update = np.zeros_like(self.kp)
        self.previous_position = self.previous_velocity = self.previous_target = None

    def predict_velocity(self, position, velocity, target, effort, disturbance, dt):
        """Four implicit nominal substeps and analytic torque-to-endpoint-velocity gain."""
        _, v, _, dv = self.predict_state(position, velocity, target, effort, disturbance, dt)
        return v, dv

    def predict_state(self, position, velocity, target, effort, disturbance, dt, dq=None, dv=None):
        """Propagate state and disturbance sensitivities across a command-history segment."""
        q, v = position.copy(), velocity.copy()
        dq = np.zeros_like(q) if dq is None else dq.copy()
        dv = np.zeros_like(v) if dv is None else dv.copy()
        h = dt / 4
        mass_h = self.inertia / h
        for _ in range(4):
            raw = self.kp * (target - q) - self.kd * v + effort
            saturated = np.abs(raw) > self.torque_limits
            # In the unsaturated sector PD spring/damping are included in the
            # implicit step. Saturated drive is modeled as a known clipped input.
            alpha = np.where(saturated, mass_h, mass_h + self.kd + self.kp * h)
            force = (
                np.where(
                    saturated,
                    mass_h * v + np.clip(raw, -self.torque_limits, self.torque_limits),
                    mass_h * v + self.kp * (target - q) + effort,
                )
                + disturbance
            )
            force_derivative = mass_h * dv + np.where(saturated, 0.0, -self.kp * dq) + 1
            trial = force / (alpha + self.friction_viscosity)
            sliding = np.abs(self.friction_viscosity * trial) > self.friction_limit
            denominator = np.where(sliding, alpha, alpha + self.friction_viscosity)
            v = np.where(sliding, (force - np.sign(force) * self.friction_limit) / alpha, trial)
            dv = force_derivative / denominator
            q += h * v
            dq += h * dv
        return q, v, dq, dv
