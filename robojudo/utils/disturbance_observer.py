"""Bounded model-based disturbance correction with explicit issued-effort reconciliation."""

import numpy as np


class BoundedDisturbanceObserver:
    """Low-pass the sampled nominal-model velocity residual into a torque estimate.

    With exact local input gain B and constant disturbance, the unbounded
    estimation error contracts by exp(-bandwidth*dt). The predictor retains
    nominal dissipative friction. It uses only measured q/qd, existing targets
    and its own previous issued correction; no force/phase signal is accepted.
    """

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

    def acknowledge_effort(self, issued_effort, *, freeze_limited_estimate=True):
        """Reconcile a downstream limiter (or shadow zero) without inventing applied torque.

        Freeze estimate changes on limited joints. The next predictor and slew
        calculation use the accepted command, rather than the proposed effort.
        """
        issued = np.asarray(issued_effort, dtype=float)
        if issued.shape != self.kp.shape or not np.isfinite(issued).all() or np.any(abs(issued) > self.limits):
            self.reset()
            raise ValueError("Issued effort must be finite and within observer bounds")
        limited = issued != self.effort
        if freeze_limited_estimate:
            self.estimated_disturbance_nm = np.where(
                limited, self._estimate_before_update, self.estimated_disturbance_nm
            )
        self.antiwindup |= limited
        self.effort = issued.copy()

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

    def update(self, target, position, velocity, dt, *, prediction=None, feedback_dt=None, fresh=None):
        values = [np.asarray(x, dtype=float) for x in (target, position, velocity)]
        if not np.isfinite(dt) or dt <= 0 or any(x.shape != self.kp.shape or not np.isfinite(x).all() for x in values):
            self.reset()
            raise ValueError("Observer feedback must be finite matching vectors with positive dt")
        target, position, velocity = values
        previous_estimate = self.estimated_disturbance_nm.copy()
        self._estimate_before_update = previous_estimate.copy()
        if self.previous_position is None and prediction is None:
            self.velocity_prediction = velocity.copy()
            self.velocity_innovation[:] = 0
            estimate = previous_estimate
        else:
            prediction, input_gain = (
                prediction
                if prediction is not None
                else self.predict_velocity(
                    self.previous_position,
                    self.previous_velocity,
                    self.previous_target,
                    self.effort,
                    previous_estimate,
                    dt,
                )
            )
            if not np.isfinite(prediction).all() or not np.isfinite(input_gain).all() or np.any(input_gain <= 0):
                self.reset()
                raise ValueError("Invalid nominal observer prediction or torque input gain")
            self.velocity_prediction = prediction
            self.velocity_innovation = velocity - prediction
            estimate = np.clip(
                previous_estimate
                + -np.expm1(-self.bandwidth * (dt if feedback_dt is None else feedback_dt))
                * self.velocity_innovation
                / input_gain,
                -self.limits,
                self.limits,
            )
        if fresh is not None:
            estimate = np.where(fresh, estimate, previous_estimate)
        desired = -estimate
        increment = np.clip(desired - self.effort, -self.slew * dt, self.slew * dt)
        if fresh is not None:
            increment = np.where(fresh, increment, 0.0)
        candidate = np.clip(self.effort + increment, -self.limits, self.limits)
        total = self.kp * (target - position) - self.kd * velocity + candidate
        self.antiwindup = ((total > self.torque_limits) & (increment > 0)) | (
            (total < -self.torque_limits) & (increment < 0)
        )
        self.effort = np.where(self.antiwindup, self.effort, candidate)
        self.estimated_disturbance_nm = np.where(self.antiwindup, previous_estimate, estimate)
        self.previous_target, self.previous_position, self.previous_velocity = [x.copy() for x in values]
        return self.effort.copy()
