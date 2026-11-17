"""Bounded torque-integral correction for simulation experiments."""

import numpy as np


class BoundedIntegralCorrection:
    """Fixed-gain filtered torque integral with state bounds and conditional anti-windup.

    Receives joint feedback only. Motion gates, leakage and task/force inputs are
    deliberately absent. Gains and bounds are simulation settings, not a proof
    of stability for the proprietary hardware controller.
    """

    def __init__(
        self, stiffness, damping, torque_limits, *, integral_gain_scale=0.5, filter_tau=0.25, deadband_deg=0.2
    ):
        self.kp = np.asarray(stiffness, dtype=float)
        self.kd = np.asarray(damping, dtype=float)
        self.torque_limits = np.asarray(torque_limits, dtype=float)
        if self.kp.ndim != 1 or any(x.shape != self.kp.shape for x in (self.kd, self.torque_limits)):
            raise ValueError("Gains and torque limits must be matching vectors")
        if any(not np.isfinite(x).all() for x in (self.kp, self.kd, self.torque_limits)):
            raise ValueError("Gains and limits must be finite")
        if np.any(self.kp < 0) or np.any(self.kd < 0) or np.any(self.torque_limits <= 0):
            raise ValueError("Invalid gains or limits")
        if (
            not np.isfinite(integral_gain_scale)
            or integral_gain_scale <= 0
            or not np.isfinite(filter_tau)
            or filter_tau <= 0
            or not np.isfinite(deadband_deg)
            or deadband_deg < 0
        ):
            raise ValueError("Positive finite integral gain/filter and nonnegative deadband required")
        self.ki = integral_gain_scale * self.kp
        self.limits = np.minimum(8.0, 0.8 * self.torque_limits)
        self.filter_tau = filter_tau
        self.deadband = np.deg2rad(deadband_deg)
        self.slew = 2.0
        self.reset()

    def reset(self):
        self.effort = np.zeros_like(self.kp)
        self.filtered_error = np.zeros_like(self.kp)
        self.antiwindup = np.zeros_like(self.kp, dtype=bool)

    def update(self, target, position, velocity, dt):
        values = [np.asarray(x, dtype=float) for x in (target, position, velocity)]
        if not np.isfinite(dt) or dt <= 0 or any(x.shape != self.kp.shape or not np.isfinite(x).all() for x in values):
            self.reset()
            raise ValueError("Integral inputs must be finite matching vectors with positive dt")
        target, position, velocity = values
        self.filtered_error += -np.expm1(-dt / self.filter_tau) * (target - position - self.filtered_error)
        error = np.sign(self.filtered_error) * np.maximum(np.abs(self.filtered_error) - self.deadband, 0.0)
        desired = self.effort + dt * self.ki * error
        increment = np.clip(desired - self.effort, -self.slew * dt, self.slew * dt)
        candidate = np.clip(self.effort + increment, -self.limits, self.limits)
        total = self.kp * (target - position) - self.kd * velocity + candidate
        self.antiwindup = ((total > self.torque_limits) & (increment > 0)) | (
            (total < -self.torque_limits) & (increment < 0)
        )
        self.effort = np.where(self.antiwindup, self.effort, candidate)
        return self.effort.copy()
