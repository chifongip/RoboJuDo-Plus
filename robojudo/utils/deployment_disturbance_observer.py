"""Finite-residual observer for deployment; the validated simulation DOB stays frozen."""

import numpy as np

from robojudo.utils.disturbance_observer import BoundedDisturbanceObserver


class DeploymentDisturbanceObserver(BoundedDisturbanceObserver):
    """Invert the bounded nonlinear predictor before filtering a disturbance estimate.

    Predictors receive a disturbance vector and return endpoint velocity and its
    sensitivity. Timestamped callers must replay acknowledged commands, never a
    proposed or current command substituted for a past interval.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.release_slew = None

    def reset(self):
        super().reset()
        self.instantaneous_disturbance_nm = np.zeros_like(self.kp)
        self.desired_nm = np.zeros_like(self.kp)
        self.fresh = np.zeros_like(self.kp, dtype=bool)
        self.release_limited = np.zeros_like(self.kp, dtype=bool)
        self.inversion_residual_rad_s = np.zeros_like(self.kp)

    def infer_disturbance(self, velocity, predictor, fresh):
        """Twenty bounded bisections, retaining the smallest observed residual.

        In continuous monotone sectors torque resolution is limits / 2**20.
        The legacy predictor switches its integration rule at torque saturation;
        near those discontinuities a root may not exist. Retain the best tested
        candidate (including the previous estimate), never an unbounded Newton
        step. Nonfinite predictions fail closed.
        """
        lower, upper = -self.limits.copy(), self.limits.copy()
        low_v, _ = predictor(lower)
        high_v, _ = predictor(upper)
        best = self.estimated_disturbance_nm.copy()
        best_v, _ = predictor(best)
        residual = abs(best_v - velocity)
        for candidate, predicted in ((lower, low_v), (upper, high_v)):
            better = abs(predicted - velocity) < residual
            best = np.where(better, candidate, best)
            residual = np.minimum(residual, abs(predicted - velocity))
        for _ in range(20):
            midpoint = (lower + upper) / 2
            mid_v, gain = predictor(midpoint)
            valid = np.isfinite(low_v) & np.isfinite(high_v) & np.isfinite(mid_v) & np.isfinite(gain) & (gain > 0)
            if np.any(fresh & ~valid):
                raise ValueError("Invalid nonlinear observer prediction")
            better = abs(mid_v - velocity) < residual
            best = np.where(better, midpoint, best)
            residual = np.minimum(residual, abs(mid_v - velocity))
            below = mid_v < velocity
            lower, low_v = np.where(below, midpoint, lower), np.where(below, mid_v, low_v)
            upper, high_v = np.where(below, upper, midpoint), np.where(below, high_v, mid_v)
        self.inversion_residual_rad_s = np.where(fresh, residual, 0)
        return np.where(fresh, best, self.estimated_disturbance_nm)

    @staticmethod
    def _allow_unwinding(previous, estimate, blocked_direction):
        # Work in correction coordinates. Reducing an old correction is always
        # allowed; a zero crossing retains that reduction but blocks new growth
        # in the direction whose command authority was denied.
        old, new = -previous, -estimate
        outward = new * blocked_direction > np.maximum(old * blocked_direction, 0)
        retained = np.where(old * blocked_direction >= 0, old, 0)
        return -np.where(outward, retained, new)

    def acknowledge_effort(self, issued_effort, *, freeze_limited_estimate=True):
        issued = np.asarray(issued_effort, dtype=float)
        if issued.shape != self.kp.shape or not np.isfinite(issued).all() or np.any(abs(issued) > self.limits):
            self.reset()
            raise ValueError("Issued effort must be finite and within observer bounds")
        limited = issued != self.effort
        if freeze_limited_estimate:
            self.estimated_disturbance_nm = self._allow_unwinding(
                self._estimate_before_update, self.estimated_disturbance_nm, np.sign(self.effort - issued)
            )
        self.antiwindup |= limited
        self.effort = issued.copy()

    def update(self, target, position, velocity, dt, *, prediction=None, feedback_dt=None, fresh=None, predictor=None):
        values = [np.asarray(x, dtype=float) for x in (target, position, velocity)]
        if not np.isfinite(dt) or dt <= 0 or any(x.shape != self.kp.shape or not np.isfinite(x).all() for x in values):
            self.reset()
            raise ValueError("Observer feedback must be finite matching vectors with positive dt")
        target, position, velocity = values
        previous = self.estimated_disturbance_nm.copy()
        self._estimate_before_update = previous.copy()
        self.fresh = np.ones_like(previous, dtype=bool) if fresh is None else np.asarray(fresh, dtype=bool).copy()
        if predictor is None and self.previous_position is not None:

            def predictor(disturbance):
                return self.predict_velocity(
                    self.previous_position, self.previous_velocity, self.previous_target, self.effort, disturbance, dt
                )

        if predictor is None:
            self.fresh[:] = False
        if self.fresh.any():
            predicted, gain = predictor(previous) if prediction is None else prediction
            if not np.isfinite(predicted).all() or not np.isfinite(gain).all() or np.any(gain <= 0):
                raise ValueError("Invalid nominal observer prediction")
            self.velocity_prediction = predicted.copy()
            self.velocity_innovation = np.where(self.fresh, velocity - predicted, 0)
            self.instantaneous_disturbance_nm = self.infer_disturbance(velocity, predictor, self.fresh)
            elapsed = dt if feedback_dt is None else np.asarray(feedback_dt)
            estimate = previous + -np.expm1(-self.bandwidth * elapsed) * (self.instantaneous_disturbance_nm - previous)
            estimate = np.where(self.fresh, np.clip(estimate, -self.limits, self.limits), previous)
        else:
            self.velocity_prediction = velocity.copy()
            self.velocity_innovation[:] = 0
            self.inversion_residual_rad_s[:] = 0
            self.instantaneous_disturbance_nm = previous.copy()
            estimate = previous
        self.desired_nm = -estimate
        # Opposite-sign requests release to zero this update; growth starts on
        # a later update at the original growth rate, without leftover time.
        releasing = (self.effort != 0) & (
            (self.desired_nm * self.effort <= 0) | (abs(self.desired_nm) < abs(self.effort))
        )
        destination = np.where(self.desired_nm * self.effort < 0, 0, self.desired_nm)
        rate = np.where(releasing, self.slew if self.release_slew is None else self.release_slew, self.slew)
        delta = destination - self.effort
        increment = np.where(self.fresh, np.clip(delta, -rate * dt, rate * dt), 0)
        self.release_limited = self.fresh & releasing & (abs(delta) > rate * dt)
        candidate = np.clip(self.effort + increment, -self.limits, self.limits)
        total = self.kp * (target - position) - self.kd * velocity + candidate
        self.antiwindup = ((total > self.torque_limits) & (increment > 0)) | (
            (total < -self.torque_limits) & (increment < 0)
        )
        self.effort = np.where(self.antiwindup, self.effort, candidate)
        self.estimated_disturbance_nm = self._allow_unwinding(
            previous, estimate, np.where(self.antiwindup, np.sign(total), 0)
        )
        self.previous_target, self.previous_position, self.previous_velocity = [x.copy() for x in values]
        return self.effort.copy()
