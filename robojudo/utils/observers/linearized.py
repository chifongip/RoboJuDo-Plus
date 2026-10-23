"""Frozen local-gain observer used by the historical simulation baseline."""

import numpy as np

from .model import JointDynamics


class LinearizedDisturbanceObserver(JointDynamics):
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
