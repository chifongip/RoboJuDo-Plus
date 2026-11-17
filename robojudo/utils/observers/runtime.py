"""Shared synchronous/timestamped observer lifecycle and accepted-command replay."""

import numpy as np

from .linearized import LinearizedDisturbanceObserver
from .nonlinear import NonlinearDisturbanceObserver


class _PreparedHistory:
    """Pair prediction callbacks without a per-step reference cycle."""

    def __init__(self, predict, candidate_predictor=None):
        self.predict = predict
        self._candidate_predictor = candidate_predictor

    def __call__(self, disturbance=None):
        return self.predict(disturbance)

    def predict_candidate(self, disturbance):
        return self.predict(disturbance)[0]

    @property
    def candidate_predictor(self):
        if self._candidate_predictor is not None:
            return self._candidate_predictor
        return self.predict_candidate


class JointObserver:
    def __init__(self, cfg, joint_names, stiffness, damping, *, max_dt, actuator_limits=None, control_dt=0.02):
        # Revalidate mutable Pydantic configurations at the runtime boundary.
        cfg = type(cfg).model_validate(cfg.model_dump())
        if not np.isscalar(max_dt) or not np.isfinite(max_dt) or max_dt <= 0:
            raise ValueError("Observer maximum interval must be positive and finite")
        if not np.isscalar(control_dt) or not np.isfinite(control_dt) or control_dt <= 0:
            raise ValueError("Observer control interval must be positive and finite")
        if len(set(joint_names)) != len(joint_names):
            raise ValueError("Environment joint names must be unique")
        if set(cfg.joint_names) - set(joint_names):
            raise ValueError("Observer joints must belong to the environment")
        self._mode = cfg.mode
        self.algorithm = cfg.algorithm
        self._feedback_shape = (len(joint_names),)
        self.joint_names = list(cfg.joint_names)
        self._shape = (len(self.joint_names),)
        self.indices = np.array([joint_names.index(name) for name in self.joint_names])
        self.cfg = cfg
        self.max_dt, self.control_dt = max_dt, control_dt
        limits = cfg.actuator_limits_nm
        if limits is None:
            limits = self._joint_vector(actuator_limits)
        observer_type = NonlinearDisturbanceObserver if cfg.algorithm == "nonlinear" else LinearizedDisturbanceObserver
        self.observer = observer_type(
            self._joint_vector(stiffness),
            self._joint_vector(damping),
            limits,
            inertia=cfg.inertia,
            friction_limit=cfg.friction_limit,
            friction_viscosity=cfg.friction_viscosity,
            bandwidth_rad_s=cfg.bandwidth_rad_s,
            **({"backend": cfg.backend} if cfg.algorithm == "nonlinear" else {}),
        )
        self.observer.limits = np.asarray(
            cfg.resolve_correction_limits(self.observer.torque_limits.tolist()), dtype=float
        )
        self.observer.slew = cfg.slew_nm_s
        if cfg.algorithm == "nonlinear":
            self.observer.release_slew = cfg.release_slew_nm_s
        self.reset()

    @property
    def mode(self):
        return self._mode

    def _joint_vector(self, value):
        vector = np.asarray(value, dtype=float)
        if vector.shape != self._feedback_shape:
            raise ValueError("Observer inputs must match joint_names")
        return vector[self.indices]

    def reset(self):
        self.observer.reset()
        self.last_time = None
        self.sample_times = None
        self.sample_position = self.sample_velocity = None
        self.command_history = []
        self.pending_target = None
        self.proposed_nm = np.zeros(self._shape)
        self.limited_nm = np.zeros(self._shape)
        self.issued_nm = np.zeros(self._shape)

    def _prepare_history(self, qd, stamps):
        """Prepare receive-interval geometry once for all candidate disturbances.

        The returned predictor is local to one proposal. Only time-dependent
        masks and segments are shared; state and sensitivities are replayed
        independently for every candidate, including saturation transitions.
        """
        if self.sample_times is None:

            def predict_initial(disturbance=None):
                return (qd.copy(), np.ones(self._shape)), np.zeros(self._shape), np.zeros(self._shape, dtype=bool)

            return _PreparedHistory(predict_initial)
        elapsed = stamps - self.sample_times
        fresh = (elapsed > 0) & (elapsed <= self.max_dt)
        covered = np.zeros(self._shape)
        segments = []
        for i, (start, goal, effort) in enumerate(self.command_history):
            end = self.command_history[i + 1][0] if i + 1 < len(self.command_history) else np.inf
            duration = np.maximum(0, np.minimum(stamps, end) - np.maximum(self.sample_times, start))
            active = fresh & (duration > 0)
            if not active.any():
                continue
            segments.append((active, goal, effort, np.where(active, duration, 0.02)))
            covered += duration
        covered_interval = fresh & np.isclose(covered, elapsed, rtol=0, atol=1e-9)
        feedback_dt = np.maximum(elapsed, 0)

        def predict(disturbance=None):
            position, velocity = self.sample_position.copy(), self.sample_velocity.copy()
            dq, dv = np.zeros(self._shape), np.zeros(self._shape)
            for active, goal, effort, duration in segments:
                result = self.observer.predict_state(
                    position,
                    velocity,
                    goal,
                    effort,
                    self.observer.estimated_disturbance_nm if disturbance is None else disturbance,
                    duration,
                    dq,
                    dv,
                )
                position, velocity, dq, dv = [
                    np.where(active, new, old) for new, old in zip(result, (position, velocity, dq, dv), strict=True)
                ]
            if disturbance is not None:
                # Candidate solves must expose invalid gains, never substitute
                # measured velocity and invent a perfect fit.
                return (velocity, np.where(covered_interval, dv, 1)), feedback_dt, covered_interval
            valid = covered_interval & (dv > 0)
            return (np.where(valid, velocity, qd), np.where(valid, dv, 1)), feedback_dt, valid

        candidate = None
        if self.algorithm == "nonlinear" and covered_interval.any():
            candidate = self.observer.prepare_predictor(
                lambda disturbance: predict(disturbance)[0],
                self.sample_position,
                self.sample_velocity,
                segments,
                covered_interval,
            )
        return _PreparedHistory(predict, candidate)

    def _predict_history(self, qd, stamps, disturbance=None):
        """Replay accepted commands over each joint's observed receive interval.

        Receive time is not physical acquisition time. No assumption that Python's
        monotonic clock shares an epoch with the native clock is needed.
        """
        return self._prepare_history(qd, stamps)(disturbance)

    def propose(self, target, position, velocity, now, *, sample_times=None):
        try:
            if not np.isscalar(now) or not np.isfinite(now):
                raise ValueError("Observer time must be a finite scalar")
            dt = None if self.last_time is None else now - self.last_time
            if dt is None or dt <= 0 or dt > self.max_dt:
                self.reset()
                # First sample initializes history only; this interval cannot issue torque.
                dt = min(self.max_dt, self.control_dt)
            q, qd, goal = [self._joint_vector(value) for value in (position, velocity, target)]
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                if sample_times is None:
                    # Synchronous callers use one clock and one sample per command.
                    self.proposed_nm = self.observer.update(goal, q, qd, dt)
                else:
                    stamps = self._joint_vector(sample_times)
                    if not np.isfinite(stamps).all():
                        raise ValueError("Feedback timestamps must be finite")
                    if self.sample_times is not None and np.any(stamps < self.sample_times):
                        self.reset()
                    predict_history = self._prepare_history(qd, stamps)
                    prediction, elapsed, fresh = predict_history()
                    nonlinear = (
                        {} if self.algorithm == "linearized" else {"predictor": predict_history.candidate_predictor}
                    )
                    self.proposed_nm = self.observer.update(
                        goal,
                        q,
                        qd,
                        dt,
                        prediction=prediction,
                        feedback_dt=elapsed,
                        fresh=fresh,
                        **nonlinear,
                    )
                    self.sample_times = stamps.copy()
                    self.sample_position, self.sample_velocity = q.copy(), qd.copy()
                    # Keep the command active at the oldest sample and subsequent changes.
                    while len(self.command_history) > 1 and self.command_history[1][0] <= stamps.min():
                        self.command_history.pop(0)
                self.pending_target = goal.copy()
                pd = self.observer.kp * (goal - q) - self.observer.kd * qd
                lower = np.minimum(0.0, -self.observer.torque_limits - pd)
                upper = np.maximum(0.0, self.observer.torque_limits - pd)
                self.limited_nm = np.clip(self.proposed_nm, lower, upper)
            self.last_time = now
        except (TypeError, ValueError, FloatingPointError, OverflowError):
            self.reset()
            raise
        # Safety headroom may remove effort faster than the normal slew rate.
        # Reconcile that removal before the next predictor/update.
        return (self.limited_nm if self.mode == "ACTIVE" else np.zeros(self._shape)).copy()

    def acknowledge(self, effort, *, command_time=None):
        try:
            issued = np.asarray(effort, dtype=float)
            if issued.shape != self._shape:
                raise ValueError("Command receipt must match observer joints")
            issued = issued.copy()
            if self.mode == "ACTIVE" and (
                np.any(issued < np.minimum(0, self.limited_nm)) or np.any(issued > np.maximum(0, self.limited_nm))
            ):
                raise ValueError("Command receipt exceeds or reverses the requested correction")
            if self.mode != "ACTIVE" and np.any(issued != 0):
                raise ValueError("Non-active observer cannot acknowledge nonzero torque")
            self.observer.acknowledge_effort(issued, freeze_limited_estimate=self.mode != "SHADOW")
            self.issued_nm = issued
            if command_time is not None:
                if not np.isscalar(command_time) or not np.isfinite(command_time):
                    raise ValueError("Command receipt time must be finite")
                if self.pending_target is None or (self.command_history and command_time < self.command_history[-1][0]):
                    raise ValueError("Command receipt has no proposal or moves backwards in time")
                if not self.command_history or not (
                    np.array_equal(self.pending_target, self.command_history[-1][1])
                    and np.array_equal(issued, self.command_history[-1][2])
                ):
                    self.command_history.append((command_time, self.pending_target.copy(), issued.copy()))
                if len(self.command_history) > 256:
                    raise ValueError("Observer command history exceeded its bounded capacity")
        except (TypeError, ValueError):
            self.reset()
            raise

    def get_status(self):
        """Return copied command diagnostics in configured observer joint order."""
        model = self.observer
        status = {
            "mode": self.mode,
            "algorithm": self.algorithm,
            "joint_names": self.joint_names.copy(),
            "estimated_disturbance_nm": model.estimated_disturbance_nm.tolist(),
            "proposed_nm": self.proposed_nm.tolist(),
            "limited_nm": self.limited_nm.tolist(),
            "issued_nm": self.issued_nm.tolist(),
            "velocity_prediction_rad_s": model.velocity_prediction.tolist(),
            "velocity_innovation_rad_s": model.velocity_innovation.tolist(),
            "antiwindup": model.antiwindup.tolist(),
            "estimate_at_bound": np.isclose(
                abs(model.estimated_disturbance_nm), model.limits, rtol=0, atol=1e-3
            ).tolist(),
            "headroom_limited": (self.proposed_nm != self.limited_nm).tolist(),
            "downstream_limited": (self.limited_nm != self.issued_nm).tolist(),
        }
        for attribute in (
            "desired_nm",
            "instantaneous_disturbance_nm",
            "inversion_residual_rad_s",
            "fresh",
            "release_limited",
        ):
            if hasattr(model, attribute):
                status[attribute] = getattr(model, attribute).tolist()
        return status
