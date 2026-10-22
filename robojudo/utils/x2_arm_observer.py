"""Hardware-independent lifecycle and command limiting for the X2 arm observer."""

import numpy as np

from robojudo.utils.disturbance_observer import BoundedDisturbanceObserver
from robojudo.utils.x2_observer_reference import JOINT_NAMES, selected_observer_parameters


class X2ArmObserver:
    def __init__(self, cfg, joint_names, arm_joint_names, stiffness, damping, simulation_limits, max_dt):
        # Pydantic instances can be mutated or model_construct'ed without validation.
        # Revalidate at the deployment boundary, before any simulation fallback.
        cfg = type(cfg).model_validate(cfg.model_dump())
        if not np.isscalar(max_dt) or not np.isfinite(max_dt) or max_dt <= 0:
            raise ValueError("Observer maximum interval must be positive and finite")
        self._mode = cfg.mode
        self._feedback_shape = (len(joint_names),)
        if len(set(joint_names)) != len(joint_names) or set(arm_joint_names) != set(JOINT_NAMES):
            raise ValueError("Observer requires the unique 14 X2 arm names")
        if len(arm_joint_names) != 14:
            raise ValueError("Observer requires exactly 14 arms")
        self.indices = np.asarray([joint_names.index(name) for name in JOINT_NAMES])
        self.adapter_order = np.asarray([JOINT_NAMES.index(name) for name in arm_joint_names])
        self.reference_order = np.argsort(self.adapter_order)
        # A mode change is a lifecycle change, never an in-place model/limit mutation.
        self.cfg = cfg
        self.max_dt = max_dt
        limits = cfg.actuator_limits_nm
        if limits is None:
            limits = np.asarray(simulation_limits, dtype=float)
            if limits.shape != self._feedback_shape:
                raise ValueError("Simulation torque limits must match joint_names")
            limits = limits[self.indices]
        parameters = selected_observer_parameters()
        parameters.pop("method")
        self.observer = BoundedDisturbanceObserver(
            self._arm_vector(stiffness), self._arm_vector(damping), limits, **parameters
        )
        if cfg.correction_limits_nm is not None:
            self.observer.limits = np.asarray(cfg.correction_limits_nm, dtype=float).copy()
        if cfg.slew_nm_s is not None:
            self.observer.slew = cfg.slew_nm_s
        self.reset()

    @property
    def mode(self):
        return self._mode

    def _arm_vector(self, value):
        vector = np.asarray(value, dtype=float)
        if vector.shape != self._feedback_shape:
            raise ValueError("Observer inputs must match joint_names")
        return vector[self.indices]

    def reset(self):
        self.observer.reset()
        self.last_time = None
        self.proposed_nm = np.zeros(14)
        self.limited_nm = np.zeros(14)
        self.issued_nm = np.zeros(14)

    def adapter_limits(self):
        return self.observer.limits[self.adapter_order].tolist()

    def propose(self, target, position, velocity, now):
        try:
            if not np.isscalar(now) or not np.isfinite(now):
                raise ValueError("Observer time must be a finite scalar")
            dt = None if self.last_time is None else now - self.last_time
            if dt is None or dt <= 0 or dt > self.max_dt:
                self.reset()
                # First sample initializes history only; this interval cannot issue torque.
                dt = min(self.max_dt, 0.02)
            q, qd, goal = [self._arm_vector(value) for value in (position, velocity, target)]
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                self.proposed_nm = self.observer.update(goal, q, qd, dt)
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
        return (self.limited_nm if self.mode == "ACTIVE" else np.zeros(14))[self.adapter_order].tolist()

    def acknowledge(self, adapter_effort):
        try:
            issued = np.asarray(adapter_effort, dtype=float)
            if issued.shape != (14,):
                raise ValueError("Arm command receipt must contain exactly 14 values")
            issued = issued[self.reference_order].copy()
            if self.mode == "ACTIVE" and (
                np.any(issued < np.minimum(0, self.limited_nm)) or np.any(issued > np.maximum(0, self.limited_nm))
            ):
                raise ValueError("Arm command receipt exceeds or reverses the requested correction")
            if self.mode != "ACTIVE" and np.any(issued != 0):
                raise ValueError("Non-active observer cannot acknowledge nonzero torque")
            self.observer.acknowledge_effort(issued, freeze_limited_estimate=self.mode != "SHADOW")
            self.issued_nm = issued
        except (TypeError, ValueError):
            self.reset()
            raise
