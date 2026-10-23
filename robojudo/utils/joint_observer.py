"""Synchronous joint observer lifecycle, independent of robot and transport."""

import numpy as np

from robojudo.utils.disturbance_observer import BoundedDisturbanceObserver


class JointObserver:
    def __init__(self, cfg, joint_names, stiffness, damping, torque_limits, control_dt):
        cfg = type(cfg).model_validate(cfg.model_dump())
        if len(set(joint_names)) != len(joint_names):
            raise ValueError("Environment joint names must be unique")
        self.joint_names = list(cfg.joint_names)
        self.indices = np.array([joint_names.index(name) for name in cfg.joint_names])
        self.mode = cfg.mode
        self.control_dt = float(control_dt)
        if not np.isfinite(self.control_dt) or self.control_dt <= 0:
            raise ValueError("Observer control interval must be positive and finite")
        limits = np.asarray(torque_limits)[self.indices] if cfg.actuator_limits_nm is None else cfg.actuator_limits_nm
        self.observer = BoundedDisturbanceObserver(
            np.asarray(stiffness)[self.indices],
            np.asarray(damping)[self.indices],
            limits,
            inertia=cfg.inertia,
            friction_limit=cfg.friction_limit,
            friction_viscosity=cfg.friction_viscosity,
            bandwidth_rad_s=cfg.bandwidth_rad_s,
        )
        self.observer.limits = np.broadcast_to(cfg.correction_limits_nm, self.indices.shape).copy()
        if np.any(self.observer.limits > self.observer.torque_limits):
            raise ValueError("Correction limits must not exceed actuator limits")
        self.observer.slew = cfg.slew_nm_s
        self.reset()

    def reset(self):
        self.observer.reset()
        self.last_time = None
        self.proposed_nm = np.zeros(len(self.indices))
        self.limited_nm = np.zeros(len(self.indices))
        self.issued_nm = np.zeros(len(self.indices))

    def propose(self, target, position, velocity, now):
        if not np.isfinite(now):
            raise ValueError("Observer time must be finite")
        dt = None if self.last_time is None else now - self.last_time
        if dt is None or dt <= 0 or dt > 5 * self.control_dt:
            self.reset()
            dt = self.control_dt
        goal, q, qd = [np.asarray(v)[self.indices] for v in (target, position, velocity)]
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            self.proposed_nm = self.observer.update(goal, q, qd, dt)
            pd = self.observer.kp * (goal - q) - self.observer.kd * qd
            lower = np.minimum(0.0, -self.observer.torque_limits - pd)
            upper = np.maximum(0.0, self.observer.torque_limits - pd)
            self.limited_nm = np.clip(self.proposed_nm, lower, upper)
        self.last_time = now
        return self.limited_nm if self.mode == "ACTIVE" else np.zeros(len(self.indices))

    def acknowledge(self, effort):
        issued = np.asarray(effort, dtype=float)
        if issued.shape != self.indices.shape or not np.isfinite(issued).all():
            raise ValueError("Admitted effort must be finite and match observer joints")
        if self.mode == "ACTIVE":
            if np.any(issued < np.minimum(0, self.limited_nm)) or np.any(issued > np.maximum(0, self.limited_nm)):
                raise ValueError("Admitted effort exceeds or reverses the requested correction")
        elif np.any(issued != 0):
            raise ValueError("Non-active observer cannot apply correction")
        self.observer.acknowledge_effort(issued, freeze_limited_estimate=self.mode != "SHADOW")
        self.issued_nm = issued.copy()
