"""Translation-only shared payload; evaluation state, never controller feedback."""

import numpy as np


class CompliantPayload:
    mass = 40.0 / 9.81
    gravity = np.array([0.0, 0.0, -9.81])
    stiffness = np.array([100.0, 100.0, 800.0])
    damping_ratio = 0.25

    def __init__(self, initial_sign=1):
        if initial_sign not in (-1, 1):
            raise ValueError("Initial motion sign must be +1 or -1")
        self.initial_sign = initial_sign
        self.damping = 2 * self.damping_ratio * np.sqrt(self.mass * self.stiffness)
        self.initialized = self.detached = False
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.anchor = np.zeros(3)
        self.anchor_velocity = np.zeros(3)
        self.force = np.zeros(3)
        self.support = np.zeros(3)
        self.hand_force = np.zeros(3)
        self.fraction = 0.0
        self.steps = 0
        self.anchor_work = self.support_work = self.transfer_work = self.dissipation = 0.0
        self.catch_dissipation = self.initial_energy = 0.0

    def configuration(self):
        return {
            "mass_kg": self.mass,
            "stiffness_n_m": self.stiffness.tolist(),
            "damping_ratio": self.damping_ratio,
            "damping_ns_m": self.damping.tolist(),
            "initial_sign": self.initial_sign,
            "initial_offset_m": [0.005, 0, 0.001],
            "initial_relative_velocity_m_s": [0, float(0.005 * np.sqrt(self.stiffness[1] / self.mass)), 0],
            "scope": "shared midpoint translation, equal hand reactions; no rotation, slip or release contact",
            "support": "(1-load_fraction) * weight; load fraction scales both grip reactions",
        }

    def initialize(self, anchor, velocity):
        self.anchor, self.anchor_velocity = np.array(anchor, dtype=float), np.array(velocity, dtype=float)
        self.position = self.anchor + self.mass * self.gravity / self.stiffness
        self.position += self.initial_sign * np.array([0.005, 0, 0.001])
        self.velocity = self.anchor_velocity + self.initial_sign * np.array(
            [0, 0.005 * np.sqrt(self.stiffness[1] / self.mass), 0]
        )
        self.initialized = True
        self.initial_energy = self.energy()

    def evaluate(self, anchor, velocity, fraction):
        """Cache the force at the beginning of a physics interval; do not advance state."""
        if not 0 <= fraction <= 1:
            raise ValueError("Load fraction must lie in [0,1]")
        self.anchor, self.anchor_velocity = np.array(anchor, dtype=float), np.array(velocity, dtype=float)
        extension = self.anchor - self.position
        self.transfer_work += 0.5 * (fraction - self.fraction) * np.dot(self.stiffness * extension, extension)
        self.fraction = fraction
        relative_velocity = self.anchor_velocity - self.velocity
        self.force = self.stiffness * extension + self.damping * relative_velocity
        self.support = -(1 - fraction) * self.mass * self.gravity
        self.hand_force = -0.5 * fraction * self.force
        return self.hand_force.copy()

    def advance(self, dt, next_anchor, next_velocity):
        """One semi-implicit Euler step, paired with one robot physics step."""
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Positive finite payload timestep required")
        relative_velocity = self.anchor_velocity - self.velocity
        self.anchor_work += dt * self.fraction * np.dot(self.force, self.anchor_velocity)
        self.support_work += dt * np.dot(self.support, self.velocity)
        self.dissipation += dt * self.fraction * np.dot(self.damping * relative_velocity, relative_velocity)
        self.velocity += dt * (self.mass * self.gravity + self.fraction * self.force + self.support) / self.mass
        self.position += dt * self.velocity
        self.anchor, self.anchor_velocity = np.array(next_anchor, dtype=float), np.array(next_velocity, dtype=float)
        self.steps += 1
        if not np.isfinite(self.position).all() or not np.isfinite(self.velocity).all():
            raise FloatingPointError("Nonfinite compliant payload state")

    def detach(self):
        """Virtual support catches the unloaded payload; its removed kinetic energy is explicit."""
        if not self.detached:
            self.transfer_work -= (
                0.5
                * self.fraction
                * np.dot(self.stiffness * (self.anchor - self.position), self.anchor - self.position)
            )
            self.catch_dissipation += 0.5 * self.mass * np.dot(self.velocity, self.velocity)
            self.velocity[:] = 0
            self.hand_force[:] = 0
            self.force[:] = 0
            self.fraction = 0.0
            self.support = -self.mass * self.gravity
            self.detached = True

    def energy(self):
        extension = self.anchor - self.position
        return float(
            0.5 * self.mass * np.dot(self.velocity, self.velocity)
            - self.mass * np.dot(self.gravity, self.position)
            + 0.5 * self.fraction * np.dot(self.stiffness * extension, extension)
        )

    def telemetry(self):
        row = {
            "payload_initialized": int(self.initialized),
            "payload_detached": int(self.detached),
            "payload_load_fraction": self.fraction,
            "payload_integration_steps": self.steps,
        }
        for name, value in (
            ("position_m", self.position),
            ("velocity_m_s", self.velocity),
            ("anchor_m", self.anchor),
            ("anchor_velocity_m_s", self.anchor_velocity),
            ("relative_position_m", self.position - self.anchor),
            ("support_force_n", self.support),
            ("grip_force_on_payload_n", self.fraction * self.force),
            ("left_force_n", self.hand_force),
            ("right_force_n", self.hand_force),
        ):
            for axis, component in zip("xyz", value, strict=True):
                row[f"payload_{name}_{axis}"] = float(component)
        extension = self.anchor - self.position
        row.update(
            payload_spring_energy_j=float(0.5 * self.fraction * np.dot(self.stiffness * extension, extension)),
            payload_mechanical_energy_j=self.energy() if self.initialized else 0.0,
            payload_damping_dissipation_j=self.dissipation,
            payload_anchor_work_j=self.anchor_work,
            payload_support_work_j=self.support_work,
            payload_transfer_work_j=self.transfer_work,
            payload_catch_dissipation_j=self.catch_dissipation,
            payload_energy_residual_j=(
                self.energy()
                - self.initial_energy
                - self.anchor_work
                - self.support_work
                - self.transfer_work
                + self.dissipation
                + self.catch_dissipation
            )
            if self.initialized
            else 0.0,
        )
        return row
