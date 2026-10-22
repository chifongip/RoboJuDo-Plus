"""Deterministic simulation-only arm feedback imperfections and finite study cases."""

import heapq
from dataclasses import asdict, dataclass

import numpy as np

BURST_STARTS = (12, 27.5, 35, 50, 62, 76.5, 79.1, 84, 92)


@dataclass(frozen=True)
class Perturbation:
    case: str
    position_std_deg: float = 0.0
    velocity_std_rad_s: float = 0.0
    colored: bool = False
    latency_min_s: float = 0.0
    latency_max_s: float = 0.0
    dropout_s: float = 0.0
    inertia_scale: float = 1.0
    armature_scale: float = 1.0
    friction_scale: float = 1.0
    shared_pd: bool = False

    def __post_init__(self):
        values = [v for v in asdict(self).values() if isinstance(v, int | float)]
        if not all(np.isfinite(values)) or any(v < 0 for v in values):
            raise ValueError("Perturbations must be finite and nonnegative")
        if self.latency_min_s > self.latency_max_s:
            raise ValueError("Latency range is reversed")
        if min(self.inertia_scale, self.armature_scale, self.friction_scale) <= 0:
            raise ValueError("Plant multipliers must be positive")


def study_cases():
    cases = [Perturbation("N0")]
    for prefix, field, levels in (
        ("Q", "position_std_deg", (0.01, 0.1)),
        ("V", "velocity_std_rad_s", (0.005, 0.05)),
        ("I", "inertia_scale", (0.7, 1.3)),
        ("A", "armature_scale", (0.5, 1.5)),
        ("F", "friction_scale", (0.5, 1.5)),
        ("H", "dropout_s", (0.04, 0.08)),
    ):
        cases.extend(Perturbation(f"{prefix}{i}", **{field: x}) for i, x in enumerate(levels, 1))
    colored = dict(colored=True, position_std_deg=0.03, velocity_std_rad_s=0.015)
    cases.append(Perturbation("C1", **colored))
    cases.extend(Perturbation(f"D{i}", latency_min_s=x, latency_max_s=x) for i, x in enumerate((0.005, 0.02, 0.04), 1))
    cases.append(Perturbation("J1", latency_min_s=0.01, latency_max_s=0.03))
    moderate = dict(**colored, latency_min_s=0.01, latency_max_s=0.03, inertia_scale=1.3, friction_scale=1.5)
    cases.extend(
        (
            Perturbation("X1", **moderate),
            Perturbation(
                "X2",
                colored=True,
                position_std_deg=0.1,
                velocity_std_rad_s=0.05,
                latency_min_s=0.02,
                latency_max_s=0.06,
                dropout_s=0.08,
                inertia_scale=0.7,
                armature_scale=1.5,
                friction_scale=0.5,
            ),
            Perturbation("P1", latency_min_s=0.02, latency_max_s=0.02, shared_pd=True),
            Perturbation("P2", **moderate, shared_pd=True),
        )
    )
    return cases


class FeedbackChannel:
    """A noisy acquisition queue. No simulator object or clean-state fallback is retained.

    Colored covariance is Kronecker(channel covariance, arm covariance). Both
    marginal cross-channel and within-arm correlations are 0.5; arms are independent.
    Independent RNG streams advance even for dropped acquisitions.
    """

    def __init__(self, config, seed, initial_position, initial_velocity, initial_time):
        self.config = config
        streams = np.random.SeedSequence([seed, *config.case.encode()]).spawn(2)
        self.noise_rng, self.delay_rng = [np.random.default_rng(s) for s in streams]
        joint_cov = np.kron(np.eye(2), 0.5 * np.eye(7) + 0.5 * np.ones((7, 7)))
        self.cholesky = np.linalg.cholesky(np.kron([[1, 0.5], [0.5, 1]], joint_cov))
        self.colored_state = self.cholesky @ self.noise_rng.standard_normal(28)
        self.position, self.velocity = np.array(initial_position).copy(), np.array(initial_velocity).copy()
        self.acquired_time = self.last_time = float(initial_time)
        self.last_noise = np.zeros(28)
        self.last_latency = 0.0
        self.dropped = False
        self.queue = []
        self.sequence = self.out_of_order = self.drop_count = 0

    def acquire(self, time, position, velocity, elapsed):
        c = self.config
        dt = time - self.last_time
        if dt <= 1e-10:
            return
        self.last_time = time
        self.sequence += 1
        if c.position_std_deg or c.velocity_std_rad_s:
            if c.colored:
                rho = np.exp(-dt / 0.05)
                self.colored_state = rho * self.colored_state + np.sqrt(1 - rho**2) * (
                    self.cholesky @ self.noise_rng.standard_normal(28)
                )
                noise = self.colored_state
            else:
                noise = self.noise_rng.standard_normal(28)
            self.last_noise = noise * np.repeat([np.deg2rad(c.position_std_deg), c.velocity_std_rad_s], 14)
        self.last_latency = (
            self.delay_rng.uniform(c.latency_min_s, c.latency_max_s)
            if c.latency_min_s != c.latency_max_s
            else c.latency_min_s
        )
        self.dropped = c.dropout_s > 0 and any(s - 1e-9 <= elapsed < s + c.dropout_s - 1e-9 for s in BURST_STARTS)
        if self.dropped:
            self.drop_count += 1
        else:
            heapq.heappush(
                self.queue,
                (
                    time + self.last_latency,
                    self.sequence,
                    time,
                    np.asarray(position) + self.last_noise[:14],
                    np.asarray(velocity) + self.last_noise[14:],
                ),
            )
        while self.queue and self.queue[0][0] <= time + 1e-9:
            _, _, stamp, q, v = heapq.heappop(self.queue)
            if stamp > self.acquired_time:
                self.acquired_time, self.position, self.velocity = stamp, q, v
            else:
                self.out_of_order += 1

    def measured(self):
        return self.position.copy(), self.velocity.copy()
