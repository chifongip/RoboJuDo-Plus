from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from robojudo.environment.env_cfgs import SimulatedOdometryCfg


def _pose(position, quaternion) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(position, dtype=np.float64)
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if position.shape != (3,) or quaternion.shape != (4,):
        raise ValueError("Odometry position and quaternion must have shapes (3,) and (4,)")
    if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
        raise ValueError("Odometry pose must contain only finite values")
    norm = np.linalg.norm(quaternion)
    if norm < 1e-8:
        raise ValueError("Odometry quaternion must be non-zero")
    return position, quaternion / norm


def sensor_pose_to_root(
    sensor_position,
    sensor_quaternion,
    torso_to_sensor_position,
    torso_to_sensor_quaternion,
    root_to_torso_position,
    root_to_torso_quaternion,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a world->sensor pose to world->root using the two fixed/dynamic mounts."""
    sensor_position, sensor_quaternion = _pose(sensor_position, sensor_quaternion)
    torso_to_sensor_position, torso_to_sensor_quaternion = _pose(torso_to_sensor_position, torso_to_sensor_quaternion)
    root_to_torso_position, root_to_torso_quaternion = _pose(root_to_torso_position, root_to_torso_quaternion)
    world_to_sensor = Rotation.from_quat(sensor_quaternion)
    torso_to_sensor = Rotation.from_quat(torso_to_sensor_quaternion)
    root_to_torso = Rotation.from_quat(root_to_torso_quaternion)

    world_to_torso = world_to_sensor * torso_to_sensor.inv()
    world_to_torso_position = sensor_position - world_to_torso.apply(torso_to_sensor_position)
    world_to_root = world_to_torso * root_to_torso.inv()
    world_to_root_position = world_to_torso_position - world_to_root.apply(root_to_torso_position)
    return world_to_root_position, world_to_root.as_quat()


def root_pose_to_sensor(
    root_position,
    root_quaternion,
    root_to_torso_position,
    root_to_torso_quaternion,
    torso_to_sensor_position,
    torso_to_sensor_quaternion,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct a virtual world->sensor pose from the simulated root pose."""
    root_position, root_quaternion = _pose(root_position, root_quaternion)
    root_to_torso_position, root_to_torso_quaternion = _pose(root_to_torso_position, root_to_torso_quaternion)
    torso_to_sensor_position, torso_to_sensor_quaternion = _pose(torso_to_sensor_position, torso_to_sensor_quaternion)
    world_to_root = Rotation.from_quat(root_quaternion)
    root_to_torso = Rotation.from_quat(root_to_torso_quaternion)
    torso_to_sensor = Rotation.from_quat(torso_to_sensor_quaternion)
    world_to_torso = world_to_root * root_to_torso
    sensor_position = (
        root_position + world_to_root.apply(root_to_torso_position) + world_to_torso.apply(torso_to_sensor_position)
    )
    return sensor_position, (world_to_torso * torso_to_sensor).as_quat()


@dataclass(frozen=True)
class OdometryEstimate:
    position: np.ndarray
    quaternion: np.ndarray
    linear_velocity_body: np.ndarray
    age: float
    stale: bool


class OdometryTracker:
    """Estimate root position and body-frame velocity from timestamped world poses."""

    def __init__(self, timeout: float, velocity_filter_time_constant: float):
        if timeout <= 0.0 or velocity_filter_time_constant < 0.0:
            raise ValueError("Invalid odometry tracker timing")
        self.timeout = timeout
        self.velocity_filter_time_constant = velocity_filter_time_constant
        self.reset()

    def reset(self):
        self._position: np.ndarray | None = None
        self._quaternion: np.ndarray | None = None
        self._sample_time: float | None = None
        self._receipt_time: float | None = None
        self._velocity_body = np.zeros(3, dtype=np.float64)

    @property
    def initialized(self) -> bool:
        return self._position is not None

    def update(self, position, quaternion, sample_time: float, receipt_time: float):
        position, quaternion = _pose(position, quaternion)
        if not np.isfinite(sample_time) or not np.isfinite(receipt_time):
            raise ValueError("Odometry timestamps must be finite")
        if self._sample_time is not None and sample_time <= self._sample_time:
            return
        if self._position is not None and self._sample_time is not None:
            dt = sample_time - self._sample_time
            if dt <= self.timeout:
                velocity_world = (position - self._position) / dt
                velocity_body = Rotation.from_quat(quaternion).inv().apply(velocity_world)
                tau = self.velocity_filter_time_constant
                alpha = 1.0 if tau <= 0.0 else 1.0 - np.exp(-dt / tau)
                self._velocity_body = (1.0 - alpha) * self._velocity_body + alpha * velocity_body
        self._position = position.copy()
        self._quaternion = quaternion.copy()
        self._sample_time = float(sample_time)
        self._receipt_time = float(receipt_time)

    def estimate(self, now: float) -> OdometryEstimate:
        if self._position is None or self._quaternion is None or self._receipt_time is None:
            raise RuntimeError("No odometry sample is available")
        age = max(0.0, float(now) - self._receipt_time)
        velocity_world = Rotation.from_quat(self._quaternion).apply(self._velocity_body)
        position = self._position + velocity_world * min(age, self.timeout)
        return OdometryEstimate(
            position=position.astype(np.float32),
            quaternion=self._quaternion.astype(np.float32),
            linear_velocity_body=self._velocity_body.astype(np.float32),
            age=age,
            stale=age > self.timeout,
        )


@dataclass
class _PendingSample:
    delivery_time: float
    sample_time: float
    sensor_position: np.ndarray
    sensor_quaternion: np.ndarray
    converter: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class OdometryReplayProfile:
    """Recorded delivery timing and detrended pose residuals for simulation replay."""

    sample_times: np.ndarray
    position_residuals: np.ndarray
    yaw_residuals: np.ndarray
    valid: np.ndarray

    def __post_init__(self):
        sample_times = np.asarray(self.sample_times, dtype=np.float64)
        position_residuals = np.asarray(self.position_residuals, dtype=np.float64)
        yaw_residuals = np.asarray(self.yaw_residuals, dtype=np.float64)
        valid = np.asarray(self.valid, dtype=bool)
        count = len(sample_times)
        if (
            sample_times.shape != (count,)
            or position_residuals.shape != (count, 3)
            or yaw_residuals.shape != (count,)
            or valid.shape != (count,)
        ):
            raise ValueError("Odometry replay profile arrays have incompatible shapes")
        if count == 0 or abs(sample_times[0]) > 1e-9 or np.any(np.diff(sample_times) <= 0.0):
            raise ValueError("Odometry replay sample_times must start at zero and increase strictly")
        if not (
            np.isfinite(sample_times).all()
            and np.isfinite(position_residuals).all()
            and np.isfinite(yaw_residuals).all()
        ):
            raise ValueError("Odometry replay profile must contain finite values")
        object.__setattr__(self, "sample_times", sample_times)
        object.__setattr__(self, "position_residuals", position_residuals)
        object.__setattr__(self, "yaw_residuals", yaw_residuals)
        object.__setattr__(self, "valid", valid)


class SimulatedOdometry:
    """Deterministic sample/latency/dropout model used by MuJoCo benchmarks."""

    def __init__(self, cfg: SimulatedOdometryCfg, replay_profile: OdometryReplayProfile | None = None):
        self.cfg = cfg
        self.replay_profile = replay_profile
        self.tracker = OdometryTracker(cfg.timeout, cfg.velocity_filter_time_constant)
        self.reset()

    def reset(self, start_time: float = 0.0):
        self.rng = np.random.default_rng(self.cfg.random_seed)
        self.tracker.reset()
        self.start_time = float(start_time)
        self.next_sample_time = self.start_time
        self.replay_index = 0
        self.pending: list[_PendingSample] = []
        self.generated = 0
        self.delivered = 0
        self.dropped = 0
        self.degenerate = 0

    def _is_degenerate(self, timestamp: float) -> bool:
        return any(start <= timestamp < end for start, end in self.cfg.degeneracy_windows)

    def update(
        self,
        now: float,
        sensor_position: np.ndarray,
        sensor_quaternion: np.ndarray,
        converter: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    ) -> OdometryEstimate | None:
        if self.replay_profile is not None:
            return self._update_replay(now, sensor_position, sensor_quaternion, converter)

        period = 1.0 / self.cfg.update_rate_hz
        while now + 1e-9 >= self.next_sample_time:
            sample_time = self.next_sample_time
            self.next_sample_time += period
            self.generated += 1
            if self._is_degenerate(sample_time - self.start_time):
                self.degenerate += 1
                continue
            if self.rng.random() < self.cfg.dropout_probability:
                self.dropped += 1
                continue

            noisy_position = np.asarray(sensor_position, dtype=np.float64).copy()
            noisy_position += self.rng.normal(0.0, self.cfg.position_noise_std, 3)
            sensor_rotation = Rotation.from_quat(sensor_quaternion)
            noisy_rotation = Rotation.from_euler("z", self.rng.normal(0.0, self.cfg.yaw_noise_std)) * sensor_rotation
            delay = max(0.0, self.cfg.latency + self.rng.uniform(-self.cfg.jitter, self.cfg.jitter))
            self.pending.append(
                _PendingSample(
                    delivery_time=sample_time + delay,
                    sample_time=sample_time,
                    sensor_position=noisy_position,
                    sensor_quaternion=noisy_rotation.as_quat(),
                    converter=converter,
                )
            )

        ready = [sample for sample in self.pending if sample.delivery_time <= now + 1e-9]
        self.pending = [sample for sample in self.pending if sample.delivery_time > now + 1e-9]
        for sample in sorted(ready, key=lambda item: item.sample_time):
            root_position, root_quaternion = sample.converter(sample.sensor_position, sample.sensor_quaternion)
            self.tracker.update(root_position, root_quaternion, sample.sample_time, now)
            self.delivered += 1
        return self.tracker.estimate(now) if self.tracker.initialized else None

    def _update_replay(
        self,
        now: float,
        sensor_position: np.ndarray,
        sensor_quaternion: np.ndarray,
        converter: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    ) -> OdometryEstimate | None:
        """Apply a captured delivery schedule to the current virtual sensor pose."""
        elapsed = float(now) - self.start_time
        profile = self.replay_profile
        assert profile is not None
        while (
            self.replay_index < len(profile.sample_times) and elapsed + 1e-9 >= profile.sample_times[self.replay_index]
        ):
            index = self.replay_index
            self.replay_index += 1
            self.generated += 1
            if not profile.valid[index]:
                self.degenerate += 1
                continue
            noisy_position = np.asarray(sensor_position, dtype=np.float64) + profile.position_residuals[index]
            noisy_rotation = Rotation.from_euler("z", profile.yaw_residuals[index]) * Rotation.from_quat(
                sensor_quaternion
            )
            root_position, root_quaternion = converter(noisy_position, noisy_rotation.as_quat())
            sample_time = self.start_time + profile.sample_times[index]
            self.tracker.update(root_position, root_quaternion, sample_time, now)
            self.delivered += 1
        return self.tracker.estimate(now) if self.tracker.initialized else None

    def diagnostics(self, now: float) -> dict:
        estimate = self.tracker.estimate(now) if self.tracker.initialized else None
        return {
            "generated": self.generated,
            "delivered": self.delivered,
            "dropped": self.dropped,
            "degenerate": self.degenerate,
            "pending": len(self.pending),
            "age": None if estimate is None else estimate.age,
            "stale": estimate is None or estimate.stale,
        }
