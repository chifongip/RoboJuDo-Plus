"""Portable X2 capture validation and causal synchronization for safety replay."""

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import msgpack
import mujoco
import numpy as np
from box import Box
from scipy.spatial.transform import Rotation

from robojudo.config.x2.env.x2_env_cfg import X2_31DoF
from robojudo.config.x2.env.x2_mujuco_env_cfg import X2MujocoEnvCfg
from robojudo.config.x2.env.x2_real_env_cfg import X2SuperOdomCfg
from robojudo.environment.utils.odometry import OdometryReplayProfile, OdometryTracker, sensor_pose_to_root
from robojudo.tools.kinematics import MujocoKinematics
from robojudo.utils.rotation import TransformAlignment
from robojudo.utils.util_func import get_gravity_orientation

ODOMETRY_TOPIC = "/laser_odometry"
IMU_TOPIC = "/aima/hal/imu/torso/state"
JOINT_TOPICS = (
    "/aima/hal/joint/leg/state",
    "/aima/hal/joint/waist/state",
    "/aima/hal/joint/arm/state",
    "/aima/hal/joint/head/state",
)
REQUIRED_TOPICS = (ODOMETRY_TOPIC, IMU_TOPIC, *JOINT_TOPICS)


@dataclass(frozen=True)
class ReplayFrame:
    time: float
    odometry: dict
    imu: dict
    joints: dict[str, dict]
    ages: dict[str, float]


@dataclass(frozen=True)
class CaptureSelection:
    frames: list[ReplayFrame]
    snapshot: ReplayFrame
    start_time: float
    end_time: float
    diagnostics: dict


@dataclass(frozen=True)
class SeedState:
    root_position: np.ndarray
    odometry_origin_position: np.ndarray
    root_quaternion: np.ndarray
    linear_velocity_body: np.ndarray
    angular_velocity_body: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray


def load_capture(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open("rb") as stream:
        unpacker = msgpack.Unpacker(stream, raw=False)
        records.extend(unpacker)
    if not records or records[0].get("kind") != "metadata":
        raise ValueError("Capture is empty or lacks the metadata header")
    if records[0].get("schema_version") != 1:
        raise ValueError(f"Unsupported capture schema version: {records[0].get('schema_version')}")
    return records


def _finite_vector(record: dict, key: str, size: int) -> bool:
    value = np.asarray(record.get(key, []), dtype=np.float64)
    return value.shape == (size,) and np.isfinite(value).all()


def _valid_quaternion(record: dict) -> bool:
    if not _finite_vector(record, "quaternion", 4):
        return False
    norm = np.linalg.norm(record["quaternion"])
    return 0.99 <= norm <= 1.01


def _topic_records(records: list[dict]) -> dict[str, list[dict]]:
    by_topic = {topic: [] for topic in REQUIRED_TOPICS}
    for record in records:
        topic = record.get("topic")
        if topic in by_topic and np.isfinite(record.get("receipt_time", np.nan)):
            by_topic[topic].append(record)
    for topic_records in by_topic.values():
        topic_records.sort(key=lambda record: record["receipt_time"])
    return by_topic


def _latest(records: list[dict], times: list[float], timestamp: float) -> dict | None:
    index = bisect_right(times, timestamp) - 1
    return records[index] if index >= 0 else None


def _synchronize(by_topic: dict[str, list[dict]], timestamp: float) -> ReplayFrame | None:
    selected = {}
    ages = {}
    for topic in REQUIRED_TOPICS:
        records = by_topic[topic]
        record = _latest(records, [item["receipt_time"] for item in records], timestamp)
        if record is None:
            return None
        selected[topic] = record
        ages[topic] = timestamp - record["receipt_time"]
    joints = {}
    for topic in JOINT_TOPICS:
        for joint in selected[topic].get("joints", []):
            joints[joint.get("name", "")] = joint
    return ReplayFrame(
        time=timestamp,
        odometry=selected[ODOMETRY_TOPIC],
        imu=selected[IMU_TOPIC],
        joints=joints,
        ages=ages,
    )


def _frame_motion_score(frame: ReplayFrame, joint_names: list[str]) -> tuple[bool, float]:
    if frame.ages[ODOMETRY_TOPIC] > 0.3 or frame.ages[IMU_TOPIC] > 0.1:
        return False, float("inf")
    if any(frame.ages[topic] > 0.1 for topic in JOINT_TOPICS):
        return False, float("inf")
    if set(joint_names).difference(frame.joints):
        return False, float("inf")
    odometry = frame.odometry
    if float(odometry.get("pose_covariance", [np.inf])[0]) >= 0.5:
        return False, float("inf")
    if not _valid_quaternion(odometry) or not _valid_quaternion(frame.imu):
        return False, float("inf")
    joint_velocity = np.asarray([frame.joints[name]["velocity"] for name in joint_names], dtype=np.float64)
    gyro = np.asarray(frame.imu["angular_velocity"], dtype=np.float64)
    odom_velocity = np.asarray(odometry["linear_velocity"], dtype=np.float64)
    if not (np.isfinite(joint_velocity).all() and np.isfinite(gyro).all() and np.isfinite(odom_velocity).all()):
        return False, float("inf")
    tilt = float(np.arccos(np.clip(-get_gravity_orientation(np.asarray(frame.imu["quaternion"]))[2], -1.0, 1.0)))
    stable = (
        np.max(np.abs(joint_velocity)) <= 0.3
        and np.linalg.norm(gyro) <= 0.15
        and np.linalg.norm(odom_velocity) <= 0.1
        and tilt <= 0.35
    )
    score = float(np.mean(np.abs(joint_velocity))) + float(np.linalg.norm(gyro)) + float(np.linalg.norm(odom_velocity))
    return stable, score


def validate_and_select(
    records: list[dict],
    duration: float = 2.0,
    frequency: float = 50.0,
) -> CaptureSelection:
    """Validate the capture and choose the quietest complete two-second window."""
    by_topic = _topic_records(records)
    missing_topics = [topic for topic, values in by_topic.items() if not values]
    if missing_topics:
        raise ValueError(f"Capture is missing required topics: {missing_topics}")

    odometry = by_topic[ODOMETRY_TOPIC]
    bad_frames = [
        (record.get("frame_id"), record.get("child_frame_id"))
        for record in odometry
        if record.get("frame_id") != "map" or record.get("child_frame_id") != "lidar_chest_front"
    ]
    if bad_frames:
        raise ValueError(f"Unexpected odometry frames; expected map -> lidar_chest_front, got {bad_frames[0]}")
    if any(not _valid_quaternion(record) for record in odometry + by_topic[IMU_TOPIC]):
        raise ValueError("Capture contains a non-finite or non-normalized quaternion")
    odometry_gaps = np.diff([record["receipt_time"] for record in odometry])
    max_odom_gap = float(np.max(odometry_gaps, initial=0.0))
    if max_odom_gap > 0.3:
        raise ValueError(f"Odometry delivery gap {max_odom_gap:.3f}s exceeds the 0.300s timeout")
    degenerate_count = sum(float(record.get("pose_covariance", [np.inf])[0]) >= 0.5 for record in odometry)
    if degenerate_count:
        raise ValueError(f"Capture contains {degenerate_count} degenerate odometry samples")

    start = max(by_topic[topic][0]["receipt_time"] for topic in REQUIRED_TOPICS)
    end = min(by_topic[topic][-1]["receipt_time"] for topic in REQUIRED_TOPICS)
    period = 1.0 / frequency
    all_times = np.arange(np.ceil(start / period) * period, end + 1e-9, period)
    frames = [frame for timestamp in all_times if (frame := _synchronize(by_topic, float(timestamp))) is not None]
    required_count = int(round(duration * frequency)) + 1
    if len(frames) < required_count:
        raise ValueError(f"Capture has only {len(frames)} synchronized frames; {required_count} are required")

    joint_names = X2_31DoF().joint_names
    stable_scores = [_frame_motion_score(frame, joint_names) for frame in frames]
    candidates = []
    for start_index in range(len(frames) - required_count + 1):
        window = stable_scores[start_index : start_index + required_count]
        if all(stable for stable, _ in window):
            candidates.append((float(np.mean([score for _, score in window])), start_index))
    if not candidates:
        raise ValueError(
            "No stable two-second window: require fresh streams, all 31 joints, "
            "|dq|max<=0.3rad/s, |gyro|<=0.15rad/s, |odom velocity|<=0.1m/s, tilt<=0.35rad"
        )
    _, start_index = min(candidates)
    selected = frames[start_index : start_index + required_count]
    # Launch both safety stages from the same state at the beginning of a
    # verified-stable window, leaving the full requested duration available.
    snapshot = selected[0]
    return CaptureSelection(
        frames=selected,
        snapshot=snapshot,
        start_time=selected[0].time,
        end_time=selected[-1].time,
        diagnostics={
            "record_counts": {topic: len(values) for topic, values in by_topic.items()},
            "max_odometry_gap": max_odom_gap,
            "stable_window_motion_score": min(candidates)[0],
            "stable_window_start": selected[0].time,
            "stable_window_end": selected[-1].time,
        },
    )


def build_odometry_profile(
    selection: CaptureSelection,
    all_records: list[dict] | None = None,
    duration: float | None = None,
) -> OdometryReplayProfile:
    """Detrend recorded sensor motion, retaining timing jitter and short residuals."""
    if all_records is None:
        candidates = [frame.odometry for frame in selection.frames]
    else:
        odometry = _topic_records(all_records)[ODOMETRY_TOPIC]
        before_start = [record for record in odometry if record["receipt_time"] <= selection.start_time]
        candidates = ([before_start[-1]] if before_start else []) + [
            record
            for record in odometry
            if selection.start_time < record["receipt_time"] <= selection.start_time + (duration or 0.0)
        ]
    records = []
    seen = set()
    for record in candidates:
        identity = (record["receipt_time"], record.get("stamp"))
        if identity not in seen:
            records.append(record)
            seen.add(identity)
    if len(records) < 2:
        raise ValueError("At least two odometry samples are required to build a replay profile")
    times = np.asarray([record["receipt_time"] - records[0]["receipt_time"] for record in records])
    positions = np.asarray([record["position"] for record in records], dtype=np.float64)
    yaw = np.unwrap([Rotation.from_quat(record["quaternion"]).as_euler("xyz")[2] for record in records])
    design = np.column_stack([times, np.ones_like(times)])
    position_trend = design @ np.linalg.lstsq(design, positions, rcond=None)[0]
    yaw_trend = design @ np.linalg.lstsq(design, yaw, rcond=None)[0]
    position_residuals = positions - position_trend
    yaw_residuals = yaw - yaw_trend
    position_residuals -= position_residuals[0]
    yaw_residuals -= yaw_residuals[0]
    valid = np.asarray([float(record["pose_covariance"][0]) < 0.5 for record in records])
    return OdometryReplayProfile(times, position_residuals, yaw_residuals, valid)


def grounded_mujoco_seed(model, data, seed: SeedState, clearance: float = 0.002) -> tuple[np.ndarray, np.ndarray, dict]:
    """Create a physical MuJoCo state with the lowest X2 foot contact above ground."""
    if clearance < 0.0:
        raise ValueError("Foot clearance must be non-negative")
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()
    qpos[:3] = [0.0, 0.0, 0.0]
    qpos[3:7] = seed.root_quaternion[[3, 0, 1, 2]]
    qpos[-len(seed.joint_position) :] = seed.joint_position
    qvel[:] = 0.0
    qvel[:3] = Rotation.from_quat(seed.root_quaternion).apply(seed.linear_velocity_body)
    qvel[3:6] = seed.angular_velocity_body
    qvel[-len(seed.joint_velocity) :] = seed.joint_velocity

    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    collision_geom_ids = np.flatnonzero(model.geom_group == 3)
    foot_geom_ids = [
        int(geom_id)
        for geom_id in collision_geom_ids
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id]))
        in ("left_ankle_roll_link", "right_ankle_roll_link")
        and model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_SPHERE
    ]
    if not foot_geom_ids:
        raise ValueError("No X2 foot collision spheres were found in the MuJoCo model")
    bottoms = data.geom_xpos[foot_geom_ids, 2] - model.geom_size[foot_geom_ids, 0]
    qpos[2] = clearance - float(np.min(bottoms))
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    final_bottoms = data.geom_xpos[foot_geom_ids, 2] - model.geom_size[foot_geom_ids, 0]
    return (
        qpos,
        qvel,
        {
            "root_height": float(qpos[2]),
            "minimum_foot_clearance": float(np.min(final_bottoms)),
            "maximum_foot_clearance": float(np.max(final_bottoms)),
        },
    )


def reconstruct_environment_frames(selection: CaptureSelection) -> tuple[list[Box], SeedState]:
    """Reproduce AgiBotCppEnv's transforms without constructing robot command transport."""
    env_cfg = X2MujocoEnvCfg()
    odom_cfg = X2SuperOdomCfg()
    kinematics = MujocoKinematics(cfg=env_cfg.forward_kinematic)
    joint_names = env_cfg.dof.joint_names
    tracker = OdometryTracker(
        timeout=odom_cfg.odometry_timeout,
        velocity_filter_time_constant=odom_cfg.odometry_velocity_filter_time_constant,
    )
    alignment = TransformAlignment(yaw_only=True, xy_only=True)
    last_odometry_identity = None
    output = []
    raw_states = []
    first_stamp = None
    odometry_origin_position = None
    unique_odometry = []
    seen_odometry = set()
    for frame in selection.frames:
        identity = (frame.odometry["receipt_time"], frame.odometry.get("stamp"))
        if identity not in seen_odometry:
            unique_odometry.append(frame.odometry)
            seen_odometry.add(identity)
    header_stamps = np.asarray([float(record.get("stamp", 0.0)) for record in unique_odometry])
    use_header_stamps = np.all(header_stamps > 0.0) and np.all(np.diff(header_stamps) > 0.0)

    for frame in selection.frames:
        joint_position = np.asarray([frame.joints[name]["position"] for name in joint_names], dtype=np.float64)
        joint_velocity = np.asarray([frame.joints[name]["velocity"] for name in joint_names], dtype=np.float64)
        relative_fk = kinematics.forward(
            joint_pos=joint_position,
            base_pos=np.zeros(3),
            base_quat=np.asarray([0.0, 0.0, 0.0, 1.0]),
        )
        root_to_torso = relative_fk[env_cfg.torso_name]
        odometry_identity = (frame.odometry["receipt_time"], frame.odometry.get("stamp"))
        if odometry_identity != last_odometry_identity:
            root_position, odom_root_quaternion = sensor_pose_to_root(
                frame.odometry["position"],
                frame.odometry["quaternion"],
                odom_cfg.torso_to_odometry_sensor_position,
                odom_cfg.torso_to_odometry_sensor_quaternion,
                root_to_torso["pos"],
                root_to_torso["quat"],
            )
            if odom_cfg.odometry_position_mode == "RELATIVE_START":
                if odometry_origin_position is None:
                    odometry_origin_position = root_position.copy()
                root_position = root_position - odometry_origin_position
            stamp = float(frame.odometry["stamp"]) if use_header_stamps else float(frame.odometry["receipt_time"])
            if first_stamp is None:
                first_stamp = stamp
            tracker.update(
                root_position,
                odom_root_quaternion,
                sample_time=stamp - first_stamp,
                receipt_time=float(frame.odometry["receipt_time"]),
            )
            last_odometry_identity = odometry_identity
        estimate = tracker.estimate(frame.time)
        if estimate.stale:
            raise ValueError(f"Odometry became stale while reconstructing at t={frame.time:.3f}s")
        raw_quaternion = np.asarray(frame.imu["quaternion"], dtype=np.float64)
        if not output:
            alignment.set_base(raw_quaternion, estimate.position)
        base_quaternion, base_position = alignment.align_transform(raw_quaternion, estimate.position)
        angular_velocity = np.asarray(frame.imu["angular_velocity"], dtype=np.float64)
        fk_info = kinematics.forward(
            joint_pos=joint_position,
            base_pos=base_position,
            base_quat=base_quaternion,
            base_ang_vel=angular_velocity,
            base_lin_vel=estimate.linear_velocity_body,
        )
        output.append(
            Box(
                {
                    "dof_pos": joint_position.astype(np.float32),
                    "dof_vel": joint_velocity.astype(np.float32),
                    "base_quat": np.asarray(base_quaternion, dtype=np.float32),
                    "base_ang_vel": angular_velocity.astype(np.float32),
                    "base_lin_acc": np.asarray(frame.imu["linear_acceleration"], dtype=np.float32),
                    "base_pos": np.asarray(base_position, dtype=np.float32),
                    "base_lin_vel": estimate.linear_velocity_body.astype(np.float32),
                    "torso_pos": np.asarray(fk_info[env_cfg.torso_name]["pos"], dtype=np.float32),
                    "torso_quat": np.asarray(fk_info[env_cfg.torso_name]["quat"], dtype=np.float32),
                    "torso_ang_vel": np.asarray(fk_info[env_cfg.torso_name]["ang_vel"], dtype=np.float32),
                    "fk_info": fk_info,
                }
            )
        )
        raw_states.append(
            SeedState(
                root_position=np.asarray(estimate.position, dtype=np.float64),
                odometry_origin_position=np.asarray(
                    np.zeros(3) if odometry_origin_position is None else odometry_origin_position,
                    dtype=np.float64,
                ),
                root_quaternion=raw_quaternion,
                linear_velocity_body=np.asarray(estimate.linear_velocity_body, dtype=np.float64),
                angular_velocity_body=angular_velocity,
                joint_position=joint_position,
                joint_velocity=joint_velocity,
            )
        )
    snapshot_index = min(
        range(len(selection.frames)),
        key=lambda index: abs(selection.frames[index].time - selection.snapshot.time),
    )
    return output, raw_states[snapshot_index]
