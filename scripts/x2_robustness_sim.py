"""Simulation-only plant/feedback injection and bounded-memory physics telemetry."""

import copy
import ctypes
import hashlib
import time
import zipfile
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
from evaluate_x2_transitions import PHASES, X2TransitionEvaluationEnv, sample
from x2_feedback_perturbations import FeedbackChannel, Perturbation
from x2_integral_sim import POSTURE

from robojudo.environment import env_registry
from robojudo.utils.util_func import quat_rotate_inverse_np

INITIALIZATION_GATE = None


def construct_pipeline(factory, cfg):
    # Compiler asset caches are not plant state. Bound concurrent compilation and
    # release cached meshes after both plant and FK models have been constructed.
    with INITIALIZATION_GATE if INITIALIZATION_GATE is not None else nullcontext():
        pipeline = factory(cfg)
        mujoco.mj_clearCache(mujoco.mj_getCache())
        ctypes.CDLL(None).malloc_trim(0)
    return pipeline


class PhysicsWriter:
    """Stream numeric NPY chunks into one compressed NPZ; no whole-trial row list."""

    def __init__(self, path):
        self.archive = zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1)
        self.columns = None
        self.rows = []
        self.chunks = self.count = 0

    def write(self, row):
        row = dict(row)
        phase = row.pop("phase")
        row["phase_id"] = next(i for i, p in enumerate(PHASES) if p[0] == phase)
        if self.columns is None:
            self.columns = tuple(row)
        if tuple(row) != self.columns:
            raise ValueError("Physics telemetry schema changed within a trial")
        self.rows.append(list(row.values()))
        self.count += 1
        if len(self.rows) >= 256:
            self.flush()

    def flush(self):
        if self.rows:
            with self.archive.open(f"chunk_{self.chunks:04d}.npy", "w") as stream:
                np.lib.format.write_array(stream, np.asarray(self.rows, dtype=np.float64), allow_pickle=False)
            self.rows.clear()
            self.chunks += 1

    def close(self):
        self.flush()
        with self.archive.open("columns.npy", "w") as stream:
            np.lib.format.write_array(stream, np.asarray(self.columns or (), dtype=str), allow_pickle=False)
        self.archive.close()


def load_physics(path):
    with np.load(path, allow_pickle=False) as archive:
        columns = archive["columns"].tolist()
        chunks = [archive[k] for k in sorted(archive.files) if k.startswith("chunk_")]
    data = np.concatenate(chunks) if chunks else np.empty((0, len(columns)))
    return {name: data[:, i] for i, name in enumerate(columns)}


def perturb_plant(model, joint_names, config):
    """Change actual MuJoCo arrays only; do not reinterpret observer fitted parameters."""
    joint_ids = np.array([model.joint(name).id for name in joint_names])
    dof_ids = model.jnt_dofadr[joint_ids]
    body_ids = np.unique(model.jnt_bodyid[joint_ids])
    fields = {"body_inertia": body_ids, "dof_armature": dof_ids, "dof_frictionloss": dof_ids}
    original = {name: getattr(model, name)[indices].copy() for name, indices in fields.items()}
    for field, multiplier in (
        ("body_inertia", config.inertia_scale),
        ("dof_armature", config.armature_scale),
        ("dof_frictionloss", config.friction_scale),
    ):
        if multiplier != 1:
            getattr(model, field)[fields[field]] *= multiplier
    if any(x != 1 for x in (config.inertia_scale, config.armature_scale, config.friction_scale)):
        mujoco.mj_setConst(model, mujoco.MjData(model))
    return {
        field: {
            "indices": ids.tolist(),
            "original": original[field].tolist(),
            "perturbed": getattr(model, field)[ids].tolist(),
        }
        for field, ids in fields.items()
    }


@env_registry.register
class X2RobustnessEvaluationEnv(X2TransitionEvaluationEnv):
    def configure_robustness(self, config, seed, output, condition, profile, repetition):
        self._robustness_seed = seed
        self.perturbation = Perturbation(**config)
        self.plant_metadata = perturb_plant(self.model, POSTURE, self.perturbation)
        self.feedback = None
        self.writer = None
        self.physics_path = Path(output) / profile / f"{condition}_{repetition}_physics.npz"
        self.physics_path.parent.mkdir(parents=True, exist_ok=True)
        self.telemetry_data = mujoco.MjData(self.model)
        self.wall_start = time.monotonic()
        self.feedback_origin = None
        self.exogenous_hash = hashlib.sha256()

    def start_feedback(self, warmup_s):
        self.feedback_origin = self.data.time + warmup_s
        self.feedback = FeedbackChannel(
            self.perturbation,
            self.robustness_seed,
            self.dof_pos[self.arm_indices],
            self.dof_vel[self.arm_indices],
            self.data.time,
        )

    @property
    def robustness_seed(self):
        return self._robustness_seed

    def start_telemetry(self):
        self.writer = PhysicsWriter(self.physics_path)

    def close_telemetry(self):
        if self.writer is not None:
            self.physics_sample_count = self.writer.count
            self.writer.close()
            self.writer = None

    def correction_feedback(self):
        if self.feedback is None:
            return super().correction_feedback()
        return self.feedback.measured()

    def step(self, pd_target, hand_pose=None, *, feedforward_effort=None):
        self._shared_pd_step = True
        try:
            return super().step(pd_target, hand_pose, feedforward_effort=feedforward_effort)
        finally:
            self._shared_pd_step = False

    def _simulate_torque(self, torque_fn):
        if (
            not getattr(self, "feedback", None)
            or not self.perturbation.shared_pd
            or not getattr(self, "_shared_pd_step", False)
        ):
            return super()._simulate_torque(torque_fn)

        def measured_pd():
            torque = torque_fn()
            q, v = self.feedback.measured()
            arm = self.arm_indices
            pd = self.stiffness[arm] * (self.last_target[arm] - q) - self.damping[arm] * v
            self.last_pd_torque[arm] = np.where(self._control_mask[arm], pd, 0)
            torque[arm] = self.last_pd_torque[arm] + self.last_feedforward_effort[arm]
            return torque

        return super()._simulate_torque(measured_pd)

    def update(self, simple=False):
        super().update(simple)
        if simple and hasattr(self, "evaluation_after_physics"):
            self.evaluation_after_physics()
        if not simple or not getattr(self, "feedback", None):
            return
        self.feedback.acquire(
            self.data.time,
            self.dof_pos[self.arm_indices],
            self.dof_vel[self.arm_indices],
            self.data.time - self.feedback_origin,
        )
        if self.writer is None:
            return
        # Compute evaluation kinematics on separate MjData: no forward calls on the plant.
        shadow = copy.copy(self)
        shadow.data = self.telemetry_data
        shadow.data.qpos[:] = self.data.qpos
        shadow.data.qvel[:] = self.data.qvel
        shadow.data.time = self.data.time
        mujoco.mj_fwdPosition(self.model, shadow.data)
        raw_quat = self.data.qpos.astype(np.float32)[3:7][[1, 2, 3, 0]]
        shadow._base_lin_vel = quat_rotate_inverse_np(raw_quat, self.data.qvel.astype(np.float32)[:3])
        elapsed = self.data.time - self.measurement_start
        row = sample(shadow, elapsed)
        row.update(
            feedback_acquisition_s=self.feedback.acquired_time - self.measurement_start,
            feedback_age_s=self.data.time - self.feedback.acquired_time,
            acquisition_latency_s=self.feedback.last_latency,
            acquisition_dropped=int(self.feedback.dropped),
            correction_update_s=self.reference_time,
        )
        for i, name in enumerate(POSTURE):
            row[f"{name}_measured_position_rad"] = self.feedback.position[i]
            row[f"{name}_measured_velocity_rad_s"] = self.feedback.velocity[i]
            row[f"{name}_injected_position_noise_rad"] = self.feedback.last_noise[i]
            row[f"{name}_injected_velocity_noise_rad_s"] = self.feedback.last_noise[14 + i]
        for i, name in enumerate(self.joint_names):
            row[f"{name}_saturation_fraction"] = int(self.last_saturation_mask[i])
        if hasattr(self, "evaluation_telemetry"):
            row.update(self.evaluation_telemetry())
        self.exogenous_hash.update(self.feedback.last_noise.astype("<f8").tobytes())
        self.exogenous_hash.update(
            np.asarray([self.feedback.last_latency, self.feedback.dropped], dtype="<f8").tobytes()
        )
        self.writer.write(row)

    def robustness_summary(self):
        import resource

        return {
            "case": self.perturbation.case,
            "seed": self.robustness_seed,
            "exogenous_sha256": self.exogenous_hash.hexdigest(),
            "perturbation": asdict(self.perturbation),
            "plant": self.plant_metadata,
            "physics_trace": str(self.physics_path.relative_to(self.physics_path.parent.parent)),
            "physics_samples": getattr(self, "physics_sample_count", 0),
            "feedback_drop_count": self.feedback.drop_count if self.feedback else 0,
            "discarded_out_of_order": self.feedback.out_of_order if self.feedback else 0,
            "wall_seconds": time.monotonic() - self.wall_start,
            "worker_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "worker_live_rss_mib": float(
                next(
                    line.split()[1]
                    for line in Path("/proc/self/status").read_text().splitlines()
                    if line.startswith("VmRSS:")
                )
            )
            / 1024,
        }


@env_registry.register
class X2CompliantPayloadEvaluationEnv(X2RobustnessEvaluationEnv):
    """A separate translating payload state; robot/policy dimensions are unchanged."""

    def configure_payload(self, config):
        from x2_compliant_payload import CompliantPayload

        self.payload = CompliantPayload(**config)
        self.payload_pending_at = None
        self.payload_force_time = -1.0
        self.payload_jacp = np.zeros((3, self.model.nv))
        self.payload_jacr = np.zeros_like(self.payload_jacp)

    def hand_midpoint_kinematics(self):
        velocities = []
        for body_id in self.ee_ids:
            mujoco.mj_jacBodyCom(self.model, self.data, self.payload_jacp, self.payload_jacr, body_id)
            velocities.append(self.payload_jacp @ self.data.qvel)
        return self.force_points().mean(axis=0), np.mean(velocities, axis=0)

    def update_load(self):
        if self.measurement_start is None:
            return super().update_load()
        mujoco.mj_fwdPosition(self.model, self.data)
        self.check_state()
        elapsed = round(float(self.data.time - self.measurement_start), 8)
        self.force[:] = 0
        if 26 <= elapsed < 79:
            from evaluate_x2_transitions import schedule

            anchor, velocity = self.hand_midpoint_kinematics()
            if not self.payload.initialized:
                self.payload.initialize(anchor, velocity)
            if self.payload_pending_at is None:
                self.force[:] = self.payload.evaluate(anchor, velocity, schedule(elapsed)[2])
                self.payload_pending_at = float(self.data.time)
                self.payload_force_time = elapsed
            elif abs(self.data.time - self.payload_pending_at) < 1e-9:
                self.force[:] = self.payload.hand_force
            else:
                raise RuntimeError("Payload physics interval was not integrated")
        elif elapsed >= 79 and self.payload.initialized:
            self.payload.detach()
        self.last_force_n = float(-self.force[2])
        for body_id in self.ee_ids:
            self.data.xfrc_applied[body_id] = np.r_[self.force, np.zeros(3)]

    def evaluation_after_physics(self):
        if not hasattr(self, "payload_pending_at") or self.payload_pending_at is None:
            return
        dt = float(self.data.time - self.payload_pending_at)
        if abs(dt - self.sim_dt) > 1e-8:
            raise RuntimeError("Payload must advance exactly once per robot physics step")
        mujoco.mj_fwdPosition(self.model, self.data)
        anchor, velocity = self.hand_midpoint_kinematics()
        self.payload.advance(dt, anchor, velocity)
        self.payload_pending_at = None

    def evaluation_telemetry(self):
        return {**self.payload.telemetry(), "payload_force_evaluation_s": self.payload_force_time}

    def robustness_summary(self):
        return {
            **super().robustness_summary(),
            "payload": self.payload.configuration(),
            "payload_integration_steps": self.payload.steps,
            "physics_dt_s": self.sim_dt,
            "controller_dt_s": self.control_dt,
        }

    def _update_force_visualization(self):
        super()._update_force_visualization()
        if self.viewer is None or not hasattr(self, "payload"):
            return
        visible = self.payload.initialized and not self.payload.detached
        color = np.array([0.2, 0.65, 1.0, float(visible)])
        self.viewer.add_marker(
            id=4200,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=self.payload.position.copy(),
            size=np.full(3, 0.04),
            rgba=color,
            label="compliant payload (translation only)",
        )
        for i, point in enumerate(self.force_points()):
            geom = mujoco.MjvGeom()
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.003, point, self.payload.position)
            self.viewer.add_marker(
                id=4201 + i,
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                pos=geom.pos.copy(),
                mat=geom.mat.copy(),
                size=geom.size.copy(),
                rgba=color,
                label="",
            )
