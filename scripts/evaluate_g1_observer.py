"""Fit and evaluate the stiff 23-DOF G1 arm observer in headless MuJoCo."""

import argparse
import json
import time
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
from box import Box
from scipy.optimize import least_squares

from robojudo.config.g1.g1_locomimic_observer_cfg import g1_23_locomanipulation_locomimic_observer
from robojudo.pipeline.four_mode_pipeline import ControlMode
from robojudo.pipeline.g1_locomanipulation_loco_mimic_pipeline import G1LocomanipulationLocoMimicPipeline
from robojudo.utils.observers.model import JointDynamics

LOAD_START, RELEASE, DURATION = 1.0, 7.0, 12.0
MODES = ("OFF", "SHADOW", "ACTIVE")


def run_trial(mode, speed, force_n=0.0, *, excitation=None, tuning=None):
    cfg = g1_23_locomanipulation_locomimic_observer()
    upper_body_cfg = cfg.ctrl[-1]
    cfg.ctrl = []
    cfg.device = "cpu"
    cfg.record.enabled = False
    cfg.env.headless = True
    cfg.env.visualize_extras = False
    cfg.loco_policy.command_status_interval_s = 0.0
    observer = cfg.env.observer
    cfg.env.observer = type(observer).model_validate({**observer.model_dump(), **(tuning or {}), "mode": mode})
    pipeline = G1LocomanipulationLocoMimicPipeline(cfg)
    env = pipeline.env
    names = cfg.env.observer.joint_names
    indices = [env.joint_names.index(name) for name in names]
    hands = [env.model.body(f"{side}_wrist_roll_rubber_hand").id for side in ("left", "right")]
    torso = env.model.body("torso_link").id
    pelvis = env.model.body("pelvis").id
    speed_cmd = 0.0
    neutral_arm_goal = np.asarray(cfg.upper_dof_pos_default)
    arm_goal = neutral_arm_goal
    excitation_joint_scale = np.linspace(0.7, 1.3, len(names))

    def control_data(_):
        return Box(
            {
                "COMMANDS": [],
                "VELOCITY_SOURCE": "VelocityZmqCtrl",
                "POSTURE_SOURCE": None,
                "VelocityZmqCtrl": {
                    "fresh": True,
                    "linear_velocity": [speed_cmd, 0, 0],
                    "angular_velocity": [0, 0, 0],
                },
                "UpperBodyZmqCtrl": {
                    "fresh": True,
                    "joint_positions": dict(zip(names, arm_goal.tolist(), strict=True)),
                },
            }
        )

    step = env.step
    sample = {}

    def record_target(target, *args, **kwargs):
        sample.update(q=env.dof_pos[indices].copy(), dq=env.dof_vel[indices].copy(), target=target[indices].copy())
        step(target, *args, **kwargs)

    torque = env._simulate_torque
    saturated = np.zeros(env.num_dofs, dtype=bool)

    def record_substeps(torque_fn):
        def measure():
            values = torque_fn()
            saturated[:] |= np.abs(values) > env.torque_limits
            return values

        torque(measure)

    pipeline.ctrl_manager.get_ctrl_data = control_data
    # Exercise the pipeline's upper-body target handling with synthetic data and
    # no bound controller sockets.
    pipeline._upper_body_cfg = upper_body_cfg
    pipeline._configure_upper_body_override()
    try:
        pipeline._manual_mode_override = None
        env.elastic_band.active = True
        pipeline._enter_mode(ControlMode.JOINT_DEFAULT)
        for _ in range(pipeline._joint_default_steps):
            pipeline.step()
        if not pipeline._enter_mode(ControlMode.RL_DEFAULT):
            raise RuntimeError("Could not enter RL_DEFAULT")
        env.elastic_band.active = False
        env.elastic_band.apply()
        pipeline._set_upper_body_enabled(True)
        for _ in range(round(3.0 / pipeline.dt)):
            pipeline.step()
        speed_cmd = speed
        for _ in range(round(3.0 / pipeline.dt)):
            pipeline.step()

        rows = []
        with patch.object(env, "step", record_target), patch.object(env, "_simulate_torque", record_substeps):
            for i in range(round(DURATION / pipeline.dt)):
                t = i * pipeline.dt
                if excitation is not None:
                    amplitude, frequency = excitation
                    phases = 2 * np.pi * t * frequency * excitation_joint_scale
                    arm_goal = neutral_arm_goal + amplitude * np.sin(phases)
                loaded = LOAD_START <= t < RELEASE
                env.data.xfrc_applied[hands, 2] = -force_n if loaded else 0.0
                saturated[:] = False
                sample.clear()
                start = time.perf_counter()
                pipeline.step()
                loop_ms = 1000 * (time.perf_counter() - start)
                if pipeline.mode != ControlMode.RL_DEFAULT or not sample:
                    raise RuntimeError(f"G1 did not send an RL_DEFAULT target in {mode} at {t:.2f}s: {pipeline.mode}")
                if not np.isfinite(env.data.qpos).all() or not np.isfinite(env.data.qvel).all():
                    raise RuntimeError(f"G1 simulation became nonfinite in {mode} at {t:.2f}s")
                mujoco.mj_fwdPosition(env.model, env.data)
                rotation = env.data.xmat[torso].reshape(3, 3)
                status = env.get_observer_status()
                rows.append(
                    {
                        **sample,
                        "t": t,
                        "q_next": env.dof_pos[indices].copy(),
                        "dq_next": env.dof_vel[indices].copy(),
                        "effort": env.last_feedforward_effort.copy(),
                        "estimate": status.get("estimated_disturbance_nm", np.zeros(len(indices))),
                        "proposed": status.get("proposed_nm", np.zeros(len(indices))),
                        "hand": (env.data.xipos[hands] - env.data.xpos[torso]) @ rotation,
                        "tilt": np.arccos(np.clip(env.data.xmat[pelvis, 8], -1.0, 1.0)),
                        "height": env.data.xpos[pelvis, 2],
                        "base_xy": env.data.xpos[pelvis, :2].copy(),
                        "saturated": saturated.copy(),
                        "loop_ms": loop_ms,
                    }
                )
        return {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    finally:
        try:
            pipeline.ctrl_manager.close()
        finally:
            env.shutdown()


def fit_parameters():
    cfg = g1_23_locomanipulation_locomimic_observer()
    observer = cfg.env.observer
    indices = [cfg.env.dof.joint_names.index(name) for name in observer.joint_names]
    dof = cfg.env.dof
    kp, kd, limits = (np.asarray(values)[indices] for values in (dof.stiffness, dof.damping, dof.torque_limits))
    model = mujoco.MjModel.from_xml_path(cfg.env.xml)
    data = mujoco.MjData(model)
    for name, goal in zip(cfg.env.dof.joint_names, cfg.env.dof.default_pos, strict=True):
        data.qpos[model.joint(name).qposadr[0]] = goal
    mujoco.mj_forward(model, data)
    mass = np.zeros((model.nv, model.nv))
    try:
        mujoco.mj_fullM(model, data, mass)
    except TypeError:
        mujoco.mj_fullM(model, mass, data.qM)
    dof_indices = [model.joint(name).dofadr[0] for name in observer.joint_names]
    inertia = mass.diagonal()[dof_indices]
    friction = model.dof_frictionloss[dof_indices]
    dt = cfg.env.sim_dt * cfg.env.sim_decimation
    zeros = np.zeros_like(kp)
    train = [run_trial("OFF", speed, excitation=(0.16, freq)) for speed, freq in ((0.0, 0.9), (0.25, 1.2))]
    held_out = [run_trial("OFF", speed, excitation=(0.09, freq)) for speed, freq in ((0.0, 1.4), (0.3, 0.7))]

    def residual(params, traces):
        fitted_inertia, viscosity = np.split(params, 2)
        dynamics = JointDynamics(
            kp, kd, limits, inertia=fitted_inertia, friction_limit=friction, friction_viscosity=viscosity
        )
        residuals = []
        for trace in traces:
            predicted, _ = dynamics.predict_velocity(trace["q"], trace["dq"], trace["target"], zeros, zeros, dt)
            error = predicted - trace["dq_next"]
            # A constant velocity bias per trajectory is removed for fitting. The
            # observer still sees the full innovation, including gravity and coupling.
            residuals.append(error - np.median(error, axis=0))
        return np.concatenate(residuals).ravel()

    initial = np.concatenate((inertia, np.full_like(kp, 10.0)))
    lower = np.concatenate((0.25 * inertia, zeros))
    upper = np.concatenate((4.0 * inertia, np.full_like(kp, 100.0)))
    fit = least_squares(residual, initial, bounds=(lower, upper), args=(train,), loss="soft_l1", f_scale=0.15)
    if not fit.success:
        raise RuntimeError(f"G1 arm model fit failed: {fit.message}")
    fitted_inertia, viscosity = np.split(fit.x, 2)
    return {
        "joint_names": observer.joint_names,
        "inertia": fitted_inertia.round(6).tolist(),
        "friction_limit": friction.tolist(),
        "friction_viscosity": viscosity.round(6).tolist(),
        "train_initial_centered_rmse_rad_s": float(np.sqrt(np.mean(residual(initial, train) ** 2))),
        "train_fitted_centered_rmse_rad_s": float(np.sqrt(np.mean(residual(fit.x, train) ** 2))),
        "held_out_initial_centered_rmse_rad_s": float(np.sqrt(np.mean(residual(initial, held_out) ** 2))),
        "held_out_fitted_centered_rmse_rad_s": float(np.sqrt(np.mean(residual(fit.x, held_out) ** 2))),
    }


def summarize(trace, unloaded):
    carry = (trace["t"] >= RELEASE - 2.0) & (trace["t"] < RELEASE)
    after = trace["t"] >= RELEASE
    error = trace["q_next"] - trace["target"]
    displacement = np.linalg.norm(trace["hand"] - unloaded["hand"], axis=-1)
    correction = np.max(np.abs(trace["effort"] - unloaded["effort"]), axis=1)
    residual_limit = max(0.04, float(0.1 * np.max(correction[carry])))
    remains_above = np.flatnonzero(correction[after] > residual_limit)
    settled = remains_above[-1] + 1 if remains_above.size else 0
    release_times = trace["t"][after] - RELEASE
    settles_in_window = settled < len(release_times) and release_times[-1] - release_times[settled] >= 1.0
    # Hand poses are sampled after integration, so include the last loaded
    # sample as the pose at the instant of release.
    release_pose_index = np.flatnonzero(after)[0] - 1
    return {
        "carry_arm_rms_deg": float(np.rad2deg(np.sqrt(np.mean(error[carry] ** 2)))),
        "release_hand_peak_mm": float(1000 * np.max(displacement[release_pose_index:])),
        "correction_removal_s": float(release_times[settled]) if settles_in_window else None,
        "peak_effort_nm": float(np.max(np.abs(trace["effort"]))),
        "unloaded_peak_effort_nm": float(np.max(np.abs(unloaded["effort"]))),
        "saturated_control_intervals": int(np.count_nonzero(np.any(trace["saturated"], axis=1))),
        "max_base_tilt_deg": float(np.rad2deg(np.max(trace["tilt"]))),
        "min_pelvis_height_m": float(np.min(trace["height"])),
        "mean_planar_speed_m_s": float(
            np.linalg.norm(trace["base_xy"][-1] - trace["base_xy"][0]) / (trace["t"][-1] - trace["t"][0])
        ),
        "p99_loop_ms": float(np.percentile(trace["loop_ms"], 99)),
        "max_loop_ms": float(np.max(trace["loop_ms"])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", action="store_true", help="Fit the nominal model and check held-out prediction error.")
    parser.add_argument("--force", type=float, default=10.0, help="Downward load in newtons at each simulated hand.")
    parser.add_argument("--bandwidth", type=float, default=None, help="Override observer bandwidth in rad/s.")
    parser.add_argument("--correction-cap", type=float, default=None, help="Override correction cap in Nm.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.force) or args.force <= 0:
        parser.error("--force must be positive and finite")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.fit:
        results = fit_parameters()
    else:
        tuning = {
            name: value
            for name, value in (("bandwidth_rad_s", args.bandwidth), ("correction_limit_nm", args.correction_cap))
            if value is not None
        }
        results = {"force_per_hand_n": args.force, "tuning": tuning, "trials": {}}
        for motion, speed in (("standing", 0.0), ("walking", 0.25)):
            for mode in MODES:
                unloaded = run_trial(mode, speed, tuning=tuning)
                trace = run_trial(mode, speed, args.force, tuning=tuning)
                key = f"{motion}_{mode.lower()}"
                np.savez_compressed(args.output / f"{key}.npz", **trace)
                np.savez_compressed(args.output / f"{key}_unloaded.npz", **unloaded)
                results["trials"][key] = {
                    "loaded": summarize(trace, unloaded),
                    "unloaded": summarize(unloaded, unloaded),
                }
    (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
