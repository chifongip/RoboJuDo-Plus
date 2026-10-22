"""Evaluate bounded torque-integral correction through pickup, walking and placement.

Simulation only: no SDK changes, ROS nodes, hardware commands or sockets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mujoco
import numpy as np
from x2_integral_sim import (
    POSTURE,
    SIDES,
    SimulationFailure,
    X2IntegralEvaluationEnv,
    column,
    detrended_std,
    evaluation_config,
    force_point_metadata,
    integral_sample,
    point_wrench,
    rows_in,
    synthetic_velocity,
)

from robojudo.config.x2.x2_cfg import x2_locomimic
from robojudo.controller.ctrl_cfgs import UpperBodyZmqCtrlCfg
from robojudo.environment import env_registry
from robojudo.pipeline.four_mode_pipeline import ControlMode
from robojudo.pipeline.x2_locomanipulation_loco_mimic_pipeline import X2LocomanipulationLocoMimicPipeline
from robojudo.utils.bounded_integral import BoundedIntegralCorrection

DURATION = 105.0


PHASES = (
    ("default_baseline", 0, 10),
    ("reach", 10, 16),
    ("grasp_baseline", 16, 26),
    ("lift", 26, 30),
    ("carry_standing", 30, 40),
    ("accelerate", 40, 45),
    ("walk", 45, 60),
    ("decelerate", 60, 65),
    ("preplace_standing", 65, 75),
    ("lower", 75, 79),
    ("grasp_recovery", 79, 89),
    ("return_default", 89, 95),
    ("default_recovery", 95, 105),
)


TRANSITIONS = ("reach", "lift", "lower", "return_default")


WINDOWS = {
    "carry_standing": (35, 40),
    "walk": (55, 60),
    "preplace_standing": (70, 75),
    "grasp_recovery": (86, 89),
    "default_recovery": (102, 105),
}


def make_correction(env, parameters=None):
    return BoundedIntegralCorrection(
        env.stiffness[env.arm_indices],
        env.damping[env.arm_indices],
        env.torque_limits[env.arm_indices],
        **(parameters or {}),
    )


def smoothstep(s):
    s = float(np.clip(s, 0, 1))
    return s**3 * (10 - 15 * s + 6 * s**2)


def default_posture():
    cfg = x2_locomimic().env.dof
    return np.asarray([cfg.default_pos[cfg.joint_names.index(name)] for name in POSTURE])


def deployment_upper_config():
    return next(c.model_copy(deep=True) for c in x2_locomimic().ctrl if isinstance(c, UpperBodyZmqCtrlCfg))


def schedule(t):
    phase = next((name for name, start, end in PHASES if start <= t < end), "default_recovery")
    fraction = 0.0
    if 10 <= t < 16:
        fraction = 0.8 * smoothstep((t - 10) / 6)
    elif 16 <= t < 26:
        fraction = 0.8
    elif 26 <= t < 30:
        fraction = 0.8 + 0.2 * smoothstep((t - 26) / 4)
    elif 30 <= t < 75:
        fraction = 1.0
    elif 75 <= t < 79:
        fraction = 1.0 - 0.2 * smoothstep((t - 75) / 4)
    elif 79 <= t < 89:
        fraction = 0.8
    elif 89 <= t < 95:
        fraction = 0.8 * (1 - smoothstep((t - 89) / 6))
    load = 0.0
    if 26 <= t < 30:
        load = smoothstep((t - 26) / 4)
    elif 30 <= t < 75:
        load = 1.0
    elif 75 <= t < 79:
        load = 1 - smoothstep((t - 75) / 4)
    vx = 0.0
    if 40 <= t < 45:
        vx = 0.5 * smoothstep((t - 40) / 5)
    elif 45 <= t < 60:
        vx = 0.5
    elif 60 <= t < 65:
        vx = 0.5 * (1 - smoothstep((t - 60) / 5))
    return phase, fraction, load, vx


def external_force(t, profile):
    load = schedule(t)[2] if profile != "unloaded" else 0.0
    sine = math.sin(2 * math.pi * 0.8 * (t - 30)) if profile == "sway" else 0.0
    return load * np.asarray([5 * sine, 0, -20 - 5 * sine])


@env_registry.register
class X2TransitionEvaluationEnv(X2IntegralEvaluationEnv):
    def __init__(self, cfg_env, device="cpu"):
        super().__init__(cfg_env, device)
        self.desired_arm = default_posture()
        self.reference_velocity = np.zeros(len(POSTURE))
        self.previous_reference = None
        self.reference_time = 0.0

    def update_load(self):
        mujoco.mj_fwdPosition(self.model, self.data)
        self.check_state()
        self.force[:] = (
            0
            if self.measurement_start is None
            else external_force(self.data.time - self.measurement_start, self.profile)
        )
        self.last_force_n = float(-self.force[2])
        for body_id in self.ee_ids:
            self.data.xfrc_applied[body_id] = point_wrench(
                self.data.xipos[body_id], self.data.xipos[body_id], self.force
            )

    def step(self, pd_target, hand_pose=None, *, feedforward_effort=None):
        if feedforward_effort is not None:
            raise ValueError("Evaluation owns additive effort")
        target = np.asarray(pd_target)[self.arm_indices]
        self.reference_velocity[:] = (
            0 if self.previous_reference is None else ((target - self.previous_reference) / self.control_dt)
        )
        self.previous_reference = target.copy()
        super().step(pd_target, hand_pose, feedforward_effort=feedforward_effort)


def transition_config(viewer=False):
    cfg = evaluation_config(viewer)
    cfg.env.env_type = "X2TransitionEvaluationEnv"
    defaults = default_posture()
    for name, value in zip(POSTURE, defaults, strict=True):
        cfg.joint_default_dof.default_pos[cfg.joint_default_dof.joint_names.index(name)] = float(value)
        i = cfg.env.dof.joint_names.index(name) - (cfg.env.dof.num_dofs - cfg.upper_dof_num)
        cfg.upper_dof_pos_default[i] = float(value)
    cfg.env.feedforward_effort_limits = {
        name: min(8.0, 0.8 * cfg.env.dof.torque_limits[cfg.env.dof.joint_names.index(name)]) for name in POSTURE
    }
    return cfg


def sample(env, t):
    row = integral_sample(env, t)
    row["phase"] = schedule(env.reference_time)[0]
    row["reference_time_s"] = env.reference_time
    error = env.dof_pos[env.arm_indices] - env.last_target[env.arm_indices]
    row["arm_rms_error_rad"] = float(np.sqrt(np.mean(error**2)))
    for i, name in enumerate(POSTURE):
        row[f"{name}_desired_rad"] = float(env.desired_arm[i])
        row[f"{name}_target_velocity_rad_s"] = float(env.reference_velocity[i])
    return row


def phase_statistics(rows):
    stats = {}
    for phase, _start, _end in PHASES:
        selected = [r for r in rows if r["phase"] == phase]
        if not selected:
            continue
        rms = column(selected, "arm_rms_error_rad")
        stats[phase] = {
            "samples": len(selected),
            "rms_tracking_error_rad": float(np.sqrt(np.mean(rms**2))),
            "peak_joint_error_rad": float(
                max(abs(r[f"{name}_position_rad"] - r[f"{name}_target_rad"]) for r in selected for name in POSTURE)
            ),
            "joint_error_rad": {
                name: {
                    "mean_abs": float(
                        np.mean(
                            np.abs(column(selected, f"{name}_position_rad") - column(selected, f"{name}_target_rad"))
                        )
                    ),
                    "p95_abs": float(
                        np.percentile(
                            np.abs(column(selected, f"{name}_position_rad") - column(selected, f"{name}_target_rad")),
                            95,
                        )
                    ),
                }
                for name in POSTURE
            },
        }
    return stats


def save_trace(path, rows):
    with path.open("w", newline="") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run_trial(condition, profile, repetition, output, viewer=False, parameters=None):
    np.random.seed(repetition)
    pipeline = X2LocomanipulationLocoMimicPipeline(transition_config(viewer))
    for policy in pipeline.policy_manager.policies:
        if hasattr(policy, "close_progress"):
            policy.close_progress()
    env = pipeline.env
    env.integral = make_correction(env, parameters)
    env.profile = profile
    defaults, carrying = default_posture(), np.asarray(list(POSTURE.values()))
    upper_cfg = deployment_upper_config()
    rows, failure = [], ""

    def control_data(env_data):
        data = synthetic_velocity(env.command_vx)
        data["UpperBodyZmqCtrl"] = {"fresh": True, "joint_positions": dict(zip(POSTURE, env.desired_arm, strict=True))}
        return data

    pipeline.ctrl_manager.get_ctrl_data = control_data

    def advance():
        pipeline.step()
        if pipeline.mode != ControlMode.RL_DEFAULT:
            raise SimulationFailure(f"policy left RL_DEFAULT: {pipeline.mode}")

    try:
        if [(x.name, x.shape) for x in pipeline.policy.session.get_inputs()] != [("obs", [1, 430])]:
            raise SimulationFailure("unexpected model input")
        if [(x.name, x.shape) for x in pipeline.policy.session.get_outputs()] != [("actions", [1, 15])]:
            raise SimulationFailure("unexpected model output")
        pipeline._manual_mode_override = None
        env.elastic_band.active = True
        pipeline._enter_mode(ControlMode.JOINT_DEFAULT)
        for _ in range(pipeline._joint_default_steps):
            pipeline.step()
        if not pipeline._enter_mode(ControlMode.RL_DEFAULT):
            raise SimulationFailure("could not activate policy")
        env.elastic_band.active = False
        env.elastic_band.apply()
        for _ in range(round(5 / pipeline.dt)):
            advance()
        # Reuse the actual deployment override, including EMA, snapping, clipping,
        # and max_joint_velocity_rad_s. No network controller is instantiated.
        pipeline._upper_body_cfg = upper_cfg
        pipeline._configure_upper_body_override()
        pipeline._upper_body_filtered = env.last_target[env.arm_indices].astype(np.float32).copy()
        pipeline._set_upper_body_enabled(True)
        env.integral_enabled = condition == "integral"
        env.measurement_start = env.data.time
        env.physics_steps = env.saturated_steps = 0
        for step in range(round(DURATION / pipeline.dt)):
            t = step * pipeline.dt
            env.reference_time = t
            _, fraction, _, env.command_vx = schedule(t)
            env.desired_arm = defaults + fraction * (carrying - defaults)
            advance()
            rows.append(sample(env, float(env.data.time - env.measurement_start)))
    except (SimulationFailure, ValueError, RuntimeError, FloatingPointError) as error:
        failure = str(error)
        logging.exception("%s %s repetition %d failed", condition, profile, repetition)
    finally:
        env.clear_load()
        pipeline.ctrl_manager.close()
        env.shutdown()
    directory = output / profile
    directory.mkdir(exist_ok=True)
    path = directory / f"{condition}_{repetition}.csv"
    save_trace(path, rows)
    summary = {
        "condition": condition,
        "correction": "bounded_integral",
        "correction_parameters": parameters or {},
        "profile": profile,
        "repetition": repetition,
        "status": "failed" if failure else "completed",
        "termination_reason": failure,
        "samples": len(rows),
        "trace": str(path.relative_to(output)),
        "max_base_tilt_rad": env.max_base_tilt,
        "min_base_height_m": env.min_base_height,
        "saturated_physics_fraction": env.saturated_steps / max(env.physics_steps, 1),
        "max_effort_slew_nm_s": env.max_effort_slew,
        "max_effort_bound_excess_nm": env.max_effort_bound_excess,
        "phases": phase_statistics(rows),
    }
    if rows:
        summary["max_target_speed_rad_s"] = float(
            max(abs(r[f"{name}_target_velocity_rad_s"]) for r in rows for name in POSTURE)
        )
        walking = [r for r in rows if r["phase"] == "walk"]
        summary["mean_walking_speed_m_s"] = float(np.mean(column(walking, "actual_vx_m_s"))) if walking else None
    (directory / f"{condition}_{repetition}.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def read_trace(output, summary):
    with (output / summary["trace"]).open() as stream:
        return [{k: v if k == "phase" else float(v) for k, v in row.items()} for row in csv.DictReader(stream)]


def hand_difference(rows, reference, start, end, side):
    # Compare the same time/trajectory against an unloaded run; arm movement is not sag.
    current = rows_in(rows, start, end)
    unloaded = rows_in(reference, start, end)
    if len(current) != len(unloaded) or not current:
        raise ValueError("matched reference samples are missing")
    delta = column(unloaded, f"{side}_torso_z_m") - column(current, f"{side}_torso_z_m")
    displacement = np.stack(
        [column(unloaded, f"{side}_torso_{axis}_m") - column(current, f"{side}_torso_{axis}_m") for axis in "xyz"],
        axis=1,
    )
    return float(np.mean(delta)), float(np.max(np.linalg.norm(displacement, axis=1)))


def baseline_drift(rows):
    """Report integral drift at matching held postures, without correcting the controller."""
    result = {}
    for phase, start, end in (
        ("default_baseline", 7, 10),
        ("grasp_baseline", 23, 26),
        ("grasp_recovery", 86, 89),
        ("default_recovery", 102, 105),
    ):
        selected = rows_in(rows, start, end)
        slopes = {
            name: float(np.polyfit(column(selected, "elapsed_s"), column(selected, f"{name}_integral_nm"), 1)[0])
            for name in POSTURE
        }
        result[phase] = {"effort_slope_nm_s": slopes}
    return result


def comparisons(summaries, output):
    """Report matched measurements without angular or torque-recovery cutoffs."""
    index = {(s["condition"], s["profile"], s["repetition"]): s for s in summaries}
    results = []
    for repetition in sorted({s["repetition"] for s in summaries}):
        for profile in ("constant", "sway"):
            keys = [
                (condition, load, repetition)
                for condition, load in (
                    ("pd_only", "unloaded"),
                    ("integral", "unloaded"),
                    ("pd_only", profile),
                    ("integral", profile),
                )
            ]
            result = {"profile": profile, "repetition": repetition, "status": "unavailable"}
            if not all(index.get(key, {}).get("status") == "completed" for key in keys):
                result["reason"] = "matched loaded/unloaded trials incomplete or failed"
                results.append(result)
                continue
            pu, iu, pl, il = [read_trace(output, index[key]) for key in keys]
            result["status"] = "measured"
            result["baseline_drift"] = {"unloaded_reference": baseline_drift(iu), "loaded": baseline_drift(il)}
            result["postplace_peak_diagnostic"] = {
                side: {
                    "pd_peak_m": hand_difference(pl, pu, 79, 82, side)[1],
                    "integral_peak_m": hand_difference(il, iu, 79, 82, side)[1],
                }
                for side in SIDES
            }
            phases = {}
            for phase, start, end in PHASES:
                if phase not in TRANSITIONS:
                    continue
                measured = {
                    "pd_rms_tracking_rad": index[keys[-2]]["phases"][phase]["rms_tracking_error_rad"],
                    "integral_rms_tracking_rad": index[keys[-1]]["phases"][phase]["rms_tracking_error_rad"],
                }
                for side in SIDES:
                    measured[f"{side}_pd_peak_m"] = hand_difference(pl, pu, start, end, side)[1]
                    measured[f"{side}_integral_peak_m"] = hand_difference(il, iu, start, end, side)[1]
                phases[phase] = measured
            result["transitions"] = phases
            support = {}
            for phase in ("carry_standing", "walk", "preplace_standing"):
                start, end = WINDOWS[phase]
                pd_rows, integral_rows = rows_in(pl, start, end), rows_in(il, start, end)
                measured = {}
                for signal in ("arm_rms_error_rad", "base_tilt_rad"):
                    measured[f"pd_{signal}_detrended_std"] = detrended_std(column(pd_rows, signal))
                    measured[f"integral_{signal}_detrended_std"] = detrended_std(column(integral_rows, signal))
                measured["pd_rms_rad"] = float(np.sqrt(np.mean(column(pd_rows, "arm_rms_error_rad") ** 2)))
                measured["integral_rms_rad"] = float(np.sqrt(np.mean(column(integral_rows, "arm_rms_error_rad") ** 2)))
                for side in SIDES:
                    old = hand_difference(pl, pu, start, end, side)[0]
                    new = hand_difference(il, iu, start, end, side)[0]
                    measured[f"{side}_pd_sag_m"] = old
                    measured[f"{side}_integral_sag_m"] = new
                    measured[f"{side}_reduction"] = 1 - abs(new) / abs(old) if abs(old) > 1e-12 else None
                support[phase] = measured
            result["support"] = support
            result["recovery"] = {
                phase: {
                    "joints": {
                        name: {
                            "residual_nm": float(
                                np.mean(column(rows_in(il, *WINDOWS[phase]), f"{name}_integral_nm"))
                                - np.mean(column(rows_in(iu, *WINDOWS[phase]), f"{name}_integral_nm"))
                            )
                        }
                        for name in POSTURE
                    }
                }
                for phase in ("grasp_recovery", "default_recovery")
            }
            results.append(result)
    return results


def worker(job):
    return run_trial(*job)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("logs/x2_transitions") / time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--profile", choices=("unloaded", "constant", "sway"), action="append")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--integral-gain-scale", type=float, default=0.5, help="Fixed Ki/Kp in 1/s")
    parser.add_argument("--error-filter-tau", type=float, default=0.25)
    parser.add_argument(
        "--error-deadband-deg", type=float, default=0.2, help="Integration noise band, not an accuracy cutoff"
    )
    args = parser.parse_args()
    if args.repetitions < 1 or args.jobs < 1 or (args.viewer and args.jobs != 1):
        parser.error("positive repetitions/jobs required; --viewer requires --jobs 1")
    profiles = args.profile or ["unloaded", "constant", "sway"]
    if len(set(profiles)) != len(profiles):
        parser.error("profiles must be unique")
    parameters = {
        "integral_gain_scale": args.integral_gain_scale,
        "filter_tau": args.error_filter_tau,
        "deadband_deg": args.error_deadband_deg,
    }
    try:
        BoundedIntegralCorrection([1], [1], [1], **parameters)
    except ValueError as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("robojudo").setLevel(logging.WARNING)
    cfg = transition_config()
    model = Path(cfg.loco_policy.policy_file)
    report = {
        "model": str(model),
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "robot_xml_sha256": hashlib.sha256(Path(cfg.env.xml).read_bytes()).hexdigest(),
        "controller_sha256": hashlib.sha256(Path("robojudo/utils/bounded_integral.py").read_bytes()).hexdigest(),
        "correction": "bounded_integral",
        "assessment": "baseline_relative",
        "absolute_joint_error_cutoff_rad": None,
        "hardware_validation": "not_performed",
        **force_point_metadata(cfg.env.xml),
        "comparison_peak_metric": "3D torso-frame Euclidean displacement against matched unloaded trajectory",
        "duration_s": DURATION,
        "phases": PHASES,
        "default_posture_rad": dict(zip(POSTURE, default_posture(), strict=True)),
        "carry_posture_rad": POSTURE,
        "grasp_fraction": 0.8,
        "walking_command_m_s": 0.5,
        "measured_suspension": False,
        "upper_body_deployment_config": deployment_upper_config().model_dump(mode="json"),
        "controller_parameters": {
            "ki_over_kp_per_s": args.integral_gain_scale,
            "filter_tau_s": args.error_filter_tau,
            "deadband_deg": args.error_deadband_deg,
            "effort_limit": "min(8 Nm, 0.8 * simulated actuator limit)",
            "slew_nm_s": 2,
        },
        "trials": [],
    }
    jobs = [
        (condition, profile, repetition, args.output, args.viewer, parameters)
        for profile in profiles
        for repetition in range(1, args.repetitions + 1)
        for condition in ("pd_only", "integral")
    ]

    def record(summary):
        report["trials"].append(summary)
        (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: v for k, v in summary.items() if k != "phases"}), flush=True)

    if args.jobs == 1:
        for job in jobs:
            record(worker(job))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=multiprocessing.get_context("spawn")) as executor:
            for future in as_completed([executor.submit(worker, job) for job in jobs]):
                record(future.result())
    report["trials"].sort(key=lambda s: (s["profile"], s["repetition"], s["condition"]))
    report["comparisons"] = comparisons(report["trials"], args.output)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    flat = [{k: v for k, v in s.items() if k != "phases"} for s in report["trials"]]
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({k for s in flat for k in s}))
        writer.writeheader()
        writer.writerows(flat)
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
