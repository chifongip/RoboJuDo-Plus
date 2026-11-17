"""Physics-cadence analysis of matched X2 simulation trials, including censored recovery."""

import csv
import json

import numpy as np
from evaluate_x2_transitions import PHASES, WINDOWS
from x2_integral_sim import POSTURE, SIDES, detrended_std
from x2_robustness_sim import load_physics


def mask(data, start, end):
    t = np.round(data["elapsed_s"], 8)
    return (t >= start) & (t < end)


def physics_timestep(data, summary):
    """Use recorded cadence, inferring it only for historical traces without metadata."""
    value = summary.get("physics_dt_s")
    if value is None:
        gaps = np.diff(data.get("elapsed_s", []))
        positive = gaps[gaps > 0]
        value = float(np.min(positive)) if len(positive) else None
    if value is None or not np.isfinite(value) or value <= 0:
        raise ValueError("A positive physics timestep is required to assess coverage")
    return float(value)


def complete_window(data, start, end, dt):
    times = np.round(data["elapsed_s"], 8)
    actual = times[mask(data, start, end)]
    # Physics traces contain endpoint samples: the first interval ends at dt.
    first = max(dt, start)
    expected = np.arange(first, end - dt * 1e-6, dt)
    return len(actual) == len(expected) and np.allclose(actual, expected, rtol=0, atol=1e-8)


def error_metrics(error):
    return {
        "rms": float(np.sqrt(np.mean(error**2))),
        "p95_abs": float(np.percentile(np.abs(error), 95)),
        "max_abs": float(np.max(np.abs(error))),
        "mean_abs": float(np.mean(np.abs(error))),
    }


def physics_statistics(data):
    if not len(data.get("elapsed_s", [])):
        return {}
    result = {}
    for phase_id, (phase, _, _) in enumerate(PHASES):
        selected = data["phase_id"] == phase_id
        if not np.any(selected):
            continue
        errors = np.stack(
            [data[f"{j}_position_rad"][selected] - data[f"{j}_target_rad"][selected] for j in POSTURE], axis=1
        )
        stats = {
            "samples": int(selected.sum()),
            "aggregate_error_rad": error_metrics(errors),
            "joint_error_rad": {j: error_metrics(errors[:, i]) for i, j in enumerate(POSTURE)},
        }
        stats["body"] = {
            **{
                f"{a}_{metric}_rad": float(fn(data[f"base_{a}_rad"][selected]))
                for a in ("roll", "pitch")
                for metric, fn in (("min", np.min), ("max", np.max), ("detrended_std", detrended_std))
            },
            "peak_tilt_rad": float(data["base_tilt_rad"][selected].max()),
            "min_height_m": float(data["base_height_m"][selected].min()),
            "max_height_m": float(data["base_height_m"][selected].max()),
            "mean_vx_m_s": float(data["actual_vx_m_s"][selected].mean()),
        }
        stats["joints"] = {
            j: {
                "saturation_fraction": float(data[f"{j}_saturation_fraction"][selected].mean()),
                "antiwindup_fraction": float(data[f"{j}_antiwindup"][selected].mean()),
                "bound_fraction": float(data[f"{j}_at_bound"][selected].mean()),
                "slew_limit_fraction": float(data[f"{j}_at_slew_limit"][selected].mean()),
                **{
                    f"{s}_detrended_std_nm": detrended_std(data[f"{j}_{s}_nm"][selected])
                    for s in ("correction", "applied")
                },
                "applied_max_abs_nm": float(np.max(np.abs(data[f"{j}_applied_nm"][selected]))),
            }
            for j in POSTURE
        }
        if f"{next(iter(POSTURE))}_velocity_innovation_rad_s" in data:
            stats["observer_diagnostics"] = {
                joint: {
                    "innovation_rms_rad_s": float(
                        np.sqrt(np.mean(data[f"{joint}_velocity_innovation_rad_s"][selected] ** 2))
                    ),
                    "innovation_max_abs_rad_s": float(
                        np.max(np.abs(data[f"{joint}_velocity_innovation_rad_s"][selected]))
                    ),
                    "estimate_detrended_std_nm": detrended_std(data[f"{joint}_disturbance_estimate_nm"][selected]),
                    "estimate_max_abs_nm": float(np.max(np.abs(data[f"{joint}_disturbance_estimate_nm"][selected]))),
                }
                for joint in POSTURE
            }
        result[phase] = stats
    return result


def hand_vector(data, side):
    return np.stack([data[f"{side}_torso_{a}_m"] for a in "xyz"], axis=1)


def recovery(data, unloaded, side):
    """No fitted residual subtraction; threshold is measured before any payload load."""
    if len(data["elapsed_s"]) != len(unloaded["elapsed_s"]) or not np.allclose(
        data["elapsed_s"], unloaded["elapsed_s"], rtol=0, atol=1e-8
    ):
        raise ValueError("Matched physics timestamps are missing")
    baseline = hand_vector(unloaded, side)[mask(unloaded, 20, 25)]
    if len(baseline) < 2:
        raise ValueError("Unloaded envelope interval is missing")
    center = np.median(baseline, axis=0)
    envelope = float(np.percentile(np.linalg.norm(baseline - center, axis=1), 99))
    times = np.round(data["elapsed_s"], 8)
    sel = (times >= 79) & (times <= 89)
    t = times[sel]
    if len(t) < 2 or t[-1] < 89:
        raise ValueError("Full recovery observation is missing")
    delta = (hand_vector(data, side) - hand_vector(unloaded, side))[sel]
    distance = np.linalg.norm(delta, axis=1)
    inside = distance <= envelope
    remaining = np.logical_and.accumulate(inside[::-1])[::-1]
    candidates = np.flatnonzero(remaining & (t <= t[-1] - 1 + 1e-9))
    peak = int(np.argmax(distance))
    early = t < 82 - 1e-8
    early_index = np.flatnonzero(early)[np.argmax(distance[early])]
    terminal = t >= 86 - 1e-8
    z = delta[:, 2]
    extremum = int(np.argmax(np.abs(z)))
    degenerate = envelope <= 1e-12
    return {
        "unloaded_envelope_m": envelope,
        "envelope_interval_s": [20, 25],
        "envelope_degenerate": degenerate,
        "unloaded_center_m": center.tolist(),
        "unloaded_drift_m": (np.mean(baseline[-100:], axis=0) - np.mean(baseline[:100], axis=0)).tolist(),
        "settling_time_s": float(max(0, t[candidates[0]] - 79)) if len(candidates) and not degenerate else None,
        "status": (
            "degenerate_envelope"
            if degenerate
            else "first_sample_inside"
            if len(candidates) and candidates[0] == 0
            else "settled"
            if len(candidates)
            else "not_settled_within_observation"
        ),
        "first_sample_offset_s": float(t[0] - 79),
        "observation_end_s": float(t[-1]),
        "peak_m": float(distance[peak]),
        "peak_time_after_unload_s": float(t[peak] - 79),
        "early_peak_m": float(distance[early_index]),
        "early_peak_time_after_unload_s": float(t[early_index] - 79),
        "signed_min_xyz_m": delta.min(axis=0).tolist(),
        "signed_max_xyz_m": delta.max(axis=0).tolist(),
        "vertical_below_reference_peak_m": float(max(0, -z.min())),
        "vertical_rebound_after_extremum_m": float(np.max(np.abs(z[extremum:] - z[extremum]))),
        "vertical_rebound_signed_m": float(z[extremum + np.argmax(np.abs(z[extremum:] - z[extremum]))] - z[extremum]),
        "terminal_xyz_m": delta[-1].tolist(),
        "terminal_residual_m": float(distance[-1]),
        "late_mean_xyz_m": delta[terminal].mean(axis=0).tolist(),
        "late_mean_distance_m": float(distance[terminal].mean()),
        "late_rms_distance_m": float(np.sqrt(np.mean(distance[terminal] ** 2))),
    }


def analyze_trial(root, summary):
    path = root / summary["physics_trace"]
    if not path.exists():
        return {
            "case": summary["case"],
            "condition": summary["condition"],
            "profile": summary["profile"],
            "repetition": summary["repetition"],
            "seed": summary["seed"],
            "status": summary["status"],
            "measurements_unavailable": "Trial terminated before measurement telemetry started",
            "phases": {},
        }
    data = load_physics(path)
    finite = np.logical_and.reduce([np.isfinite(v) for v in data.values()]) if data else np.array([], dtype=bool)
    invalid_samples = int((~finite).sum())
    if invalid_samples:
        data = {k: v[finite] for k, v in data.items()}
    result = {
        "case": summary["case"],
        "condition": summary["condition"],
        "profile": summary["profile"],
        "repetition": summary["repetition"],
        "seed": summary["seed"],
        "status": summary["status"],
        "phases": physics_statistics(data),
        "invalid_physics_samples": invalid_samples,
    }
    if len(data.get("elapsed_s", [])):
        dt = physics_timestep(data, summary)
        result["physics_dt_s"] = dt
        for phase, start, end in PHASES:
            if phase in result["phases"]:
                result["phases"][phase]["complete"] = complete_window(data, start, end, dt)
        result["feedback"] = {
            "age_max_s": float(data["feedback_age_s"].max()),
            "age_p95_s": float(np.percentile(data["feedback_age_s"], 95)),
            "noise": {
                j: {
                    s: float(np.std(data[f"{j}_injected_{s}_noise_{unit}"]))
                    for s, unit in (("position", "rad"), ("velocity", "rad_s"))
                }
                for j in POSTURE
            },
        }
    if len(data.get("elapsed_s", [])):
        tail = data["elapsed_s"] >= data["elapsed_s"][-1] - 1
        result["last_second"] = {
            "last_elapsed_s": float(data["elapsed_s"][-1]),
            "max_feedback_age_s": float(data["feedback_age_s"][tail].max()),
            "max_tilt_rad": float(data["base_tilt_rad"][tail].max()),
            "min_height_m": float(data["base_height_m"][tail].min()),
            "joints": {
                j: {
                    "saturation_fraction": float(data[f"{j}_saturation_fraction"][tail].mean()),
                    "peak_error_rad": float(
                        np.max(np.abs(data[f"{j}_position_rad"][tail] - data[f"{j}_target_rad"][tail]))
                    ),
                }
                for j in POSTURE
            },
        }
    if len(data.get("elapsed_s", [])):
        result["windows"] = {}
        for name, (start, end) in WINDOWS.items():
            sel = mask(data, start, end)
            # A later failure must not erase a fully observed earlier window.
            if not complete_window(data, start, end, dt):
                continue
            error = np.stack([data[f"{j}_position_rad"][sel] - data[f"{j}_target_rad"][sel] for j in POSTURE])
            result["windows"][name] = error_metrics(error)
    if summary["profile"] != "unloaded" and summary["status"] == "completed":
        reference_path = path.parent.parent / "unloaded" / path.name
        reference_summary = reference_path.with_name(reference_path.name.replace("_physics.npz", ".json"))
        if reference_path.exists() and json.loads(reference_summary.read_text())["status"] == "completed":
            unloaded = load_physics(reference_path)
            result["recovery"] = {s: recovery(data, unloaded, s) for s in SIDES}
            result["hands"] = {}
            for phase, start, end in PHASES:
                sel = mask(data, start, end)
                if not np.any(sel):
                    continue
                result["hands"][phase] = {}
                for side in SIDES:
                    delta = (hand_vector(data, side) - hand_vector(unloaded, side))[sel]
                    result["hands"][phase][side] = {
                        "rms_displacement_m": float(np.sqrt(np.mean(np.sum(delta**2, axis=1)))),
                        "p95_displacement_m": float(np.percentile(np.linalg.norm(delta, axis=1), 95)),
                        "peak_displacement_m": float(np.max(np.linalg.norm(delta, axis=1))),
                        "mean_signed_z_m": float(np.mean(delta[:, 2])),
                    }
        else:
            result["recovery_unavailable"] = "Matched unloaded trial failed or missing"
    return result


def analyze_case(root):
    summaries = [
        json.loads(p.read_text())
        for p in sorted(root.glob("*/*.json"))
        if p.parent.name in ("unloaded", "constant", "sway", "compliant_payload")
    ]
    results = []
    with (root / "joint_errors_5ms.csv").open("w", newline="") as stream:
        writer = None
        for summary in summaries:
            result = analyze_trial(root, summary)
            results.append(result)
            for phase, stats in result.get("phases", {}).items():
                for joint, errors in stats["joint_error_rad"].items():
                    row = {
                        "case": summary["case"],
                        "condition": summary["condition"],
                        "profile": summary["profile"],
                        "seed": summary["seed"],
                        "warmup_s": summary["warmup_s"],
                        "phase": phase,
                        "trial_status": summary["status"],
                        "phase_samples": stats["samples"],
                        "phase_complete": stats["complete"],
                        "physics_dt_s": result["physics_dt_s"],
                        "joint": joint,
                        **{f"{key}_deg": float(np.rad2deg(value)) for key, value in errors.items()},
                        **stats["joints"][joint],
                    }
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
    (root / "physics_summary.json").write_text(json.dumps({"trials": results}, indent=2, allow_nan=False) + "\n")
    return summaries, results
