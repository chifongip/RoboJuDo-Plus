"""Matched summaries, explicit unfavorable cases and plots for the fixed X2 study."""

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from evaluate_x2_transitions import recovery_measurement
from plot_x2_transitions import load_numeric
from x2_integral_sim import POSTURE, SIDES
from x2_robustness_analysis import analyze_case


def legacy_rows(path):
    with path.open() as stream:
        return [
            {k: float(v) for k, v in row.items() if k == "elapsed_s" or "_torso_" in k}
            for row in csv.DictReader(stream)
        ]


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def flattened_metrics(trial):
    metrics = {}
    for phase, stats in trial.get("phases", {}).items():
        for key, value in stats["aggregate_error_rad"].items():
            metrics[f"{phase}.error.{key}_rad"] = value
        for key, value in stats["body"].items():
            metrics[f"{phase}.body.{key}"] = value
        for joint, values in stats["joints"].items():
            for key, value in values.items():
                metrics[f"{phase}.{joint}.{key}"] = value
        for joint, values in stats["joint_error_rad"].items():
            for key, value in values.items():
                metrics[f"{phase}.{joint}.error.{key}_rad"] = value
    for side, values in trial.get("recovery", {}).items():
        for key in ("peak_m", "early_peak_m", "terminal_residual_m", "late_rms_distance_m", "settling_time_s"):
            metrics[f"recovery.{side}.{key}"] = values[key]
    for phase, sides in trial.get("hands", {}).items():
        for side, values in sides.items():
            for key, value in values.items():
                metrics[f"{phase}.{side}.{key}"] = value
    return metrics


def analyze_study(root):
    # Acquisition snapshots remain immutable; preserve the exact analysis revision
    # separately when reports are refined after a long-running acquisition starts.
    revision = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    analysis_root = root / "analysis_sources" / revision
    analysis_hashes = {}
    for source in (
        Path(__file__),
        Path(__file__).with_name("x2_robustness_analysis.py"),
        Path(__file__).with_name("plot_x2_transitions.py"),
        Path("tests/test_x2_robustness.py"),
    ):
        target = analysis_root / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        with source.open("rb") as stream:
            analysis_hashes[str(source)] = hashlib.file_digest(stream, "sha256").hexdigest()
    (root / "analysis_source_sha256.json").write_text(
        json.dumps({"snapshot": str(analysis_root.relative_to(root)), "sha256": analysis_hashes}, indent=2) + "\n"
    )
    manifest = json.loads((root / "manifest.json").read_text())
    results, summaries, legacy = [], [], []
    case_names = [c["case"] for c in manifest["cases"]]
    for case in case_names:
        directory = root / case
        print(f"Analyze {case}", flush=True)
        trial_summaries, measured = analyze_case(directory)
        results.extend(measured)
        summaries.extend(trial_summaries)
        index = {(s["condition"], s["profile"], s["seed"]): s for s in trial_summaries}
        trace_cache = {}
        for summary in trial_summaries:
            if summary["profile"] == "unloaded":
                continue
            method, profile, seed = summary["condition"], summary["profile"], summary["seed"]
            keys = [
                (method, profile, seed),
                (method, "unloaded", seed),
                ("pd_only", profile, seed),
                ("pd_only", "unloaded", seed),
            ]
            record = {"case": case, "condition": method, "profile": profile, "seed": seed}
            if all(index.get(k, {}).get("status") == "completed" for k in keys):
                paths = [directory / index[k]["trace"] for k in keys]
                for path in paths:
                    if path not in trace_cache:
                        trace_cache[path] = legacy_rows(path)
                a, b, pd, pu = [trace_cache[path] for path in paths]
                record["hands"] = {s: recovery_measurement(a, b, pd, pu, s) for s in SIDES}
            else:
                record["unavailable"] = "Matched trial failed or missing"
            legacy.append(record)
    (root / "legacy_pd_envelope.json").write_text(json.dumps(legacy, indent=2, allow_nan=False) + "\n")
    idx = {(r["case"], r["condition"], r["profile"], r["seed"]): r for r in results if "seed" in r}
    exogenous_checks = []
    for case in case_names:
        for seed in (1, 2, 3):
            all_group = [s for s in summaries if s["case"] == case and s["seed"] == seed]
            completed = [s for s in all_group if s["status"] == "completed"]
            matched = len({s["exogenous_sha256"] for s in completed}) <= 1
            observed = [s for s in all_group if s["physics_samples"]]
            failed = [s for s in observed if s["status"] != "completed"]
            if failed:
                reference_summary = max(observed, key=lambda s: s["physics_samples"])
                reference = load_numeric(root / case / reference_summary["physics_trace"])
                fields = [k for k in reference if "_injected_" in k] + ["acquisition_latency_s", "acquisition_dropped"]
                for trial in failed:
                    data = load_numeric(root / case / trial["physics_trace"])
                    n = len(data["elapsed_s"])
                    matched = matched and all(np.array_equal(data[k], reference[k][:n]) for k in fields)
            exogenous_checks.append(
                {
                    "case": case,
                    "seed": seed,
                    "completed_trials": len(completed),
                    "observed_measurement_trials": len(observed),
                    "failed_prefixes_checked": len(failed),
                    "matched": matched,
                }
            )
    (root / "perturbation_matching.json").write_text(json.dumps(exogenous_checks, indent=2) + "\n")
    if not all(c["matched"] for c in exogenous_checks):
        raise RuntimeError("Exogenous perturbations differ between matched trials")
    pd_isolation = []
    baseline_hashes = {}
    for config in manifest["cases"]:
        if config["shared_pd"] or any(config[k] != 1 for k in ("inertia_scale", "armature_scale", "friction_scale")):
            continue
        for trial in (s for s in summaries if s["case"] == config["case"] and s["condition"] == "pd_only"):
            path = root / trial["case"] / trial["trace"]
            reference = root / "N0" / trial["trace"]
            if reference not in baseline_hashes:
                with reference.open("rb") as stream:
                    baseline_hashes[reference] = hashlib.file_digest(stream, "sha256").hexdigest()
            with path.open("rb") as stream:
                same = hashlib.file_digest(stream, "sha256").hexdigest() == baseline_hashes[reference]
            pd_isolation.append(
                {"case": trial["case"], "profile": trial["profile"], "seed": trial["seed"], "exact": same}
            )
    (root / "pd_feedback_isolation.json").write_text(json.dumps(pd_isolation, indent=2) + "\n")
    if not all(c["exact"] for c in pd_isolation):
        raise RuntimeError("Correction-only feedback perturbations changed a PD-only trajectory")
    pairs = []
    for trial in results:
        if "seed" not in trial or trial["status"] != "completed":
            continue
        current = flattened_metrics(trial)
        for reference_name in ("N0", "pd_only", "integral"):
            key = (
                ("N0", trial["condition"], trial["profile"], trial["seed"])
                if reference_name == "N0"
                else (trial["case"], reference_name, trial["profile"], trial["seed"])
            )
            reference = idx.get(key)
            if reference is None or reference is trial or reference["status"] != "completed":
                continue
            baseline = flattened_metrics(reference)
            for metric in sorted(current.keys() & baseline.keys()):
                a, b = current[metric], baseline[metric]
                if a is not None and b is not None:
                    pairs.append(
                        {
                            "case": trial["case"],
                            "condition": trial["condition"],
                            "profile": trial["profile"],
                            "seed": trial["seed"],
                            "reference": reference_name,
                            "metric": metric,
                            "value": a,
                            "reference_value": b,
                            "delta": a - b,
                        }
                    )
    write_csv(root / "paired_changes.csv", pairs)
    summary_index = {(s["case"], s["condition"], s["profile"], s["seed"]): s for s in summaries}
    overview = []
    for trial in results:
        if "seed" not in trial:
            continue
        s = summary_index[trial["case"], trial["condition"], trial["profile"], trial["seed"]]
        recovery_values = list(trial.get("recovery", {}).values())
        overview.append(
            {
                "case": trial["case"],
                "condition": trial["condition"],
                "profile": trial["profile"],
                "seed": trial["seed"],
                "warmup_s": s["warmup_s"],
                "status": s["status"],
                "termination_reason": s["termination_reason"],
                "termination_elapsed_s": s["termination_elapsed_s"],
                "walking_rms_deg": float(np.rad2deg(trial["windows"]["walk"]["rms"]))
                if "walk" in trial.get("windows", {})
                else None,
                "unload_peak_mm": max((1000 * r["early_peak_m"] for r in recovery_values), default=None),
                "unload_full_peak_mm": max((1000 * r["peak_m"] for r in recovery_values), default=None),
                "unloaded_envelope_mm": max((1000 * r["unloaded_envelope_m"] for r in recovery_values), default=None),
                "terminal_residual_mm": max((1000 * r["terminal_residual_m"] for r in recovery_values), default=None),
                "settling_time_s": max((r["settling_time_s"] for r in recovery_values), default=None)
                if recovery_values and all(r["settling_time_s"] is not None for r in recovery_values)
                else None,
                "tilt_peak_deg": float(np.rad2deg(s["max_base_tilt_rad"])),
                "minimum_height_m": s["min_base_height_m"],
                "walking_speed_m_s": s.get("mean_walking_speed_m_s"),
                "saturated_percent": 100 * s["saturated_physics_fraction"],
                "max_arm_saturation_percent": 100 * max(s["joint_saturated_physics_fraction"][j] for j in POSTURE),
                "feedback_age_max_ms": 1000 * trial.get("feedback", {}).get("age_max_s", 0),
                "wall_seconds": s["wall_seconds"],
                "live_rss_mib": s["worker_live_rss_mib"],
            }
        )
    write_csv(root / "overview.csv", overview)
    failures = [r for r in overview if r["status"] != "completed"]
    (root / "failures.json").write_text(json.dumps(failures, indent=2, allow_nan=False) + "\n")
    write_report(root, manifest, overview, pairs, failures)
    plotting_config = root / "plotting_config.json"
    plotting = json.loads(plotting_config.read_text()) if plotting_config.exists() else {"python": sys.executable}
    subprocess.run(
        [plotting["python"], str(Path(__file__).with_name("plot_x2_transitions.py")), str(root), "--physics"],
        check=True,
        env={**os.environ, "MPLCONFIGDIR": str((root / "plot_cache").resolve())},
    )
    hashes = {
        str(p.relative_to(root)): __import__("hashlib").file_digest(p.open("rb"), "sha256").hexdigest()
        for p in root.glob("*/*/*_physics.npz")
    }
    (root / "physics_trace_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    acquisition_sources = json.loads((root / "source_sha256.json").read_text())
    changed = [
        name
        for name, sha in acquisition_sources.items()
        if Path(name).exists() and __import__("hashlib").file_digest(Path(name).open("rb"), "sha256").hexdigest() != sha
    ]
    (root / "completion_audit.json").write_text(
        json.dumps(
            {
                "planned_trials": manifest["planned_trials"],
                "recorded_trials": len(overview),
                "completed_trials": len(overview) - len(failures),
                "failed_trials": len(failures),
                "controller_retuned": False,
                "source_changes_since_snapshot": changed,
                "trace_count": len(hashes),
                "hardware_commands": False,
                "nominal_verification_passed": json.loads((root / "nominal_verification.json").read_text())["passed"],
                "matched_perturbations_verified": all(c["matched"] for c in exogenous_checks),
                "pd_feedback_isolation_verified": all(c["exact"] for c in pd_isolation),
                "max_correction_bound_excess_nm": max(s["max_effort_bound_excess_nm"] for s in summaries),
                "max_correction_slew_nm_s": max(s["max_effort_slew_nm_s"] for s in summaries),
                "acquisition_controller_sources_intact": not any(
                    name in changed
                    for name in (
                        "scripts/evaluate_x2_transitions.py",
                        "scripts/x2_integral_sim.py",
                        "scripts/x2_robustness_sim.py",
                        "scripts/x2_feedback_perturbations.py",
                        "scripts/x2_observer_reference.py",
                        "robojudo/utils/disturbance_observer.py",
                        "robojudo/utils/bounded_integral.py",
                    )
                ),
            },
            indent=2,
        )
        + "\n"
    )


def write_report(root, manifest, overview, pairs, failures):
    def span(rows, metric, digits=3):
        values = [r[metric] for r in rows if r[metric] is not None]
        return f"{min(values):.{digits}f}–{max(values):.{digits}f}" if values else "unavailable"

    lines = [
        "# X2 observer robustness study",
        "",
        f"{len(overview)} primary trials; {len(failures)} failed. Coefficients remain fixed at 8 rad/s.",
        "",
        "## Interpretation",
        "",
        "Simulation only. Feedback magnitudes and physical parameter multipliers are exploratory assumptions, "
        "not measured hardware distributions. Results describe discrete tested points; they do not establish "
        "a continuous operating region or nonlinear stability proof. Physical torque ratings remain unverified.",
        "",
        "Correction-only cases preserve clean fast PD and locomotion sensing; P1/P2 also perturb arm PD. "
        "Feedback delay is not actuator delay. The observer continues ordinary elapsed-time updates on held packets.",
        "",
        "## Measured ranges across seeds and warmups",
        "",
        "Walking RMS uses 55–60 s; phase tables use their full scheduled intervals. "
        "Unloading peak uses 79–82 s at 5 ms cadence, including the release state. "
        "Tracking windows require full observation; recovery and paired comparisons require complete trials.",
        "",
        (
            "| Case | Profile | Controller | Completed | Walk RMS ° | Unload peak mm | "
            "Peak tilt ° | Saturated % | Speed m/s |"
        ),
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in [c["case"] for c in manifest["cases"]]:
        for profile in ("constant", "sway"):
            for method in ("pd_only", "integral", "observer"):
                rows = [r for r in overview if (r["case"], r["profile"], r["condition"]) == (case, profile, method)]
                lines.append(
                    f"| {case} | {profile} | {method} | {sum(r['status'] == 'completed' for r in rows)}/3 | "
                    f"{span(rows, 'walking_rms_deg')} | {span(rows, 'unload_peak_mm')} | "
                    f"{span(rows, 'tilt_peak_deg')} | {span(rows, 'saturated_percent')} | "
                    f"{span(rows, 'walking_speed_m_s')} |"
                )
    lines += [
        "",
        "## Unfavorable cases and failures",
        "",
        "Every paired change is retained in `paired_changes.csv`; positive changes in error, oscillation, "
        "displacement and saturation are unfavorable. "
        "Lower unloading displacement alone is not an overall improvement.",
    ]
    for metric in (
        "walk.error.rms_rad",
        "carry_standing.body.pitch_detrended_std_rad",
        "carry_standing.body.peak_tilt_rad",
        "walk.body.peak_tilt_rad",
    ):
        rows = sorted(
            [
                r
                for r in pairs
                if r["condition"] == "observer" and r["reference"] == "integral" and r["metric"] == metric
            ],
            key=lambda r: r["delta"],
            reverse=True,
        )[:3]
        for r in rows:
            lines.append(
                f"- {r['case']} / {r['profile']} / seed {r['seed']}: {metric}, observer "
                f"{np.rad2deg(r['value']):.4f}° versus integral {np.rad2deg(r['reference_value']):.4f}°."
            )
    if failures:
        lines += [
            f"- Failure: {r['case']} / {r['condition']} / {r['profile']} / seed {r['seed']}: {r['termination_reason']}"
            for r in failures
        ]
    else:
        lines.append(
            "- No trial reached the existing fall, height, numerical-warning or policy-mode termination criteria."
        )
    lines += [
        "",
        "## Recovery and accepted tradeoffs",
        "",
        "New recovery uses 5 ms physics telemetry. The threshold is each matched unloaded trial’s p99 hand "
        "excursion around its median XYZ position during 20–25 s, without detrending. This is an unloaded "
        "motion envelope, not an isolated sensor-noise estimate. Drift and degeneracy are reported. "
        "A first-sample entry is resolution-limited envelope membership, not measured physical settling.",
        "",
        "Full peak time, signed extrema, below-reference overshoot, rebound and terminal/late residuals are in "
        "each case’s `physics_summary.json`. Settling requires remaining inside through 89 s with at least "
        "1 s observed; absent settlement is censored. The original 20 ms PD-envelope results remain in "
        "`legacy_pd_envelope.json`, and cannot be interpreted as 20 ms physical settling.",
        "",
        "Previously accepted nominal carrying error, correction ripple, sway transients and local balance/"
        "saturation tradeoffs remain applicable. Aggregate RMS does not bound individual errors or transients "
        "below one degree. This study introduces no blanket angular pass/fail cutoff.",
        "",
        "## Evidence and limitations",
        "",
        "- `manifest.json`, `source_sha256.json`, source snapshots and trace hashes preserve configurations, "
        "seeds, model/XML assets and software provenance. Nominal and serial/parallel checks are separate artifacts.",
        "- `overview.csv`, per-case 5 ms joint tables, phase summaries and raw NPZ telemetry retain all measured "
        "channels; legacy CSV traces preserve historical comparisons. Applied torque is logged for the interval "
        "ending at each state timestamp.",
        "- Three paired seed/warmup settings provide a finite sensitivity study, not independent estimates of "
        "seed and warmup effects or rare-event probabilities. Feedback imperfections affect arms only; IMU, "
        "locomotion sensing, actuator delay and hardware safety logic are outside this study.",
        "- Inertia tensors, armature and Coulomb friction loss vary separately; masses, centers of mass and "
        "observer coefficients remain fixed. Identification effective inertias are not verified physical values.",
        "- Mechanism diagnosis uses measured innovations, stale age, clipping, bound/slew occupancy and body "
        "motion. Temporal association alone does not establish a unique causal explanation.",
        "",
        "References: [Chen et al.](https://doi.org/10.1109/TIE.2015.2478397); "
        "[Hespanha et al.](https://web.ece.ucsb.edu/~hespanha/published/ncs_v15p.pdf).",
    ]
    (root / "experiment_report.md").write_text("\n".join(lines) + "\n")
