"""Analyze saved ablation traces without importing robot or simulator modules."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_columns(path, names):
    with path.open() as stream:
        header = next(csv.reader(stream))
    values = np.loadtxt(path, delimiter=",", skiprows=1, usecols=[header.index(n) for n in names], ndmin=2)
    return {name: values[:, i] for i, name in enumerate(names)}


def command_check(effort, factors, bounds, *, headroom=None):
    previous = np.vstack([np.zeros_like(effort[:1]), effort[:-1]])
    releasing = (effort * previous >= 0) & (abs(effort) < abs(previous))
    rate = np.where(releasing, factors["release"], 2.0)
    violation = abs(effort - previous) / 0.02 - rate
    # The runtime may remove correction immediately for PD headroom safety.
    safety_removal = np.zeros(effort.shape, dtype=bool)
    if headroom is not None:
        safety_removal = headroom & releasing
    excess = float(np.max(np.where(safety_removal, 0, violation)))
    reversals = (effort * previous < 0) & (abs(previous) > 1e-10) & (abs(effort) > 1e-10)
    return {
        "finite": bool(np.isfinite(effort).all()),
        "correction_bound_excess_nm": float(np.max(abs(effort) - bounds)),
        "slew_excess_nm_s": excess,
        "safety_removals": int(np.count_nonzero(safety_removal & (violation > 1e-8))),
        "direct_reversals": int(np.count_nonzero(reversals)),
        "reversal_policy_passed": not factors["zero_before_reversal"] or not reversals.any(),
    }


def transition_rows(result, joints, hashes, checks):
    root = Path(result["directory"])
    trials = result["trials"]
    columns = [
        "elapsed_s",
        "arm_rms_error_rad",
        *[f"{joint}_correction_nm" for joint in joints],
        *[f"{joint}_applied_nm" for joint in joints],
        *[f"{side}_torso_{axis}_m" for side in ("left", "right") for axis in "xyz"],
    ]
    data = {}
    for trial in trials:
        if trial["status"] != "completed":
            continue
        path = root / trial["trace"]
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        values = read_columns(path, columns)
        data[trial["profile"]] = values
        effort = np.column_stack([values[f"{j}_correction_nm"] for j in joints])
        applied = np.column_stack([values[f"{j}_applied_nm"] for j in joints])
        bounds = [trial["torque_limits"]["joints"][j]["integral_bound_nm"] for j in joints]
        torque = [trial["torque_limits"]["joints"][j]["configured_total_limit_nm"] for j in joints]
        check = command_check(effort, result["factors"], bounds)
        check.update(
            study="transition",
            variant=result["variant"],
            profile=trial["profile"],
            seed=result["seed"],
            samples_complete=len(effort) == 5250,
            duration_complete=abs(trial["termination_elapsed_s"] - 105) < 1e-6,
            applied_torque_bound_excess_nm=float(np.max(abs(applied) - torque)),
            finite=check["finite"] and all(np.isfinite(v).all() for v in values.values()),
        )
        checks.append(check)
    rows = []
    for trial in trials:
        profile = trial["profile"]
        if profile not in data or "unloaded" not in data:
            continue
        values, reference = data[profile], data["unloaded"]
        t = values["elapsed_s"]
        walk, after = (t >= 55) & (t < 60), (t >= 79) & (t < 82)
        peak = max(
            np.max(
                np.linalg.norm(
                    np.column_stack(
                        [values[f"{side}_torso_{axis}_m"] - reference[f"{side}_torso_{axis}_m"] for axis in "xyz"]
                    )[after],
                    axis=1,
                )
            )
            for side in ("left", "right")
        )
        rows.append(
            {
                "study": "transition",
                "variant": result["variant"],
                **result["factors"],
                "profile": profile,
                "seed": result["seed"],
                "warmup_s": result["warmup_s"],
                "carry_rms_deg": float(np.rad2deg(np.sqrt(np.mean(values["arm_rms_error_rad"][walk] ** 2)))),
                "carry_ripple_nm": float(max(np.std(values[f"{j}_correction_nm"][walk]) for j in joints)),
                "release_hand_peak_mm": None if profile == "unloaded" else float(1000 * peak),
                "saturation_fraction": trial["saturated_physics_fraction"],
                "max_base_tilt_deg": float(np.rad2deg(trial["max_base_tilt_rad"])),
            }
        )
    return rows


def span(rows, key):
    values = [r.get(key) for r in rows]
    if any(v is None for v in values):
        return "not established"
    low, high = min(values), max(values)
    return f"{low:.3f}" if f"{low:.3f}" == f"{high:.3f}" else f"{low:.3f}–{high:.3f}"


def paired_effects(rows):
    """Change one factor at a time, preserving plant, seed and other factors."""
    contexts = ("study", "profile", "case", "walking", "load", "seed", "warmup_s")
    factors = {"estimator": ("linearized", "nonlinear"), "release": (2.0, 16.0), "unwinding": ("frozen", "directional")}
    effects = []
    factorial = [r for r in rows if r["variant"] != "original"]
    for factor, (before, after) in factors.items():
        keys = (*contexts, *(f for f in factors if f != factor))
        index = {tuple(r.get(key) for key in keys): r for r in factorial if r[factor] == after}
        for left in factorial:
            if left[factor] != before:
                continue
            right = index.get(tuple(left.get(key) for key in keys))
            if right is None:
                continue
            effect = {key: left[key] for key in keys if key in left}
            effect.update(
                factor=factor,
                before=before,
                after=after,
                before_variant=left["variant"],
                after_variant=right["variant"],
            )
            for metric in (
                "carry_rms_deg",
                "carry_ripple_nm",
                "release_hand_peak_mm",
                "torque_removal_s",
                "release_peak_deg",
            ):
                if left.get(metric) is not None and right.get(metric) is not None:
                    effect[f"delta_{metric}"] = right[metric] - left[metric]
            effects.append(effect)
    return effects


def plot_results(directory, rows, variants):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    settings = (
        ("abrupt", "release_hand_peak_mm", "Abrupt unloading: hand peak (mm)"),
        ("abrupt", "torque_removal_s", "Abrupt unloading: correction removal (s)"),
        ("transition", "carry_rms_deg", "Gradual placement, compliant load: walking RMS (deg)"),
        ("transition", "carry_ripple_nm", "Gradual placement, compliant load: correction SD (Nm)"),
    )
    labels = [
        "Original"
        if name == "original"
        else f"{'L' if v['estimator'] == 'linearized' else 'N'} / "
        f"{'F' if v['unwinding'] == 'frozen' else 'D'} / {v['release']:g}"
        for name, v in variants.items()
    ]
    for axis, (study, metric, title) in zip(axes.flat, settings, strict=True):
        for index, (variant, factors) in enumerate(variants.items()):
            selected = [
                r
                for r in rows
                if r["variant"] == variant
                and r["study"] == study
                and (
                    (r.get("walking") and r.get("case") == "nominal")
                    if study == "abrupt"
                    else r.get("profile") == "compliant_payload"
                )
            ]
            values = [r[metric] for r in selected if r.get(metric) is not None]
            color = (
                "gray"
                if variant == "original"
                else ("tab:blue" if factors["estimator"] == "linearized" else "tab:purple")
            )
            if not values:
                if selected and metric == "torque_removal_s":
                    axis.bar(index, 5, facecolor="none", edgecolor=color, linestyle="--")
                    axis.text(index, 5.05, "Not\nestablished", ha="center", fontsize=8)
                continue
            middle = float(np.median(values))
            axis.bar(index, middle, color=color, alpha=0.75, hatch="//" if factors["unwinding"] == "frozen" else None)
            axis.errorbar(
                index,
                middle,
                yerr=[[middle - min(values)], [max(values) - middle]],
                fmt="none",
                color="black",
                capsize=3,
            )
        axis.set_title(title)
        axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        axis.set_ylim(bottom=0)
        if metric == "torque_removal_s":
            axis.set_ylim(0, 6)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        "X2 observer ablation — fixed growth, bandwidth and bounds\n"
        "L/N: linearized/nonlinear; F/D: frozen/directional anti-windup; number: release Nm/s\n"
        "Abrupt: nominal walking; gradual: median and range across 4/5/6 s warmups",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure.savefig(directory / "comparison.png", dpi=170)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    analysis_source = Path(__file__)
    (args.directory / "sources" / analysis_source.name).write_bytes(analysis_source.read_bytes())
    (args.directory / "analysis_metadata.json").write_text(
        json.dumps(
            {
                "source_sha256": hashlib.sha256(analysis_source.read_bytes()).hexdigest(),
                "numpy_version": np.__version__,
            },
            indent=2,
        )
        + "\n"
    )
    metadata = json.loads((args.directory / "metadata.json").read_text())
    results = json.loads((args.directory / "results.json").read_text())
    rows, checks, hashes = [], [], {}
    transition = next(r for r in results if r["study"] == "transition" and r["status"] == "completed")
    joints = metadata["joint_names"]
    bounds = np.array([transition["trials"][0]["torque_limits"]["joints"][j]["integral_bound_nm"] for j in joints])
    for result in results:
        if result["status"] != "completed":
            continue
        if result["study"] == "transition":
            rows.extend(transition_rows(result, joints, hashes, checks))
            continue
        rows.append(
            {k: v for k, v in result.items() if k not in ("directory", "metrics", "factors")}
            | result["factors"]
            | result["metrics"]
        )
        for suffix in ("loaded", "unloaded"):
            path = Path(result["directory"]) / f"{suffix}.npz"
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as trace:
                step = 4 if result["study"] == "abrupt" else 1
                effort = trace["effort"][::step]
                headroom = trace["proposed"][::step] != trace["headroom_limited_effort"][::step]
                check = command_check(effort, result["factors"], bounds, headroom=headroom)
                check.update(
                    study=result["study"],
                    variant=result["variant"],
                    case=result["case"],
                    seed=result["seed"],
                    load=result["load"],
                    walking=result["walking"],
                    trace=suffix,
                    samples_complete=len(trace["t"]) == (2400 if step == 4 else 601),
                    duration_complete=bool(abs(trace["t"][-1] - (11.995 if step == 4 else 12)) < 1e-6),
                    finite=check["finite"] and all(np.isfinite(trace[k]).all() for k in ("q", "target", "estimate")),
                )
                checks.append(check)
    for check in checks:
        check["passed"] = (
            all(check[k] for k in ("finite", "samples_complete", "duration_complete", "reversal_policy_passed"))
            and check["correction_bound_excess_nm"] <= 1e-8
            and check["slew_excess_nm_s"] <= 1e-8
            and check.get("applied_torque_bound_excess_nm", 0) <= 1e-8
        )
    complete = (
        len(results) == len(metadata["jobs"])
        and {r["directory"] for r in results} == {job["directory"] for job in metadata["jobs"]}
        and all(r["status"] == "completed" for r in results)
    )
    expected_checks = sum(3 if job["study"] == "transition" else 2 for job in metadata["jobs"])
    validation = {
        "passed": complete and len(checks) == expected_checks and all(c["passed"] for c in checks),
        "checks": checks,
        "scope": "Command samples; existing harness clips total torque every 5 ms. "
        "Diagnostic instantaneous estimates in linearized variants may exceed correction bounds before filtering.",
    }
    (args.directory / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    (args.directory / "trace_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    (args.directory / "metrics.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    (args.directory / "paired_effects.json").write_text(
        json.dumps(paired_effects(rows), indent=2, allow_nan=False) + "\n"
    )
    with (args.directory / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for r in rows for key in r}))
        writer.writeheader()
        writer.writerows(rows)
    groups = defaultdict(list)
    for row in rows:
        context = (
            f"transition / {row['profile']}"
            if row["study"] == "transition"
            else f"{row['study']} / {'walking' if row['walking'] else 'standing'} / "
            f"{row['case']} / load {row['load']:g}"
        )
        groups[context].append(row)
    lines = [
        "# Observer factorial ablation",
        "",
        f"Completed jobs: {sum(r['status'] == 'completed' for r in results)}/{len(metadata['jobs'])}. "
        f"Trace validation passed: {validation['passed']}.",
        "",
        "Factors: estimator (linearized/nonlinear), release (2/16 Nm/s), "
        "anti-windup (frozen/directional). Growth remains 2 Nm/s and bandwidth 8 rad/s. "
        "All factorial variants share zero-before-reversal; original retains its historical output policy.",
        "",
    ]
    for context, selected in sorted(groups.items()):
        lines.extend(
            [
                f"## {context}",
                "",
                "| Variant | Carry RMS (deg) | Correction SD (Nm) | Hand peak (mm) | "
                "Torque removal (s) | Joint peak (deg) |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for variant in metadata["variants"]:
            trials = [r for r in selected if r["variant"] == variant]
            if trials:
                lines.append(
                    f"| {variant} | "
                    + " | ".join(
                        span(trials, key)
                        for key in (
                            "carry_rms_deg",
                            "carry_ripple_nm",
                            "release_hand_peak_mm",
                            "torque_removal_s",
                            "release_peak_deg",
                        )
                    )
                    + " |"
                )
        lines.append("")
    lines.extend(
        [
            "## Limits",
            "",
            "Ranges are observed minima/maxima, not confidence intervals. Nominal and clipped release trials "
            "are deterministic single-seed runs; impaired walking uses three seeds. Transitions use paired "
            "4/5/6 s warmups. Abrupt and gradual-placement peaks use different protocols and windows; "
            "compare methods within a protocol. Unestablished/irrelevant metrics are not zeros. "
            "Original and factorial variants also differ in zero-crossing policy. "
            "No hardware qualification, real-time timing guarantee or model-uncertainty sweep is claimed.",
            "",
        ]
    )
    (args.directory / "findings.md").write_text("\n".join(lines))
    if args.plot:
        plot_results(args.directory, rows, metadata["variants"])
    print(
        json.dumps(
            {"jobs": len(results), "rows": len(rows), "trace_checks": len(checks), "passed": validation["passed"]}
        )
    )


if __name__ == "__main__":
    main()
