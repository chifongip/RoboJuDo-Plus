"""Summarize the four-method transition benchmark without importing robot code."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

LABELS = {
    "pd_only": "PD only",
    "integral": "Integral",
    "observer": "Linearized observer",
    "nonlinear_observer": "Nonlinear observer",
}
COMPARISONS = {
    "integral": "comparisons",
    "observer": "observer_comparisons",
    "nonlinear_observer": "nonlinear_observer_comparisons",
}


def trial_metrics(root, report, trial):
    row = {key: trial[key] for key in ("condition", "profile", "repetition", "warmup_s", "status")}
    if trial["status"] != "completed":
        row["reason"] = trial["termination_reason"]
        return row
    joints = list(report["carry_posture_rad"])
    fields = ["elapsed_s", "arm_rms_error_rad", *[f"{name}_correction_nm" for name in joints]]
    path = root / trial["trace"]
    with path.open() as stream:
        header = next(csv.reader(stream))
    data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=[header.index(f) for f in fields], ndmin=2)
    walking = data[(data[:, 0] >= 55) & (data[:, 0] < 60)]
    row.update(
        walking_rms_deg=float(np.rad2deg(np.sqrt(np.mean(walking[:, 1] ** 2)))),
        correction_std_nm=float(np.max(np.std(walking[:, 2:], axis=0))),
        peak_joint_error_deg=float(np.rad2deg(max(p["peak_joint_error_rad"] for p in trial["phases"].values()))),
        peak_tilt_deg=float(np.rad2deg(trial["max_base_tilt_rad"])),
        saturated_percent=100 * trial["saturated_physics_fraction"],
        max_effort_slew_nm_s=trial["max_effort_slew_nm_s"],
        max_effort_bound_excess_nm=trial["max_effort_bound_excess_nm"],
    )
    method = trial["condition"]
    lookup_method = "integral" if method == "pd_only" else method
    comparison = next(
        (
            c
            for c in report.get(COMPARISONS[lookup_method], [])
            if c["profile"] == trial["profile"] and c["repetition"] == trial["repetition"] and c["status"] == "measured"
        ),
        None,
    )
    if comparison:
        peak_key = "pd_peak_m" if method == "pd_only" else f"{method}_peak_m"
        row["unloading_peak_mm"] = 1000 * max(v[peak_key] for v in comparison["postplace_peak_diagnostic"].values())
        recovery_key = "pd" if method == "pd_only" else method
        recovery = [v[recovery_key]["recovery_time_s"] for v in comparison["unloading"].values()]
        row["recovery_s"] = max(recovery) if all(v is not None for v in recovery) else None
    return row


def span(rows, key, digits=3):
    values = [row.get(key) for row in rows]
    if not values or any(v is None for v in values):
        return "not established"
    low, high = min(values), max(values)
    low_text, high_text = f"{low:.{digits}f}", f"{high:.{digits}f}"
    return low_text if low_text == high_text else f"{low_text}–{high_text}"


def run_description(report):
    """Describe the recorded settings, including runs with selected methods/profiles."""
    trials = report["trials"]
    methods = report.get("conditions", sorted({t["condition"] for t in trials}))
    profiles = report.get("profiles", sorted({t["profile"] for t in trials}))
    warmups = sorted({t["warmup_s"] for t in trials} or set(report.get("warmup_seconds", [])))
    warmup_text = "/".join(f"{value:g}" for value in warmups) if warmups else "not recorded"
    return [
        "Methods: " + ", ".join(LABELS[method] for method in methods) + ".",
        "",
        "Profiles: " + ", ".join(profiles) + f". Warmup settings (seconds): {warmup_text}.",
    ]


def plot_comparison(root, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    colors = ("tab:orange", "tab:green", "tab:blue", "tab:purple")
    labels = ("PD only", "Integral", "Linearized\nobserver", "Nonlinear\nobserver")
    for column, (profile, title) in enumerate(
        (("constant", "Constant load"), ("compliant_payload", "Compliant payload"))
    ):
        for row, (metric, ylabel) in enumerate(
            (
                ("walking_rms_deg", "Walking RMS (deg)"),
                ("unloading_peak_mm", "Post-unload hand peak (mm)"),
            )
        ):
            axis = axes[row, column]
            for index, (method, color) in enumerate(zip(LABELS, colors, strict=True)):
                values = np.array(
                    [
                        r[metric]
                        for r in rows
                        if r["condition"] == method
                        and r["profile"] == profile
                        and r["status"] == "completed"
                        and metric in r
                    ]
                )
                if not len(values):
                    continue
                middle = np.median(values)
                axis.errorbar(
                    index,
                    middle,
                    yerr=np.array([[middle - values.min()], [values.max() - middle]]),
                    fmt="o",
                    color=color,
                    capsize=5,
                    markersize=7,
                )
            axis.set_xticks(range(4), labels)
            axis.set_yscale("log")
            axis.set_ylabel(ylabel + "; log scale")
            axis.grid(axis="y", which="both", alpha=0.22)
            axis.set_xlim(-0.5, 3.5)
            if row == 0:
                axis.set_title(title)
    figure.suptitle(
        "X2 full transition benchmark — four fixed methods\n"
        "Median and min–max across warmups; load ramped down over 75–79 s",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(root / "methods_comparison.png", dpi=170)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--plot", action="store_true", help="Also export a compact comparison figure; requires matplotlib"
    )
    args = parser.parse_args()
    report = json.loads((args.directory / "summary.json").read_text())
    rows = [trial_metrics(args.directory, report, trial) for trial in report["trials"]]
    with (args.directory / "methods_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    completed = sum(r["status"] == "completed" for r in rows)
    lines = [
        "# X2 transition comparison",
        "",
        f"Completed **{completed}/{len(rows)}** trials. "
        f"Scheduled duration (seconds): {report.get('duration_s', 'not recorded')}.",
        "",
        *run_description(report),
        "",
        "This benchmark exercises the synchronous correction cores under the same final-target filtering, "
        "feedback and MuJoCo total-torque clipping harness. It does not emulate native command transport.",
        "",
    ]
    for profile in report.get("profiles", sorted({r["profile"] for r in rows})):
        lines.extend(
            [
                f"## {profile}",
                "",
                "| Method | Completed | Walk RMS (deg) | Peak joint error (deg) | Post-unload hand peak (mm) | "
                "Recovery to PD envelope (s) | Peak tilt (deg) | Saturated physics (%) | Correction SD (Nm) |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for method, label in LABELS.items():
            selected = [r for r in rows if r["condition"] == method and r["profile"] == profile]
            if not selected:
                continue
            good = [r for r in selected if r["status"] == "completed"]
            metrics = [
                "—" if profile == "unloaded" and k in ("unloading_peak_mm", "recovery_s") else span(good, k)
                for k in (
                    "walking_rms_deg",
                    "peak_joint_error_deg",
                    "unloading_peak_mm",
                    "recovery_s",
                    "peak_tilt_deg",
                    "saturated_percent",
                    "correction_std_nm",
                )
            ]
            lines.append(f"| {label} | {len(good)}/{len(selected)} | " + " | ".join(metrics) + " |")
        lines.append("")
    lines.extend(
        [
            "## Definitions and limits",
            "",
            "- Ranges cover the completed trials for each method/profile; they are not confidence intervals.",
            "- Walk RMS uses 55–60 s and all 14 arm joints against the final issued target. "
            "Peak joint error and tilt cover the entire measured trial.",
            "- Post-unload hand peak uses 79–82 s, comparing torso-frame 3D hand positions with the same "
            "method's matched unloaded trajectory; the worse hand is reported.",
            "- Recovery requires both hands to enter and remain in the matched PD displacement envelope "
            "over 79–89 s, with at least one second remaining. "
            "Immediate entry does not prove zero mechanical settling.",
            "- Load is ramped down over 75–79 s, rather than removed abruptly. Physics is 5 ms, "
            "and these transition peak/recovery metrics use 20 ms control samples.",
            "- Correction SD is the largest per-joint standard deviation over 55–60 s. "
            "Saturation is the percentage of physics steps with any actuator clipped.",
            "- Unloaded trials have no load-induced peak/recovery comparison. Failed or unmatched trials "
            "cannot establish a recovery result.",
            "- Compliant-payload results cover the existing translation-only model and positive initial-motion "
            "direction, not arbitrary payloads, poses, disturbances or contact mechanics.",
            "- Simulation only; no hardware qualification is claimed.",
            "",
        ]
    )
    failures = [r for r in rows if r["status"] != "completed"]
    if failures:
        lines.extend(["## Failures", ""])
        lines.extend(
            f"- {r['condition']} / {r['profile']} / repetition {r['repetition']}: {r['reason']}" for r in failures
        )
    (args.directory / "methods_comparison.md").write_text("\n".join(lines) + "\n")
    if args.plot:
        plot_comparison(args.directory, rows)
    print(args.directory / "methods_comparison.md")


if __name__ == "__main__":
    main()
