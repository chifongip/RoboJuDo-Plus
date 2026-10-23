"""Plot saved posture-transition trials using NumPy/Matplotlib only."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_numeric(path):
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            fields = archive["columns"].tolist()
            chunks = [archive[key] for key in sorted(archive.files) if key.startswith("chunk_")]
        data = np.concatenate(chunks) if chunks else np.empty((0, len(fields)))
        return {name: data[:, i] for i, name in enumerate(fields)}
    with path.open() as stream:
        fields = next(csv.reader(stream))
    numeric = [i for i, name in enumerate(fields) if name != "phase"]
    data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=numeric, ndmin=2)
    return {fields[index]: data[:, j] for j, index in enumerate(numeric)}


def plot_profile(root, report, profile):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trials = [t for t in report["trials"] if t["profile"] == profile and t["samples"]]
    reference_trials = {
        (t["condition"], t["repetition"]): t for t in report["trials"] if t["profile"] == "unloaded" and t["samples"]
    }
    fig, axes = plt.subplots(10, 1, figsize=(13, 23), sharex=True)
    pose_fig, pose_axes = plt.subplots(7, 2, figsize=(16, 19), sharex=True)
    target_trial = next((t for t in trials if t["condition"] == "pd_only"), trials[0] if trials else None)
    for trial in trials:
        rows = load_numeric(root / trial["trace"])
        condition = trial["condition"]
        color, style = {
            "pd_only": ("tab:orange", "--"),
            "integral": ("tab:green", ":"),
            "observer": ("tab:blue", "-"),
            "nonlinear_observer": ("tab:purple", "-."),
        }[condition]
        t = rows["reference_time_s"]
        label = f"{condition}, seed {trial.get('seed', trial['repetition'])}"
        for name, axis in zip(report["carry_posture_rad"], pose_axes.flat, strict=True):
            axis.plot(t, np.rad2deg(rows[f"{name}_position_rad"]), style, color=color, alpha=0.8)
            if trial is target_trial:
                axis.plot(t, np.rad2deg(rows[f"{name}_target_rad"]), ":", color="black", label="issued target")
                axis.plot(t, np.rad2deg(rows[f"{name}_desired_rad"]), color="gray", alpha=0.6, label="desired")
            axis.set_title(name.removesuffix("_joint"), fontsize=9)
            axis.set_ylabel("deg")
            axis.grid(alpha=0.3)
        elbow = "left_elbow_joint"
        axes[0].plot(t, np.rad2deg(rows[f"{elbow}_position_rad"]), style, color=color, label=label)
        if trial is target_trial:
            axes[0].plot(t, np.rad2deg(rows[f"{elbow}_target_rad"]), ":", color="black", label="issued target")
            axes[0].plot(t, np.rad2deg(rows[f"{elbow}_desired_rad"]), color="gray", alpha=0.6, label="desired")
        reference = reference_trials.get((condition, trial["repetition"]))
        if reference:
            unloaded = load_numeric(root / reference["trace"])
            n = min(len(t), len(unloaded["elapsed_s"]))
            for index, side in enumerate(("left", "right")):
                key = f"{side}_torso_z_m"
                axes[index + 1].plot(t[:n], 1000 * (unloaded[key][:n] - rows[key][:n]), style, color=color)
            if condition in ("integral", "observer", "nonlinear_observer"):
                key = f"{elbow}_{'observer' if condition in ('observer', 'nonlinear_observer') else 'integral'}_nm"
                axes[4].plot(t[:n], rows[key][:n], style, color=color, label=f"{condition} total")
                axes[4].plot(t[:n], unloaded[key][:n], ":", color="gray", label="unloaded reference")
                axes[5].plot(t[:n], rows[key][:n] - unloaded[key][:n], style, color=color)
        axes[3].plot(t, np.rad2deg(rows["arm_rms_error_rad"]), style, color=color)
        if condition in ("integral", "observer", "nonlinear_observer"):
            axes[6].plot(t, rows[f"{elbow}_target_velocity_rad_s"], color=color)
            axes[7].plot(t, rows["force_z_n"], color="black", alpha=0.5)
            axes[7].plot(t, rows["force_x_n"], color="tab:green", alpha=0.5)
        axes[8].plot(t, rows["actual_vx_m_s"], style, color=color)
        if trial is target_trial:
            axes[8].plot(t, rows["command_vx_m_s"], ":", color="black")
        axes[9].plot(t, np.rad2deg(rows["base_tilt_rad"]), style, color=color)
    labels = (
        "Left elbow (deg)",
        "Left load-induced\ndisplacement (mm)",
        "Right load-induced\ndisplacement (mm)",
        "Arm tracking RMS (deg)",
        "Elbow correction (Nm)",
        "Elbow correction minus\nunloaded reference (Nm)",
        "Elbow target speed (rad/s)",
        "Hand force (N)\nz black; x green",
        "Forward velocity (m/s)",
        "Base tilt (deg)",
    )
    for axis, label in zip(axes, labels, strict=True):
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
    for axis in list(axes) + list(pose_axes.flat):
        for name, start, end in report["phases"]:
            if name in ("reach", "lift", "lower", "return_default"):
                axis.axvspan(start, end, alpha=0.06, color="red" if name == "lower" else "green")
    axes[0].legend(fontsize=8, ncol=3)
    axes[-1].set_xlabel("Time (s); shaded regions: arm posture transitions")
    point = "hand center (body COM)" if "center of mass" in report.get("force_point", "") else "wrist origin"
    fig.suptitle(
        f"{report.get('case', {}).get('case', 'Nominal')}: {profile}; walking command 0.5 m/s\n"
        f"Force and measurement point: {point}"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(root / profile / "comparison.png", dpi=130)
    plt.close(fig)
    handles, labels = axes[0].get_legend_handles_labels()
    pose_fig.legend(handles, labels, loc="upper center", fontsize=8, ncol=4)
    pose_fig.tight_layout(rect=(0, 0, 1, 0.97))
    pose_fig.savefig(root / profile / "posture_tracking.png", dpi=130)
    plt.close(pose_fig)


def plot_overview(root, rows, case_names):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex=True)
    specs = [
        ("walking_rms_deg", "Walking aggregate RMS (deg)"),
        ("unload_peak_mm", "Unloading 79–82 s peak (mm)"),
        ("tilt_peak_deg", "Whole-trial peak tilt (deg)"),
        ("saturated_percent", "Any actuator saturated (%)"),
    ]
    for column, profile in enumerate(("constant", "sway")):
        for axrow, (metric, label) in enumerate(specs):
            ax = axes[axrow, column]
            for method, color in (("pd_only", "tab:orange"), ("integral", "tab:green"), ("observer", "tab:blue")):
                vals = [
                    [
                        r[metric]
                        for r in rows
                        if r["case"] == case
                        and r["condition"] == method
                        and r["profile"] == profile
                        and r[metric] is not None
                    ]
                    for case in case_names
                ]
                center = [np.median(x) if x else np.nan for x in vals]
                ax.plot(range(len(case_names)), center, ".-", color=color, label=method)
                for i, values in enumerate(vals):
                    if values:
                        ax.vlines(i, min(values), max(values), color=color, alpha=0.5)
                    failed_values = [
                        r[metric]
                        for r in rows
                        if r["case"] == case_names[i]
                        and r["condition"] == method
                        and r["profile"] == profile
                        and r["status"] != "completed"
                        and r[metric] is not None
                    ]
                    if failed_values:
                        ax.scatter([i] * len(failed_values), failed_values, marker="x", color="red", s=40, zorder=5)
            ax.set_ylabel(label)
            ax.grid(alpha=0.25)
            ax.set_xticks(range(len(case_names)), case_names, rotation=60)
            if axrow == 0:
                ax.set_title(profile)
                if any(r["status"] != "completed" for r in rows):
                    ax.scatter([], [], marker="x", color="red", label="terminated trial")
                ax.legend()
    fig.suptitle("Exploratory simulation robustness; markers median, bars range across paired seeds/warmups")
    fig.tight_layout()
    fig.savefig(root / "robustness_overview.png", dpi=130)
    plt.close(fig)


def plot_recovery(root, case_names):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [c for c in ("N0", "V2", "D3", "X2", "P1", "P2") if c in case_names]
    fig, axes = plt.subplots(len(selected), 2, figsize=(14, 3 * len(selected)), squeeze=False)
    for row, case in enumerate(selected):
        summary_path = root / case / "physics_summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"trials": []}
        index = {(t.get("condition"), t.get("profile"), t.get("seed")): t for t in summary["trials"]}
        for col, profile in enumerate(("constant", "sway")):
            ax = axes[row, col]
            for method, color in (("pd_only", "tab:orange"), ("integral", "tab:green"), ("observer", "tab:blue")):
                path = root / case / profile / f"{method}_2_physics.npz"
                ref = root / case / "unloaded" / path.name
                if not path.exists() or not ref.exists():
                    continue
                data, unloaded = load_numeric(path), load_numeric(ref)
                n = min(len(data["elapsed_s"]), len(unloaded["elapsed_s"]))
                t = data["elapsed_s"][:n]
                sel = (t >= 75) & (t <= 89)
                if not np.any(sel):
                    continue
                delta = np.stack(
                    [data[f"left_torso_{a}_m"][:n] - unloaded[f"left_torso_{a}_m"][:n] for a in "xyz"], axis=1
                )
                ax.plot(t[sel], 1000 * np.linalg.norm(delta, axis=1)[sel], color=color, label=method)
                envelope = (
                    index.get((method, profile, 2), {}).get("recovery", {}).get("left", {}).get("unloaded_envelope_m")
                )
                if envelope is not None:
                    ax.axhline(1000 * envelope, color=color, linestyle=":", alpha=0.55)
            ax.axvline(79, color="gray", linestyle=":")
            ax.set_title(f"{case} / {profile}, seed 2; left hand")
            ax.set_ylabel("3D displacement (mm; symlog)")
            ax.set_yscale("symlog", linthresh=0.1)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.25)
            if row == 0:
                ax.legend()
    fig.tight_layout()
    fig.suptitle("5 ms telemetry; dotted horizontal lines: measured unloaded motion envelopes")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(root / "unloading_5ms.png", dpi=130)
    plt.close(fig)


def plot_failure(root, failure):
    """Show measured precursors, with matched comparators, without asserting unique causality."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    case, method, profile, seed = (failure[k] for k in ("case", "condition", "profile", "seed"))
    path = root / case / profile / f"{method}_{seed}_physics.npz"
    if not path.exists():
        return
    failed = load_numeric(path)
    if not len(failed.get("elapsed_s", [])):
        return
    end = float(failed["elapsed_s"][-1])
    start = max(0, end - 8)
    fig, axes = plt.subplots(7, 1, figsize=(13, 17), sharex=True)
    variants = [
        (case, m, color, "-")
        for m, color in (("pd_only", "tab:orange"), ("integral", "tab:green"), ("observer", "tab:blue"))
    ]
    if case != "N0":
        variants.append(("N0", method, "gray", "--"))
    diagnostic = []
    for variant, controller, color, style in variants:
        trace = root / variant / profile / f"{controller}_{seed}_physics.npz"
        if not trace.exists():
            continue
        d = load_numeric(trace)
        t = d["elapsed_s"]
        sel = (t >= start - 1e-8) & (t <= end + 1e-8)
        if not np.any(sel):
            continue
        names = [key.removesuffix("_target_rad") for key in d if key.endswith("_target_rad")]
        power = sum(d[f"{j}_correction_nm"][sel] * d[f"{j}_velocity_rad_s"][sel] for j in names)
        error = np.max(
            np.stack([np.abs(d[f"{j}_position_rad"][sel] - d[f"{j}_target_rad"][sel]) for j in names]), axis=0
        )
        saturation = np.mean(np.stack([d[f"{j}_saturation_fraction"][sel] for j in names]), axis=0)
        signals = [
            np.rad2deg(d["base_tilt_rad"][sel]),
            np.rad2deg(d["base_pitch_rad"][sel]),
            d["base_height_m"][sel],
            power,
            np.rad2deg(error),
            100 * saturation,
            1000 * d["feedback_age_s"][sel],
        ]
        for ax, values in zip(axes, signals, strict=True):
            ax.plot(t[sel], values, style, color=color, label=f"{variant} / {controller}", alpha=0.85)
        diagnostic.append(
            {
                "case": variant,
                "condition": controller,
                "window_s": [start, end],
                "max_tilt_deg": float(signals[0].max()),
                "mean_correction_joint_power_w": float(np.mean(power)),
                "mean_arm_joint_clipping_fraction": float(np.mean(saturation)),
                "max_feedback_age_ms": float(signals[-1].max()),
            }
        )
    labels = [
        "Base tilt (deg)",
        "Base pitch (deg)",
        "Pelvis height (m)",
        "Correction joint power (W)",
        "Maximum arm error (deg)",
        "Arm joints clipped (%)",
        "Feedback age (ms)",
    ]
    for ax, label in zip(axes, labels, strict=True):
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[0].legend(ncol=2)
    axes[-1].set_xlabel("Elapsed simulation time (s)")
    fig.suptitle(
        f"Failure: {case} / {method} / {profile} / seed {seed}, t={end:.3f} s\n"
        "Power uses issued correction times end-of-step joint velocity; temporal association is not unique causality"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    directory = root / "failure_diagnostics"
    directory.mkdir(exist_ok=True)
    name = f"{case}_{method}_{profile}_{seed}"
    fig.savefig(directory / f"{name}.png", dpi=130)
    plt.close(fig)
    (directory / f"{name}.json").write_text(json.dumps(diagnostic, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--case", action="append", help="Study case; repeat to select cases")
    parser.add_argument("--physics", action="store_true", help="Plot 5 ms NPZ telemetry")
    args = parser.parse_args()
    if (args.directory / "manifest.json").exists():
        manifest = json.loads((args.directory / "manifest.json").read_text())
        case_names = [case["case"] for case in manifest["cases"]]
        with (args.directory / "overview.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            for field in ("walking_rms_deg", "unload_peak_mm", "tilt_peak_deg", "saturated_percent"):
                row[field] = float(row[field]) if row[field] else None
        plot_overview(args.directory, rows, case_names)
        plot_recovery(args.directory, case_names)
        failures_path = args.directory / "failures.json"
        if failures_path.exists():
            for failure in json.loads(failures_path.read_text()):
                plot_failure(args.directory, failure)
        import sys

        import matplotlib

        (args.directory / "plotting_versions.json").write_text(
            json.dumps(
                {"python": sys.executable, "numpy": np.__version__, "matplotlib": matplotlib.__version__}, indent=2
            )
            + "\n"
        )
        cases = args.case or [c for c in ("N0", "X2", "P2") if c in case_names]
        for case in cases:
            root = args.directory / case
            report = json.loads((root / "summary.json").read_text())
            report.setdefault("force_point", "hand center: wrist_roll_link center of mass")
            if args.physics:
                report["trials"] = [{**t, "trace": t["physics_trace"]} for t in report["trials"]]
            for profile in ("constant", "sway"):
                plot_profile(root, report, profile)
        return
    report = json.loads((args.directory / "summary.json").read_text())
    if "case" in report:
        report.setdefault("force_point", "hand center: wrist_roll_link center of mass")
    if args.physics:
        if not all("physics_trace" in t for t in report["trials"]):
            parser.error("This report has no physics traces")
        report["trials"] = [{**t, "trace": t["physics_trace"]} for t in report["trials"]]
    for profile in sorted({t["profile"] for t in report["trials"]}):
        plot_profile(args.directory, report, profile)
        print(args.directory / profile / "comparison.png")


if __name__ == "__main__":
    main()
