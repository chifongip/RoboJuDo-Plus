"""Plot saved posture-transition trials using NumPy/Matplotlib only."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_numeric(path):
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
    for trial in trials:
        rows = load_numeric(root / trial["trace"])
        condition = trial["condition"]
        color, style = ("tab:orange", "--") if condition == "pd_only" else ("tab:blue", "-")
        t = rows["reference_time_s"]
        label = f"{condition} {trial['repetition']}"
        for name, axis in zip(report["carry_posture_rad"], pose_axes.flat, strict=True):
            axis.plot(t, np.rad2deg(rows[f"{name}_position_rad"]), style, color=color, alpha=0.8)
            if condition == "pd_only" and trial["repetition"] == 1:
                axis.plot(t, np.rad2deg(rows[f"{name}_target_rad"]), ":", color="black", label="issued target")
                axis.plot(t, np.rad2deg(rows[f"{name}_desired_rad"]), color="gray", alpha=0.6, label="desired")
            axis.set_title(name.removesuffix("_joint"), fontsize=9)
            axis.set_ylabel("deg")
            axis.grid(alpha=0.3)
        elbow = "left_elbow_joint"
        axes[0].plot(t, np.rad2deg(rows[f"{elbow}_position_rad"]), style, color=color, label=label)
        if condition == "pd_only" and trial["repetition"] == 1:
            axes[0].plot(t, np.rad2deg(rows[f"{elbow}_target_rad"]), ":", color="black", label="issued target")
            axes[0].plot(t, np.rad2deg(rows[f"{elbow}_desired_rad"]), color="gray", alpha=0.6, label="desired")
        reference = reference_trials.get((condition, trial["repetition"]))
        if reference:
            unloaded = load_numeric(root / reference["trace"])
            n = min(len(t), len(unloaded["elapsed_s"]))
            for index, side in enumerate(("left", "right")):
                key = f"{side}_torso_z_m"
                axes[index + 1].plot(t[:n], 1000 * (unloaded[key][:n] - rows[key][:n]), style, color=color)
            if condition == "integral":
                key = f"{elbow}_integral_nm"
                axes[4].plot(t[:n], rows[key][:n], color=color, label="total integral")
                axes[4].plot(t[:n], unloaded[key][:n], ":", color="gray", label="unloaded reference")
                axes[5].plot(t[:n], rows[key][:n] - unloaded[key][:n], color=color)
        axes[3].plot(t, np.rad2deg(rows["arm_rms_error_rad"]), style, color=color)
        if condition == "integral":
            axes[6].plot(t, rows[f"{elbow}_target_velocity_rad_s"], color=color)
            axes[7].plot(t, rows["force_z_n"], color="black", alpha=0.5)
            axes[7].plot(t, rows["force_x_n"], color="tab:green", alpha=0.5)
        axes[8].plot(t, rows["actual_vx_m_s"], style, color=color)
        if condition == "pd_only" and trial["repetition"] == 1:
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
    fig.suptitle(f"Stop-to-pick/place: {profile}; walking command 0.5 m/s\nForce and measurement point: {point}")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(root / profile / "comparison.png", dpi=130)
    plt.close(fig)
    handles, labels = axes[0].get_legend_handles_labels()
    pose_fig.legend(handles, labels, loc="upper center", fontsize=8, ncol=4)
    pose_fig.tight_layout(rect=(0, 0, 1, 0.97))
    pose_fig.savefig(root / profile / "posture_tracking.png", dpi=130)
    plt.close(pose_fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report = json.loads((args.directory / "summary.json").read_text())
    for profile in sorted({t["profile"] for t in report["trials"]}):
        plot_profile(args.directory, report, profile)
        print(args.directory / profile / "comparison.png")


if __name__ == "__main__":
    main()
