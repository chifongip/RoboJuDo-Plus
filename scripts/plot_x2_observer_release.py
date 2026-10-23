"""Plot paired abrupt-unloading traces from evaluate_x2_observer_release.py."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--case", default="nominal")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rate", type=float, default=16)
    args = parser.parse_args()
    names = json.loads((args.directory / "metadata.json").read_text())["joint_names"]
    traces = []
    for condition, rate, label in (
        ("pd", 2, "PD only"),
        ("legacy", 2, "Legacy deployment"),
        ("improved", args.rate, f"Improved, release {args.rate:g} Nm/s"),
    ):
        stem = args.directory / f"{args.case}_{args.seed}_{condition}_{rate:g}"
        with np.load(f"{stem}_loaded.npz") as loaded, np.load(f"{stem}_unloaded.npz") as unloaded:
            traces.append((label, dict(loaded), dict(unloaded)))
    _, legacy, baseline = traces[1]
    after = legacy["t"] >= 7
    worst_joint = int(np.argmax(np.max(abs(legacy["q"][after] - baseline["q"][after]), axis=0)))
    figure, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for label, loaded, unloaded in traces:
        t = loaded["t"] - 7
        axes[0].plot(t, np.rad2deg(loaded["q"][:, worst_joint] - unloaded["q"][:, worst_joint]), label=label)
        axes[1].plot(t, loaded["effort"][:, worst_joint] - unloaded["effort"][:, worst_joint], label=label)
        if "hand" in loaded:
            axes[2].plot(t, 1000 * np.max(np.linalg.norm(loaded["hand"] - unloaded["hand"], axis=2), axis=1))
        else:
            axes[2].plot(t, loaded["estimate"][:, worst_joint] - unloaded["estimate"][:, worst_joint])
    axes[0].set_ylabel("Joint displacement (deg)")
    axes[1].set_ylabel("Added correction (Nm)")
    axes[2].set_ylabel("Worst hand displacement (mm)" if "hand" in legacy else "Disturbance estimate (Nm)")
    axes[2].set_xlabel("Time from abrupt unloading (s)")
    axes[0].set_title(f"{args.case}: {names[worst_joint]}\nRelative to each method's matched unloaded motion")
    axes[0].legend()
    for axis in axes:
        axis.axvline(0, color="black", linestyle="--", linewidth=0.8)
        axis.set_xlim(-0.5, 4)
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(args.directory / f"{args.case}_{args.seed}_comparison.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
