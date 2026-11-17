"""Run the fixed, simulation-only X2 robustness matrix after exact nominal verification."""

# ruff: noqa: E402 -- Thread caps must precede NumPy/ONNX imports.

import os

# Set before importing NumPy, ONNX or the spawned worker module.
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import ctypes
import gc
import hashlib
import importlib.metadata
import json
import logging
import multiprocessing
import resource
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
from evaluate_x2_transitions import PHASES, condition_parameters, run_trial, transition_config
from plot_x2_transitions import load_numeric
from x2_feedback_perturbations import BURST_STARTS, Perturbation, study_cases
from x2_integral_sim import POSTURE


def digest(path):
    return hashlib.file_digest(Path(path).open("rb"), "sha256").hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def initialize_worker(gate):
    import x2_robustness_sim

    x2_robustness_sim.INITIALIZATION_GATE = gate


def worker(job):
    logging.getLogger("robojudo").setLevel(logging.WARNING)
    result = run_trial(*job)
    gc.collect()
    ctypes.CDLL(None).malloc_trim(0)
    return result


def jobs_for(case, root):
    return [
        (
            condition,
            profile,
            repetition,
            root / case.case,
            False,
            condition_parameters(condition),
            float(warmup),
            asdict(case),
            repetition,
        )
        for profile in ("unloaded", "constant", "sway")
        for repetition, warmup in enumerate((4, 5, 6), 1)
        for condition in ("pd_only", "integral", "observer")
    ]


def compare_numeric(a_path, b_path):
    a, b = load_numeric(a_path), load_numeric(b_path)
    common = sorted(a.keys() & b.keys())
    different = [k for k in common if not np.array_equal(a[k], b[k])]
    return {
        "exact": not different and len(a["elapsed_s"]) == len(b["elapsed_s"]),
        "shared_fields": len(common),
        "different_fields": different,
        "reference": str(b_path),
        "reference_sha256": digest(b_path),
        "max_absolute_difference": max(
            (float(np.max(np.abs(a[k] - b[k]))) for k in different if a[k].shape == b[k].shape), default=0
        ),
    }


def nominal_gate(root):
    results = []
    for path in sorted((root / "N0").glob("*/*.json")):
        trial = json.loads(path.read_text())
        reference_root = Path("logs/x2_transitions") / (
            "observer-validation-20261022/candidate_01"
            if trial["condition"] == "observer"
            else "torque-id-baselines-20261022"
        )
        check = compare_numeric(root / "N0" / trial["trace"], reference_root / trial["trace"])
        check.update(condition=trial["condition"], profile=trial["profile"], seed=trial["seed"])
        check["completed"] = trial["status"] == "completed"
        check["physics_samples"] = trial["physics_samples"]
        results.append(check)
    gate = {
        "passed": len(results) == 27
        and all(r["exact"] and r["completed"] and r["physics_samples"] == 21000 for r in results),
        "trials": results,
    }
    write_json(root / "nominal_verification.json", gate)
    return gate["passed"]


def snapshot(root, cases):
    cfg = transition_config()
    files = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    files += [str(p) for p in Path("scripts").glob("*robustness*.py")]
    files += ["scripts/x2_feedback_perturbations.py", "tests/test_x2_robustness.py"]
    files = sorted(
        {p for p in files if Path(p).is_file() and (p.endswith((".py", ".toml", ".yaml", ".txt")) or p == "AGENTS.md")}
    )
    hashes = {}
    for name in files:
        path = Path(name)
        target = root / "sources" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[name] = digest(path)
    write_json(root / "source_sha256.json", hashes)
    manifest = {
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        "cases": [asdict(c) for c in cases],
        "planned_trials": 27 * len(cases),
        "seed_warmup_pairs": [[1, 4], [2, 5], [3, 6]],
        "burst_starts_s": BURST_STARTS,
        "acquisition_and_telemetry_dt_s": 0.005,
        "correction_dt_s": 0.02,
        "feedback_ranges": "exploratory assumptions; no recorded sensor or transport statistics found",
        "noise_covariance": (
            "white independent; colored AR(1) tau=0.05s, same-arm rho=0.5, q/v rho=0.5, independent arms"
        ),
        "reference_model": "fixed selected coefficients; uncertainty changes plant arrays only",
        "config": cfg.model_dump(mode="json"),
        "hardware_commands": False,
        "model_sha256": digest(cfg.loco_policy.policy_file),
        "xml_sha256": digest(cfg.env.xml),
        "robot_asset_sha256": {str(p): digest(p) for p in Path(cfg.env.xml).parent.rglob("*") if p.is_file()},
        "software": {p: importlib.metadata.version(p) for p in ("numpy", "mujoco", "onnxruntime", "torch")},
        "thread_environment": {
            k: os.environ[k]
            for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
        "compiler_initialization_workers": 2,
        "compiler_asset_cache": "cleared after construction; no plant-state changes",
        "onnx_threads": [cfg.loco_policy.onnx_intra_op_num_threads, cfg.loco_policy.onnx_inter_op_num_threads],
    }
    write_json(root / "manifest.json", manifest)


def verify_serial(root, case):
    output = root / "execution_verification" / case.case
    output.mkdir(parents=True, exist_ok=True)
    summary = run_trial("observer", "constant", 2, output, False, condition_parameters("observer"), 5, asdict(case), 2)
    parallel = root / case.case / summary["trace"]
    check = compare_numeric(output / summary["trace"], parallel)
    from x2_robustness_sim import load_physics

    a = load_physics(output / summary["physics_trace"])
    b = load_physics(root / case.case / summary["physics_trace"])
    check["physics_exact"] = a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)
    write_json(output / "equivalence.json", check)
    gc.collect()
    ctypes.CDLL(None).malloc_trim(0)
    if not check["exact"] or not check["physics_exact"]:
        raise RuntimeError(f"Serial/parallel equivalence failed for {case.case}")


def execute_batch(pool, case, root):
    jobs = jobs_for(case, root)
    (root / case.case).mkdir(exist_ok=True)
    start = time.monotonic()
    summaries = []
    futures = {pool.submit(worker, job): job for job in jobs}
    for future in as_completed(futures):
        job = futures[future]
        try:
            summary = future.result()
        except Exception as error:
            write_json(
                root / case.case / "worker_failure.json",
                {
                    "job": [str(x) for x in job[:5]],
                    "error": repr(error),
                    "action": "Preserved failure; study stopped before automatic replacement",
                },
            )
            raise
        summaries.append(summary)
        with (root / "progress.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        k: summary[k]
                        for k in (
                            "case",
                            "condition",
                            "profile",
                            "seed",
                            "status",
                            "wall_seconds",
                            "worker_peak_rss_mib",
                            "termination_reason",
                        )
                    }
                )
                + "\n"
            )
        with (root / "resources.jsonl").open("a") as stream:
            mem_available_kib = int(
                next(
                    line.split()[1]
                    for line in Path("/proc/meminfo").read_text().splitlines()
                    if line.startswith("MemAvailable:")
                )
            )
            stream.write(
                json.dumps(
                    {
                        "elapsed_wall_s": time.monotonic() - start,
                        "case": case.case,
                        "available_memory_mib": mem_available_kib / 1024,
                        "free_disk_bytes": shutil.disk_usage(root).free,
                    }
                )
                + "\n"
            )
        print(
            json.dumps(
                {
                    "case": case.case,
                    "done": len(summaries),
                    "condition": summary["condition"],
                    "profile": summary["profile"],
                    "seed": summary["seed"],
                    "status": summary["status"],
                    "wall_seconds": round(summary["wall_seconds"], 2),
                    "live_rss_mib": round(summary["worker_live_rss_mib"]),
                }
            ),
            flush=True,
        )
    summaries.sort(key=lambda s: (s["profile"], s["repetition"], s["condition"]))
    write_json(
        root / case.case / "summary.json",
        {
            "trials": summaries,
            "case": asdict(case),
            "phases": PHASES,
            "carry_posture_rad": POSTURE,
            "batch_wall_seconds": time.monotonic() - start,
        },
    )
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--case-file", type=Path, help="JSON list of complete perturbation configurations")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("jobs must be positive")
    if args.analyze_only:
        from x2_robustness_report import analyze_study

        analyze_study(args.output)
        return
    cases = [Perturbation(**c) for c in json.loads(args.case_file.read_text())] if args.case_file else study_cases()
    if len({c.case for c in cases}) != len(cases) or not any(c == Perturbation("N0") for c in cases):
        parser.error("Unique cases and an unchanged N0 case are required")
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot(args.output, cases)
    start = time.monotonic()
    n0 = next(c for c in cases if c.case == "N0")
    context = multiprocessing.get_context("spawn")
    gate = context.BoundedSemaphore(2)
    with ProcessPoolExecutor(
        max_workers=args.jobs, mp_context=context, initializer=initialize_worker, initargs=(gate,)
    ) as pool:
        execute_batch(pool, n0, args.output)
        if not nominal_gate(args.output):
            raise RuntimeError("Nominal reproduction failed; perturbed cases were not started")
        verify_serial(args.output, n0)
        for case in cases:
            if case.case == "N0":
                continue
            execute_batch(pool, case, args.output)
            if case.case == "Q1":
                verify_serial(args.output, case)
    write_json(
        args.output / "execution.json",
        {
            "workers": args.jobs,
            "wall_seconds": time.monotonic() - start,
            "parent_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "disk_bytes": sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file()),
            "status": "acquisition_complete",
        },
    )
    from x2_robustness_report import analyze_study

    analyze_study(args.output)


if __name__ == "__main__":
    main()
