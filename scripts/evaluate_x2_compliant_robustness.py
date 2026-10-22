"""Run all 22 uncertainty cases with both compliant-payload motion directions."""

# ruff: noqa: E402 -- Native thread caps precede simulation imports.
import os

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import json
import multiprocessing
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
from evaluate_x2_robustness import compare_numeric, digest, initialize_worker, snapshot, worker, write_json
from evaluate_x2_transitions import condition_parameters
from x2_compliant_payload import CompliantPayload
from x2_feedback_perturbations import study_cases
from x2_robustness_analysis import analyze_case
from x2_robustness_sim import load_physics

METHODS = ("pd_only", "integral", "observer")


def jobs_for(case, root):
    jobs = []
    for variant, profiles, sign in (
        ("reference", ("unloaded", "constant"), None),
        ("positive", ("compliant_payload",), 1),
        ("negative", ("compliant_payload",), -1),
    ):
        directory = root / case.case / variant
        directory.mkdir(parents=True, exist_ok=True)
        for profile in profiles:
            for seed, warmup in ((1, 4), (2, 5), (3, 6)):
                for method in METHODS:
                    jobs.append(
                        (
                            method,
                            profile,
                            seed,
                            directory,
                            False,
                            condition_parameters(method),
                            warmup,
                            asdict(case),
                            seed,
                            {"initial_sign": sign} if sign else None,
                        )
                    )
    return jobs


def verify_nominal(root, historical):
    checks = []
    for variant in ("reference", "positive", "negative"):
        for path in sorted((root / "N0" / variant).glob("*/*.json")):
            summary = json.loads(path.read_text())
            current = root / "N0" / variant
            archived = historical / variant
            check = compare_numeric(current / summary["trace"], archived / summary["trace"])
            a = load_physics(current / summary["physics_trace"])
            b = load_physics(archived / summary["physics_trace"])
            check.update(
                variant=variant,
                condition=summary["condition"],
                seed=summary["seed"],
                completed=summary["status"] == "completed",
                physics_exact=a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a),
            )
            checks.append(check)
    passed = len(checks) == 36 and all(c["exact"] and c["physics_exact"] and c["completed"] for c in checks)
    write_json(root / "nominal_verification.json", {"passed": passed, "trials": checks})
    if not passed:
        raise RuntimeError("Nominal reproduction failed; perturbed cases were not started")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("jobs must be positive")
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    cases = study_cases()
    snapshot(root, cases)
    hashes = json.loads((root / "source_sha256.json").read_text())
    for name in (
        "scripts/evaluate_x2_compliant_robustness.py",
        "scripts/x2_compliant_payload.py",
        "tests/test_x2_compliant_payload.py",
    ):
        shutil.copy2(name, root / "sources" / name)
        hashes[name] = digest(name)
    write_json(root / "source_sha256.json", hashes)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest.update(
        planned_trials=792,
        jobs=args.jobs,
        payload=CompliantPayload().configuration(),
        variants={"reference": ["unloaded", "constant"], "positive": 1, "negative": -1},
        nominal_reference="logs/x2_transitions/compliant-payload-20261022",
        controller="fixed selected 8 rad/s observer; unchanged integral and PD",
        matching="same feedback perturbations, plant parameters and relative payload initial states; forces endogenous",
        scope="5 ms physics, 50 Hz correction; prior timestep convergence limitation remains unresolved",
    )
    write_json(root / "manifest.json", manifest)
    started = time.monotonic()
    context = multiprocessing.get_context("spawn")
    recorded = 0
    with ProcessPoolExecutor(
        max_workers=args.jobs,
        mp_context=context,
        initializer=initialize_worker,
        initargs=(context.BoundedSemaphore(2),),
    ) as pool:
        for case in cases:
            if shutil.disk_usage(root).free < 3 * 1024**3:
                raise RuntimeError("Less than 3 GiB free; stopped without deleting historical evidence")
            jobs = jobs_for(case, root)
            futures = {pool.submit(worker, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    summary = future.result()
                except Exception as error:
                    write_json(root / "worker_failure.json", {"case": case.case, "error": repr(error)})
                    raise
                recorded += 1
                progress = {
                    "recorded": recorded,
                    "planned": 792,
                    "case": case.case,
                    "variant": job[3].name,
                    **{k: summary[k] for k in ("condition", "profile", "seed", "status", "wall_seconds")},
                }
                with (root / "progress.jsonl").open("a") as stream:
                    stream.write(json.dumps(progress) + "\n")
                print(json.dumps(progress), flush=True)
            if case.case == "N0":
                verify_nominal(root, Path(manifest["nominal_reference"]))
            for variant in ("positive", "negative"):
                (root / case.case / variant / "unloaded").symlink_to(
                    "../reference/unloaded",
                    target_is_directory=True,
                )
    write_json(
        root / "execution.json",
        {
            "status": "acquisition_complete",
            "recorded_trials": recorded,
            "workers": args.jobs,
            "wall_seconds": time.monotonic() - started,
        },
    )
    for case in cases:
        for variant in ("reference", "positive", "negative"):
            analyze_case(root / case.case / variant)
    traces = {
        str(p.relative_to(root)): digest(p)
        for p in root.glob("*/*/*/*_physics.npz")
        if p.parent.name != "unloaded" or "reference" in p.parts
    }
    write_json(root / "physics_trace_sha256.json", traces)
    write_json(
        root / "completion.json",
        {
            "status": "analysis_complete",
            "recorded_trials": recorded,
            "physics_traces": len(traces),
            "wall_seconds": time.monotonic() - started,
            "controller_retuned": False,
            "hardware_commands": False,
        },
    )


if __name__ == "__main__":
    main()
