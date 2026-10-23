"""Factorial observer comparison using existing local/MuJoCo simulation harnesses.

Runs matched loaded/unloaded pairs, preserving original harness safety checks and
metrics. No hardware environment, SDK, network controller or socket is created.
"""

import argparse
import gc
import hashlib
import json
import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import patch

import evaluate_x2_observer_release as release
import evaluate_x2_transitions as transition
import mujoco
import numpy as np
from x2_observer_ablation import VARIANTS, make_observer

from robojudo.config.x2.observer_parameters import JOINT_NAMES

_BASE_RELEASE_RUNTIME = release.runtime


def release_runtime(variant, names, kp, kd, limits):
    wrapper = _BASE_RELEASE_RUNTIME("legacy" if variant == "original" else "improved", 2, names, kp, kd, limits)
    indices = wrapper.indices
    wrapper.observer = make_observer(
        variant, np.asarray(kp)[indices], np.asarray(kd)[indices], np.asarray(limits)[indices]
    )
    wrapper.reset()
    return wrapper


def worker(job):
    root = Path(job["directory"])
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    variant = job["variant"]
    result = {**job, "factors": VARIANTS[variant]}
    try:
        if job["study"] == "transition":

            def correction(env, parameters=None):
                return make_observer(
                    variant,
                    env.stiffness[env.arm_indices],
                    env.damping[env.arm_indices],
                    env.torque_limits[env.arm_indices],
                )

            trials = []
            # Preserve harness column names; factor metadata is authoritative.
            with patch.object(transition, "make_correction", correction):
                for profile in ("unloaded", "constant", "compliant_payload"):
                    trials.append(
                        transition.run_trial(
                            "nonlinear_observer",
                            profile,
                            job["seed"],
                            root,
                            parameters=transition.condition_parameters("nonlinear_observer"),
                            warmup_s=job["warmup_s"],
                        )
                    )
            result.update(
                status="completed" if all(t["status"] == "completed" for t in trials) else "failed", trials=trials
            )
        else:

            def runtime(_condition, _rate, names, kp, kd, limits):
                return release_runtime(variant, names, kp, kd, limits)

            traces = []
            with patch.object(release, "runtime", runtime):
                for load in (0, job["load"]):
                    if job["study"] == "local":
                        trace = release.local_trial("improved", 2, load, job["case"], job["seed"])
                    else:
                        trace = release.mujoco_trial("improved", 2, load, job["case"], job["seed"], job["walking"])
                    name = "unloaded" if load == 0 else "loaded"
                    np.savez_compressed(root / f"{name}.npz", **trace)
                    traces.append(trace)
            result.update(status="completed", metrics=release.metrics(traces[1], traces[0]))
    except Exception as error:
        # Preserve failures as results and continue the predeclared experiment.
        result.update(status="failed", reason=f"{type(error).__name__}: {error}")
    result["wall_s"] = time.monotonic() - started
    (root / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    mujoco.mj_clearCache(mujoco.mj_getCache())
    gc.collect()
    return result


def jobs_for(root, variants):
    jobs = []
    for variant in variants:
        for case in ("nominal", "clipped", "reversal"):
            for load in (-4.0, 4.0):
                jobs.append(dict(study="local", variant=variant, case=case, seed=1, load=load, walking=False))
        for walking, case, seeds in (
            (False, "nominal", (1,)),
            (True, "nominal", (1,)),
            (True, "clipped", (1,)),
            (True, "impaired", (1, 2, 3)),
        ):
            for seed in seeds:
                jobs.append(dict(study="abrupt", variant=variant, case=case, seed=seed, load=-20.0, walking=walking))
        for seed, warmup in enumerate((4.0, 5.0, 6.0), 1):
            jobs.append(dict(study="transition", variant=variant, seed=seed, warmup_s=warmup))
    for index, job in enumerate(jobs):
        job["directory"] = str(root / job["study"] / f"{index:03d}_{job['variant']}")
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--variant", action="append", choices=tuple(VARIANTS))
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("jobs must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    variants = args.variant or list(VARIANTS)
    jobs = jobs_for(args.output.resolve(), variants)
    source_paths = [
        Path(__file__),
        Path("scripts/x2_observer_ablation.py"),
        Path(release.__file__),
        Path(transition.__file__),
        Path("scripts/x2_integral_sim.py"),
        Path("scripts/x2_compliant_payload.py"),
        Path("scripts/x2_robustness_sim.py"),
        *Path("robojudo/utils/observers").glob("*.py"),
        Path("robojudo/config/x2/observer_parameters.py"),
    ]
    sources = args.output / "sources"
    sources.mkdir()
    hashes = {}
    for path in source_paths:
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        (sources / path.name).write_bytes(path.read_bytes())
    metadata = {
        "variants": {name: VARIANTS[name] for name in variants},
        "jobs": jobs,
        "growth_slew_nm_s": 2,
        "bandwidth_rad_s": 8,
        "joint_names": JOINT_NAMES,
        "hardware_validation": "not_performed",
        "source_sha256": hashes,
        "numpy_version": np.__version__,
        "mujoco_version": mujoco.__version__,
        "notes": "Deterministic nominal/clipped cases use one seed; impaired walking uses three. "
        "All factorial variants use zero-before-reversal; original is a separate reference. "
        "Transition harness condition/column names are retained; variant factors identify the actual controller.",
    }
    cfg = transition.transition_config()
    metadata["asset_sha256"] = {
        str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in (cfg.env.xml, cfg.loco_policy.policy_file)
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    results = []
    logging.getLogger("robojudo").setLevel(logging.WARNING)
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(worker, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            (args.output / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
            print(
                json.dumps(
                    {
                        "completed_jobs": len(results),
                        "total_jobs": len(jobs),
                        **{key: result[key] for key in ("study", "variant", "status", "wall_s")},
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
