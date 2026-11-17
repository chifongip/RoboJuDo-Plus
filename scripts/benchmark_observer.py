"""Offline observer timing with synthetic feedback; never connects to hardware.

Run with PYTHONPATH=. python scripts/benchmark_observer.py --profile /tmp/observer.prof
Timing excludes physics and policy inference and is not a deployment deadline guarantee.
"""

import argparse
import cProfile
import json
import platform
import time
from pathlib import Path

import numpy as np

from robojudo.config.x2 import x2_locomimic_observer
from robojudo.utils.observers import JointObserver


def workload(kind, steps, warmup, backend="auto"):
    cfg = x2_locomimic_observer().env
    cfg.observer.backend = backend
    dof = cfg.dof
    construction_start = time.perf_counter_ns()
    ctrl = JointObserver(
        cfg.observer,
        dof.joint_names,
        dof.stiffness,
        dof.damping,
        actuator_limits=dof.torque_limits,
        max_dt=0.1,
    )
    construction_ms = (time.perf_counter_ns() - construction_start) / 1e6
    size = len(dof.joint_names)
    samples = []
    histories = []
    for step in range(steps + warmup):
        now = 100 + step * 0.02
        goal = np.asarray(dof.default_pos) + 0.1 * np.sin(step * 0.13)
        q = np.asarray(dof.default_pos) + 0.03 * np.sin(step * 0.11 + np.arange(size))
        v = 0.165 * np.cos(step * 0.11 + np.arange(size))
        # Delayed per-joint feedback spans several accepted commands. Every
        # fifth sample repeats; all generated timestamps remain monotone.
        sample_step = step - 1 if step % 5 == 4 else step
        stamps = 100 + sample_step * 0.02 - np.linspace(0.001, 0.035, size)
        start = time.perf_counter_ns()
        effort = ctrl.propose(goal, q, v, now, sample_times=None if kind == "synchronous" else stamps)
        ctrl.acknowledge(effort * (0.5 if step % 9 == 0 else 1), command_time=None if kind == "synchronous" else now)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        if step >= warmup:
            samples.append(elapsed)
            histories.append(len(ctrl.command_history))
    return {
        "samples": steps,
        "backend": ctrl.observer.backend,
        "construction_ms": construction_ms,
        "latency_ms": dict(
            zip(("p50", "p95", "p99", "max"), map(float, np.percentile(samples, [50, 95, 99, 100])), strict=True)
        ),
        "over_20_ms": int(np.count_nonzero(np.asarray(samples) > 20)),
        "max_retained_commands": max(histories),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("auto", "numpy", "numba"), default="auto")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", type=Path, help="Separate profiled run; excluded from latency measurements")
    args = parser.parse_args()
    if args.steps <= 0 or args.warmup < 0:
        parser.error("steps must be positive and warmup nonnegative")
    report = {"platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__}
    for kind in ("synchronous", "timestamped"):
        report[kind] = workload(kind, args.steps, args.warmup, args.backend)
    result = json.dumps(report, indent=2)
    print(result)
    if args.output:
        args.output.write_text(result + "\n")
    if args.profile:
        profiler = cProfile.Profile()
        profiler.runcall(workload, "timestamped", args.steps, args.warmup, args.backend)
        profiler.dump_stats(str(args.profile))


if __name__ == "__main__":
    main()
