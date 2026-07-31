#!/usr/bin/env python3
"""Evaluate low-rate odometry for the state-estimator BeyondMimic export in MuJoCo."""

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from robojudo.config.x2.policy.x2_beyondmimic_policy_cfg import X2BeyondMimicPolicyCfg
from robojudo.config.x2.x2_cfg import x2_beyondmimic
from robojudo.environment.env_cfgs import SimulatedOdometryCfg
from robojudo.pipeline.rl_pipeline import RlPipeline
from robojudo.utils.util_func import get_gravity_orientation, quat_rotate_inverse_np

SCENARIOS = {
    "nominal": {},
    "latency_jitter": {"latency": 0.04, "jitter": 0.02},
    "noise": {"position_noise_std": (0.02, 0.02, 0.01), "yaw_noise_std": math.radians(1.0)},
    "dropout": {"dropout_probability": 0.1},
    "combined": {
        "latency": 0.04,
        "jitter": 0.02,
        "position_noise_std": (0.02, 0.02, 0.01),
        "yaw_noise_std": math.radians(1.0),
        "dropout_probability": 0.1,
        "degeneracy_windows": [(2.0, 2.2)],
    },
    "timeout": {"degeneracy_windows": [(2.0, 2.4)]},
}


@dataclass
class Result:
    scenario: str
    rate_hz: float
    heading_degrees: float
    seed: int
    requested_steps: int
    completed_steps: int
    fallen: bool
    stale_abort: bool
    error: str | None
    min_base_height: float
    max_tilt_radians: float
    anchor_position_rmse: float
    anchor_orientation_rmse: float
    odometry_position_rmse: float
    odometry_velocity_rmse: float
    odometry_max_age: float
    delivered_samples: int
    dropped_samples: int


def _rmse(values: list[float]) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values else float("nan")


def _make_pipeline(rate: float, heading: float, seed: int, scenario: str, visualize: bool) -> RlPipeline:
    cfg = x2_beyondmimic()
    odometry = SimulatedOdometryCfg(
        enabled=True,
        update_rate_hz=rate,
        random_seed=seed,
        **SCENARIOS[scenario],
    )
    cfg.env = cfg.env.model_copy(
        update={
            "sim_dt": 0.005,
            "sim_decimation": 4,
            "headless": not visualize,
            "visualize_extras": visualize,
            "elastic_band": cfg.env.elastic_band.model_copy(update={"active": False, "visualize": False}),
            "random_heading": False,
            "initial_heading_degrees": heading,
            "simulated_odometry": odometry,
        }
    )
    cfg.policy = X2BeyondMimicPolicyCfg(
        max_timestep=6747,
        policy_name="Solo_dance",
        without_state_estimator=False,
    )
    cfg.ctrl = []
    cfg.do_safety_check = False
    cfg.run_fullspeed = True
    return RlPipeline(cfg=cfg)


def run_case(rate: float, heading: float, seed: int, scenario: str, steps: int, visualize: bool = False) -> Result:
    pipeline = _make_pipeline(rate, heading, seed, scenario, visualize)
    anchor_position_errors: list[float] = []
    anchor_orientation_errors: list[float] = []
    odometry_position_errors: list[float] = []
    odometry_velocity_errors: list[float] = []
    odometry_ages: list[float] = []
    min_height = float("inf")
    max_tilt = 0.0
    fallen = False
    stale_abort = False
    error = None
    completed = 0

    try:
        for step in range(steps):
            step_start = time.monotonic()
            pipeline.env.update()
            env_data = pipeline.env.get_data()
            ctrl_data = pipeline.ctrl_manager.get_ctrl_data(env_data)
            obs, extras = pipeline.policy.get_observation(env_data, ctrl_data)
            pd_target = pipeline.policy.get_pd_target(obs)
            if pipeline.env.visualizer is not None:
                pipeline.policy.debug_viz(pipeline.env.visualizer, env_data, ctrl_data, extras)
            pipeline.env.step(pd_target, extras.get("hand_pose"))
            pipeline.policy.post_step_callback([])
            completed = step + 1

            gravity = get_gravity_orientation(env_data.base_quat)
            tilt = float(np.arccos(np.clip(-gravity[2], -1.0, 1.0)))
            height = float(pipeline.env.data.qpos[2])
            max_tilt = max(max_tilt, tilt)
            min_height = min(min_height, height)
            anchor_position_errors.append(float(np.linalg.norm(extras["pos"])))
            anchor_orientation_errors.append(float(2.0 * np.arccos(np.clip(abs(extras["ori"][3]), 0.0, 1.0))))

            raw_pos = pipeline.env.data.qpos[:3]
            aligned_pos = pipeline.env.base_align.align_pos(raw_pos)
            odometry_position_errors.append(float(np.linalg.norm(env_data.base_pos - aligned_pos)))
            raw_quat = pipeline.env.data.qpos[3:7][[1, 2, 3, 0]]
            perfect_velocity = quat_rotate_inverse_np(raw_quat, pipeline.env.data.qvel[:3])
            odometry_velocity_errors.append(float(np.linalg.norm(env_data.base_lin_vel - perfect_velocity)))
            diagnostics = pipeline.env.odometry_diagnostics
            if diagnostics is not None and diagnostics["age"] is not None:
                odometry_ages.append(float(diagnostics["age"]))

            if tilt > 1.0 or height < 0.45:
                fallen = True
                break
            if visualize:
                remaining = pipeline.dt - (time.monotonic() - step_start)
                if remaining > 0.0:
                    time.sleep(remaining)
    except RuntimeError as exc:
        error = str(exc)
        stale_abort = "odometry became stale" in error.lower()
    finally:
        diagnostics = pipeline.env.odometry_diagnostics or {}
        pipeline.policy.close_progress()
        pipeline.env.shutdown()

    return Result(
        scenario=scenario,
        rate_hz=rate,
        heading_degrees=heading,
        seed=seed,
        requested_steps=steps,
        completed_steps=completed,
        fallen=fallen,
        stale_abort=stale_abort,
        error=error,
        min_base_height=min_height,
        max_tilt_radians=max_tilt,
        anchor_position_rmse=_rmse(anchor_position_errors),
        anchor_orientation_rmse=_rmse(anchor_orientation_errors),
        odometry_position_rmse=_rmse(odometry_position_errors),
        odometry_velocity_rmse=_rmse(odometry_velocity_errors),
        odometry_max_age=max(odometry_ages, default=float("nan")),
        delivered_samples=int(diagnostics.get("delivered", 0)),
        dropped_samples=int(diagnostics.get("dropped", 0)),
    )


def _gate(results: list[Result]) -> tuple[bool, list[str]]:
    failures = []
    nominal_10 = [result for result in results if result.scenario == "nominal" and result.rate_hz == 10.0]
    baseline = [result for result in results if result.scenario == "nominal" and result.rate_hz == 50.0]
    baseline_position = (
        float(np.nanmean([result.anchor_position_rmse for result in baseline])) if baseline else float("nan")
    )
    baseline_orientation = (
        float(np.nanmean([result.anchor_orientation_rmse for result in baseline])) if baseline else float("nan")
    )

    for result in nominal_10:
        if result.fallen or result.error:
            failures.append(f"10 Hz nominal failed at heading={result.heading_degrees}, seed={result.seed}")
        if result.anchor_position_rmse > baseline_position * 1.25 + 0.02:
            failures.append(f"10 Hz anchor-position regression at heading={result.heading_degrees}, seed={result.seed}")
        if result.anchor_orientation_rmse > baseline_orientation * 1.15 + 0.05:
            failures.append(
                f"10 Hz anchor-orientation regression at heading={result.heading_degrees}, seed={result.seed}"
            )
        if result.odometry_position_rmse > 0.08 or result.odometry_velocity_rmse > 0.35:
            failures.append(
                f"10 Hz estimator error exceeded limit at heading={result.heading_degrees}, seed={result.seed}"
            )

    for result in results:
        if result.scenario == "timeout":
            if not result.stale_abort:
                failures.append(
                    f"timeout case did not abort safely at heading={result.heading_degrees}, seed={result.seed}"
                )
        elif result.scenario != "nominal" and (result.fallen or result.error):
            failures.append(f"{result.scenario} failed at heading={result.heading_degrees}, seed={result.seed}")
    return not failures, failures


def _write_reports(output: Path, results: list[Result], passed: bool, failures: list[str]):
    output.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    report = {"passed": passed, "failures": failures, "results": rows}
    (output / "results.json").write_text(json.dumps(report, indent=2))
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# BeyondMimic odometry benchmark",
        "",
        f"Overall safety gate: **{'PASS' if passed else 'FAIL'}**",
        "",
        "| Scenario | Rate | Heading | Seed | Steps | Fallen | Stale abort | Position RMSE | Velocity RMSE |",
        "|---|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.scenario} | {result.rate_hz:g} | {result.heading_degrees:g} | {result.seed} | "
            f"{result.completed_steps}/{result.requested_steps} | {result.fallen} | {result.stale_abort} | "
            f"{result.odometry_position_rmse:.4f} | {result.odometry_velocity_rmse:.4f} |"
        )
    if failures:
        lines.extend(["", "## Gate failures", "", *[f"- {failure}" for failure in failures]])
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates", type=float, nargs="+", default=[50, 20, 10, 5])
    parser.add_argument("--headings", type=float, nargs="+", default=[0, 90, 180, 270])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--steps", type=int, default=6747)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/beyondmimic_odometry"))
    parser.add_argument("--smoke", action="store_true", help="Run a short 50/10 Hz, 0/180 degree deterministic check")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open the MuJoCo viewer, draw BeyondMimic anchors, and run in real time",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.rates = [50.0, 10.0]
        args.headings = [0.0, 180.0]
        args.seeds = [0]
        args.scenarios = ["nominal", "timeout"]
        args.steps = min(args.steps, 250)

    results = []
    for scenario in args.scenarios:
        rates = [10.0] if scenario != "nominal" else args.rates
        for rate in rates:
            for heading in args.headings:
                for seed in args.seeds:
                    result = run_case(rate, heading, seed, scenario, args.steps, visualize=args.visualize)
                    results.append(result)
                    print(
                        f"{scenario} {rate:g}Hz heading={heading:g} seed={seed}: "
                        f"{result.completed_steps}/{result.requested_steps}, "
                        f"fallen={result.fallen}, stale_abort={result.stale_abort}"
                    )

    passed, failures = _gate(results)
    _write_reports(args.output, results, passed, failures)
    print(f"Safety gate: {'PASS' if passed else 'FAIL'}; reports: {args.output}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
