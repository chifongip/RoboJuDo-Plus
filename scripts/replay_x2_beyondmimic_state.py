#!/usr/bin/env python3
"""Replay an X2 capture through the production BeyondMimic transition in MuJoCo."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from box import Box

import robojudo.pipeline
from robojudo.config.x2.env.x2_env_cfg import X2_31DoF
from robojudo.config.x2.policy.x2_beyondmimic_policy_cfg import X2BeyondMimicPolicyCfg
from robojudo.config.x2.x2_cfg import x2_locomimic_beyondmimic
from robojudo.environment.env_cfgs import SimulatedOdometryCfg
from robojudo.pipeline.four_mode_pipeline import ControlMode
from robojudo.pipeline.rl_loco_mimic_pipeline import PolicyInterpManager
from robojudo.pipeline.rl_pipeline import PolicyWrapper
from robojudo.tools.dof import merge_dof_cfgs
from robojudo.tools.x2_replay import (
    build_odometry_profile,
    grounded_mujoco_seed,
    load_capture,
    reconstruct_environment_frames,
    validate_and_select,
)
from robojudo.utils.util_func import get_gravity_orientation

MEASURED_JOINT_LIMIT_TOLERANCE = 0.10
JOINT_DEFAULT_STEPS = 75
EXPECTED_MIMIC_TRANSITION_STEPS = 101


@dataclass
class RolloutResult:
    name: str
    requested_active_steps: int
    completed_active_steps: int
    passed: bool
    failures: list[str]
    phase_steps: dict[str, int]
    preparation_support: str
    spawn: dict
    min_base_height: float
    max_tilt_radians: float
    max_pd_torque_ratio: float
    torque_saturated_joints: list[str]
    max_pd_target_limit_excess: float
    pd_target_clipped_joints: list[str]
    anchor_position_rmse: float
    anchor_orientation_rmse: float
    odometry_max_age: float


def _rmse(values: list[float]) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values else float("nan")


def _policy_cfg(policy_name: str) -> X2BeyondMimicPolicyCfg:
    return X2BeyondMimicPolicyCfg(
        policy_name=policy_name,
        max_timestep=-1,
        without_state_estimator=False,
    )


def audit_capture_and_immediate_launch(env_frames, policy_name: str) -> dict:
    """Gate captured state validity and retain direct activation as a diagnostic only."""
    base_dof = X2_31DoF()
    position_limits = np.asarray(base_dof.position_limits, dtype=np.float64)
    measured = np.asarray(env_frames[0].dof_pos, dtype=np.float64)
    low_excess = np.maximum(position_limits[:, 0] - measured, 0.0)
    high_excess = np.maximum(measured - position_limits[:, 1], 0.0)
    measured_excess = np.maximum(low_excess, high_excess)
    measured_violations = [
        {
            "joint": base_dof.joint_names[index],
            "position": float(measured[index]),
            "limits": position_limits[index].tolist(),
            "excess": float(measured_excess[index]),
        }
        for index in np.flatnonzero(measured_excess > 0.0)
    ]
    failures = [
        f"captured {item['joint']} exceeds its joint limit by {item['excess']:.4f}rad "
        f"(tolerance {MEASURED_JOINT_LIMIT_TOLERANCE:.4f}rad)"
        for item in measured_violations
        if item["excess"] > MEASURED_JOINT_LIMIT_TOLERANCE
    ]

    immediate = {"passed": False, "violations": [], "error": None}
    wrapper = None
    try:
        wrapper = PolicyWrapper(_policy_cfg(policy_name), base_dof, device="cpu")
        effective_dof = merge_dof_cfgs(base_dof, wrapper.cfg_action_dof)
        limits = np.asarray(effective_dof.position_limits, dtype=np.float64)
        observation, _ = wrapper.get_observation(env_frames[0], Box({}))
        target = np.asarray(wrapper.get_pd_target(observation), dtype=np.float64)
        if not np.isfinite(observation).all() or not np.isfinite(target).all():
            raise FloatingPointError("immediate-launch observation or target is non-finite")
        immediate["violations"] = [
            {
                "joint": base_dof.joint_names[index],
                "target": float(target[index]),
                "limits": limits[index].tolist(),
            }
            for index in np.flatnonzero((target < limits[:, 0]) | (target > limits[:, 1]))
        ]
        immediate["passed"] = not immediate["violations"]
    except Exception as exc:
        immediate["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if wrapper is not None:
            wrapper.close_progress()

    return {
        "passed": not failures,
        "failures": failures,
        "measured_joint_limit_tolerance_rad": MEASURED_JOINT_LIMIT_TOLERANCE,
        "measured_joint_limit_violations": measured_violations,
        "immediate_launch_diagnostic": immediate,
    }


def _make_pipeline(policy_name: str, visualize: bool):
    cfg = x2_locomimic_beyondmimic()
    cfg.env = cfg.env.model_copy(
        update={
            "sim_dt": 0.005,
            "sim_decimation": 4,
            "headless": not visualize,
            "visualize_extras": visualize,
            "clip_position_targets": True,
            "elastic_band": cfg.env.elastic_band.model_copy(update={"active": True, "visualize": visualize}),
            "random_heading": False,
            "initial_heading_degrees": None,
            "simulated_odometry": SimulatedOdometryCfg(enabled=True, update_rate_hz=10.0),
        }
    )
    cfg.mimic_policies = [_policy_cfg(policy_name)]
    cfg.ctrl = []
    cfg.do_safety_check = False
    cfg.run_fullspeed = True
    pipeline_class = getattr(robojudo.pipeline, cfg.pipeline_type)
    return pipeline_class(cfg=cfg)


def _seed_pipeline(pipeline, seed) -> dict:
    qpos, qvel, spawn = grounded_mujoco_seed(pipeline.env.model, pipeline.env.data, seed)
    pipeline.env.reborn(qpos, qvel)
    pipeline.env.reset_alignment()
    pipeline.policy_manager.reset_to_loco(refresh_env=True)
    pipeline.policy_locomotion_mimic_flag = 0
    return spawn


def _target_violation(target: np.ndarray, limits: np.ndarray) -> tuple[float, list[str]]:
    excess = np.maximum(np.maximum(limits[:, 0] - target, target - limits[:, 1]), 0.0)
    names = X2_31DoF().joint_names
    indices = np.flatnonzero(excess > 1e-8)
    return float(np.max(excess, initial=0.0)), [names[index] for index in indices]


def run_rollout(policy_name: str, seed, profile, steps: int, visualize: bool, name: str) -> RolloutResult:
    pipeline = _make_pipeline(policy_name, visualize)
    if profile is not None:
        pipeline.env.set_odometry_replay_profile(profile)
    spawn = _seed_pipeline(pipeline, seed)
    failures = []
    phase_steps = {"joint_default": 0, "mimic_transition": 0, "active_mimic": 0}
    position_errors = []
    orientation_errors = []
    ages = []
    min_height = float("inf")
    max_tilt = 0.0
    max_torque_ratio = 0.0
    max_target_excess = 0.0
    torque_saturated_joints = set()
    pd_target_clipped_joints = set()
    limits = np.asarray(pipeline.env.position_limits, dtype=np.float64)

    def run_step(phase: str):
        nonlocal min_height, max_tilt, max_torque_ratio, max_target_excess
        pre_position = pipeline.env.dof_pos
        pre_velocity = pipeline.env.dof_vel
        stiffness = pipeline.env.stiffness
        damping = pipeline.env.damping
        target, extras = pipeline.step()
        target = np.asarray(target, dtype=np.float64)
        if not np.isfinite(target).all():
            raise FloatingPointError(f"{phase}: non-finite PD target")
        max_excess, clipped_names = _target_violation(target, limits)
        max_target_excess = max(max_target_excess, max_excess)
        pd_target_clipped_joints.update(clipped_names)
        applied_target = np.clip(target, limits[:, 0], limits[:, 1])
        torque = stiffness * (applied_target - pre_position) - damping * pre_velocity
        ratio = np.abs(torque) / pipeline.env.torque_limits
        max_torque_ratio = max(max_torque_ratio, float(np.max(ratio)))
        torque_saturated_joints.update(pipeline.env.joint_names[index] for index in np.flatnonzero(ratio > 1.0 + 1e-6))

        tilt = float(np.arccos(np.clip(-get_gravity_orientation(pipeline.env.base_quat)[2], -1.0, 1.0)))
        height = float(pipeline.env.data.qpos[2])
        min_height = min(min_height, height)
        max_tilt = max(max_tilt, tilt)
        diagnostics = pipeline.env.odometry_diagnostics or {}
        if diagnostics.get("age") is not None:
            ages.append(float(diagnostics["age"]))
        if diagnostics.get("stale"):
            raise RuntimeError(f"{phase}: simulated odometry became stale")
        if tilt > 1.0:
            raise RuntimeError(f"{phase}: base tilt {tilt:.3f}rad exceeds 1.000rad")
        if height < 0.45:
            raise RuntimeError(f"{phase}: base height {height:.3f}m is below 0.450m")
        return extras

    try:
        if not (0.0 <= spawn["minimum_foot_clearance"] <= 0.01):
            raise RuntimeError(
                f"physical seed foot clearance {spawn['minimum_foot_clearance']:.4f}m is outside [0, 0.01]m"
            )
        if spawn["root_height"] < 0.45:
            raise RuntimeError(f"physical seed root height {spawn['root_height']:.3f}m is below 0.450m")

        if not pipeline._enter_mode(ControlMode.JOINT_DEFAULT):
            raise RuntimeError("failed to enter JOINT_DEFAULT")
        while not pipeline._joint_default_complete and phase_steps["joint_default"] < JOINT_DEFAULT_STEPS + 1:
            run_step("joint_default")
            phase_steps["joint_default"] += 1
        if phase_steps["joint_default"] != JOINT_DEFAULT_STEPS:
            raise RuntimeError(
                f"JOINT_DEFAULT completed in {phase_steps['joint_default']} steps, expected {JOINT_DEFAULT_STEPS}"
            )
        if not pipeline._enter_mode(ControlMode.RL_DEFAULT):
            raise RuntimeError("failed to enter RL_DEFAULT")
        if not pipeline.policy_manager.switch_to_mimic():
            raise RuntimeError("failed to request the BeyondMimic policy")
        pipeline.policy_locomotion_mimic_flag = 1

        mimic_id = pipeline.policy_manager.policy_mimic_ids[0]
        while phase_steps["mimic_transition"] <= EXPECTED_MIMIC_TRANSITION_STEPS:
            run_step("mimic_transition")
            phase_steps["mimic_transition"] += 1
            if (
                pipeline.policy_manager.current_policy_id == mimic_id
                and pipeline.policy_manager.interp_state == PolicyInterpManager.InterpState.IDLE
            ):
                break
        if phase_steps["mimic_transition"] != EXPECTED_MIMIC_TRANSITION_STEPS:
            raise RuntimeError(
                f"mimic transition completed in {phase_steps['mimic_transition']} steps, "
                f"expected {EXPECTED_MIMIC_TRANSITION_STEPS}"
            )
        if pipeline.policy.robot_anchor_align is None:
            raise RuntimeError("BeyondMimic anchor was not realigned at policy activation")

        # The capture was made while the real robot was mechanically supported.
        # Match that condition only while reaching the production launch pose;
        # all active BeyondMimic steps below are unsupported.
        pipeline.env.elastic_band.active = False
        pipeline.env.data.xfrc_applied[pipeline.env.elastic_band.body_id, :3] = 0.0
        for _ in range(steps):
            extras = run_step("active_mimic")
            phase_steps["active_mimic"] += 1
            position_errors.append(float(np.linalg.norm(extras["pos"])))
            orientation_errors.append(float(2.0 * np.arccos(np.clip(abs(extras["ori"][3]), 0.0, 1.0))))
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
    finally:
        for policy in pipeline.policy_manager.policies:
            close_progress = getattr(policy, "close_progress", None)
            if close_progress is not None:
                close_progress()
        pipeline.env.shutdown()

    return RolloutResult(
        name=name,
        requested_active_steps=steps,
        completed_active_steps=phase_steps["active_mimic"],
        passed=not failures and phase_steps["active_mimic"] == steps,
        failures=failures,
        phase_steps=phase_steps,
        preparation_support="production_sim_elastic_band_until_mimic_activation",
        spawn=spawn,
        min_base_height=min_height,
        max_tilt_radians=max_tilt,
        max_pd_torque_ratio=max_torque_ratio,
        torque_saturated_joints=sorted(torque_saturated_joints),
        max_pd_target_limit_excess=max_target_excess,
        pd_target_clipped_joints=sorted(pd_target_clipped_joints),
        anchor_position_rmse=_rmse(position_errors),
        anchor_orientation_rmse=_rmse(orientation_errors),
        odometry_max_age=max(ages, default=float("nan")),
    )


def _write_report(output: Path, report: dict):
    def json_safe(value):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        return value

    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    lines = [
        "# X2 real-state BeyondMimic safety replay",
        "",
        f"Overall gate: **{'PASS' if report['passed'] else 'FAIL'}**",
        "",
        "The capture stage was subscriber-only. No real-robot command transport was instantiated.",
        "",
        f"- Captured-state audit: {'PASS' if report['capture_audit']['passed'] else 'FAIL'}",
        "- Immediate launch: diagnostic only",
        f"- Ideal 10 Hz production sequence: {'PASS' if report['rollouts']['ideal_10hz']['passed'] else 'FAIL'}",
        f"- Captured-profile production sequence: "
        f"{'PASS' if report['rollouts']['captured_profile']['passed'] else 'FAIL'}",
        "",
        "## Gate failures",
        "",
    ]
    if report["failures"]:
        lines.extend(f"- {failure}" for failure in report["failures"])
    else:
        lines.append("- None")
    lines.extend(["", "## Non-gating diagnostics", ""])
    lines.extend(f"- {warning}" for warning in report["warnings"])
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--policy-name", default="Solo_dance")
    parser.add_argument("--steps", type=int, default=100, help="active 50 Hz BeyondMimic steps after preparation")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/x2_real_state_replay"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    records = load_capture(args.capture)
    selection = validate_and_select(records, duration=2.0)
    env_frames, seed = reconstruct_environment_frames(selection)
    audit = audit_capture_and_immediate_launch(env_frames, args.policy_name)
    total_duration = (JOINT_DEFAULT_STEPS + EXPECTED_MIMIC_TRANSITION_STEPS + args.steps) / 50.0
    profile = build_odometry_profile(selection, records, total_duration)
    ideal = run_rollout(args.policy_name, seed, None, args.steps, args.visualize, "ideal_10hz")
    captured = run_rollout(
        args.policy_name,
        seed,
        profile,
        args.steps,
        args.visualize,
        "captured_profile",
    )
    failures = list(dict.fromkeys([*audit["failures"], *ideal.failures, *captured.failures]))
    if ideal.passed and captured.passed:
        position_limit = ideal.anchor_position_rmse * 1.25 + 0.02
        orientation_limit = ideal.anchor_orientation_rmse * 1.15 + 0.05
        if captured.anchor_position_rmse > position_limit:
            failures.append(
                f"captured-profile anchor position RMSE {captured.anchor_position_rmse:.4f}m "
                f"exceeds {position_limit:.4f}m"
            )
        if captured.anchor_orientation_rmse > orientation_limit:
            failures.append(
                f"captured-profile anchor orientation RMSE {captured.anchor_orientation_rmse:.4f}rad "
                f"exceeds {orientation_limit:.4f}rad"
            )
    warnings = []
    if audit["measured_joint_limit_violations"]:
        warnings.append(
            f"{len(audit['measured_joint_limit_violations'])} captured joints were outside nominal limits "
            f"but within the {MEASURED_JOINT_LIMIT_TOLERANCE:.2f} rad capture tolerance"
        )
    if not audit["immediate_launch_diagnostic"]["passed"]:
        warnings.append("immediate BeyondMimic activation produced invalid targets; use the production transition")
    if ideal.spawn["maximum_foot_clearance"] - ideal.spawn["minimum_foot_clearance"] > 0.02:
        warnings.append("the supported captured posture was not a two-foot self-supporting MuJoCo seed")
    for rollout in (ideal, captured):
        if rollout.pd_target_clipped_joints:
            warnings.append(
                f"{rollout.name}: position targets were clamped on {rollout.pd_target_clipped_joints}; "
                f"maximum excess {rollout.max_pd_target_limit_excess:.3f}rad"
            )
        if rollout.torque_saturated_joints:
            warnings.append(
                f"{rollout.name}: torque saturated on {rollout.torque_saturated_joints}; "
                f"maximum demand ratio {rollout.max_pd_torque_ratio:.3f}"
            )
    report = {
        "passed": not failures,
        "capture": {
            **selection.diagnostics,
            "raw_converted_odometry_origin": seed.odometry_origin_position.tolist(),
            "rebased_root_position": seed.root_position.tolist(),
        },
        "policy": args.policy_name,
        "active_steps": args.steps,
        "capture_audit": audit,
        "rollouts": {
            "ideal_10hz": asdict(ideal),
            "captured_profile": asdict(captured),
        },
        "comparison_limits": {
            "position_multiplier": 1.25,
            "position_allowance_m": 0.02,
            "orientation_multiplier": 1.15,
            "orientation_allowance_rad": 0.05,
        },
        "warnings": warnings,
        "failures": failures,
    }
    _write_report(args.output, report)
    print(f"Safety gate: {'PASS' if report['passed'] else 'FAIL'}; reports: {args.output}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
