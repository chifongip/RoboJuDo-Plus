"""Abrupt unloading through the deployment wrapper, using local and MuJoCo plants only.

No native adapter, hardware transport, or network controller is constructed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from x2_integral_sim import (
    SimulationFailure,
    X2IntegralEvaluationEnv,
    evaluation_config,
    point_wrench,
    synthetic_velocity,
)

from robojudo.config.observer_cfg import JointObserverCfg
from robojudo.config.x2.observer_parameters import JOINT_NAMES, selected_observer_parameters
from robojudo.config.x2.x2_cfg import x2_locomimic
from robojudo.environment import env_registry
from robojudo.environment.mujoco_env import MujocoEnv
from robojudo.pipeline.four_mode_pipeline import ControlMode
from robojudo.pipeline.x2_locomanipulation_loco_mimic_pipeline import X2LocomanipulationLocoMimicPipeline
from robojudo.utils.observers import JointObserver

LOAD_START, RELEASE, DURATION = 1.0, 7.0, 12.0


def runtime(condition, release_rate, names, kp, kd, limits):
    indices = np.array([names.index(n) for n in JOINT_NAMES])
    parameters = selected_observer_parameters()
    parameters.pop("method")
    cfg = JointObserverCfg(
        algorithm="linearized" if condition == "legacy" else "nonlinear",
        joint_names=list(JOINT_NAMES),
        **parameters,
        mode="ACTIVE",
        actuator_limits_nm=np.asarray(limits)[indices].tolist(),
        # Preserve the bounds used by the historical release comparison.
        correction_limit_fraction=0.8,
        correction_limit_nm=8.0,
        slew_nm_s=2,
        release_slew_nm_s=None if condition == "legacy" else release_rate,
    )
    return JointObserver(cfg, names, kp, kd, actuator_limits=limits, max_dt=0.1)


class SimulationAdapter:
    """A command receipt boundary with optional downstream correction clipping."""

    def __init__(self, wrapper, clip=None):
        self.wrapper, self.clip = wrapper, clip

    def issue(self, target, q, v, now, stamps):
        started = time.perf_counter()
        proposed = self.wrapper.propose(target, q, v, now, sample_times=stamps)
        issued = proposed if self.clip is None else np.clip(proposed, -self.clip, self.clip)
        self.wrapper.acknowledge(issued, command_time=now)
        return issued, time.perf_counter() - started


def observer_trace(wrapper):
    model = wrapper.observer
    return {
        "estimate": model.estimated_disturbance_nm.copy(),
        "desired": getattr(model, "desired_nm", -model.estimated_disturbance_nm).copy(),
        "proposed": wrapper.proposed_nm.copy(),
        "headroom_limited_effort": wrapper.limited_nm.copy(),
        "prediction": model.velocity_prediction.copy(),
        "innovation": model.velocity_innovation.copy(),
        "instantaneous": getattr(model, "instantaneous_disturbance_nm", np.full(14, np.nan)).copy(),
    }


def metrics(trace, unloaded):
    t, q, effort = trace["t"], trace["q"], trace["effort"]
    carry = (t >= RELEASE - 2) & (t < RELEASE)
    after = t >= RELEASE - 1e-8
    error = q - trace["target"]
    # Matched unloaded motion removes gravity/model bias and gait oscillation.
    dq = q - unloaded["q"]
    release_times = t[after] - RELEASE
    residual = np.max(abs(effort[after] - unloaded["effort"][after]), axis=1)
    threshold = max(0.1 * float(np.max(abs(effort[carry] - unloaded["effort"][carry]))), 0.04)

    def sustained(values, bound):
        outside = np.flatnonzero(values > bound)
        index = int(outside[-1] + 1) if len(outside) else 0
        if index >= len(release_times) or release_times[-1] - release_times[index] < 1:
            return None
        return float(max(0, release_times[index]))

    result = {
        "carry_rms_deg": float(np.rad2deg(np.sqrt(np.mean(error[carry] ** 2)))),
        "release_positive_peak_deg": float(np.rad2deg(np.max(dq[after]))),
        "release_negative_peak_deg": float(np.rad2deg(np.min(dq[after]))),
        "release_peak_deg": float(np.rad2deg(np.max(abs(dq[after])))),
        "torque_removal_s": sustained(residual, threshold),
        "torque_threshold_nm": threshold,
        "joint_settling_s": sustained(np.max(abs(dq[after]), axis=1), np.deg2rad(0.5)),
        "carry_ripple_nm": float(np.max(np.std(effort[carry], axis=0))),
        "estimate_bound_fraction": float(np.mean(trace["bound"][carry])),
        "limited_fraction": float(np.mean(trace["limited"][carry])),
        "observer_p99_ms": float(1000 * np.percentile(trace["wall_s"], 99)),
        "observer_max_ms": float(1000 * np.max(trace["wall_s"])),
    }
    if "hand" in trace:
        hand = trace["hand"] - unloaded["hand"]
        result["release_hand_peak_mm"] = float(1000 * np.max(np.linalg.norm(hand[after], axis=2)))
        result["hand_settling_s"] = sustained(np.max(np.linalg.norm(hand[after], axis=2), axis=1), 0.005)
        result["max_base_tilt_deg"] = float(np.rad2deg(np.max(trace["tilt"])))
        result["saturation_fraction"] = float(np.mean(np.any(trace["saturated"], axis=1)))
    return result


def local_trial(condition, rate, load, case, seed):
    cfg = x2_locomimic().env.dof
    names, kp, kd, limits = cfg.joint_names, cfg.stiffness, cfg.damping, cfg.torque_limits
    wrapper = runtime(condition, rate, names, kp, kd, limits)
    adapter = SimulationAdapter(wrapper, clip=1 if case == "clipped" else None)
    arm = wrapper.indices
    q, v, goal = np.zeros(len(names)), np.zeros(len(names)), np.zeros(len(names))
    rng = np.random.default_rng(seed)
    records, samples = [], []
    received = None
    effort = np.zeros(14)
    for step in range(round(DURATION / 0.02) + 1):
        elapsed = step * 0.02
        now = 100 + elapsed
        acquired_q, acquired_v = q.copy(), v.copy()
        if case == "impaired":
            acquired_v[arm] = np.round((acquired_v[arm] + rng.normal(0, 0.005, 14)) / 0.02442) * 0.02442
        samples.append((now, acquired_q, acquired_v))
        lag = 1 if case == "impaired" else 0
        dropped = case == "impaired" and (7 <= elapsed < 7.06 or step % 31 == 0)
        if received is None or not dropped:
            received = samples[max(0, len(samples) - 1 - lag)]
        stamp, measured_q, measured_v = received
        issued, wall = adapter.issue(goal, measured_q, measured_v, now, np.full(len(names), stamp))
        effort = issued if condition != "pd" else np.zeros(14)
        if condition == "pd":
            wrapper.acknowledge(np.zeros(14), command_time=now)
        model = wrapper.observer
        records.append(
            dict(
                t=elapsed,
                q=q[arm].copy(),
                target=goal[arm].copy(),
                effort=effort.copy(),
                wall_s=wall,
                bound=abs(model.estimated_disturbance_nm) >= model.limits - 1e-3,
                limited=model.antiwindup.copy(),
                **observer_trace(wrapper),
            )
        )
        disturbance = np.zeros(14)
        if LOAD_START <= elapsed < RELEASE:
            disturbance[0] = load
            disturbance[7] = -load * 0.5
        elif case == "reversal" and RELEASE <= elapsed:
            disturbance[0] = -load
            disturbance[7] = load * 0.5
        q[arm], v[arm], _, _ = model.predict_state(q[arm], v[arm], goal[arm], effort, disturbance, 0.02)
    return {key: np.asarray([r[key] for r in records]) for key in records[0]}


@env_registry.register
class X2ReleaseEvaluationEnv(X2IntegralEvaluationEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.release_adapter = None
        self.records = []
        self.wall_s = 0.0

    def configure(self, condition, rate, load, case, seed):
        self.condition, self.load, self.case = condition, load, case
        self.rng = np.random.default_rng(seed)
        self.release_adapter = SimulationAdapter(
            runtime(condition, rate, self.joint_names, self.stiffness, self.damping, self.torque_limits),
            clip=1 if case == "clipped" else None,
        )
        self.samples = []
        self.received = None
        self.measurement_start = self.data.time
        self.records = []

    def step(self, pd_target, hand_pose=None, *, feedforward_effort=None):
        if feedforward_effort is not None:
            raise ValueError("Release evaluation owns feedforward effort")
        self.last_target = np.clip(np.asarray(pd_target), *self.position_limits.T)
        effort = np.zeros(self.num_dofs)
        if self.release_adapter is not None:
            now = 100 + self.data.time
            acquired_q, acquired_v = self.dof_pos.copy(), self.dof_vel.copy()
            if self.case == "impaired":
                acquired_v[self.arm_indices] = (
                    np.round((acquired_v[self.arm_indices] + self.rng.normal(0, 0.005, 14)) / 0.02442) * 0.02442
                )
            self.samples.append((now, acquired_q, acquired_v))
            lag = 1 if self.case == "impaired" else 0
            elapsed = round(float(self.data.time - self.measurement_start), 9)
            dropped = self.case == "impaired" and 7 <= elapsed < 7.06
            if self.received is None or not dropped:
                self.received = self.samples[max(0, len(self.samples) - 1 - lag)]
            stamp, q, v = self.received
            issued, self.wall_s = self.release_adapter.issue(self.last_target, q, v, now, np.full(self.num_dofs, stamp))
            effort[self.release_adapter.wrapper.indices] = issued
            if self.condition == "pd":
                effort[:] = 0
                self.release_adapter.wrapper.acknowledge(np.zeros(14), command_time=now)
            self.samples = self.samples[-4:]
        MujocoEnv.step(self, self.last_target, hand_pose, feedforward_effort=effort)

    def update_load(self):
        mujoco.mj_fwdPosition(self.model, self.data)
        self.check_state()
        if self.release_adapter is None:
            return
        elapsed = round(float(self.data.time - self.measurement_start), 9)
        for i, body_id in enumerate(self.ee_ids):
            force = (
                np.array([0, 0, self.load * (1 if i == 0 else 0.5)]) if LOAD_START <= elapsed < RELEASE else np.zeros(3)
            )
            self.data.xfrc_applied[body_id] = point_wrench(self.data.xipos[body_id], self.data.xipos[body_id], force)
        wrapper = self.release_adapter.wrapper
        indices, model = wrapper.indices, wrapper.observer
        _, hand = self.end_effector_positions()
        tilt = np.arccos(np.clip(self.data.xmat[self.pelvis_id].reshape(3, 3)[2, 2], -1, 1))
        self.records.append(
            dict(
                t=elapsed,
                q=self.dof_pos[indices].copy(),
                target=self.last_target[indices].copy(),
                hand=hand.copy(),
                tilt=tilt,
                effort=wrapper.issued_nm.copy(),
                wall_s=self.wall_s,
                bound=abs(model.estimated_disturbance_nm) >= model.limits - 1e-3,
                limited=model.antiwindup.copy(),
                saturated=abs(
                    self.stiffness * (self.last_target - self.dof_pos)
                    - self.damping * self.dof_vel
                    + np.array(
                        [wrapper.issued_nm[JOINT_NAMES.index(n)] if n in JOINT_NAMES else 0 for n in self.joint_names]
                    )
                )
                > self.torque_limits,
                **observer_trace(wrapper),
            )
        )


def mujoco_trial(condition, rate, load, case, seed, walking):
    np.random.seed(seed)
    cfg = evaluation_config()
    cfg.env.env_type = "X2ReleaseEvaluationEnv"
    cfg.env.sim_dt, cfg.env.sim_decimation = 0.005, 4
    cfg.env.feedforward_effort_limits = {
        n: min(8, 0.8 * cfg.env.dof.torque_limits[cfg.env.dof.joint_names.index(n)]) for n in JOINT_NAMES
    }
    pipeline = X2LocomanipulationLocoMimicPipeline(cfg)
    for policy in pipeline.policy_manager.policies:
        if hasattr(policy, "close_progress"):
            policy.close_progress()
    env = pipeline.env
    command = [0.0]
    pipeline.ctrl_manager.get_ctrl_data = lambda _: synthetic_velocity(command[0])
    try:
        pipeline._manual_mode_override = None
        env.elastic_band.active = True
        pipeline._enter_mode(ControlMode.JOINT_DEFAULT)
        for _ in range(pipeline._joint_default_steps):
            pipeline.step()
        if not pipeline._enter_mode(ControlMode.RL_DEFAULT):
            raise SimulationFailure("could not enter RL_DEFAULT")
        env.elastic_band.active = False
        env.elastic_band.apply()
        for _ in range(round(5 / pipeline.dt)):
            pipeline.step()
        command[0] = 0.3 if walking else 0
        for _ in range(round(3 / pipeline.dt)):
            pipeline.step()
        env.configure(condition, rate, load, case, seed)
        for _ in range(round(DURATION / pipeline.dt)):
            pipeline.step()
            if pipeline.mode != ControlMode.RL_DEFAULT:
                raise SimulationFailure(f"policy left RL_DEFAULT: {pipeline.mode}")
        return {key: np.asarray([r[key] for r in env.records]) for key in env.records[0]}
    finally:
        pipeline.ctrl_manager.close()
        env.shutdown()
        mujoco.mj_clearCache(mujoco.mj_getCache())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plant", choices=("local", "mujoco"), default="local")
    parser.add_argument("--rates", type=float, nargs="+", default=[2, 8, 16, 20])
    parser.add_argument(
        "--cases", nargs="+", choices=("nominal", "impaired", "clipped", "reversal"), default=["nominal"]
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--load", type=float, default=-4, help="Nm for local, N per left hand for MuJoCo")
    parser.add_argument("--walking", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(not np.isfinite(r) or r <= 0 for r in args.rates) or not np.isfinite(args.load):
        parser.error("finite load and positive finite rates required")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "joint_names": list(JOINT_NAMES),
        "load_start_s": LOAD_START,
        "release_s": RELEASE,
        "duration_s": DURATION,
        "control_dt_s": 0.02,
        "physics_dt_s": 0.005 if args.plant == "mujoco" else None,
        "observer_parameters": selected_observer_parameters(),
        "growth_slew_nm_s": 2,
        "numpy_version": np.__version__,
        "mujoco_version": mujoco.__version__,
        "hardware_qualified": False,
        "plant": args.plant,
        "load": args.load,
        "walking": args.walking,
        "seeds": args.seeds,
        "cases": args.cases,
        "release_rates_nm_s": args.rates,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    results = []
    for case in args.cases:
        if case == "reversal" and args.plant == "mujoco":
            parser.error("reversal is a local-model diagnostic")
        for seed in args.seeds:
            for condition, rate in [("pd", 2), ("legacy", 2), *[("improved", r) for r in args.rates]]:
                traces = []
                entry = dict(
                    case=case,
                    seed=seed,
                    condition=condition,
                    rate=rate,
                    plant=args.plant,
                    walking=args.walking,
                    load=args.load,
                )
                try:
                    for load in (0, args.load):
                        trace = (
                            local_trial(condition, rate, load, case, seed)
                            if args.plant == "local"
                            else mujoco_trial(condition, rate, load, case, seed, args.walking)
                        )
                        suffix = "unloaded" if load == 0 else "loaded"
                        np.savez_compressed(args.output / f"{case}_{seed}_{condition}_{rate:g}_{suffix}.npz", **trace)
                        traces.append(trace)
                    entry.update(status="completed", **metrics(traces[1], traces[0]))
                except (ValueError, RuntimeError, SimulationFailure, FloatingPointError) as error:
                    entry.update(status="failed", reason=str(error))
                results.append(entry)
                (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
                print(json.dumps(entry), flush=True)


if __name__ == "__main__":
    main()
