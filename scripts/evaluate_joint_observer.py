"""Fit and evaluate G1 and X2 arm observers in headless MuJoCo."""

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
from box import Box
from scipy.optimize import least_squares

from robojudo.config.g1.carry_pose import G1_CARRY_POSE
from robojudo.config.g1.g1_locomimic_observer_cfg import g1_23_locomanipulation_locomimic_observer
from robojudo.config.x2.carry_pose import X2_CARRY_POSE
from robojudo.config.x2.x2_locomimic_observer_cfg import x2_locomimic_observer
from robojudo.controller.ctrl_cfgs import UpperBodyZmqCtrlCfg
from robojudo.environment import env_registry
from robojudo.environment.mujoco_env import MujocoEnv
from robojudo.pipeline.four_mode_pipeline import FALL_TILT_THRESHOLD_RAD, ControlMode
from robojudo.pipeline.g1_locomanipulation_loco_mimic_pipeline import G1LocomanipulationLocoMimicPipeline
from robojudo.pipeline.x2_locomanipulation_loco_mimic_pipeline import X2LocomanipulationLocoMimicPipeline

LOAD_START, RELEASE, DURATION = 1.0, 7.0, 12.0
MODES = ("OFF", "SHADOW", "ACTIVE")
MOTIONS = (("standing", 0.0), ("walking", 0.25))
SIM_DT = 0.005
PROBE_DT = 0.02
PROBE_SUBSTEPS = round(PROBE_DT / SIM_DT)
PROBE_SEED = 0
PROBE_SLEW_NM_S = 2.0
CARRY_FORCE_PER_HAND_N = 20.0
CLOSED_LOOP_FORCE_PER_HAND_N = 10.0
ROBOTS = {
    "g1": (g1_23_locomanipulation_locomimic_observer, G1LocomanipulationLocoMimicPipeline, "wrist_roll_rubber_hand"),
    "x2": (x2_locomimic_observer, X2LocomanipulationLocoMimicPipeline, "wrist_roll_link"),
}


def configure_headless(cfg):
    cfg.ctrl = []
    cfg.device = "cpu"
    cfg.record.enabled = False
    cfg.env.headless = True
    cfg.env.visualize_extras = False
    cfg.env.random_heading = False
    cfg.loco_policy.command_status_interval_s = 0.0


def synthetic_controls(speed, names, arm_target):
    return Box(
        {
            "COMMANDS": [],
            "VELOCITY_SOURCE": "VelocityZmqCtrl",
            "POSTURE_SOURCE": None,
            "VelocityZmqCtrl": {
                "fresh": True,
                "linear_velocity": [speed, 0.0, 0.0],
                "angular_velocity": [0.0] * 3,
            },
            "UpperBodyZmqCtrl": {"fresh": True, "joint_positions": dict(zip(names, arm_target, strict=True))},
        }
    )


def enter_unassisted_rl(pipeline):
    for policy in pipeline.policy_manager.policies:
        if hasattr(policy, "close_progress"):
            policy.close_progress()
    pipeline._manual_mode_override = None
    band = pipeline.env.elastic_band
    band.active = True
    pipeline._enter_mode(ControlMode.JOINT_DEFAULT)
    for _ in range(pipeline._joint_default_steps):
        pipeline.step()
    if not pipeline._enter_mode(ControlMode.RL_DEFAULT):
        raise RuntimeError("Could not enter RL_DEFAULT")
    band.active = False
    band.apply()


def run_trial(mode, speed, force_n=0.0, *, tuning=None, robot="g1"):
    config_type, pipeline_type, hand_body = ROBOTS[robot]
    cfg = config_type()
    upper_body_cfg = next(ctrl for ctrl in cfg.ctrl if isinstance(ctrl, UpperBodyZmqCtrlCfg))
    configure_headless(cfg)
    observer = cfg.env.observer
    cfg.env.observer = type(observer).model_validate({**observer.model_dump(), **(tuning or {}), "mode": mode})
    np.random.seed(PROBE_SEED)
    pipeline = pipeline_type(cfg)
    env = pipeline.env
    names = cfg.env.observer.joint_names
    indices = [env.joint_names.index(name) for name in names]
    hands = [env.model.body(f"{side}_{hand_body}").id for side in ("left", "right")]
    torso = env.model.body("torso_link").id
    pelvis = env.model.body("pelvis").id
    speed_cmd = 0.0
    upper_offset = env.num_dofs - cfg.upper_dof_num
    neutral_arm_goal = [cfg.upper_dof_pos_default[index - upper_offset] for index in indices]

    def control_data(_):
        return synthetic_controls(speed_cmd, names, neutral_arm_goal)

    step = env.step
    sample = {}

    def record_target(target, *args, **kwargs):
        sample.update(q=env.dof_pos[indices].copy(), dq=env.dof_vel[indices].copy(), target=target[indices].copy())
        step(target, *args, **kwargs)

    torque = env._simulate_torque
    substep_clipping = []

    def record_substeps(torque_fn):
        def measure():
            values = torque_fn()
            substep_clipping.append(np.abs(values) > env.torque_limits)
            return values

        torque(measure)

    def warm_up(phase):
        for index in range(round(3.0 / pipeline.dt)):
            pipeline.step()
            if pipeline.mode != ControlMode.RL_DEFAULT:
                raise RuntimeError(
                    f"{robot.upper()} exited RL_DEFAULT during {phase} warm-up at {(index + 1) * pipeline.dt:.2f}s "
                    f"(mode {pipeline.mode.name}, tilt {pipeline._robot_tilt():.3f} rad)"
                )

    try:
        pipeline.ctrl_manager.get_ctrl_data = control_data
        pipeline._upper_body_cfg = upper_body_cfg
        pipeline._configure_upper_body_override()
        enter_unassisted_rl(pipeline)
        pipeline._set_upper_body_enabled(True)
        warm_up("standing")
        speed_cmd = speed
        warm_up("commanded motion")

        rows = []
        with patch.object(env, "step", record_target), patch.object(env, "_simulate_torque", record_substeps):
            for i in range(round(DURATION / pipeline.dt)):
                t = i * pipeline.dt
                loaded = LOAD_START <= t < RELEASE
                env.data.xfrc_applied[hands, 2] = -force_n if loaded else 0.0
                substep_clipping.clear()
                sample.clear()
                start = time.perf_counter()
                pipeline.step()
                loop_ms = 1000 * (time.perf_counter() - start)
                if pipeline.mode != ControlMode.RL_DEFAULT or not sample:
                    raise RuntimeError(f"{robot.upper()} left RL_DEFAULT in {mode} at {t:.2f}s: {pipeline.mode}")
                if not np.isfinite(env.data.qpos).all() or not np.isfinite(env.data.qvel).all():
                    raise RuntimeError(f"{robot.upper()} simulation became nonfinite in {mode} at {t:.2f}s")
                mujoco.mj_fwdPosition(env.model, env.data)
                rotation = env.data.xmat[torso].reshape(3, 3)
                pelvis_height = env.data.xpos[pelvis, 2]
                tilt = np.arccos(np.clip(env.data.xmat[pelvis, 8], -1.0, 1.0))
                if pelvis_height < 0.45 or tilt > FALL_TILT_THRESHOLD_RAD:
                    raise RuntimeError(
                        f"{robot.upper()} fell in {mode} at {t:.2f}s (height {pelvis_height:.3f}, tilt {tilt:.3f})"
                    )
                status = env.get_observer_status()
                rows.append(
                    {
                        **sample,
                        "t": t,
                        "q_next": env.dof_pos[indices].copy(),
                        "dq_next": env.dof_vel[indices].copy(),
                        "effort": env.last_feedforward_effort.copy(),
                        "estimate": status.get("estimated_disturbance_nm", np.zeros(len(indices))),
                        "proposed": status.get("proposed_nm", np.zeros(len(indices))),
                        "hand": (env.data.xipos[hands] - env.data.xpos[torso]) @ rotation,
                        "tilt": tilt,
                        "height": pelvis_height,
                        "base_xy": env.data.xpos[pelvis, :2].copy(),
                        "saturated": np.any(substep_clipping, axis=0),
                        "substep_clipping": np.asarray(substep_clipping),
                        "loop_ms": loop_ms,
                    }
                )
        return {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    finally:
        env.data.xfrc_applied[hands, 2] = 0.0
        try:
            pipeline.ctrl_manager.close()
        finally:
            env.shutdown()


MULTISINE_BLOCK_DURATION = 40
MULTISINE_END = 2 * MULTISINE_BLOCK_DURATION
PROBE_DURATION = 100
PROBE_SAMPLES = round(PROBE_DURATION / PROBE_DT)
OPERATING_TIMES = {"default": 0, "grasp": 23, "carry": 35}
FREQUENCIES = np.array([0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4])
PROBE_PROTOCOL = {
    "operating_times_s": OPERATING_TIMES,
    "frequencies_hz": FREQUENCIES.tolist(),
    "control_dt_s": PROBE_DT,
    "physics_dt_s": SIM_DT,
    "duration_s": PROBE_DURATION,
}
_PHASES = -np.pi * np.arange(len(FREQUENCIES)) * (np.arange(len(FREQUENCIES)) - 1) / len(FREQUENCIES)
_NORMALIZER = np.max(
    np.abs(np.sin(2 * np.pi * np.arange(0, 10, PROBE_DT)[:, None] * FREQUENCIES + _PHASES).sum(axis=1))
)


def excitation(t, amplitude):
    if 0 <= t < MULTISINE_END:
        sine = np.sin(2 * np.pi * (t % MULTISINE_BLOCK_DURATION) * FREQUENCIES + _PHASES).sum() / _NORMALIZER
        return float(amplitude * (1 if t < MULTISINE_BLOCK_DURATION else 2) * sine)
    if 84 <= t < 88:
        return 2 * amplitude
    if 92 <= t < 96:
        return -2 * amplitude
    return 0.0


def smoothstep(value):
    value = float(np.clip(value, 0, 1))
    return value**3 * (10 - 15 * value + 6 * value**2)


def standing_target(t):
    """The original approach, grasp and lift schedule through the standing carry point."""
    if t < 26:
        return 0.8 * smoothstep((t - 10) / 6), 0.0
    load = smoothstep((t - 26) / 4)
    return 0.8 + 0.2 * load, load


@env_registry.register
class ObserverCalibrationEnv(MujocoEnv):
    """Apply the standing load and probe with the observer disabled."""

    def __init__(self, cfg_env, device="cpu"):
        super().__init__(cfg_env, device)
        self.probe_effort = np.zeros(self.num_dofs)
        self.requested = self.issued = 0.0
        self.last_target = np.asarray(self.default_pos).copy()
        self.saturated_steps = 0
        self.load_n = 0.0
        self.hands = []
        self.pelvis = self.model.body("pelvis").id
        self.probe_joint = None

    def step(self, pd_target, hand_pose=None, *, feedforward_effort=None):
        if feedforward_effort is not None:
            raise ValueError("Calibration owns feedforward torque")
        self.last_target = np.asarray(pd_target).copy()
        max_change = PROBE_SLEW_NM_S * PROBE_DT
        self.issued += float(np.clip(self.requested - self.issued, -max_change, max_change))
        if self.probe_joint is not None:
            self.probe_effort[self.probe_joint] = self.issued
        super().step(pd_target, hand_pose, feedforward_effort=self.probe_effort)

    def check_state(self):
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("Nonfinite simulation state")
        if self.data.xpos[self.pelvis, 2] < 0.30:
            raise RuntimeError("Pelvis height below 0.30 m")
        tilt = np.arccos(np.clip(self.data.xmat[self.pelvis, 8], -1.0, 1.0))
        if tilt >= FALL_TILT_THRESHOLD_RAD:
            raise RuntimeError("Simulation reached the fall tilt threshold")
        for warning in (
            mujoco.mjtWarning.mjWARN_BADQPOS,
            mujoco.mjtWarning.mjWARN_BADQVEL,
            mujoco.mjtWarning.mjWARN_BADQACC,
        ):
            if self.data.warning[warning].number:
                raise RuntimeError(f"MuJoCo numerical warning: {warning.name}")

    def _simulate_torque(self, torque_fn):
        def loaded_torque():
            mujoco.mj_fwdPosition(self.model, self.data)
            self.check_state()
            self.data.xfrc_applied[self.hands, 2] = -self.load_n
            torque = torque_fn()
            self.saturated_steps += int(np.any(np.abs(torque) > self.torque_limits))
            return torque

        super()._simulate_torque(loaded_torque)
        mujoco.mj_fwdPosition(self.model, self.data)
        self.check_state()


def calibration_setup(robot):
    config_type, pipeline_type, hand_body = ROBOTS[robot]
    observer_cfg = config_type()
    names = observer_cfg.env.observer.joint_names
    upper_body = next(ctrl for ctrl in observer_cfg.ctrl if isinstance(ctrl, UpperBodyZmqCtrlCfg))
    if robot == "x2":
        from scripts.x2_integral_sim import evaluation_config

        cfg = evaluation_config()
        carrying = np.array([X2_CARRY_POSE[name] for name in names])
    else:
        cfg = observer_cfg
        carrying = np.array([G1_CARRY_POSE[name] for name in names])

    dof = cfg.env.dof
    indices = [dof.joint_names.index(name) for name in names]
    defaults = np.asarray(dof.default_pos)[indices]
    cfg.joint_default_dof = cfg.joint_default_dof.model_copy(deep=True)
    cfg.upper_dof_pos_default = list(cfg.upper_dof_pos_default)
    cfg.env.feedforward_effort_limits = {
        name: min(8.0, 0.8 * dof.torque_limits[index]) for name, index in zip(names, indices, strict=True)
    }
    for name, index, value in zip(names, indices, defaults, strict=True):
        cfg.joint_default_dof.default_pos[cfg.joint_default_dof.joint_names.index(name)] = float(value)
        cfg.upper_dof_pos_default[index - (dof.num_dofs - cfg.upper_dof_num)] = float(value)
    cfg.env.env_type = "ObserverCalibrationEnv"
    configure_headless(cfg)
    return cfg, pipeline_type, upper_body, hand_body, names, defaults, carrying, CARRY_FORCE_PER_HAND_N


def acquire_probe(robot, condition, joint, output, *, scale=1.0):
    cfg, pipeline_type, upper_body, hand_body, names, defaults, carrying, load_n = calibration_setup(robot)
    np.random.seed(PROBE_SEED)
    pipeline = pipeline_type(cfg)
    env = pipeline.env
    indices = [env.joint_names.index(name) for name in names]
    env.hands = [env.model.body(f"{side}_{hand_body}").id for side in ("left", "right")]
    amplitude = 0.0
    if joint is not None:
        env.probe_joint = indices[joint]
        amplitude = scale * min(0.25, 0.01 * env.torque_limits[env.probe_joint])

    arm_target = defaults

    def controls(_):
        return synthetic_controls(0.0, names, arm_target)

    def advance():
        pipeline.step()
        if pipeline.mode != ControlMode.RL_DEFAULT:
            raise RuntimeError(f"{robot.upper()} exited RL_DEFAULT: {pipeline.mode.name}")

    try:
        if abs(env.sim_dt - SIM_DT) > 1e-9 or env.sim_decimation != PROBE_SUBSTEPS:
            raise ValueError("Calibration requires 5 ms physics steps and 50 Hz control")
        pipeline.ctrl_manager.get_ctrl_data = controls
        enter_unassisted_rl(pipeline)
        for _ in range(round(5 / PROBE_DT)):
            advance()
        pipeline._upper_body_cfg = upper_body
        pipeline._configure_upper_body_override()
        pipeline._upper_body_filtered = env.last_target[indices].astype(np.float32)
        pipeline._set_upper_body_enabled(True)
        operating_time = OPERATING_TIMES[condition]
        for step in range(round((operating_time + 10) / PROBE_DT)):
            fraction, load = standing_target(min(step * PROBE_DT, operating_time))
            arm_target = defaults + fraction * (carrying - defaults)
            env.load_n = load_n * load
            advance()

        env.saturated_steps = 0
        q = np.empty((PROBE_SAMPLES, len(indices)))
        dq = np.empty_like(q)
        effort = np.zeros(PROBE_SAMPLES)
        for step in range(PROBE_SAMPLES):
            env.requested = excitation(step * PROBE_DT, amplitude)
            advance()
            q[step] = env.dof_pos[indices]
            dq[step] = env.dof_vel[indices]
            if joint is not None:
                effort[step] = env.last_feedforward_effort[env.probe_joint]
                if abs(effort[step] - env.issued) > 1e-9:
                    raise ValueError("Probe feedforward torque clipped")
        if env.saturated_steps:
            raise ValueError(f"Actuator torque clipped on {env.saturated_steps} physics steps")
        name = "reference" if joint is None else names[joint]
        suffix = "_half" if scale != 1.0 else ""
        target = output / condition / f"{name}{suffix}.npz"
        target.parent.mkdir(exist_ok=True)
        np.savez_compressed(target, q=q, dq=dq, effort=effort)
        return str(target.relative_to(output))
    finally:
        env.data.xfrc_applied[env.hands] = 0.0
        try:
            pipeline.ctrl_manager.close()
        finally:
            env.shutdown()


MULTISINE_SAMPLES = round(MULTISINE_END / PROBE_DT)
BLOCK_SAMPLES = MULTISINE_SAMPLES // 2
HELD_OUT_SAMPLES = round(10 / PROBE_DT)
TRAIN = np.r_[0 : BLOCK_SAMPLES - HELD_OUT_SAMPLES, BLOCK_SAMPLES : MULTISINE_SAMPLES - HELD_OUT_SAMPLES]
VALIDATE = np.r_[
    BLOCK_SAMPLES - HELD_OUT_SAMPLES : BLOCK_SAMPLES, MULTISINE_SAMPLES - HELD_OUT_SAMPLES : MULTISINE_SAMPLES
]
LOWER = np.log([0.005, 0.05, 0.1, 0.001, 0.1])
UPPER = np.log([5.0, 100.0, 300.0, 2.0, 5000.0])
G1_FRICTION_SEED_NM = 0.10
G1_WRIST_ROLL_INERTIA_KG_M2 = 0.01020


def rollout(log_parameters, effort):
    """Implicit dissipative velocity update at the actual four physics substeps."""
    mass, damping, stiffness, friction, viscous = np.exp(log_parameters)
    mass_h = mass / SIM_DT
    alpha = mass_h + damping + stiffness * SIM_DT
    position = velocity = 0.0
    result = np.empty((len(effort), 2))
    for step, torque in enumerate(effort):
        for _ in range(PROBE_SUBSTEPS):
            force = mass_h * velocity + torque - stiffness * position
            velocity = force / (alpha + viscous)
            if abs(viscous * velocity) > friction:
                velocity = (force - np.copysign(friction, force)) / alpha
            position += SIM_DT * velocity
        result[step] = position, velocity
    return result


def relative_rmse(predicted, measured):
    return np.sqrt(np.mean((predicted - measured) ** 2, axis=0)) / np.maximum(
        np.sqrt(np.mean(measured**2, axis=0)), 1e-12
    )


def fit_response(effort, measured, *, friction_seed_nm=0.30, fixed_inertia=None):
    """Fit two amplitude blocks and select the initialization on held-out periods."""
    scales = np.maximum(np.sqrt(np.mean(measured[:MULTISINE_SAMPLES] ** 2, axis=0)), 1e-12)
    first_free = int(fixed_inertia is not None)
    lower, upper = LOWER[first_free:], UPPER[first_free:]
    inertia_seeds = (0.03, 0.1, 0.3) if fixed_inertia is None else (fixed_inertia,)
    candidates = []

    def full_parameters(free_parameters):
        if fixed_inertia is None:
            return free_parameters
        return np.r_[np.log(fixed_inertia), free_parameters]

    def residual(free_parameters):
        parameters = full_parameters(free_parameters)
        return ((rollout(parameters, effort[:MULTISINE_SAMPLES])[TRAIN] - measured[TRAIN]) / scales).ravel()

    for inertia in inertia_seeds:
        initial = np.log([inertia, 3.0, 50.0, friction_seed_nm, 30.0])
        fit = least_squares(
            residual,
            initial[first_free:],
            bounds=(lower, upper),
            max_nfev=100,
            diff_step=1e-4,
        )
        predicted = rollout(full_parameters(fit.x), effort)
        validation = relative_rmse(predicted[VALIDATE], measured[VALIDATE])
        candidates.append((float(np.max(validation)), fit, predicted, validation))
    _, fit, predicted, validation = min(candidates, key=lambda candidate: (not candidate[1].success, candidate[0]))
    pulse = relative_rmse(predicted[MULTISINE_SAMPLES:], measured[MULTISINE_SAMPLES:])
    absolute_pulse = np.sqrt(np.mean((predicted[MULTISINE_SAMPLES:] - measured[MULTISINE_SAMPLES:]) ** 2, axis=0))
    parameters = dict(
        zip(
            ("inertia", "damping", "stiffness", "friction_nm", "friction_viscosity"),
            np.exp(full_parameters(fit.x)).tolist(),
            strict=True,
        )
    )
    if fixed_inertia is not None:
        parameters["inertia"] = fixed_inertia
    return {
        "parameters": parameters,
        "validation_relative_rmse_position_velocity": validation.tolist(),
        "pulse_relative_rmse_position_velocity": pulse.tolist(),
        "pulse_absolute_rmse_rad_rad_s": absolute_pulse.tolist(),
        "adequate": bool(fit.success and np.all(validation <= 0.25) and np.all(pulse <= 0.25)),
        "optimizer_success": bool(fit.success),
        "evaluations": fit.nfev,
        "parameter_at_search_boundary": bool(np.any(np.minimum(fit.x - lower, upper - fit.x) < 1e-3)),
    }


def acquire_trial(robot, stance, joint, output):
    try:
        trace = acquire_probe(robot, stance, joint, output)
        return stance, joint, trace, None
    except (ValueError, RuntimeError, FloatingPointError) as first_error:
        if joint is None:
            return stance, joint, None, str(first_error)
        try:
            trace = acquire_probe(robot, stance, joint, output, scale=0.5)
            return stance, joint, trace, None
        except (ValueError, RuntimeError, FloatingPointError) as retry_error:
            return stance, joint, None, f"{first_error}; half-amplitude retry: {retry_error}"


def probe_metadata(names, carry_pose):
    return {
        **PROBE_PROTOCOL,
        "carry_force_per_hand_n": CARRY_FORCE_PER_HAND_N,
        "carry_target_rad": dict(zip(names, carry_pose.tolist(), strict=True)),
    }


def acquire_traces(robot, output, names, jobs, carry_pose):
    traces = {stance: {} for stance in OPERATING_TIMES}
    failures = []
    manifest = {
        "robot": robot,
        "joint_names": names,
        "traces": traces,
        "failures": failures,
        "probe": probe_metadata(names, carry_pose),
    }
    manifest_path = output / "manifest.json"
    pending = manifest_path.with_suffix(".tmp")
    tasks = [(robot, stance, joint, output) for stance in traces for joint in (None, *range(len(names)))]
    with ProcessPoolExecutor(max_workers=jobs, max_tasks_per_child=1) as pool:
        for future in as_completed(pool.submit(acquire_trial, *task) for task in tasks):
            stance, index, trace, error = future.result()
            joint = "reference" if index is None else names[index]
            if error is None:
                traces[stance][joint] = trace
            else:
                failures.append(f"{stance} {joint}: {error}")
            pending.write_text(json.dumps(manifest, indent=2) + "\n")
            pending.replace(manifest_path)
            print(f"{robot} {stance} {joint}: {error or 'measured'}", flush=True)
    return manifest


def read_manifest(root, robot, names, carry_pose):
    manifest = json.loads((root / "manifest.json").read_text())

    def pose_matches(target):
        return (
            isinstance(target, dict)
            and target.keys() == set(names)
            and np.allclose([target[name] for name in names], carry_pose, rtol=0, atol=1e-6)
        )

    if "traces" in manifest:
        if manifest.get("robot") != robot or manifest.get("joint_names") != names:
            raise ValueError("Calibration traces have a different robot or joint order")
        probe = manifest.get("probe")
        if probe is None and (root / "summary.json").is_file():
            probe = json.loads((root / "summary.json").read_text()).get("probe")
        if (
            probe is None
            or probe.get("carry_force_per_hand_n") != CARRY_FORCE_PER_HAND_N
            or not pose_matches(probe.get("carry_target_rad"))
        ):
            raise ValueError("Calibration traces do not match the configured carry load and pose")
        # Earlier manifests stored only the carry load and target. Validate the
        # protocol for fields that are present in the manifest or its summary.
        if any(probe.get(key, expected) != expected for key, expected in PROBE_PROTOCOL.items()):
            raise ValueError("Calibration traces do not match the configured probe protocol")
        return manifest

    posture = manifest.get("posture_rad", {})
    if robot != "x2" or list(posture) != names:
        raise ValueError("Historical calibration traces require the original X2 joint order")
    if (
        manifest.get("dt_s") != PROBE_DT
        or manifest.get("duration_s") != PROBE_DURATION
        or manifest.get("frequencies_hz") != FREQUENCIES.tolist()
        or any(manifest.get("operating_times", {}).get(key) != time for key, time in OPERATING_TIMES.items())
    ):
        raise ValueError("Historical calibration traces do not match the configured probe protocol")
    if manifest.get("force_per_hand_n") != [0, 0, -CARRY_FORCE_PER_HAND_N] or not pose_matches(posture):
        raise ValueError("Historical X2 traces do not match the configured carry load and pose")
    trials = {}
    for trial in sorted(manifest["trials"], key=lambda trial: trial["amplitude_scale"]):
        if (
            trial["status"] == "completed"
            and trial["samples"] == PROBE_SAMPLES
            and trial["saturated_physics_fraction"] == 0
            and trial["max_excitation_clip_nm"] <= 1e-9
        ):
            trials.setdefault((trial["operating_point"], trial["joint_name"]), trial["trace"])
    traces = {stance: {} for stance in OPERATING_TIMES}
    failures = []
    for stance in traces:
        for name in ["reference", *names]:
            trial = trials.get((stance, name))
            if trial is None:
                failures.append(f"{stance} {name}: historical trace is missing, failed or clipped")
            else:
                traces[stance][name] = trial
    return {"robot": robot, "joint_names": names, "traces": traces, "failures": failures}


def load_probe(path, index, name):
    if path.suffix == ".csv":
        from scripts.plot_x2_transitions import load_numeric

        data = load_numeric(path)
        times = np.arange(1, PROBE_SAMPLES + 1) * PROBE_DT
        if not np.array_equal(data["elapsed_s"], times):
            raise ValueError(f"Unexpected sample times in {path.name}")
        effort = data["actual_excitation_nm"]
        measured = np.stack((data[f"{name}_position_rad"], data[f"{name}_velocity_rad_s"]), axis=1)
    else:
        with np.load(path, allow_pickle=False) as trace:
            effort = trace["effort"]
            measured = np.stack((trace["q"][:, index], trace["dq"][:, index]), axis=1)
    if measured.shape != (PROBE_SAMPLES, 2) or effort.shape != (PROBE_SAMPLES,):
        raise ValueError(f"Incomplete trace {path.name}")
    if not np.isfinite(measured).all() or not np.isfinite(effort).all():
        raise ValueError(f"Nonfinite samples in {path.name}")
    return effort, measured


def fit_trace_pair(root, trace, reference, index, name, **fit_options):
    effort, measured = load_probe(root / trace, index, name)
    baseline_effort, baseline = load_probe(root / reference, index, name)
    if np.any(baseline_effort):
        raise ValueError("Reference trace has nonzero probe torque")
    return fit_response(effort, measured - baseline, **fit_options)


def select_joint(fits, name, armature, kd, *, fixed_inertia=None):
    if not all(fit["adequate"] and not fit["parameter_at_search_boundary"] for fit in fits):
        raise ValueError(f"Standing validation required for {name}")
    parameters = [fit["parameters"] for fit in fits]
    if fixed_inertia is not None:
        inertia = fixed_inertia
    elif "wrist_pitch" in name or "wrist_roll" in name:
        inertia = armature
    else:
        inertia = max(armature, float(np.median([p["inertia"] for p in parameters])))
    viscosity = max(0.0, float(np.median([p["damping"] + p["friction_viscosity"] - kd for p in parameters])))
    return {"inertia": inertia, "friction_viscosity": viscosity}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fit_parameters(output, force_n=CLOSED_LOOP_FORCE_PER_HAND_N, *, robot="g1", jobs=3, traces=None):
    cfg, _, _, _, names, _, carrying, _ = calibration_setup(robot)
    observer = ROBOTS[robot][0]().env.observer
    dof = cfg.env.dof
    model = mujoco.MjModel.from_xml_path(cfg.env.xml)
    indices = [dof.joint_names.index(name) for name in names]
    addresses = [model.joint(name).dofadr[0] for name in names]
    armature = model.dof_armature[addresses]
    friction = model.dof_frictionloss[addresses]
    manifest = (
        acquire_traces(robot, output, names, jobs, carrying)
        if traces is None
        else read_manifest(traces, robot, names, carrying)
    )
    root = output if traces is None else traces
    failures = list(manifest["failures"])
    joints = {}
    report = {
        "robot": robot,
        "joint_names": names,
        "policy": cfg.loco_policy.policy_name,
        "trace_directory": str(root),
        "sha256": {
            key: sha256(path)
            for key, path in (("mjcf", cfg.env.xml), ("policy", cfg.loco_policy.policy_file), ("runner", __file__))
        },
        "mujoco_version": mujoco.__version__,
        "probe": probe_metadata(names, carrying),
        "joints": joints,
        "failures": failures,
    }
    for index, name in enumerate(names):
        fits = {}
        incumbent = {"inertia": observer.inertia[index], "friction_viscosity": observer.friction_viscosity[index]}
        entry = {"incumbent": incumbent, "standing_fits": fits, "armature_kg_m2": float(armature[index])}
        joints[name] = entry
        fixed_inertia = None
        fit_options = {}
        if robot == "g1":
            fit_options["friction_seed_nm"] = G1_FRICTION_SEED_NM
            if name.endswith("_wrist_roll_joint"):
                fixed_inertia = G1_WRIST_ROLL_INERTIA_KG_M2
                fit_options["fixed_inertia"] = fixed_inertia
                entry["fixed_inertia_kg_m2"] = fixed_inertia
        for stance in OPERATING_TIMES:
            stance_traces = manifest["traces"].get(stance, {})
            if name in stance_traces and "reference" in stance_traces:
                try:
                    fits[stance] = fit_trace_pair(
                        root, stance_traces[name], stance_traces["reference"], index, name, **fit_options
                    )
                except (OSError, KeyError, IndexError, ValueError, RuntimeError, FloatingPointError) as error:
                    failures.append(f"{stance} {name}: {error}")
        if set(fits) != OPERATING_TIMES.keys():
            failures.append(f"{name}: incomplete standing fits")
            continue
        try:
            candidate = select_joint(
                fits.values(), name, float(armature[index]), dof.damping[indices[index]], fixed_inertia=fixed_inertia
            )
        except ValueError as error:
            failures.append(str(error))
            continue
        entry["candidate"] = candidate
        entry["delta"] = {key: value - incumbent[key] for key, value in candidate.items()}
    if all("candidate" in entry for entry in joints.values()):
        tuning = {
            key: [entry["candidate"][key] for entry in joints.values()] for key in ("inertia", "friction_viscosity")
        }
        tuning["friction_limit"] = float(friction[0]) if np.all(friction == friction[0]) else friction.tolist()
        tuning["bandwidth_rad_s"] = 8.0
        report["candidate_parameters"] = tuning
        report["matches_frozen"] = (
            bool(np.all(friction == observer.friction_limit))
            and tuning["bandwidth_rad_s"] == observer.bandwidth_rad_s
            and all(delta == 0.0 for entry in joints.values() for delta in entry["delta"].values())
        )
        report["closed_loop_force_per_hand_n"] = force_n
        report["closed_loop"] = compare_closed_loop(output, force_n, tuning, failures, robot=robot)
    report["passed"] = not failures
    return report


def summarize(trace, unloaded):
    carry = (trace["t"] >= RELEASE - 2.0) & (trace["t"] < RELEASE)
    after = trace["t"] >= RELEASE
    error = trace["q_next"] - trace["target"]
    displacement = np.linalg.norm(trace["hand"] - unloaded["hand"], axis=-1)
    correction = np.max(np.abs(trace["effort"] - unloaded["effort"]), axis=1)
    residual_limit = max(0.04, float(0.1 * np.max(correction[carry])))
    remains_above = np.flatnonzero(correction[after] > residual_limit)
    settled = remains_above[-1] + 1 if remains_above.size else 0
    release_times = trace["t"][after] - RELEASE
    settles_in_window = settled < len(release_times) and release_times[-1] - release_times[settled] >= 1.0
    # Hand poses are sampled after integration, so include the last loaded
    # sample as the pose at the instant of release.
    release_pose_index = np.flatnonzero(after)[0] - 1
    return {
        "carry_arm_rms_deg": float(np.rad2deg(np.sqrt(np.mean(error[carry] ** 2)))),
        "release_hand_peak_mm": float(1000 * np.max(displacement[release_pose_index:])),
        "correction_removal_s": float(release_times[settled]) if settles_in_window else None,
        "peak_effort_nm": float(np.max(np.abs(trace["effort"]))),
        "unloaded_peak_effort_nm": float(np.max(np.abs(unloaded["effort"]))),
        "saturated_control_intervals": int(np.count_nonzero(np.any(trace["saturated"], axis=1))),
        "saturated_physics_steps": int(np.count_nonzero(np.any(trace["substep_clipping"], axis=-1))),
        "max_base_tilt_deg": float(np.rad2deg(np.max(trace["tilt"]))),
        "min_pelvis_height_m": float(np.min(trace["height"])),
        "mean_planar_speed_m_s": float(
            np.linalg.norm(trace["base_xy"][-1] - trace["base_xy"][0]) / (trace["t"][-1] - trace["t"][0])
        ),
        "p99_loop_ms": float(np.percentile(trace["loop_ms"], 99)),
        "max_loop_ms": float(np.max(trace["loop_ms"])),
    }


def compare_closed_loop(output, force_n, tuning, failures, *, robot="g1"):
    results = {}
    for motion, speed in MOTIONS:
        results[motion] = scores = {}
        for label, parameters in (("incumbent", None), ("candidate", tuning)):
            try:
                unloaded = run_trial("ACTIVE", speed, tuning=parameters, robot=robot)
                loaded = run_trial("ACTIVE", speed, force_n, tuning=parameters, robot=robot)
                for load, trace in (("unloaded", unloaded), ("loaded", loaded)):
                    np.savez_compressed(output / f"closed_loop_{motion}_{label}_{load}.npz", **trace)
                scores[label] = {
                    "loaded": summarize(loaded, unloaded),
                    "unloaded": summarize(unloaded, unloaded),
                }
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                failures.append(f"{motion} {label} closed loop: {exc}")
        if set(scores) != {"incumbent", "candidate"}:
            continue
        old, new = scores["incumbent"], scores["candidate"]
        for metric, floor in (("carry_arm_rms_deg", 0.05), ("release_hand_peak_mm", 2.0)):
            baseline = old["loaded"][metric]
            candidate = new["loaded"][metric]
            limit = baseline + max(0.05 * baseline, floor)
            if candidate > limit:
                failures.append(f"{motion}: closed-loop {metric} was {candidate:.4f}, limit {limit:.4f}")
        for load in ("loaded", "unloaded"):
            if new[load]["saturated_physics_steps"] > old[load]["saturated_physics_steps"]:
                failures.append(f"{motion} {load}: closed-loop actuator saturation increased")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=ROBOTS, default="g1", help="Select the simulation preset (default: g1).")
    parser.add_argument("--fit", action="store_true", help="Fit from additive torque and run validation checks.")
    parser.add_argument("--jobs", type=int, default=3, help="Parallel simulation workers during fitting.")
    parser.add_argument("--traces", type=Path, help="Reuse recorded calibration traces with --fit.")
    parser.add_argument(
        "--force",
        type=float,
        default=CLOSED_LOOP_FORCE_PER_HAND_N,
        help="Downward load per hand for closed-loop trials in newtons (default: %(default)s).",
    )
    parser.add_argument("--bandwidth", type=float, default=None, help="Override observer bandwidth in rad/s.")
    parser.add_argument("--correction-cap", type=float, default=None, help="Override correction cap in Nm.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.force) or args.force <= 0:
        parser.error("--force must be positive and finite")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.traces is not None and not args.fit:
        parser.error("--traces requires --fit")
    if args.fit:
        if args.bandwidth is not None or args.correction_cap is not None:
            parser.error("--fit selects the bandwidth and uses the registered correction limits")
        try:
            args.output.mkdir(parents=True)
        except FileExistsError:
            parser.error(f"--fit requires a new output directory: {args.output}")
        results = fit_parameters(args.output, args.force, robot=args.robot, jobs=args.jobs, traces=args.traces)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        tuning = {
            name: value
            for name, value in (("bandwidth_rad_s", args.bandwidth), ("correction_limit_nm", args.correction_cap))
            if value is not None
        }
        results = {"robot": args.robot, "force_per_hand_n": args.force, "tuning": tuning, "trials": {}}
        for motion, speed in MOTIONS:
            for mode in MODES:
                unloaded = run_trial(mode, speed, tuning=tuning, robot=args.robot)
                trace = run_trial(mode, speed, args.force, tuning=tuning, robot=args.robot)
                key = f"{motion}_{mode.lower()}"
                np.savez_compressed(args.output / f"{key}.npz", **trace)
                np.savez_compressed(args.output / f"{key}_unloaded.npz", **unloaded)
                results["trials"][key] = {
                    "loaded": summarize(trace, unloaded),
                    "unloaded": summarize(unloaded, unloaded),
                }
    (args.output / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    if args.fit and not results["passed"]:
        parser.exit(1, f"{args.robot.upper()} candidate failed validation. See {args.output / 'summary.json'}\n")


if __name__ == "__main__":
    main()
