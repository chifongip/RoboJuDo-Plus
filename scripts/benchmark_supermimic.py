"""Offline, batch-one SuperMimic inference comparison; never opens a robot environment."""

import argparse
import contextlib
import csv
import hashlib
import json
import logging
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import torch
from robojudo.config import cfg_registry
from robojudo.policy.supermimic_policy import (
    SuperMimicMotion,
    SuperMimicMotionPolicy,
    SuperMimicVelocityPolicy,
    quat_apply,
    quat_conjugate,
)
from torch import nn

ROUTES = ("A", "B", "C", "D", "E", "F", "G")
LABELS = dict(
    zip(
        ROUTES,
        ("torch_cpu", "ort_cpu", "torch_cuda", "ort_cuda_io", "ort_cuda_graph", "cached_torch_cpu", "cached_ort_cpu"),
        strict=True,
    )
)
LAYERS = ("network", "interface", "strategy")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def command_output(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


def environment():
    try:
        import onnxruntime as ort

        ort_info = {"version": ort.__version__, "providers": ort.get_available_providers()}
    except ImportError:
        ort_info = None
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "ort": ort_info,
        "affinity": sorted(os.sched_getaffinity(0)),
        "cpu_threads": torch.get_num_threads(),
        "model": Path("/proc/device-tree/model").read_bytes().decode().strip("\0")
        if Path("/proc/device-tree/model").exists()
        else platform.machine(),
        "l4t": Path("/etc/nv_tegra_release").read_text() if Path("/etc/nv_tegra_release").exists() else None,
        "jetpack": command_output(["dpkg-query", "-W", "nvidia-jetpack"]),
        "power_mode": command_output(["nvpmodel", "-q"]),
        "clocks": command_output(["jetson_clocks", "--show"]),
        "gpu": command_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
        "git_head": command_output(["git", "rev-parse", "HEAD"]),
    }


def config(task):
    return cfg_registry.get(f"g1_23_supermimic_{task}")().policy


def load_policy(task):
    return (SuperMimicMotionPolicy if task == "motion" else SuperMimicVelocityPolicy)(config(task))


def controls(command):
    return {
        "VELOCITY_SOURCE": "VelocityZmqCtrl",
        "VelocityZmqCtrl": {
            "fresh": True,
            "linear_velocity": [float(command[0]), float(command[1]), 0.0],
            "angular_velocity": [0.0, 0.0, float(command[2])],
        },
    }


def restore(policy, tape, index):
    policy.timestep = int(tape["frame"][index])
    policy.last_action = tape["previous_action"][index].copy()
    policy.history = deque((x.copy() for x in tape["history_before"][index]), maxlen=10)
    if tape["reset"][index]:
        policy.history.clear()
    return SimpleNamespace(**{k: tape[k][index] for k in ("dof_pos", "dof_vel", "base_quat", "base_ang_vel")})


def postprocess(action, policy):
    action = np.asarray(action, dtype=np.float32).reshape(23)
    if not np.isfinite(action).all():
        raise ValueError("Non-finite action")
    if policy.action_clip is not None:
        action = np.clip(action, -policy.action_clip, policy.action_clip)
    return action * policy.action_scales + policy.default_pos


def make_tape(task, directory, seed):
    """Freeze histories generated exclusively by the CPU baseline, including resets."""
    policy = load_policy(task)
    motion_cfg = config("motion")
    motion = SuperMimicMotion(motion_cfg.motion_file, policy.cfg_obs_dof.joint_names)
    rng = np.random.default_rng(seed)
    rows = []
    limits = np.asarray(config("velocity").command_limits)
    commands = [np.zeros(3), *np.eye(3), *(-np.eye(3)), np.array([0.5, -0.5, 0.5])]
    for segment, command in enumerate(commands):
        policy.reset()
        for frame in range(motion.length):
            q = motion.data["body_quat_w"][frame, 0]
            omega = quat_apply(quat_conjugate(q), motion.data["body_ang_vel_w"][frame, 0]).numpy()
            factor = frame / (motion.length - 1) if segment == len(commands) - 1 else 1.0
            cmd = (command * limits * factor).astype(np.float32)
            raw = {
                "dof_pos": motion.data["joint_pos"][frame].numpy() + rng.normal(0, 0.01, 23),
                "dof_vel": motion.data["joint_vel"][frame].numpy() + rng.normal(0, 0.1, 23),
                "base_quat": q.numpy()[[1, 2, 3, 0]],
                "base_ang_vel": omega + rng.normal(0, 0.02, 3),
            }
            raw = {k: np.asarray(v, dtype=np.float32) for k, v in raw.items()}
            policy.timestep = frame
            history = np.asarray(policy.history) if policy.history else np.zeros((10, 75), dtype=np.float32)
            previous = policy.last_action.copy()
            obs, _ = policy.get_observation(SimpleNamespace(**raw), controls(cmd))
            delta = policy.get_action(obs)
            action = policy.last_action.copy()
            rows.append(
                {
                    **raw,
                    "frame": frame,
                    "reset": frame == 0,
                    "command": cmd,
                    "history_before": history,
                    "previous_action": previous,
                    "input": obs,
                    "action": action,
                    "target": delta + policy.default_pos,
                }
            )
    write_json(directory / "policy_config.json", policy.cfg_policy.model_dump(mode="json"))
    path = directory / "tape.npz"
    np.savez(path, **{k: np.asarray([row[k] for row in rows]) for k in rows[0]})
    write_json(
        directory / "inputs.json",
        {
            "seed": seed,
            "samples": len(rows),
            "tape_sha256": digest(path),
            "checkpoint": str(policy.cfg_policy.checkpoint),
            "checkpoint_sha256": digest(policy.cfg_policy.checkpoint),
            "motion": str(motion_cfg.motion_file),
            "motion_sha256": digest(motion_cfg.motion_file),
        },
    )


class CachedDecoder(nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, history, latent):
        normalized = (
            self.actor.obs_norm(history) if self.actor.input_normalization else history / self.actor.observation_scale
        )
        return self.actor.decoder(torch.cat((latent, normalized.flatten(-2)), -1))


def error_metrics(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    finite = bool(np.isfinite(actual).all())
    delta = actual.astype(np.float64) - expected
    return {
        "passed": finite and bool(np.allclose(actual, expected, atol=1e-4, rtol=1e-4)),
        "max_abs": float(np.max(np.abs(delta))) if finite else None,
        "rmse": float(np.sqrt(np.mean(delta**2))) if finite else None,
    }


def summarize(values):
    values = np.asarray(values)
    return {
        **{f"p{q}_ms": float(np.percentile(values, q)) for q in (50, 95, 99)},
        "max_ms": float(np.max(values)),
        "over_20ms_fraction": float(np.mean(values > 20)),
        "count": len(values),
    }


class Backend:
    def __init__(self, route, policy, directory, tape):
        self.route, self.policy = route, policy
        self.cuda = route in ("C", "D", "E")
        if self.cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no CPU fallback permitted")
        self.device = "cuda" if self.cuda else "cpu"
        self.actor = policy.model.to(self.device).eval()
        self.cache = None
        self.setup = {}
        if route in ("F", "G"):
            if policy.velocity or policy.cfg_policy.motion_anchor_ori_b:
                raise RuntimeError("Cached route requires motion without anchor")
            start = time.perf_counter_ns()
            goals = torch.from_numpy(tape["input"][:, 750:])
            # Compute each phase once using the same GEMV shape as the baseline.
            count = policy.motion.length
            with torch.inference_mode():
                self.cache = torch.cat(
                    [self.actor.encoder(self.actor.goal_norm(g[None])).tanh() for g in goals[:count]]
                ).numpy()
            self.setup.update(cache_ms=(time.perf_counter_ns() - start) / 1e6, cache_bytes=self.cache.nbytes)
            np.save(directory / "latents.npy", self.cache)
            self.setup["cache_sha256"] = digest(directory / "latents.npy")
            self.actor = CachedDecoder(self.actor).eval()
        self.history = torch.zeros((1, 10, 75), device=self.device)
        self.condition = torch.zeros((1, 32 if route in ("F", "G") else policy.condition_dim), device=self.device)
        self.session = None
        if route in ("B", "D", "E", "G"):
            import onnx  # noqa: F401
            import onnxruntime as ort

            provider = "CUDAExecutionProvider" if self.cuda else "CPUExecutionProvider"
            if provider not in ort.get_available_providers():
                raise RuntimeError(f"{provider} unavailable; no CPU fallback permitted")
            path = directory / "actor.onnx"
            start = time.perf_counter_ns()
            # Export CPU constants; the execution provider owns device placement.
            self.actor.cpu()
            torch.onnx.export(
                self.actor,
                (self.history.cpu(), self.condition.cpu()),
                str(path),
                input_names=["history", "condition"],
                output_names=["action"],
                opset_version=17,
                dynamo=False,
            )
            self.setup["export_ms"] = (time.perf_counter_ns() - start) / 1e6
            self.setup["onnx_sha256"] = digest(path)
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            if self.cuda:
                options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            options.enable_profiling = True
            options.profile_file_prefix = str(directory / "ort-profile")
            start = time.perf_counter_ns()
            providers = (
                [
                    (
                        provider,
                        {
                            "enable_cuda_graph": "1" if route == "E" else "0",
                            "user_compute_stream": str(torch.cuda.current_stream().cuda_stream),
                            "use_tf32": "0",
                        },
                    )
                ]
                if self.cuda
                else [provider]
            )
            self.session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
            self.session.disable_fallback()
            self.setup["session_ms"] = (time.perf_counter_ns() - start) / 1e6
            self.setup["providers"] = self.session.get_providers()
            self.setup["provider_options"] = self.session.get_provider_options()
            if self.cuda:
                actual = self.setup["provider_options"].get("CUDAExecutionProvider", {})
                if actual.get("use_tf32") != "0":
                    raise RuntimeError("Cannot confirm TF32 is disabled with this CUDA provider version")
                if route == "E" and actual.get("enable_cuda_graph") != "1":
                    raise RuntimeError("CUDA Graph option was not enabled")
                self.binding = self.session.io_binding()
                self.output = torch.empty((1, 23), device="cuda")
                for name, tensor in (("history", self.history), ("condition", self.condition)):
                    self.binding.bind_input(
                        name, "cuda", torch.cuda.current_device(), np.float32, tuple(tensor.shape), tensor.data_ptr()
                    )
                self.binding.bind_output(
                    "action", "cuda", torch.cuda.current_device(), np.float32, (1, 23), self.output.data_ptr()
                )
        start = time.perf_counter_ns()
        self.run(self.split(tape["input"][0], int(tape["frame"][0])))
        self.setup["first_run_ms"] = (time.perf_counter_ns() - start) / 1e6
        # Provider profiling is only for placement verification, never timed iterations.
        if self.session:
            profile = Path(self.session.end_profiling())
            events = json.loads(profile.read_text())
            used = {e.get("args", {}).get("provider") for e in events} - {None}
            self.setup["executed_providers"] = sorted(used)
            shutil.move(str(profile), directory / "placement.json")
            if self.cuda and used != {"CUDAExecutionProvider"}:
                raise RuntimeError(f"Unexpected provider placement: {used}")

    def split(self, obs, frame):
        condition = self.cache[frame] if self.cache is not None else obs[750:]
        return np.ascontiguousarray(obs[:750].reshape(1, 10, 75)), np.ascontiguousarray(condition.reshape(1, -1))

    def upload(self, inputs):
        self.history.copy_(torch.from_numpy(inputs[0]))
        self.condition.copy_(torch.from_numpy(inputs[1]))
        if self.cuda:
            torch.cuda.synchronize()

    def forward(self):
        if self.session:
            if self.cuda:
                self.session.run_with_iobinding(self.binding)
                return self.output
            return self.session.run(["action"], {"history": self.history.numpy(), "condition": self.condition.numpy()})[
                0
            ]
        return self.actor(self.history, self.condition)

    def run(self, inputs):
        if not self.cuda:
            with torch.inference_mode():
                if self.session:
                    return self.session.run(["action"], dict(zip(("history", "condition"), inputs, strict=True)))[0][0]
                return self.actor(*(torch.from_numpy(x) for x in inputs)).numpy()[0]
        self.upload(inputs)
        with torch.inference_mode():
            result = self.forward()
        return result.cpu().numpy().copy()[0] if isinstance(result, torch.Tensor) else result[0].copy()

    def strategy_input(self, tape, index):
        env = restore(self.policy, tape, index)
        return env

    def observe(self, env, tape, index):
        if self.cache is None:
            obs, _ = self.policy.get_observation(env, controls(tape["command"][index]))
            return self.split(obs, int(tape["frame"][index]))
        history = self.policy._pack(self.policy._proprioception(env), self.cache[int(tape["frame"][index])])
        return history[:750].reshape(1, 10, 75), history[750:].reshape(1, 32)


def sleep_until(deadline):
    remaining = deadline - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)


def hardware_sample():
    paths = list(Path("/sys/class/thermal").glob("thermal_zone*/temp"))
    paths += list(Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_cur_freq"))
    sample = {"time": time.time()}
    for path in paths:
        with contextlib.suppress(OSError):
            sample[str(path)] = path.read_text().strip()
    return sample


def worker(args):
    task_dir = args.output_dir / args.task
    directory = task_dir / args.route / str(args.repeat)
    directory.mkdir(parents=True, exist_ok=True)
    result = {"task": args.task, "route": args.route, "repeat": args.repeat, "label": LABELS[args.route]}
    result["hardware_start"] = hardware_sample()
    start = time.perf_counter_ns()
    try:
        policy = load_policy(args.task)
        result["load_ms"] = (time.perf_counter_ns() - start) / 1e6
        with np.load(task_dir / "tape.npz", allow_pickle=False) as source:
            tape = {k: source[k] for k in source.files}
        setup_start = time.perf_counter_ns()
        backend = Backend(args.route, policy, directory, tape)
        result["backend_setup_ms"] = (time.perf_counter_ns() - setup_start) / 1e6
        result["setup"] = backend.setup
        actions, targets = [], []
        for i, obs in enumerate(tape["input"]):
            action = backend.run(backend.split(obs, int(tape["frame"][i])))
            actions.append(action)
            targets.append(postprocess(action, policy))
        result["correctness"] = {
            "action": error_metrics(actions, tape["action"]),
            "pd_target": error_metrics(targets, tape["target"]),
        }
        # Also validate the real preprocessing path, including reset/last-frame samples.
        for i in range(len(tape["input"])):
            env = backend.strategy_input(tape, i)
            action = backend.run(backend.observe(env, tape, i))
            if not error_metrics(action, tape["action"][i])["passed"]:
                raise ValueError(f"Strategy preprocessing mismatch at sample {i}")
        if not all(v["passed"] for v in result["correctness"].values()):
            raise ValueError("Output tolerance check failed")
        for i in range(args.warmup):
            index = i % len(tape["input"])
            backend.run(backend.split(tape["input"][index], int(tape["frame"][index])))
        result["measurements"] = {}
        with open(directory / "timings.csv", "w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("mode", "layer", "sample", "tape_index", "elapsed_ms", "start_lateness_ms"))
            for mode, count in (("continuous", args.iterations), ("paced", args.paced_iterations)):
                if count == 0:
                    continue
                # Stabilize at the deployment rate before measuring paced behavior.
                if mode == "paced":
                    deadline = time.perf_counter()
                    for i in range(args.paced_warmup):
                        backend.run(
                            backend.split(
                                tape["input"][i % len(tape["input"])], int(tape["frame"][i % len(tape["input"])])
                            )
                        )
                        deadline += 0.02
                        sleep_until(deadline)
                for layer in LAYERS:
                    begin = torch.cuda.Event(enable_timing=True) if backend.cuda else None
                    end = torch.cuda.Event(enable_timing=True) if backend.cuda else None
                    timings = []
                    deadline = time.perf_counter()
                    for step in range(count):
                        index = step % len(tape["input"])
                        env = backend.strategy_input(tape, index) if layer == "strategy" else None
                        inputs = backend.split(tape["input"][index], int(tape["frame"][index]))
                        if layer == "network":
                            backend.upload(inputs)
                        if mode == "paced":
                            sleep_until(deadline)
                        lateness = max(0.0, time.perf_counter() - deadline) * 1000 if mode == "paced" else 0.0
                        start = time.perf_counter_ns()
                        if layer == "network" and backend.cuda:
                            begin.record()
                            with torch.inference_mode():
                                backend.forward()
                            end.record()
                            end.synchronize()
                            elapsed = begin.elapsed_time(end)
                        elif layer == "network":
                            with torch.inference_mode():
                                backend.forward()
                            elapsed = (time.perf_counter_ns() - start) / 1e6
                        else:
                            if layer == "strategy":
                                inputs = backend.observe(env, tape, index)
                            postprocess(backend.run(inputs), policy)
                            elapsed = (time.perf_counter_ns() - start) / 1e6
                        timings.append(elapsed)
                        writer.writerow((mode, layer, step, index, elapsed, lateness))
                        deadline += 0.02
                    result["measurements"][f"{mode}/{layer}"] = summarize(timings)
        result["status"] = "passed"
    except Exception as exc:
        unavailable = isinstance(exc, ImportError) or "unavailable; no CPU fallback" in str(exc)
        result.update(status="skipped" if unavailable else "failed", reason=f"{type(exc).__name__}: {exc}")
    result["hardware_end"] = hardware_sample()
    write_json(directory / "result.json", result)
    return result


def report(directory, results):
    lines = [
        "# SuperMimic inference comparison",
        "",
        "FP32, batch=1; offline inference only. No hardware validation.",
        "",
        "| Task | Route | Repeat | Status | Paced strategy P50/P99 (ms) | >20ms |",
        "|---|---|---|---|---|---|",
    ]
    failures = []
    for r in results:
        m = r.get("measurements", {}).get("paced/strategy", {})
        latency = f"{m['p50_ms']:.3f}/{m['p99_ms']:.3f}" if m else "—"
        lines.append(
            f"| {r['task']} | {r['route']} {r['label']} | {r['repeat']} | {r['status']} | {latency} | "
            f"{m.get('over_20ms_fraction', '—')} |"
        )
        if r.get("reason"):
            failures.append(f"- {r['task']}/{r['route']}/{r['repeat']}: {r['reason']}")
    lines += ["", *failures, "", "## Decision", ""]
    for task in ("motion", "velocity"):
        baselines = [r for r in results if r["task"] == task and r["route"] == "A"]
        eligible = []
        for route in ROUTES[1:]:
            candidates = [r for r in results if r["task"] == task and r["route"] == route]
            if len(baselines) != 3 or len(candidates) != 3:
                continue
            pairs = [(next(b for b in baselines if b["repeat"] == c["repeat"]), c) for c in candidates]
            if all(
                b["status"] == c["status"] == "passed"
                and "paced/strategy" in b["measurements"]
                and "paced/strategy" in c["measurements"]
                and c["measurements"]["paced/strategy"]["p99_ms"] <= b["measurements"]["paced/strategy"]["p99_ms"] * 0.9
                and c["measurements"]["paced/strategy"]["over_20ms_fraction"]
                <= b["measurements"]["paced/strategy"]["over_20ms_fraction"]
                for b, c in pairs
            ):
                eligible.append(route)
        lines.append(
            f"- {task}: routes meeting the 10% P99 improvement rule in all three repeats: "
            f"{', '.join(eligible) if eligible else 'none / insufficient data'}. "
            "This is an inference result, not deployment approval."
        )
    write_json(directory / "summary.json", results)
    (directory / "report.md").write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("motion", "velocity", "all"), default="all")
    parser.add_argument("--routes", nargs="+", choices=ROUTES, default=list(ROUTES))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmark_results/supermimic") / time.strftime("%Y%m%d-%H%M%S")
    )
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--paced-iterations", type=int, default=3000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--paced-warmup", type=int, default=250)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--route", choices=ROUTES, help=argparse.SUPPRESS)
    parser.add_argument("--repeat", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.iterations, args.paced_iterations, args.warmup, args.paced_warmup) < 0 or args.repeats < 1:
        parser.error("counts must be nonnegative and repeats positive")
    if args.iterations + args.paced_iterations == 0:
        parser.error("at least one measurement mode must have samples")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    logging.getLogger("robojudo.policy.supermimic_policy").setLevel(logging.ERROR)
    if args.route:
        worker(args)
        return
    if args.output_dir.exists():
        raise SystemExit("Output directory already exists; use a new directory to avoid mixing runs")
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "environment.json", environment())
    write_json(
        args.output_dir / "arguments.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    )
    tasks = ("motion", "velocity") if args.task == "all" else (args.task,)
    jobs = []
    for task in tasks:
        directory = args.output_dir / task
        directory.mkdir()
        make_tape(task, directory, args.seed)
        jobs.extend(
            (task, route, repeat)
            for route in args.routes
            for repeat in range(args.repeats)
            if not (route in ("F", "G") and task == "velocity")
        )
    random.Random(args.seed).shuffle(jobs)
    results = []
    for task, route, repeat in jobs:
        print(f"Running {task}/{route}/{repeat}", flush=True)
        monitor = None
        path = args.output_dir / task / route / str(repeat)
        path.mkdir(parents=True)
        with open(path / "telemetry.log", "w") as telemetry, open(path / "worker.log", "w") as log:
            if shutil.which("tegrastats"):
                monitor = subprocess.Popen(["tegrastats", "--interval", "1000"], stdout=telemetry, stderr=telemetry)
            try:
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--task",
                        task,
                        "--route",
                        route,
                        "--repeat",
                        str(repeat),
                        "--output-dir",
                        str(args.output_dir.resolve()),
                        *sum(
                            (
                                [f"--{k.replace('_', '-')}", str(getattr(args, k))]
                                for k in ("iterations", "paced_iterations", "warmup", "paced_warmup")
                            ),
                            [],
                        ),
                    ],
                    stdout=log,
                    stderr=log,
                    check=False,
                )
            finally:
                if monitor:
                    monitor.terminate()
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        monitor.wait(timeout=5)
                    if monitor.poll() is None:
                        monitor.kill()
                        monitor.wait()
        result_file = path / "result.json"
        result = (
            json.loads(result_file.read_text())
            if result_file.exists()
            else {
                "task": task,
                "route": route,
                "repeat": repeat,
                "label": LABELS[route],
                "status": "failed",
                "reason": "Worker exited without a result; see worker.log",
            }
        )
        results.append(result)
        report(args.output_dir, results)
    print(args.output_dir / "report.md")


if __name__ == "__main__":
    main()
