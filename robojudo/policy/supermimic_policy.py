"""Direct checkpoint deployment of SuperMimic motion and velocity actors."""

import io
import logging
from collections import deque
from itertools import pairwise
from pathlib import Path

import numpy as np

import torch
from robojudo.config.g1.policy.g1_supermimic_policy_cfg import G1SuperMimicPolicyCfg
from robojudo.controller.velocity_source import get_pressed_velocity_keys, get_selected_velocity_source
from robojudo.policy import Policy, policy_registry
from robojudo.policy.utils.velocity_command import get_fresh_zmq_velocity
from robojudo.tools.tool_cfgs import DoFConfig
from torch import nn

logger = logging.getLogger(__name__)


def mlp(input_dim, hidden_dims, output_dim):
    layers = []
    for left, right in pairwise((input_dim, *hidden_dims)):
        layers.extend((nn.Linear(left, right), nn.ELU()))
    layers.append(nn.Linear(hidden_dims[-1], output_dim))
    return nn.Sequential(*layers)


class FrozenRunningNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim, dtype=torch.float64))
        self.register_buffer("var", torch.ones(dim, dtype=torch.float64))
        self.register_buffer("std", torch.ones(dim))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    def forward(self, inputs):
        return (inputs - self.mean.to(inputs.dtype)) / (self.std.to(inputs.dtype) + 1e-2)


class VelocityEncoder(nn.Module):
    def __init__(self, command_scale, observation_dim=93):
        super().__init__()
        self.register_buffer("command_scale", torch.tensor(command_scale, dtype=torch.float32))
        self.network = mlp(10 * observation_dim + 3, (256, 256), 32)

    def forward(self, history, command):
        return self.network(torch.cat((command / self.command_scale, history.flatten(-2)), -1)).tanh()


class SuperMimicActor(nn.Module):
    def __init__(self, config, default_q, *, velocity=False, command_scale=(2.0, 1.32, 3.35)):
        super().__init__()
        self.velocity = velocity
        self.num_actions = default_q.numel()
        if self.num_actions not in (23, 29):
            raise ValueError("SuperMimic supports G1 23-DoF and 29-DoF actors")
        self.reference_dim = config.get("reference_dim", 10 * (15 + 2 * self.num_actions))
        self.observation_dim = (
            6
            + 3 * self.num_actions
            + 6 * (config.get("relative_orientation", False) + config.get("motion_anchor_ori_b", False))
        )
        self.input_normalization = config.get("actor_input_normalization", False)
        self.obs_norm = FrozenRunningNorm(self.observation_dim) if self.input_normalization else nn.Identity()
        self.register_buffer("default_q", default_q.detach().clone().float())
        self.register_buffer(
            "observation_scale",
            torch.tensor(
                [0.3] * self.num_actions
                + [5.0] * self.num_actions
                + [3.14] * 3
                + [1.0] * (self.observation_dim - (2 * self.num_actions + 3))
            ),
        )
        hidden = tuple(config["actor_dims"])
        self.decoder = mlp(32 + 10 * self.observation_dim, hidden, self.num_actions)
        if velocity:
            self.encoder = VelocityEncoder(command_scale, self.observation_dim)
        else:
            self.goal_norm = FrozenRunningNorm(self.reference_dim) if self.input_normalization else nn.Identity()
            self.encoder = mlp(self.reference_dim, hidden, 32)

    def forward(self, history, command_or_goal):
        normalized = self.obs_norm(history) if self.input_normalization else history / self.observation_scale
        latent = (
            self.encoder(normalized, command_or_goal)
            if self.velocity
            else self.encoder(self.goal_norm(command_or_goal)).tanh()
        )
        return self.decoder(torch.cat((latent, normalized.flatten(-2)), -1))


class SuperMimicCachedDecoder(nn.Module):
    """Decode live history using a motion latent computed at startup."""

    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, history, latent):
        normalized = (
            self.actor.obs_norm(history) if self.actor.input_normalization else history / self.actor.observation_scale
        )
        return self.actor.decoder(torch.cat((latent, normalized.flatten(-2)), -1))


def onnx_cpu_session(model, observation_dim, condition_dim, threads):
    """Export FP32 in memory; startup fails explicitly if dependencies are missing."""
    try:
        import onnx  # noqa: F401
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "SuperMimic ONNX CPU inference requires onnx and onnxruntime; install both in this environment"
        ) from exc
    with io.BytesIO() as stream:
        torch.onnx.export(
            model.cpu().eval(),
            (torch.zeros(1, 10, observation_dim), torch.zeros(1, condition_dim)),
            stream,
            input_names=["history", "condition"],
            output_names=["action"],
            opset_version=17,
            dynamo=False,
        )
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        session = ort.InferenceSession(stream.getvalue(), sess_options=options, providers=["CPUExecutionProvider"])
    session.disable_fallback()
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("SuperMimic inference requires CPUExecutionProvider only")
    warmup = session.run(
        ["action"],
        {
            "history": np.zeros((1, 10, observation_dim), dtype=np.float32),
            "condition": np.zeros((1, condition_dim), dtype=np.float32),
        },
    )[0]
    if not np.isfinite(warmup).all():
        raise ValueError("SuperMimic ONNX warmup returned a non-finite action")
    return session


def quat_conjugate(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def quat_mul(a, b):
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return torch.cat((aw * bw - (av * bv).sum(-1, keepdim=True), aw * bv + bw * av + torch.linalg.cross(av, bv)), -1)


def quat_apply(q, v):
    axis, v = torch.broadcast_tensors(q[..., 1:], v)
    cross = 2 * torch.linalg.cross(axis, v)
    return v + q[..., :1] * cross + torch.linalg.cross(axis, cross)


def yaw_quat(q):
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    zero = torch.zeros_like(yaw)
    return torch.stack((torch.cos(yaw / 2), zero, zero, torch.sin(yaw / 2)), -1)


def orientation_columns(q, *, interleaved=False):
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
    x = quat_apply(q, q.new_tensor([1, 0, 0]))
    y = quat_apply(q, q.new_tensor([0, 1, 0]))
    return torch.stack((x, y), -1).flatten(-2) if interleaved else torch.cat((x, y), -1)


def reference_features(reference, default_q, reference_dim):
    """Preserve the full-body and root/joint layouts for both G1 variants."""
    root_quat = reference["body_quat_w"][..., 0, :]
    inverse_yaw = quat_conjugate(yaw_quat(root_quat[..., 0, :]))
    if reference_dim == 10 * (210 + 2 * default_q.numel()):
        root = reference["body_pos_w"][..., 0, 0, :]
        origin = torch.cat((root[..., :2], torch.zeros_like(root[..., 2:3])), -1)
        rotation = inverse_yaw[..., None, None, :]
        position = quat_apply(rotation, reference["body_pos_w"] - origin[..., None, None, :])
        orientation = orientation_columns(quat_mul(rotation, reference["body_quat_w"]))
        parts = (
            (reference["joint_pos"] - default_q) / 0.3,
            reference["joint_vel"] / 5,
            (position / 0.3).flatten(-2),
            orientation.flatten(-2),
            quat_apply(rotation, reference["body_lin_vel_w"]).flatten(-2),
            (quat_apply(rotation, reference["body_ang_vel_w"]) / 3.14).flatten(-2),
        )
    elif reference_dim == 10 * (15 + 2 * default_q.numel()):
        root = reference["body_pos_w"][..., 0, :]
        rotation = inverse_yaw[..., None, :]
        parts = (
            quat_apply(rotation, root - root[..., :1, :]),
            orientation_columns(quat_mul(rotation, root_quat)),
            quat_apply(quat_conjugate(root_quat), reference["body_lin_vel_w"][..., 0, :]),
            quat_apply(quat_conjugate(root_quat), reference["body_ang_vel_w"][..., 0, :]),
            reference["joint_pos"],
            reference["joint_vel"] * 0.1,
        )
    else:
        raise ValueError(f"Unsupported SuperMimic reference width: {reference_dim}")
    return torch.cat(parts, -1).flatten(-2)


TRACKED_BODIES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)


MOTION_FIELDS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def tracked_bodies(num_joints):
    if num_joints == 23:
        return tuple(name.replace("wrist_yaw_link", "wrist_roll_rubber_hand") for name in TRACKED_BODIES)
    if num_joints == 29:
        return TRACKED_BODIES
    raise ValueError("SuperMimic motions require 23 or 29 joints")


class SuperMimicMotion:
    def __init__(self, motion_file, joint_names):
        self.body_names = tracked_bodies(len(joint_names))
        if motion_file is None:
            raise ValueError("SuperMimic requires a motion file; set policy.motion_file in g1_supermimic_cfg.py")
        path = Path(motion_file)
        self.name = path.stem
        if path.suffix.lower() != ".npz":
            raise ValueError("SuperMimic requires a converted motion NPZ file")
        if not path.is_file():
            raise FileNotFoundError(f"SuperMimic motion file not found: {path}")
        self._load_clip(path, joint_names)

    def _load_clip(self, path, joint_names):
        self.data = {}
        with np.load(path, allow_pickle=False) as source:
            required = set(MOTION_FIELDS) | {"fps", "quaternion_order", "joint_names", "body_names"}
            missing = required.difference(source.files)
            if missing:
                raise ValueError(f"Missing SuperMimic motion fields: {sorted(missing)}")
            if source["quaternion_order"].item() != "wxyz" or not np.isclose(source["fps"].item(), 50):
                raise ValueError("SuperMimic clip must use 50 Hz and wxyz quaternions")
            if source["joint_pos"].ndim != 2:
                raise ValueError("SuperMimic joint_pos must have shape (frames, joints)")
            self.length = source["joint_pos"].shape[0]
            if self.length < 2:
                raise ValueError("SuperMimic motion must contain at least two frames")
            stored_joints, stored_bodies = source["joint_names"].tolist(), source["body_names"].tolist()
            if (
                len(stored_joints) != len(joint_names)
                or len(set(stored_joints)) != len(joint_names)
                or set(stored_joints) != set(joint_names)
            ):
                raise ValueError("SuperMimic motion joints do not match the checkpoint")
            if len(set(stored_bodies)) != len(stored_bodies) or not set(self.body_names).issubset(stored_bodies):
                raise ValueError("SuperMimic motion is missing tracked bodies or has duplicate body names")
            joint_ids = [stored_joints.index(name) for name in joint_names]
            body_ids = [stored_bodies.index(name) for name in self.body_names]
            for key in MOTION_FIELDS:
                raw = source[key]
                expected = (
                    (self.length, len(stored_joints))
                    if key.startswith("joint_")
                    else (self.length, len(stored_bodies), 4 if key == "body_quat_w" else 3)
                )
                if raw.shape != expected or not np.isfinite(raw).all():
                    raise ValueError(f"Invalid SuperMimic motion array {key}: {raw.shape}")
                if key == "body_quat_w" and not np.allclose(np.linalg.norm(raw, axis=-1), 1, atol=1e-5):
                    raise ValueError("SuperMimic reference quaternions must be unit quaternions")
                self.data[key] = torch.from_numpy(
                    raw[:, joint_ids if key.startswith("joint_") else body_ids].copy()
                ).float()

    def future(self, frame):
        indices = torch.arange(frame, frame + 10).clamp(max=self.length - 1)
        return {name: values[indices] for name, values in self.data.items()}


def _vector(value, size, name):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"SuperMimic {name} must be a finite ({size},) vector")
    return result


class SuperMimicPolicyBase(Policy):
    velocity = False

    def __init__(self, cfg_policy: G1SuperMimicPolicyCfg, device="cpu"):
        self.ort_session = None
        self.inference_backend = cfg_policy.inference_backend
        if self.inference_backend == "onnx_cpu":
            device = "cpu"
        if not cfg_policy.checkpoint.is_file():
            raise FileNotFoundError(f"SuperMimic checkpoint not found: {cfg_policy.checkpoint}; set policy.checkpoint")
        saved = torch.load(cfg_policy.checkpoint, map_location="cpu", weights_only=True, mmap=True)
        kinds = ("supermimic_velocity_actor",) if self.velocity else ("supermimic_actor", "supermimic_online")
        if saved.get("kind") not in kinds or saved.get("schema_version") != 1:
            raise ValueError(f"Expected schema-1 SuperMimic checkpoint of kind {kinds}")
        config = dict(saved["config"])
        metadata = saved["robot_metadata"]
        names = metadata["joint_names"]
        bootstrap = cfg_policy.obs_dof
        num_actions = bootstrap.num_dofs
        if (
            num_actions not in (23, 29)
            or len(names) != num_actions
            or len(set(names)) != num_actions
            or set(names) != set(bootstrap.joint_names)
        ):
            raise ValueError(f"SuperMimic checkpoint joints must match the configured {num_actions}-DoF G1")
        if config.get("robot", f"g1_{num_actions}dof") != f"g1_{num_actions}dof":
            raise ValueError("SuperMimic checkpoint robot variant does not match the policy configuration")
        if config.get("relative_orientation", False):
            raise ValueError("This deployment requires relative_orientation=False")
        if self.velocity:
            if (
                saved.get("observation_schema") != f"proprioceptive_{6 + 3 * num_actions}_v1"
                or saved.get("history_length") != 10
                or config.get("motion_anchor_ori_b", False)
            ):
                raise ValueError("Velocity checkpoint must use ten proprioceptive observations without a motion anchor")
        else:
            width = saved["actor"]["encoder.0.weight"].shape[1]
            schema = "full_body_256_v1" if num_actions == 23 else "full_body_268"
            if width != cfg_policy.reference_dim or saved.get("reference_schema", schema) != schema:
                raise ValueError(f"Motion deployment requires {schema} / {cfg_policy.reference_dim}-D references")
            if config.get("motion_anchor_ori_b", False) != cfg_policy.motion_anchor_ori_b:
                raise ValueError("Motion checkpoint motion_anchor_ori_b does not match the policy configuration")
            config["reference_dim"] = width
        if not np.isclose(saved.get("simulation", {}).get("control_dt", 0.02), 1 / cfg_policy.freq):
            raise ValueError("Checkpoint control period does not match the 50 Hz deployment")
        if cfg_policy.action_dof.joint_names != bootstrap.joint_names:
            raise ValueError("SuperMimic observation and action joint layouts must match")
        default = _vector(metadata["default_q"], num_actions, "checkpoint default_q")
        order = [bootstrap.joint_names.index(name) for name in names]
        dof = DoFConfig(
            joint_names=names,
            default_pos=default.tolist(),
            **{
                key: [getattr(bootstrap, key)[i] for i in order]
                for key in ("stiffness", "damping", "torque_limits", "position_limits")
            },
        )
        cfg_policy = cfg_policy.model_copy(update={"obs_dof": dof, "action_dof": dof})
        super().__init__(cfg_policy, device)
        if self.action_beta != 1.0 or self.cfg_policy.action_scale != 1.0:
            raise ValueError("SuperMimic requires action_beta=1 and action_scale=1; use per-joint action_scales")
        self.action_scales = _vector(cfg_policy.action_scales, num_actions, "action_scales")[order]
        if (self.action_scales <= 0).any():
            raise ValueError("SuperMimic action_scales must be positive")
        if str(device).startswith("cpu"):
            torch.set_num_threads(cfg_policy.torch_num_threads)
        command_scale = saved.get("velocity_config", {}).get("command_scale", (2.0, 1.32, 3.35))
        if (_vector(command_scale, 3, "checkpoint command_scale") <= 0).any():
            raise ValueError("SuperMimic command_scale must be positive")
        self.model = SuperMimicActor(
            config, torch.from_numpy(default), velocity=self.velocity, command_scale=command_scale
        )
        self.model.load_state_dict(saved["actor"], strict=True)
        if not torch.equal(self.model.default_q, torch.from_numpy(default)):
            raise ValueError("Checkpoint actor default_q and robot metadata disagree")
        if self.velocity and not torch.equal(self.model.encoder.command_scale, torch.tensor(command_scale).float()):
            raise ValueError("Velocity command scale weights and metadata disagree")
        for name, value in self.model.state_dict().items():
            if not torch.isfinite(value).all():
                raise ValueError(f"Non-finite checkpoint tensor: {name}")
        self.model.to(device).eval()
        self.observation_dim = self.model.observation_dim
        self.condition_dim = 3 if self.velocity else self.model.reference_dim
        self.input_size = 10 * self.observation_dim + self.condition_dim
        self.saved = saved
        self.reset()
        if self.velocity and self.inference_backend == "onnx_cpu":
            self.ort_session = onnx_cpu_session(self.model, self.observation_dim, 3, cfg_policy.torch_num_threads)
        logger.info(
            "Loaded %s from %s (iteration %s)",
            type(self).__name__,
            cfg_policy.checkpoint,
            saved.get("iteration", "unknown"),
        )

    def reset(self):
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.history = deque(maxlen=10)
        self.timestep = 0

    def _proprioception(self, env_data, anchor=None):
        quat = torch.from_numpy(_vector(env_data.base_quat, 4, "base_quat")[[3, 0, 1, 2]])
        if not torch.isclose(quat.norm(), torch.tensor(1.0), atol=1e-4):
            raise ValueError("SuperMimic base_quat must be a unit quaternion")
        gravity = quat_apply(quat_conjugate(quat), torch.tensor([0.0, 0.0, -1.0])).numpy()
        return np.concatenate(
            (
                _vector(env_data.dof_pos, self.num_dofs, "dof_pos") - self.default_dof_pos,
                _vector(env_data.dof_vel, self.num_dofs, "dof_vel"),
                _vector(env_data.base_ang_vel, 3, "base_ang_vel"),
                gravity,
                *([] if anchor is None else [anchor]),
                self.last_action,
            )
        ).astype(np.float32)

    def _pack(self, observation, condition):
        if not self.history:
            self.history.extend(observation.copy() for _ in range(10))
        else:
            self.history.append(observation.copy())
        return np.concatenate((*self.history, np.asarray(condition).reshape(-1))).astype(np.float32)

    def get_action(self, obs):
        values = _vector(obs, self.input_size, "policy input")
        history = values[: 10 * self.observation_dim].reshape(1, 10, self.observation_dim)
        condition = values[10 * self.observation_dim :][None]
        if self.ort_session is not None:
            actions = self.ort_session.run(["action"], {"history": history, "condition": condition})[0][0]
        else:
            with torch.inference_mode():
                actions = (
                    self.model(torch.from_numpy(history).to(self.device), torch.from_numpy(condition).to(self.device))
                    .cpu()
                    .numpy()[0]
                )
        actions = _vector(actions, self.num_actions, "policy action")
        if self.action_clip is not None:
            actions = np.clip(actions, -self.action_clip, self.action_clip)
        self.last_action = actions.copy()
        return actions * self.action_scales

    def post_step_callback(self, commands=None):
        self.timestep += 1


@policy_registry.register
class SuperMimicVelocityPolicy(SuperMimicPolicyBase):
    velocity = True

    def reset(self):
        super().reset()
        self._last_command_was_zero = None

    def _get_command(self, ctrl_data):
        source = get_selected_velocity_source(ctrl_data)
        command = np.zeros(3, dtype=np.float32)
        if source == "VelocityZmqCtrl":
            fresh = get_fresh_zmq_velocity(ctrl_data[source])
            if fresh is not None:
                command = fresh
        elif source == "KeyboardCtrl" and ctrl_data[source].get("fresh", True):
            keys = get_pressed_velocity_keys(ctrl_data[source])
            command = (
                np.asarray(
                    [("w" in keys) - ("s" in keys), ("a" in keys) - ("d" in keys), ("q" in keys) - ("e" in keys)]
                )
                * self.cfg_policy.keyboard_command
            )
        elif source in ("JoystickCtrl", "RosJoystickCtrl", "UnitreeCtrl") and ctrl_data[source].get("fresh", True):
            axes = ctrl_data[source]["axes"]
            # RoboJuDo joystick axes: forward/left/counterclockwise need opposite signs.
            command = np.asarray([axes["LeftY"], -axes["LeftX"], -axes["RightX"]]) * self.cfg_policy.command_limits
        limit = _vector(self.cfg_policy.command_limits, 3, "command_limits")
        if (limit <= 0).any():
            raise ValueError("Velocity command limits must be positive")
        return np.clip(_vector(command, 3, "velocity command"), -limit, limit)

    def get_observation(self, env_data, ctrl_data):
        command = self._get_command(ctrl_data)
        is_zero = not np.any(command)
        if is_zero != self._last_command_was_zero:
            logger.info(
                "SuperMimic velocity command: vx=%.3f m/s, vy=%.3f m/s, wz=%.3f rad/s (source=%s)",
                *command,
                get_selected_velocity_source(ctrl_data) or "none",
            )
            self._last_command_was_zero = is_zero
        return self._pack(self._proprioception(env_data), command), {"commands": command}


@policy_registry.register
class SuperMimicMotionPolicy(SuperMimicPolicyBase):
    def __init__(self, cfg_policy, device="cpu"):
        super().__init__(cfg_policy, device)
        self.motion = SuperMimicMotion(cfg_policy.motion_file, self.cfg_obs_dof.joint_names)
        self.latent_cache = None
        if self.inference_backend == "onnx_cpu":
            logger.info("Preparing ONNX CPU motion decoder and %s cached latents", self.motion.length)
            with torch.inference_mode():
                latents = []
                for frame in range(self.motion.length):
                    goal = reference_features(self.motion.future(frame), self.model.default_q, self.model.reference_dim)
                    latents.append(self.model.encoder(self.model.goal_norm(goal[None])).tanh()[0].numpy().copy())
                self.latent_cache = np.stack(latents)
            if not np.isfinite(self.latent_cache).all():
                raise ValueError("Non-finite SuperMimic cached motion latent")
            self.condition_dim = 32
            self.input_size = 10 * self.observation_dim + 32
            self.ort_session = onnx_cpu_session(
                SuperMimicCachedDecoder(self.model), self.observation_dim, 32, cfg_policy.torch_num_threads
            )
            logger.info(
                "Cached %s motion latents (%s bytes); using ONNX CPU decoder",
                len(self.latent_cache),
                self.latent_cache.nbytes,
            )

    def reset(self):
        super().reset()
        self.heading_offset = None
        self._paused = False

    def post_step_callback(self, commands=None):
        commands = commands or []
        if "[MOTION_RESET]" in commands or "[MOTION_FADE_IN]" in commands:
            self.reset()
            return
        if "[MOTION_FADE_OUT]" in commands:
            self._paused = True
        if not self._paused:
            self.timestep += 1

    def reset_alignment(self, env_data=None):
        del env_data
        self.heading_offset = None
        self.history.clear()

    def get_init_dof_pos(self):
        return self.motion.data["joint_pos"][0].numpy().copy()

    def get_observation(self, env_data, ctrl_data):
        del ctrl_data
        frame = (
            self.timestep % self.motion.length
            if self.cfg_policy.loop_motion
            else min(self.timestep, self.motion.length - 1)
        )
        anchor = None
        if self.cfg_policy.motion_anchor_ori_b:
            robot = torch.from_numpy(_vector(env_data.torso_quat, 4, "torso_quat")[[3, 0, 1, 2]])
            if not torch.isclose(robot.norm(), torch.tensor(1.0), atol=1e-4):
                raise ValueError("SuperMimic torso_quat must be a unit quaternion")
            target = self.motion.data["body_quat_w"][frame, self.motion.body_names.index("torso_link")]
            if self.heading_offset is None:
                self.heading_offset = quat_mul(yaw_quat(robot), quat_conjugate(yaw_quat(target)))
            aligned = quat_mul(self.heading_offset, target)
            anchor = orientation_columns(quat_mul(quat_conjugate(robot), aligned), interleaved=True).numpy()
        goal = (
            self.latent_cache[frame]
            if self.latent_cache is not None
            else reference_features(
                self.motion.future(frame), self.model.default_q.detach().cpu(), self.model.reference_dim
            ).numpy()
        )
        done = not self.cfg_policy.loop_motion and self.timestep >= self.motion.length
        return self._pack(self._proprioception(env_data, anchor), goal), {
            "motion_name": self.motion.name,
            "frame": frame,
            "CALLBACK": ["[MOTION_DONE]"] if done else [],
        }
