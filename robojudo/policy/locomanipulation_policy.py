import logging
from collections import deque

import numpy as np

from robojudo.controller.velocity_source import JOYSTICK_SOURCE_TYPES, get_selected_velocity_source
from robojudo.policy import Policy, PolicyCfg
from robojudo.policy.onnx_runtime import create_onnx_session
from robojudo.policy.utils.velocity_command import clip_velocity, get_fresh_zmq_velocity
from robojudo.utils.util_func import command_remap, quat_rotate_inverse_np

logger = logging.getLogger(__name__)


class LocomanipulationPolicyBase(Policy):
    """Robot-neutral ONNX Locomanipulation runtime shared by G1 and X2."""

    def __init__(self, cfg_policy: PolicyCfg, device: str):
        super().__init__(cfg_policy=cfg_policy, device=device)
        self.session = create_onnx_session(
            cfg_policy.policy_file,
            cfg_policy,
            providers=self._providers_for_device(device),
        )
        self.action_scales = np.asarray(cfg_policy.action_scales, dtype=np.float32)
        self._model_label = f"{cfg_policy.robot.upper()} locomanipulation"
        self._validate_model_contract()
        logger.info("Loaded %s ONNX model from %s", self._model_label, cfg_policy.policy_file)

        self.commands_map = cfg_policy.commands_map
        self._command_defaults = np.asarray([entry[1] for entry in self.commands_map], dtype=np.float32)
        self.cmd = self._command_defaults.copy()
        self.base_height_default = float(self._command_defaults[3])
        self.reset()

    @staticmethod
    def _providers_for_device(device: str) -> list[str]:
        if device == "cpu":
            return ["CPUExecutionProvider"]
        if device == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "tensorrt":
            return ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        raise ValueError(f"Unknown device: {device}")

    def reset(self):
        self.timestep = 0
        self.cmd = self._command_defaults.copy()
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.current_vel_cmd = np.zeros(3, dtype=np.float32)
        self._target_height = self.base_height_default
        self._target_waist_yaw = self.commands_map[4][1]
        self._held_keys: set[str] = set()
        self.history_term_bufs: dict[str, deque[np.ndarray]] = {
            key: deque(maxlen=self.cfg_policy.history_length) for key in self.cfg_policy.history_obs_dims
        }

    def post_step_callback(self, commands: list[str] | None = None):
        del commands
        self.timestep += 1

    def _get_commands(self, ctrl_data) -> np.ndarray:
        commands = self.cmd.copy()
        target_vel = np.zeros(3, dtype=np.float32)
        selected = get_selected_velocity_source(ctrl_data)

        for key in JOYSTICK_SOURCE_TYPES.intersection(ctrl_data.keys()):
            for event in ctrl_data[key]["button_event"]:
                if event["type"] != "button" or not event["pressed"]:
                    continue
                if event["name"] in ("Up", "Down"):
                    sign = 1.0 if event["name"] == "Up" else -1.0
                    self._target_height = self._clip_command(
                        self._target_height + sign * self.cfg_policy.height_step,
                        self.commands_map[3],
                    )
                elif event["name"] in ("Left", "Right"):
                    sign = 1.0 if event["name"] == "Left" else -1.0
                    self._target_waist_yaw = self._clip_command(
                        self._target_waist_yaw + sign * self.cfg_policy.waist_yaw_step,
                        self.commands_map[4],
                    )
                elif event["name"] in ("Back", "Select", "F1"):
                    self._reset_commands()

        keyboard_entry = ctrl_data.get("KeyboardCtrl")
        if keyboard_entry is not None:
            events = keyboard_entry["keyboard_event"]
            if "pressed_keys" in keyboard_entry:
                self._held_keys = set(keyboard_entry["pressed_keys"])
            else:
                for event in events:
                    if event["type"] != "keyboard":
                        continue
                    if event["pressed"]:
                        self._held_keys.add(event["name"])
                    else:
                        self._held_keys.discard(event["name"])

            for event in events:
                if event["type"] != "keyboard" or not event["pressed"]:
                    continue
                if event["name"] == "r":
                    self._target_height = self._clip_command(
                        self._target_height + self.cfg_policy.height_step, self.commands_map[3]
                    )
                elif event["name"] == "f":
                    self._target_height = self._clip_command(
                        self._target_height - self.cfg_policy.height_step, self.commands_map[3]
                    )
                elif event["name"] == "z":
                    self._target_waist_yaw = self._clip_command(
                        self._target_waist_yaw + self.cfg_policy.waist_yaw_step,
                        self.commands_map[4],
                    )
                elif event["name"] == "c":
                    self._target_waist_yaw = self._clip_command(
                        self._target_waist_yaw - self.cfg_policy.waist_yaw_step,
                        self.commands_map[4],
                    )
                elif event["name"] == "x":
                    self._reset_commands()

        if selected == "VelocityZmqCtrl":
            velocity = get_fresh_zmq_velocity(ctrl_data[selected])
            if velocity is not None:
                target_vel = clip_velocity(velocity, self.commands_map[:3])
        elif selected in JOYSTICK_SOURCE_TYPES:
            axes = ctrl_data[selected]["axes"]
            lx, ly, rx = (
                axis if abs(axis) >= 0.1 else 0.0
                for axis in (axes["LeftX"], axes["LeftY"], axes["RightX"])
            )
            target_vel[0] = command_remap(ly, self.commands_map[0])
            target_vel[1] = command_remap(lx, self.commands_map[1])
            target_vel[2] = command_remap(rx, self.commands_map[2])
        elif selected == "KeyboardCtrl":
            vel_keys = {
                "w": (0, 1.0),
                "s": (0, -1.0),
                "a": (1, 1.0),
                "d": (1, -1.0),
                "q": (2, 1.0),
                "e": (2, -1.0),
            }
            for held_key in self._held_keys:
                if held_key in vel_keys:
                    axis, sign = vel_keys[held_key]
                    low = min(self.commands_map[axis][0], self.commands_map[axis][2])
                    high = max(self.commands_map[axis][0], self.commands_map[axis][2])
                    target_vel[axis] += sign * (high if sign > 0 else abs(low))
        if np.linalg.norm(target_vel) < 1e-6:
            self.current_vel_cmd *= self.cfg_policy.command_decay
            if np.linalg.norm(self.current_vel_cmd) < self.cfg_policy.standing_command_threshold:
                self.current_vel_cmd[:] = 0.0
        else:
            for index in range(3):
                self.current_vel_cmd[index] = self._clip_command(target_vel[index], self.commands_map[index])

        commands[:3] = self.current_vel_cmd
        commands[3] = self._smooth_command(self.cmd[3], self._target_height)
        commands[4] = self._smooth_command(self.cmd[4], self._target_waist_yaw)
        self.cmd[3:5] = commands[3:5]

        print(
            f"\rvel=({commands[0]:+.1f}, {commands[1]:+.1f}, "
            f"{commands[2]:+.1f}) h={commands[3]:.3f} wy={commands[4]:+.2f}",
            end="",
            flush=True,
        )
        return commands

    @staticmethod
    def _clip_command(value: float, command_map: list[float]) -> float:
        return float(np.clip(value, min(command_map[0], command_map[2]), max(command_map[0], command_map[2])))

    def _smooth_command(self, current: float, target: float) -> float:
        value = self.cfg_policy.command_decay * current + (1.0 - self.cfg_policy.command_decay) * target
        return float(target if abs(value - target) < 0.001 else value)

    def _reset_commands(self):
        self.current_vel_cmd[:] = 0.0
        self._target_height = self.base_height_default
        self._target_waist_yaw = self.commands_map[4][1]

    def _get_phase(self, commands: np.ndarray) -> np.ndarray:
        phase_fraction = (self.timestep * self.dt) % self.cfg_policy.gait_period / self.cfg_policy.gait_period
        phase = np.asarray(
            [np.sin(phase_fraction * 2.0 * np.pi), np.cos(phase_fraction * 2.0 * np.pi)],
            dtype=np.float32,
        )
        if np.linalg.norm(commands[:3]) < self.cfg_policy.standing_command_threshold:
            phase[:] = 0.0
        return phase

    def get_observation(self, env_data, ctrl_data):
        commands = self._get_commands(ctrl_data)
        projected_gravity = quat_rotate_inverse_np(
            env_data.base_quat,
            np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
        ).astype(np.float32)
        obs_terms = {
            "base_ang_vel": np.asarray(env_data.base_ang_vel, dtype=np.float32),
            "projected_gravity": projected_gravity,
            "command": commands[:3],
            "base_height_command": commands[3:4],
            "waist_yaw_command": commands[4:5],
            "phase": self._get_phase(commands),
            "joint_pos": np.asarray(env_data.dof_pos - self.default_dof_pos, dtype=np.float32),
            "joint_vel": np.asarray(env_data.dof_vel, dtype=np.float32),
            "actions": self.last_action.copy(),
        }

        parts = []
        for key, expected_dim in self.cfg_policy.history_obs_dims.items():
            term = obs_terms[key]
            if term.shape != (expected_dim,):
                raise ValueError(
                    f"{self._model_label} observation term {key} has shape {term.shape}, "
                    f"expected {(expected_dim,)}"
                )
            history = self.history_term_bufs[key]
            if not history:
                history.extend(term.copy() for _ in range(self.cfg_policy.history_length))
            else:
                history.append(term.copy())
            parts.extend(history)

        obs = np.concatenate(parts).astype(np.float32)
        if obs.shape != (self.cfg_policy.num_obs,):
            raise ValueError(
                f"{type(self).__name__} observation shape {obs.shape} != ({self.cfg_policy.num_obs},)"
            )
        if not np.isfinite(obs).all():
            raise FloatingPointError(f"{self._model_label} observation contains non-finite values")
        return obs, {"locomotion_command": commands.copy()}

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        model_obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        if model_obs.shape != (1, self.cfg_policy.num_obs):
            raise ValueError(
                f"{self._model_label} ONNX observation shape {model_obs.shape} "
                f"!= (1, {self.cfg_policy.num_obs})"
            )
        raw_action = self.session.run(["actions"], {"obs": model_obs})[0].reshape(-1).astype(np.float32)
        if raw_action.shape != (self.num_actions,):
            raise ValueError(
                f"{self._model_label} action shape {raw_action.shape} != ({self.num_actions},)"
            )
        if not np.isfinite(raw_action).all():
            raise FloatingPointError(f"{self._model_label} policy produced a non-finite action")
        if self.action_clip is not None:
            raw_action = np.clip(raw_action, -self.action_clip, self.action_clip)
        self.last_action = raw_action.copy()
        return raw_action * self.action_scales

    def _validate_model_contract(self):
        input_shapes = {value.name: value.shape for value in self.session.get_inputs()}
        output_shapes = {value.name: value.shape for value in self.session.get_outputs()}
        if input_shapes != {"obs": [1, self.cfg_policy.num_obs]}:
            raise ValueError(f"Unexpected {self._model_label} ONNX inputs: {input_shapes}")
        if output_shapes.get("actions") != [1, self.num_actions]:
            raise ValueError(f"Unexpected {self._model_label} ONNX outputs: {output_shapes}")

        metadata = self.session.get_modelmeta().custom_metadata_map
        joint_names = metadata.get("joint_names", "").split(",")
        if joint_names != self.cfg_obs_dof.joint_names:
            raise ValueError(f"{self._model_label} ONNX joint_names do not match the policy configuration")
        observation_names = metadata.get("observation_names", "").split(",")
        if observation_names != list(self.cfg_policy.history_obs_dims):
            raise ValueError(
                f"{self._model_label} ONNX observation_names do not match the policy configuration"
            )
        command_names = metadata.get("command_names", "").split(",")
        if command_names != ["twist", "base_height", "waist_yaw"]:
            raise ValueError(f"Unexpected {self._model_label} command metadata: {command_names}")

        self._validate_rounded_metadata(metadata, "action_scale", self.action_scales)
        self._validate_rounded_metadata(metadata, "default_joint_pos", self.default_dof_pos)
        self._validate_rounded_metadata(metadata, "joint_stiffness", self.cfg_obs_dof.stiffness)
        self._validate_rounded_metadata(metadata, "joint_damping", self.cfg_obs_dof.damping)

    def _validate_rounded_metadata(self, metadata: dict[str, str], key: str, expected) -> None:
        actual = np.asarray([float(value) for value in metadata.get(key, "").split(",")], dtype=np.float64)
        expected_array = np.asarray(expected, dtype=np.float64)
        tolerance = 5e-4 + np.finfo(np.float64).eps
        if actual.shape != expected_array.shape or not np.allclose(
            actual, expected_array, rtol=0.0, atol=tolerance
        ):
            raise ValueError(
                f"{self._model_label} ONNX {key} metadata does not match the policy configuration"
            )
