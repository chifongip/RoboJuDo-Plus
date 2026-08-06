import logging

import numpy as np
import onnxruntime as ort

from robojudo.environment.utils.mujoco_viz import MujocoVisualizer
from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import BeyondMimicPolicyCfg
from robojudo.tools.dof import DoFConfig
from robojudo.utils.progress import ProgressBar
from robojudo.utils.rotation import TransformAlignment
from robojudo.utils.util_func import matrix_from_quat, subtract_frame_transforms

logger = logging.getLogger(__name__)


def _parse_metadata_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_metadata_floats(value: str) -> list[float]:
    return [float(item) for item in _parse_metadata_strings(value)]


class BeyondMimicPolicyBase(Policy):
    """Robot-neutral runtime for mjlab BeyondMimic-style tracking exports."""

    cfg_policy: BeyondMimicPolicyCfg

    _required_metadata = {
        "action_scale",
        "anchor_body_name",
        "body_names",
        "default_joint_pos",
        "joint_damping",
        "joint_names",
        "joint_stiffness",
        "observation_names",
    }
    _reference_output_names = (
        "actions",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
    )

    def __init__(self, cfg_policy: BeyondMimicPolicyCfg, device):
        sess_options = ort.SessionOptions()
        self.session = ort.InferenceSession(
            cfg_policy.policy_file,
            sess_options,
            providers=self._providers_for_device(device),
        )

        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        input_shapes = {value.name: value.shape for value in self.session.get_inputs()}
        obs_shape = input_shapes.get("obs")
        self.expected_observation_size = obs_shape[-1] if obs_shape and isinstance(obs_shape[-1], int) else None
        self.motion_anchor_body_index = -1
        self.modelmeta_dict = self.session.get_modelmeta().custom_metadata_map
        self.observation_names = _parse_metadata_strings(self.modelmeta_dict.get("observation_names", ""))
        self.model_joint_names = _parse_metadata_strings(self.modelmeta_dict.get("joint_names", ""))
        self.model_joint_count = len(self.model_joint_names)
        self._validate_model_contract(cfg_policy)
        self._validate_observation_contract(cfg_policy)

        cfg_policy_new = cfg_policy.model_copy()
        if cfg_policy_new.use_modelmeta_config:
            logger.info("[%s] Using ONNX metadata as configuration", type(self).__name__)
            dof_config = DoFConfig(
                joint_names=self.model_joint_names,
                default_pos=_parse_metadata_floats(self.modelmeta_dict["default_joint_pos"]),
                stiffness=_parse_metadata_floats(self.modelmeta_dict["joint_stiffness"]),
                damping=_parse_metadata_floats(self.modelmeta_dict["joint_damping"]),
            )

            anchor_body_name = self.modelmeta_dict["anchor_body_name"]
            body_names = _parse_metadata_strings(self.modelmeta_dict["body_names"])
            self.motion_anchor_body_index = body_names.index(anchor_body_name)

            cfg_policy_new.action_dof = dof_config
            cfg_policy_new.obs_dof = dof_config
            cfg_policy_new.action_scales = _parse_metadata_floats(self.modelmeta_dict["action_scale"])
        else:
            if cfg_policy_new.obs_dof.joint_names != self.model_joint_names:
                raise ValueError("BeyondMimic ONNX joint_names do not match the observation DoF configuration")
            if cfg_policy_new.action_dof.joint_names != self.model_joint_names:
                raise ValueError("BeyondMimic ONNX joint_names do not match the action DoF configuration")

        super().__init__(cfg_policy=cfg_policy_new, device=device)
        self.action_scales = np.asarray(self.cfg_policy.action_scales, dtype=np.float32)
        if self.action_scales.shape != (self.num_actions,):
            raise ValueError(
                f"BeyondMimic action_scales shape {self.action_scales.shape} != ({self.num_actions},)"
            )
        self.without_state_estimator = self.cfg_policy.without_state_estimator
        self.override_robot_anchor_pos = self.cfg_policy.override_robot_anchor_pos
        self.use_motion_from_model = self.cfg_policy.use_motion_from_model

        self.max_timestep = self.cfg_policy.max_timestep
        self.command = None
        self.reset()

        if self.use_motion_from_model:
            assert self.motion_anchor_body_index >= 0, "motion_anchor_body_index not set"
            assert self.command is not None, "command not initialized"
            command_init = self.command.copy()

            # motion init2anchor alignment
            anchor_pos_w_init = command_init["body_pos_w"][self.motion_anchor_body_index, :]
            anchor_quat_w_init = command_init["body_quat_w"][self.motion_anchor_body_index, :][[1, 2, 3, 0]]

            self.motion_init_align = TransformAlignment(
                quat=anchor_quat_w_init, pos=anchor_pos_w_init, yaw_only=True, xy_only=False
            )

    @staticmethod
    def _providers_for_device(device: str) -> list[str]:
        if device == "cpu":
            return ["CPUExecutionProvider"]
        if device == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "tensorrt":
            return ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        raise ValueError(f"Unknown device: {device}")

    def _validate_model_contract(self, cfg_policy: BeyondMimicPolicyCfg):
        input_shapes = {value.name: value.shape for value in self.session.get_inputs()}
        if set(input_shapes) != {"obs", "time_step"}:
            raise ValueError(f"Unexpected BeyondMimic ONNX inputs: {input_shapes}")
        if len(input_shapes["obs"]) != 2 or input_shapes["obs"][0] != 1:
            raise ValueError(f"BeyondMimic ONNX obs input must have shape [1, N], got {input_shapes['obs']}")
        if input_shapes["time_step"] != [1, 1]:
            raise ValueError(
                f"BeyondMimic ONNX time_step input must have shape [1, 1], got {input_shapes['time_step']}"
            )

        missing_outputs = sorted(set(self._reference_output_names).difference(self.output_names))
        if missing_outputs:
            raise ValueError(f"BeyondMimic ONNX is missing outputs: {missing_outputs}")

        missing_metadata = sorted(self._required_metadata.difference(self.modelmeta_dict))
        if missing_metadata:
            raise ValueError(f"BeyondMimic ONNX is missing metadata: {missing_metadata}")

        if not self.model_joint_names:
            raise ValueError("BeyondMimic ONNX joint_names metadata must not be empty")
        if len(set(self.model_joint_names)) != self.model_joint_count:
            raise ValueError("BeyondMimic ONNX joint_names metadata contains duplicates")

        output_shapes = {value.name: value.shape for value in self.session.get_outputs()}
        for output_name in ("actions", "joint_pos", "joint_vel"):
            expected_shape = [1, self.model_joint_count]
            if output_shapes.get(output_name) != expected_shape:
                raise ValueError(
                    f"BeyondMimic ONNX {output_name} output must have shape {expected_shape}, "
                    f"got {output_shapes.get(output_name)}"
                )

        body_names = _parse_metadata_strings(self.modelmeta_dict.get("body_names", ""))
        anchor_body_name = self.modelmeta_dict.get("anchor_body_name", "")
        if not body_names or anchor_body_name not in body_names:
            raise ValueError(
                f"BeyondMimic ONNX anchor body {anchor_body_name!r} is not present in body_names metadata"
            )
        expected_body_shapes = {
            "body_pos_w": [1, len(body_names), 3],
            "body_quat_w": [1, len(body_names), 4],
        }
        for output_name, expected_shape in expected_body_shapes.items():
            if output_shapes.get(output_name) != expected_shape:
                raise ValueError(
                    f"BeyondMimic ONNX {output_name} output must have shape {expected_shape}, "
                    f"got {output_shapes.get(output_name)}"
                )

        for metadata_name in ("action_scale", "default_joint_pos", "joint_stiffness", "joint_damping"):
            values = _parse_metadata_floats(self.modelmeta_dict.get(metadata_name, ""))
            if len(values) != self.model_joint_count:
                raise ValueError(
                    f"BeyondMimic ONNX {metadata_name} metadata has {len(values)} values, "
                    f"expected {self.model_joint_count}"
                )

    def _validate_observation_contract(self, cfg_policy: BeyondMimicPolicyCfg):
        if not self.observation_names:
            raise ValueError("BeyondMimic ONNX observation_names metadata must not be empty")

        has_anchor_position = "motion_anchor_pos_b" in self.observation_names
        has_base_linear_velocity = "base_lin_vel" in self.observation_names
        if has_anchor_position != has_base_linear_velocity:
            raise ValueError(
                f"BeyondMimic model {cfg_policy.policy_name!r} has an incomplete state-estimator observation contract: "
                "motion_anchor_pos_b and base_lin_vel must either both be present or both be absent"
            )

        model_without_state_estimator = not has_anchor_position
        if cfg_policy.without_state_estimator != model_without_state_estimator:
            mode = "without" if model_without_state_estimator else "with"
            raise ValueError(
                f"BeyondMimic model {cfg_policy.policy_name!r} was exported {mode} state-estimator observations, "
                f"but without_state_estimator={cfg_policy.without_state_estimator}"
            )
        if not model_without_state_estimator and cfg_policy.override_robot_anchor_pos:
            raise ValueError(
                f"BeyondMimic model {cfg_policy.policy_name!r} includes motion_anchor_pos_b, "
                "so override_robot_anchor_pos must be False"
            )

        expected_observation_names = [
            "command",
            *([] if model_without_state_estimator else ["motion_anchor_pos_b"]),
            "motion_anchor_ori_b",
            *([] if model_without_state_estimator else ["base_lin_vel"]),
            "base_ang_vel",
            "joint_pos",
            "joint_vel",
            "actions",
        ]
        if self.observation_names != expected_observation_names:
            raise ValueError(
                f"BeyondMimic model {cfg_policy.policy_name!r} declares observation_names "
                f"{self.observation_names}, expected {expected_observation_names}"
            )

        observation_sizes = {
            "command": 2 * self.model_joint_count,
            "motion_anchor_pos_b": 3,
            "motion_anchor_ori_b": 6,
            "base_lin_vel": 3,
            "base_ang_vel": 3,
            "joint_pos": self.model_joint_count,
            "joint_vel": self.model_joint_count,
            "actions": self.model_joint_count,
        }
        unknown_observations = sorted(set(self.observation_names).difference(observation_sizes))
        if unknown_observations:
            raise ValueError(f"BeyondMimic ONNX declares unsupported observations: {unknown_observations}")
        metadata_observation_size = sum(observation_sizes[name] for name in self.observation_names)
        if self.expected_observation_size != metadata_observation_size:
            raise ValueError(
                f"BeyondMimic model {cfg_policy.policy_name!r} declares observation_names totaling "
                f"{metadata_observation_size} values, but its ONNX input expects {self.expected_observation_size}"
            )

    def _prepare_policy(self):
        obs_shape = self.session.get_inputs()[0].shape  # e.g. [1, 154]
        obs = np.zeros(obs_shape[1], dtype=np.float32)
        self.get_action(obs)

    def reset(self):
        self.close_progress()
        self.timestep: float = self.cfg_policy.start_timestep
        if self.use_motion_from_model and self.max_timestep > 0:
            self.pbar = ProgressBar(f"Beyondmimic {self.cfg_policy.policy_name}", self.max_timestep)
        else:
            self.pbar = None
        self.play_speed: float = 1.0
        self.flag_motion_done = False
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.robot_anchor_align = None
        self._prepare_policy()

    def reset_alignment(self, env_data=None):
        """Align the reference motion to the current robot anchor without changing estimator state."""
        self.robot_anchor_align = None
        if env_data is not None and self.use_motion_from_model:
            robot_anchor_pos = self._get_robot_anchor_position(env_data)
            robot_anchor_quat = self._get_robot_anchor_orientation(env_data)
            self.robot_anchor_align = TransformAlignment(
                quat=robot_anchor_quat,
                pos=robot_anchor_pos,
                yaw_only=True,
                xy_only=False,
            )

    def close_progress(self):
        pbar = getattr(self, "pbar", None)
        if pbar is not None:
            pbar.close()
            self.pbar = None

    def post_step_callback(self, commands: list[str] | None = None):
        self.timestep += 1 * self.play_speed
        if self.pbar:
            self.pbar.set(self.timestep)

        if 0 < self.max_timestep <= self.timestep:
            self.play_speed = 0.0
            self.flag_motion_done = True
            self.close_progress()

        for command in commands or []:
            match command:
                case "[MOTION_RESET]":
                    self.reset()
                case "[MOTION_FADE_IN]":
                    self.play_speed = 1.0
                case "[MOTION_FADE_OUT]":
                    self.play_speed = 0.0
                case "[POLICY_LOCO]":
                    self.close_progress()

    def _get_command(self, env_data, ctrl_data):
        if not self.use_motion_from_model:
            assert "BeyondMimicCtrl" in ctrl_data, "BeyondMimicCtrl not found in ctrl_data"
            command = ctrl_data.get("BeyondMimicCtrl")
            self.command = command
            # print(command.time_steps[0])
            return (
                command.command,
                command.robot_anchor_pos_w,
                command.robot_anchor_quat_w,
                command.anchor_pos_w,
                command.anchor_quat_w,
                command.get("hand_pose", None),
            )
        else:
            assert self.command is not None, "command not initialized"
            if self.robot_anchor_align is None:
                self.reset_alignment(env_data)

            # print(self.command["time_step"])
            command = np.concatenate([self.command["joint_pos"], self.command["joint_vel"]], axis=-1)

            anchor_pos_w = self.command["body_pos_w"][self.motion_anchor_body_index, :]
            anchor_quat_w = self.command["body_quat_w"][self.motion_anchor_body_index, :][[1, 2, 3, 0]]

            anchor_quat_w, anchor_pos_w = self.motion_init_align.align_transform(anchor_quat_w, anchor_pos_w)
            anchor_quat_w, anchor_pos_w = self.robot_anchor_align.unalign_transform(anchor_quat_w, anchor_pos_w)

            if self.override_robot_anchor_pos:
                robot_anchor_pos_w = anchor_pos_w.copy()
            else:
                robot_anchor_pos_w = self._get_robot_anchor_position(env_data)

            robot_anchor_quat_w = self._get_robot_anchor_orientation(env_data)

            return command, robot_anchor_pos_w, robot_anchor_quat_w, anchor_pos_w, anchor_quat_w, None

    def _get_robot_anchor_position(self, env_data) -> np.ndarray:
        position = getattr(env_data, "torso_pos", None)
        if position is None:
            if self.without_state_estimator:
                return np.zeros(3, dtype=np.float32)
            raise ValueError(f"{type(self).__name__} requires torso_pos for state-estimator observations")
        return self._as_vector(position, "torso_pos", 3)

    def _get_robot_anchor_orientation(self, env_data) -> np.ndarray:
        orientation = getattr(env_data, "torso_quat", None)
        if orientation is None:
            raise ValueError(f"{type(self).__name__} requires torso_quat for motion anchor orientation")
        return self._as_vector(orientation, "torso_quat", 4)

    @staticmethod
    def _as_vector(value, name: str, size: int) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32)
        if vector.shape != (size,):
            raise ValueError(f"{name} must have shape ({size},), got {vector.shape}")
        if not np.isfinite(vector).all():
            raise FloatingPointError(f"{name} contains non-finite values")
        return vector

    def get_observation(self, env_data, ctrl_data):
        dof_pos = env_data.dof_pos
        dof_vel = env_data.dof_vel
        ang_vel = env_data.base_ang_vel
        lin_vel = env_data.base_lin_vel

        command, robot_anchor_pos_w, robot_anchor_quat_w, anchor_pos_w, anchor_quat_w, hand_pose = self._get_command(
            env_data, ctrl_data
        )

        pos, ori = subtract_frame_transforms(
            robot_anchor_pos_w,
            robot_anchor_quat_w,
            anchor_pos_w,
            anchor_quat_w,
        )
        mat = matrix_from_quat(ori)

        obs_command = command
        obs_motion_anchor_pos_b = pos
        obs_motion_anchor_ori_b = mat[:, :2].flatten()

        obs_base_lin_vel = lin_vel
        obs_base_ang_vel = ang_vel
        obs_joint_pos_rel = dof_pos - self.default_dof_pos
        obs_joint_vel_rel = dof_vel
        obs_last_action = self.last_action

        obs_prop = np.concatenate(
            [
                obs_command,
                obs_motion_anchor_pos_b if not self.without_state_estimator else [],
                obs_motion_anchor_ori_b,
                obs_base_lin_vel if not self.without_state_estimator else [],
                obs_base_ang_vel,
                obs_joint_pos_rel,
                obs_joint_vel_rel,
                obs_last_action,
            ]
        )

        obs = obs_prop
        extras = {
            "pos": pos,
            "ori": ori,
            "robot_anchor_pos_w": robot_anchor_pos_w,
            "robot_anchor_quat_w": robot_anchor_quat_w,
            "anchor_pos_w": anchor_pos_w,
            "anchor_quat_w": anchor_quat_w,
            "command": command,
            "hand_pose": hand_pose,
            "CALLBACK": ["[MOTION_DONE]"] if self.flag_motion_done else [],
        }
        return obs, extras

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if self.expected_observation_size is not None and obs.shape != (self.expected_observation_size,):
            raise ValueError(
                f"BeyondMimic model {self.cfg_policy.policy_name!r} expects observation shape "
                f"({self.expected_observation_size},), got {obs.shape}; "
                f"without_state_estimator={self.without_state_estimator}, "
                f"observation_names={self.observation_names}"
            )
        ort_inputs = {
            "obs": np.expand_dims(obs, axis=0).astype(np.float32),
            "time_step": np.expand_dims(np.array([int(self.timestep)]), axis=0).astype(np.float32),
        }

        ort_outputs = self.session.run(
            [
                "actions",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
            ],
            ort_inputs,
        )
        actions = np.asarray(ort_outputs[0], dtype=np.float32).reshape(-1)
        if actions.shape != (self.num_actions,):
            raise ValueError(f"BeyondMimic action shape {actions.shape} != ({self.num_actions},)")
        if not np.isfinite(actions).all():
            raise FloatingPointError("BeyondMimic policy produced a non-finite action")

        actions = (1 - self.action_beta) * self.last_action + self.action_beta * actions
        if self.action_clip is not None:
            actions = np.clip(actions, -self.action_clip, self.action_clip)
        self.last_action = actions.copy()

        scaled_actions = actions * self.action_scales

        if self.use_motion_from_model:
            command = {
                "time_step": self.timestep,
                "joint_pos": np.asarray(ort_outputs[1], dtype=np.float32).squeeze(0),
                "joint_vel": np.asarray(ort_outputs[2], dtype=np.float32).squeeze(0),
                "body_pos_w": np.asarray(ort_outputs[3], dtype=np.float32).squeeze(0),
                "body_quat_w": np.asarray(ort_outputs[4], dtype=np.float32).squeeze(0),  # [w, x, y, z]
            }
            if not all(np.isfinite(value).all() for key, value in command.items() if key != "time_step"):
                raise FloatingPointError("BeyondMimic reference motion contains non-finite values")
            self.command = command
        return scaled_actions

    def get_init_dof_pos(self) -> np.ndarray:
        """
        Return first frame of the reference motion.
        """
        if self.command is not None:
            joint_pos = self.command["joint_pos"]
            return joint_pos.copy()
        else:
            return self.default_dof_pos.copy()

    def debug_viz(self, visualizer: MujocoVisualizer, env_data, ctrl_data, extras):
        required = {
            "robot_anchor_pos_w",
            "robot_anchor_quat_w",
            "anchor_pos_w",
            "anchor_quat_w",
            "pos",
        }
        missing = required.difference(extras)
        if missing:
            logger.debug("Skip BeyondMimic debug visualization; missing extras: %s", sorted(missing))
            return

        robot_anchor_pos_w = extras["robot_anchor_pos_w"]
        robot_anchor_quat_w = extras["robot_anchor_quat_w"]
        anchor_pos_w = extras["anchor_pos_w"]
        anchor_quat_w = extras["anchor_quat_w"]

        pos = extras["pos"]
        # ori = extras["ori"]

        visualizer.draw_arrow(
            anchor_pos_w,
            anchor_quat_w,
            [0.2, 0, 0],
            color=[1, 0, 0, 1],
            scale=2,
            id=0,
            aligned_frame=True,
        )
        visualizer.draw_arrow(
            robot_anchor_pos_w,
            robot_anchor_quat_w,
            [0.2, 0, 0],
            color=[0, 1, 0, 1],
            scale=2,
            id=1,
            aligned_frame=True,
        )
        visualizer.draw_arrow(
            robot_anchor_pos_w,
            robot_anchor_quat_w,
            pos,
            color=[0, 1, 1, 1],
            scale=2,
            id=2,
            aligned_frame=True,
        )

        torso_pos = env_data.get("torso_pos")
        torso_quat = env_data.get("torso_quat")
        if torso_pos is not None and torso_quat is not None:
            visualizer.draw_arrow(
                torso_pos,
                torso_quat,
                [0.2, 0, 0],
                color=[1, 1, 0, 1],
                scale=2,
                id=3,
                aligned_frame=True,
            )


@policy_registry.register
class BeyondMimicPolicy(BeyondMimicPolicyBase):
    """Backward-compatible generic policy entry point."""
