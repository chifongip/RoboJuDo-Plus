import logging

import numpy as np

from robojudo.policy import Policy, policy_registry
from robojudo.policy.onnx_runtime import create_onnx_session
from robojudo.utils.progress import ProgressBar
from robojudo.utils.util_func import get_gravity_orientation

logger = logging.getLogger(__name__)


@policy_registry.register
class X2DeployPolicy(Policy):
    def __init__(self, cfg_policy, device):
        super().__init__(cfg_policy=cfg_policy, device=device)
        providers = ["CPUExecutionProvider"]
        self.session = create_onnx_session(self.cfg_policy.policy_file, self.cfg_policy, providers=providers)
        self.obs_scales = self.cfg_policy.obs_scales
        self.obs_clip = self.cfg_policy.obs_clip
        self.warmup_frames = self.cfg_policy.warmup_frames
        self.max_timestep = self.cfg_policy.max_timestep
        self._default_pose_mode = False
        self._validate_onnx_io()
        self.reset()

    def reset(self):
        self.close_progress()
        self.heart_count = self.cfg_policy.phase_start_count - 1.0
        self.mimic_ref_pos = np.zeros(self.num_dofs, dtype=np.float32)
        self.mimic_ref_vel = np.zeros(self.num_dofs, dtype=np.float32)
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self._needs_warmup = self.warmup_frames > 0
        self._use_warmup_action = False
        self.flag_motion_done = False
        self.pbar = (
            ProgressBar(f"X2Deploy {self.cfg_policy.policy_name}", self.max_timestep)
            if self.max_timestep > 0
            else None
        )

    def post_step_callback(self, commands=None):
        if "[POLICY_LOCO]" in (commands or []):
            self.close_progress()

    def close_progress(self):
        if getattr(self, "pbar", None) is not None:
            self.pbar.close()
            self.pbar = None

    def set_default_pose_mode(self, enabled: bool):
        enabled = bool(enabled)
        if self._default_pose_mode == enabled:
            return
        self._default_pose_mode = enabled
        if not enabled:
            # Start the reference motion from its first frame when the robot is armed.
            self.reset()

    def get_observation(self, env_data, ctrl_data):
        obs = self._build_observation(env_data)
        if self._needs_warmup and not self._default_pose_mode:
            self._needs_warmup = False
            for _ in range(self.warmup_frames):
                self._run_inference(obs)
                obs = self._build_observation(env_data)
            self._use_warmup_action = True
        extras = {"CALLBACK": ["[MOTION_DONE]"] if self.flag_motion_done else []}
        return obs, extras

    def _build_observation(self, env_data):
        gravity_orientation = get_gravity_orientation(env_data.base_quat)
        dof_pos = (env_data.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"]
        dof_vel = env_data.dof_vel * self.obs_scales["dof_vel"]
        base_ang_vel = env_data.base_ang_vel * self.obs_scales["ang_vel"]
        actions = self.last_action * self.obs_scales["actions"]

        obs = np.concatenate(
            [
                self.mimic_ref_pos,
                self.mimic_ref_vel,
                gravity_orientation,
                base_ang_vel,
                dof_pos,
                dof_vel,
                actions,
            ]
        ).astype(np.float32)
        obs = np.clip(obs, -self.obs_clip, self.obs_clip)
        if obs.shape[0] != self.cfg_policy.num_obs:
            raise ValueError(f"X2DeployPolicy obs size {obs.shape[0]} != {self.cfg_policy.num_obs}")
        return obs

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if self._default_pose_mode:
            return np.zeros(self.num_actions, dtype=np.float32)
        if self._use_warmup_action:
            self._use_warmup_action = False
            raw_action = self.last_action.copy()
        else:
            raw_action = self._run_inference(obs)
        return raw_action * self.action_scale

    def _run_inference(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.reshape(1, -1).astype(np.float32)
        if obs.shape != (1, self.cfg_policy.num_obs):
            raise ValueError(f"X2DeployPolicy ONNX obs shape {obs.shape} != (1, {self.cfg_policy.num_obs})")

        phase_limit = self.cfg_policy.phase_end_count
        if self.max_timestep > 0:
            phase_limit = min(phase_limit, float(self.max_timestep))
        self.heart_count = min(self.heart_count + 1.0, phase_limit)
        if self.pbar is not None:
            self.pbar.set(self.heart_count)
        outputs = self.session.run(
            None,
            {
                "obs": obs,
                "time_step": np.asarray([[self.heart_count]], dtype=np.float32),
            },
        )
        output_by_name = {out.name: value for out, value in zip(self.session.get_outputs(), outputs, strict=True)}
        raw_action = output_by_name["actions"].reshape(-1).astype(np.float32)
        if raw_action.shape != (self.num_actions,):
            raise ValueError(f"X2DeployPolicy action shape {raw_action.shape} != ({self.num_actions},)")
        if self.action_clip is not None:
            raw_action = np.clip(raw_action, -self.action_clip, self.action_clip)
        if not np.isfinite(raw_action).all():
            raise FloatingPointError("X2 policy produced a non-finite action")
        self.last_action = raw_action.copy()

        if "joint_pos" in output_by_name:
            mimic_ref_pos = output_by_name["joint_pos"].reshape(-1).astype(np.float32)
            if mimic_ref_pos.shape == (self.num_dofs,):
                if not np.isfinite(mimic_ref_pos).all():
                    raise FloatingPointError("X2 policy produced non-finite joint_pos")
                self.mimic_ref_pos = mimic_ref_pos
        if "joint_vel" in output_by_name:
            mimic_ref_vel = output_by_name["joint_vel"].reshape(-1).astype(np.float32)
            if mimic_ref_vel.shape == (self.num_dofs,):
                if not np.isfinite(mimic_ref_vel).all():
                    raise FloatingPointError("X2 policy produced non-finite joint_vel")
                self.mimic_ref_vel = mimic_ref_vel

        if 0 < self.max_timestep <= self.heart_count:
            self.flag_motion_done = True
            self.close_progress()

        return raw_action

    def _validate_onnx_io(self):
        input_shapes = {inp.name: inp.shape for inp in self.session.get_inputs()}
        output_shapes = {out.name: out.shape for out in self.session.get_outputs()}
        if input_shapes.get("obs") != [1, self.cfg_policy.num_obs]:
            raise ValueError(
                f"X2 policy ONNX obs input must be [1, {self.cfg_policy.num_obs}], "
                f"got {input_shapes.get('obs')}"
            )
        if input_shapes.get("time_step") != [1, 1]:
            raise ValueError(f"X2 policy ONNX time_step input must be [1, 1], got {input_shapes.get('time_step')}")
        if output_shapes.get("actions") != [1, self.num_actions]:
            raise ValueError(
                f"X2 policy ONNX actions output must be [1, {self.num_actions}], "
                f"got {output_shapes.get('actions')}"
            )
