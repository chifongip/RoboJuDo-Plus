import logging
import os
from collections import deque

import numpy as np
import onnxruntime as ort

from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import AmpRecoveryPolicyCfg
from robojudo.utils.util_func import get_gravity_orientation

logger = logging.getLogger(__name__)

_OBSERVATION_NAMES = (
    "base_ang_vel",
    "projected_gravity",
    "command",
    "joint_pos",
    "joint_vel",
    "actions",
)


@policy_registry.register
class AmpRecoveryPolicy(Policy):
    """Deploy an mjlab AMP recovery actor with term-major observation history."""

    cfg_policy: AmpRecoveryPolicyCfg

    def __init__(self, cfg_policy: AmpRecoveryPolicyCfg, device: str):
        if not os.path.isfile(cfg_policy.policy_file):
            raise FileNotFoundError(f"Model file not found at {cfg_policy.policy_file}")

        super().__init__(cfg_policy=cfg_policy, device=device)
        self.session = ort.InferenceSession(cfg_policy.policy_file, providers=["CPUExecutionProvider"])
        self.action_scales = np.asarray(cfg_policy.action_scales, dtype=np.float32)
        self._validate_model_io()
        logger.info("Loaded AMP Recovery ONNX model from %s", cfg_policy.policy_file)
        self.reset()

    def reset(self):
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self._history = {name: deque(maxlen=self.history_length) for name in _OBSERVATION_NAMES}

    def post_step_callback(self, commands: list[str] | None = None):
        del commands

    def get_observation(self, env_data, ctrl_data):
        del ctrl_data
        terms = {
            "base_ang_vel": np.asarray(env_data.base_ang_vel, dtype=np.float32),
            "projected_gravity": get_gravity_orientation(env_data.base_quat).astype(np.float32),
            "command": np.zeros(3, dtype=np.float32),
            "joint_pos": (env_data.dof_pos - self.default_dof_pos).astype(np.float32),
            "joint_vel": np.asarray(env_data.dof_vel, dtype=np.float32),
            "actions": self.last_action.astype(np.float32),
        }

        flattened_terms = []
        for name in _OBSERVATION_NAMES:
            value = terms[name]
            history = self._history[name]
            if not history:
                history.extend(value.copy() for _ in range(self.history_length))
            else:
                history.append(value.copy())
            flattened_terms.append(np.concatenate(tuple(history)))

        obs = np.concatenate(flattened_terms).astype(np.float32)
        if obs.shape != (self.cfg_policy.num_obs,):
            raise ValueError(f"AMP Recovery observation shape {obs.shape} does not match ({self.cfg_policy.num_obs},)")
        if not np.isfinite(obs).all():
            raise FloatingPointError("AMP Recovery observation contains a non-finite value")
        return obs, {}

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        obs_batch = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        if obs_batch.shape != (1, self.cfg_policy.num_obs):
            raise ValueError(
                f"AMP Recovery ONNX observation shape {obs_batch.shape} does not match (1, {self.cfg_policy.num_obs})"
            )

        actions = self.session.run(["actions"], {"obs": obs_batch})[0].reshape(-1).astype(np.float32)
        if actions.shape != (self.num_actions,):
            raise ValueError(f"AMP Recovery action shape {actions.shape} does not match ({self.num_actions},)")
        if not np.isfinite(actions).all():
            raise FloatingPointError("AMP Recovery policy produced a non-finite action")

        self.last_action = actions.copy()
        return actions * self.action_scales

    def _validate_model_io(self):
        inputs = {item.name: item.shape for item in self.session.get_inputs()}
        outputs = {item.name: item.shape for item in self.session.get_outputs()}
        self._validate_tensor_shape(inputs, "obs", self.cfg_policy.num_obs, "input")
        self._validate_tensor_shape(outputs, "actions", self.num_actions, "output")

    @staticmethod
    def _validate_tensor_shape(tensors: dict[str, list], name: str, width: int, kind: str):
        shape = tensors.get(name)
        if shape is None or len(shape) != 2 or shape[1] != width:
            raise ValueError(f"AMP Recovery ONNX {kind} '{name}' must have shape [batch, {width}], got {shape}")
