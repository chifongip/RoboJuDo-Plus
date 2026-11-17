"""X2 body policy and independently commanded fingers in one MuJoCo world."""

import time

import numpy as np

from robojudo.controller.omnihand_runtime import ALL_HAND_NAMES, OMNIHAND_JOINT_NAMES, measured_command_baseline
from robojudo.environment import env_registry
from robojudo.environment.observer_mujoco_env import ObserverMujocoEnv


@env_registry.register
class X2OmniHandMujocoEnv(ObserverMujocoEnv):
    def __init__(self, cfg_env, device="cpu"):
        self._hand_targets = None
        super().__init__(cfg_env, device)
        self._hand_actuators = self._resolve_actuator_indices(self.model, OMNIHAND_JOINT_NAMES)
        self._hand_qpos = np.array([self.model.joint(n).qposadr[0] for n in ALL_HAND_NAMES])
        self._reset_hand_targets()

    def _reset_hand_targets(self):
        measured = {n: float(self.data.qpos[self.model.joint(n).qposadr[0]]) for n in OMNIHAND_JOINT_NAMES}
        self._hand_targets = np.array(list(measured_command_baseline(measured).values()))

    def _apply_auxiliary_targets(self):
        if self._hand_targets is not None:
            self.data.ctrl[self._hand_actuators] = self._hand_targets

    def submit_hand_targets(self, positions):
        self._hand_targets = np.array([positions[n] for n in OMNIHAND_JOINT_NAMES])

    def hand_snapshot(self):
        stamp = min(self._joint_sample_times)
        measured = dict(zip(ALL_HAND_NAMES, self.data.qpos[self._hand_qpos].tolist(), strict=True))
        return {
            "positions": measured,
            "sample_times": dict.fromkeys(ALL_HAND_NAMES, stamp),
            "targets": dict(zip(OMNIHAND_JOINT_NAMES, self._hand_targets.tolist(), strict=True)),
            "fresh": time.monotonic() - stamp <= 0.1,
        }

    def reborn(self, init_qpos=None):
        super().reborn(init_qpos)
        if self._hand_targets is not None:
            self._reset_hand_targets()
