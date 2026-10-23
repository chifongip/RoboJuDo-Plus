"""MuJoCo simulation with configurable disturbance correction on selected joints."""

import logging

import numpy as np

from robojudo.environment import env_registry
from robojudo.environment.mujoco_env import MujocoEnv
from robojudo.utils.joint_observer import JointObserver

logger = logging.getLogger(__name__)


@env_registry.register
class ObserverMujocoEnv(MujocoEnv):
    def __init__(self, cfg_env, device="cpu"):
        self._observer = None
        self._observer_initialized = False
        self._observer_eligible = True
        self._last_observer_log_time = float("-inf")
        super().__init__(cfg_env, device=device)
        self._observer_initialized = True
        self._rebuild_observer()

    def _rebuild_observer(self):
        cfg = self.cfg_env.observer
        self._observer = (
            None
            if cfg.mode == "OFF"
            else JointObserver(
                cfg,
                self.joint_names,
                self.stiffness,
                self.damping,
                self.torque_limits,
                self.control_dt,
            )
        )

    def reset_observer(self):
        if self._observer is not None:
            self._observer.reset()
        self._last_observer_log_time = float("-inf")

    def set_observer_eligible(self, eligible):
        eligible = bool(eligible)
        if eligible != self._observer_eligible:
            self.reset_observer()
            self._observer_eligible = eligible

    def reset(self):
        super().reset()
        self.reset_observer()

    def reborn(self, init_qpos=None):
        super().reborn(init_qpos)
        self.reset_observer()
        # reborn changes MuJoCo state outside the usual step/update loop.
        self.update()

    def set_gains(self, stiffness, damping):
        super().set_gains(stiffness, damping)
        if self._observer_initialized:
            self._rebuild_observer()

    def set_control_joint_names(self, joint_names):
        super().set_control_joint_names(joint_names)
        self.reset_observer()

    def command_passive(self):
        self.reset_observer()
        super().command_passive()

    def command_damping(self, damping=5.0):
        self.reset_observer()
        super().command_damping(damping)

    def get_observer_status(self):
        observer = self._observer
        if observer is None:
            return {"mode": "OFF", "eligible": False}
        return {
            "mode": observer.mode,
            "eligible": self._observer_eligible,
            "joint_names": observer.joint_names.copy(),
            "estimated_disturbance_nm": observer.observer.estimated_disturbance_nm.tolist(),
            "proposed_nm": observer.proposed_nm.tolist(),
            "limited_nm": observer.limited_nm.tolist(),
            "issued_nm": observer.issued_nm.tolist(),
            "model_qualification": "simulation_only",
        }

    def step(self, pd_target, hand_pose=None, *, feedforward_effort=None):
        try:
            if feedforward_effort is not None:
                raise ValueError("Observer simulation owns feedforward effort")
            target = np.asarray(pd_target, dtype=np.float64)
            if target.shape != (self.num_dofs,) or not np.isfinite(target).all():
                raise ValueError("PD target must be a finite joint vector")
            observer = self._observer
            if observer is None or not self._observer_eligible:
                self.reset_observer()
                return super().step(target, hand_pose)

            now = float(self.data.time)
            effort = np.zeros(self.num_dofs)
            effort[observer.indices] = observer.propose(target, self.dof_pos, self.dof_vel, now)
            admitted = np.where(self._control_mask, self._bounded_feedforward(effort), 0.0)
            # This is admitted additive effort, not measured torque after PD clipping.
            observer.acknowledge(admitted[observer.indices])
            super().step(target, hand_pose, feedforward_effort=admitted)
            if now - self._last_observer_log_time >= 1.0 and logger.isEnabledFor(logging.INFO):
                self._last_observer_log_time = now
                at_bound = np.isclose(
                    np.abs(observer.observer.estimated_disturbance_nm), observer.observer.limits, rtol=0, atol=1e-3
                )
                limited = observer.observer.antiwindup | ~np.isclose(
                    observer.proposed_nm, observer.issued_nm, rtol=0, atol=1e-6
                )
                logger.info(
                    "Observer %s Nm: est=[%s] issued=[%s] bound=%d limited=%d",
                    observer.mode,
                    ",".join(f"{value:.1f}" for value in observer.observer.estimated_disturbance_nm),
                    ",".join(f"{value:.1f}" for value in observer.issued_nm),
                    np.count_nonzero(at_bound),
                    np.count_nonzero(limited),
                )
        except (TypeError, ValueError, RuntimeError, FloatingPointError, OverflowError):
            self.reset_observer()
            raise
