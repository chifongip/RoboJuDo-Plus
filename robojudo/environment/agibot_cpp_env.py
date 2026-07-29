import logging
import time

import numpy as np

from robojudo.environment import Environment, env_registry
from robojudo.environment.env_cfgs import AgiBotEnvCfg

logger = logging.getLogger(__name__)


@env_registry.register
class AgiBotCppEnv(Environment):
    cfg_env: AgiBotEnvCfg

    def __init__(self, cfg_env: AgiBotEnvCfg, device="cpu"):
        self.enabled: bool = cfg_env.act
        self.aimdk = None
        self._control_joint_names: set[str] = set()
        self._last_clamp_log_time = 0.0
        super().__init__(cfg_env=cfg_env, device=device)
        self._validate_gains(self.stiffness, self.damping)

        try:
            from aimdk_cpp import AimdkController
        except ImportError as e:
            raise ImportError(
                "AgiBotCppEnv requires the optional aimdk_cpp extension. "
                "Install it with `python submodule_install.py aimdk` after sourcing ROS 2, then source "
                "`third_party/aimdk/install/setup.bash`."
            ) from e

        cfg = cfg_env.aimdk.to_dict()
        cfg.update(
            {
                "act": self.enabled,
                "enable_odometry": cfg_env.odometry_type == "AIMDK",
                "joint_names": self.joint_names,
                "leg_joint_names": cfg_env.leg_joint_names,
                "waist_joint_names": cfg_env.waist_joint_names,
                "arm_joint_names": cfg_env.arm_joint_names,
                "head_joint_names": cfg_env.head_joint_names,
                "stiffness": self.stiffness.tolist(),
                "damping": self.damping.tolist(),
            }
        )
        self.aimdk = AimdkController(cfg)
        self._odometry_type = cfg_env.odometry_type
        self.self_check()

    def self_check(self):
        if self.aimdk is None:
            return
        if not self.aimdk.self_check():
            raise RuntimeError("AgiBotCppEnv did not receive AimDK joint/IMU state.")

    def reset(self):
        if self.born_place_align:
            self.born_place_align = False
            self.update()
            self.born_place_align = True
            self.set_born_place()
            self.update()

    def set_born_place(self, quat: np.ndarray | None = None, pos: np.ndarray | None = None):
        quat_ = self.base_quat if quat is None else quat
        pos_ = self.base_pos if pos is None else pos
        super().set_born_place(quat_, pos_)

    def update(self):
        if self.aimdk is None:
            return

        if self.enabled and not self.aimdk.state_is_fresh(self.cfg_env.aimdk.state_timeout):
            self.command_damping(self.cfg_env.aimdk.shutdown_damping)
            raise RuntimeError("AgiBotCppEnv joint, IMU, or odometry state became stale; damping commands were sent.")

        state = self.aimdk.get_robot_state()
        self._dof_pos = np.asarray(state.motor_state.q, dtype=np.float32)
        self._dof_vel = np.asarray(state.motor_state.dq, dtype=np.float32)

        quat = np.asarray(state.imu_state.quaternion, dtype=np.float32)
        if self.born_place_align:
            quat = self.base_align.align_quat(quat)

        self._base_quat = quat
        self._base_ang_vel = np.asarray(state.imu_state.gyroscope, dtype=np.float32)
        self._base_lin_acc = np.asarray(state.imu_state.accelerometer, dtype=np.float32)

        measured_torso_pos = None
        measured_torso_quat = None
        if self._odometry_type == "AIMDK":
            odometry = state.odometry_state
            if odometry.valid:
                odometry_pos = np.asarray(odometry.position, dtype=np.float32)
                odometry_quat = np.asarray(odometry.quaternion, dtype=np.float32)
                if self.born_place_align:
                    odometry_quat, odometry_pos = self.base_align.align_transform(
                        odometry_quat,
                        odometry_pos,
                    )
                self._base_pos = odometry_pos
                # nav_msgs/Odometry specifies twist in child_frame_id. The X2
                # publisher uses lidar_imu_chest_front, so this is already a
                # body-frame velocity and must not be rotated again.
                self._base_lin_vel = np.asarray(odometry.linear_velocity, dtype=np.float32)
                measured_torso_pos = odometry_pos.copy()
                measured_torso_quat = odometry_quat.copy()
            else:
                self._base_pos = None
                self._base_lin_vel = None
        elif self._odometry_type == "DUMMY":
            self._base_pos = np.array([0.0, 0.0, 0.8], dtype=np.float32)
            self._base_lin_vel = np.zeros(3, dtype=np.float32)
        else:
            self._base_pos = None
            self._base_lin_vel = None

        if self.update_with_fk:
            fk_info = self.fk()
            if measured_torso_pos is not None:
                fk_info[self._torso_name]["pos"] = measured_torso_pos
                fk_info[self._torso_name]["quat"] = measured_torso_quat
            self._fk_info = fk_info.copy()
            self._torso_pos = fk_info[self._torso_name]["pos"]
            self._torso_quat = fk_info[self._torso_name]["quat"]
            self._torso_ang_vel = fk_info[self._torso_name]["ang_vel"]

    def step(self, pd_target, hand_pose=None):
        assert len(pd_target) == self.num_dofs, "pd_target len should be num_dofs of env"
        if hand_pose is not None:
            logger.debug("AgiBotCppEnv ignores hand_pose until AimDK hand control is wired.")
        if self.enabled and self.aimdk is not None:
            pd_target = np.asarray(pd_target, dtype=np.float64)
            if not np.isfinite(pd_target).all():
                self.command_damping(self.cfg_env.aimdk.shutdown_damping)
                raise FloatingPointError("AgiBotCppEnv received a non-finite PD target; damping commands were sent.")

            clipped_target = np.clip(pd_target, self.position_limits[:, 0], self.position_limits[:, 1])
            clipped_indices = np.flatnonzero(clipped_target != pd_target)
            now = time.monotonic()
            if len(clipped_indices) and now - self._last_clamp_log_time >= 1.0:
                details = [
                    f"{self.joint_names[i]}={pd_target[i]:.4f}->{clipped_target[i]:.4f}"
                    for i in clipped_indices
                    if self.joint_names[i] in self._control_joint_names
                ]
                if details:
                    logger.warning("Clamped X2 PD targets: %s", ", ".join(details))
                    self._last_clamp_log_time = now
            self.aimdk.step(clipped_target.tolist())

    def command_passive(self):
        if self.enabled and self.aimdk is not None:
            self.aimdk.set_passive()

    def command_damping(self, damping=5.0):
        if not np.isfinite(damping) or damping < 0.0:
            raise ValueError("Damping must be finite and non-negative")
        if self.enabled and self.aimdk is not None:
            self.aimdk.set_damping(float(damping))

    def arm_position_control(self):
        if self.enabled and self.aimdk is not None:
            self.aimdk.arm_position_control()

    def set_control_joint_names(self, joint_names):
        self._control_joint_names = set(joint_names)
        if self.aimdk is not None:
            self.aimdk.set_control_joint_names(list(joint_names))

    def shutdown(self):
        self.enabled = False
        if self.aimdk is not None:
            self.aimdk.shutdown()

    def set_gains(self, stiffness, damping):
        self.stiffness, self.damping = self._validate_gains(stiffness, damping)
        if self.aimdk is not None and self.enabled:
            self.aimdk.set_gains(self.stiffness.tolist(), self.damping.tolist())

    def _validate_gains(self, stiffness, damping):
        stiffness = np.asarray(stiffness, dtype=np.float64)
        damping = np.asarray(damping, dtype=np.float64)
        expected_shape = (self.num_dofs,)
        if stiffness.shape != expected_shape or damping.shape != expected_shape:
            raise ValueError(f"Stiffness and damping must each have shape {expected_shape}")
        if not np.isfinite(stiffness).all() or not np.isfinite(damping).all():
            raise ValueError("Stiffness and damping must contain only finite values")
        if np.any(stiffness < 0.0) or np.any(damping < 0.0):
            raise ValueError("Stiffness and damping must be non-negative")
        return stiffness, damping
