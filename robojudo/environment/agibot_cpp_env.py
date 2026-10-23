import logging
import time

import numpy as np
from scipy.spatial.transform import Rotation

from robojudo.environment import Environment, env_registry
from robojudo.environment.env_cfgs import AgiBotEnvCfg

logger = logging.getLogger(__name__)


def format_state_freshness_report(report) -> str:
    """Render an AimDK freshness report as a compact, actionable error detail."""
    details: list[str] = []
    if "imu_missing" in report.reasons:
        details.append("IMU never received")
    elif "imu_stale" in report.reasons:
        details.append(f"IMU stale ({report.imu_age_sec:.3f}s old)")

    if report.missing_joint_names:
        details.append(f"joints never received: {', '.join(report.missing_joint_names)}")
    if report.stale_joint_names:
        stale = ", ".join(f"{name} ({report.joint_age_sec[name]:.3f}s)" for name in report.stale_joint_names)
        details.append(f"stale joints: {stale}")

    if "odometry_missing" in report.reasons:
        details.append("odometry never accepted")
    elif "odometry_invalid" in report.reasons:
        state = "degenerate" if report.odometry_degenerate else "invalid"
        details.append(f"odometry {state}")
    elif "odometry_stale" in report.reasons:
        details.append(f"odometry stale ({report.odometry_age_sec:.3f}s old)")

    if report.last_odometry_rejection_reason:
        rejection_age = report.last_odometry_rejection_age_sec
        age = "unknown age" if rejection_age is None else f"{rejection_age:.3f}s ago"
        details.append(f"last odometry rejection: {report.last_odometry_rejection_reason} ({age})")

    return "; ".join(details) if details else "all required streams are fresh"


@env_registry.register
class AgiBotCppEnv(Environment):
    cfg_env: AgiBotEnvCfg

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, enabled):
        self._enabled = bool(enabled)
        if not self._enabled:
            self.reset_arm_observer()

    def __init__(self, cfg_env: AgiBotEnvCfg, device="cpu"):
        self.enabled: bool = cfg_env.act
        self.aimdk = None
        self._arm_observer = None
        self._arm_observer_eligible = False
        self._observer_reset_generation = None
        self._observer_feedback_time = None
        self._observer_sample_times = None
        self._last_shadow_log_time = float("-inf")
        self._control_joint_names: set[str] = set()
        self._last_clamp_log_time = 0.0
        self._last_safety_state: str | None = None
        super().__init__(cfg_env=cfg_env, device=device)
        self._validate_gains(self.stiffness, self.damping)
        self._last_odometry_sequence: int | None = None
        self._last_odometry_stamp: float | None = None
        self._last_odometry_root_pos: np.ndarray | None = None
        self._last_odometry_root_quat: np.ndarray | None = None
        self._odometry_position_origin: np.ndarray | None = None
        self._filtered_base_lin_vel = np.zeros(3, dtype=np.float32)
        self._last_odometry_receipt_time: float | None = None

        try:
            from aimdk_cpp import AimdkController
        except ImportError as e:
            raise ImportError(
                "AgiBotCppEnv requires the optional aimdk_cpp extension. "
                "Install it with `python submodule_install.py aimdk` after sourcing ROS 2, then source "
                "`third_party/aimdk/install/setup.bash`."
            ) from e

        observer_cfg = getattr(cfg_env, "arm_observer", None)
        if observer_cfg is not None and observer_cfg.mode != "OFF":
            from robojudo.utils.x2_arm_observer import X2ArmObserver

            self._arm_observer = X2ArmObserver(
                observer_cfg,
                self.joint_names,
                cfg_env.arm_joint_names,
                self.stiffness,
                self.damping,
                self.torque_limits,
                min(cfg_env.aimdk.command_timeout, cfg_env.aimdk.state_timeout),
            )
            logger.warning(
                "X2 arm observer %s: nominal model is simulation-derived and not hardware-qualified",
                observer_cfg.mode,
            )
        cfg = cfg_env.aimdk.to_dict()
        if self._arm_observer is not None:
            cfg["arm_effort_enabled"] = self._arm_observer.mode == "ACTIVE"
            cfg["arm_effort_limits"] = self._arm_observer.adapter_limits()
        cfg.update(
            {
                "act": self.enabled,
                "enable_odometry": cfg_env.odometry_type in ("AIMDK", "SUPERODOM"),
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
        required_safety_methods = ("get_state_freshness_report", "get_safety_status", "arm_position_control")
        if self._arm_observer is not None:
            required_safety_methods += ("clear_arm_effort",)
        missing_safety_methods = [name for name in required_safety_methods if not hasattr(self.aimdk, name)]
        if missing_safety_methods:
            self.aimdk.shutdown()
            raise RuntimeError(
                "The installed aimdk_cpp binding is missing X2 safety state-machine APIs "
                f"{missing_safety_methods}; rebuild it with `python submodule_install.py aimdk`"
            )
        if self._arm_observer is not None and not hasattr(self.aimdk.get_safety_status(), "reset_generation"):
            self.aimdk.shutdown()
            raise RuntimeError("Rebuild aimdk_cpp for observer command receipts and reset generations")
        if self._arm_observer is not None and not hasattr(self.aimdk.get_robot_state(), "joint_received_time_sec"):
            self.aimdk.shutdown()
            raise RuntimeError("Rebuild aimdk_cpp for timestamped observer joint snapshots")
        self._odometry_type = cfg_env.odometry_type
        self.self_check()

    def self_check(self):
        if self.aimdk is None:
            return
        if self.aimdk.self_check(self.cfg_env.aimdk.startup_state_timeout):
            return

        report = self.aimdk.get_state_freshness_report(self.cfg_env.aimdk.state_timeout)
        if report.required_streams_fresh:
            logger.warning("AimDK state recovered immediately after the startup self-check timeout; continuing.")
            return

        required_streams = "joint/IMU"
        if self._odometry_type in ("AIMDK", "SUPERODOM"):
            required_streams += "/odometry"
        detail = format_state_freshness_report(report)
        raise RuntimeError(f"AgiBotCppEnv did not receive fresh AimDK {required_streams} state: {detail}.")

    def reset(self):
        self.reset_arm_observer()
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

        if self.enabled:
            state_damping_timeout = getattr(
                self.cfg_env.aimdk, "state_damping_timeout", self.cfg_env.aimdk.state_timeout
            )
            odometry_damping_timeout = getattr(
                self.cfg_env.aimdk,
                "odometry_damping_timeout",
                getattr(self.cfg_env.aimdk, "odometry_timeout", state_damping_timeout),
            )
            report = self.aimdk.get_state_freshness_report(state_damping_timeout, odometry_damping_timeout)
            if not report.required_streams_fresh:
                self.reset_arm_observer()
                detail = format_state_freshness_report(report)
                raise RuntimeError(f"AgiBotCppEnv state exceeded its hard timeout ({detail}); damping is latched.")

            get_safety_status = getattr(self.aimdk, "get_safety_status", None)
            if get_safety_status is not None:
                safety_status = get_safety_status()
                safety_state = safety_status.state
                if getattr(self, "_arm_observer", None) is not None:
                    self._check_observer_safety(safety_status)
                if safety_state != getattr(self, "_last_safety_state", None):
                    if safety_state == "HOLD":
                        logger.warning("AimDK safety hold entered: %s", safety_status.fault)
                    elif safety_state == "ACTIVE" and getattr(self, "_last_safety_state", None) == "HOLD":
                        logger.info("AimDK safety hold recovered")
                    self._last_safety_state = safety_state
                if safety_status.latched:
                    raise RuntimeError(
                        "AgiBotCppEnv safety damping is latched "
                        f"({safety_status.fault}); re-arm position control before sending targets."
                    )

        snapshot_request_time = time.monotonic()
        state = self.aimdk.get_robot_state()
        self._dof_pos = np.asarray(state.motor_state.q, dtype=np.float32)
        self._dof_vel = np.asarray(state.motor_state.dq, dtype=np.float32)
        observer = getattr(self, "_arm_observer", None)
        if observer is not None:
            ages = np.asarray(state.joint_age_sec, dtype=float)
            self._observer_sample_times = np.asarray(state.joint_received_time_sec, dtype=float)
            self._observer_feedback_time = None
            if ages.shape == (self.num_dofs,):
                arm_ages = ages[observer.indices]
                if np.isfinite(arm_ages).all() and np.all(arm_ages >= 0):
                    # Start before the native call, so callback/copy latency cannot
                    # make a cached sample look newer than it actually is.
                    self._observer_feedback_time = snapshot_request_time - float(np.max(arm_ages))
            if (
                self._observer_feedback_time is None
                or snapshot_request_time - self._observer_feedback_time > observer.max_dt
            ):
                self.reset_arm_observer()

        quat = np.asarray(state.imu_state.quaternion, dtype=np.float32)
        if self.born_place_align:
            quat = self.base_align.align_quat(quat)

        self._base_quat = quat
        self._base_ang_vel = np.asarray(state.imu_state.gyroscope, dtype=np.float32)
        self._base_lin_acc = np.asarray(state.imu_state.accelerometer, dtype=np.float32)

        if self._odometry_type in ("AIMDK", "SUPERODOM"):
            odometry = state.odometry_state
            if odometry.valid:
                root_pos, root_quat = self._update_odometry_state(odometry)
                if self.born_place_align:
                    root_pos = self.base_align.align_pos(root_pos)
                self._base_pos = root_pos
                self._base_lin_vel = self._filtered_base_lin_vel.copy()
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
            self._fk_info = fk_info.copy()
            self._torso_pos = fk_info[self._torso_name]["pos"]
            self._torso_quat = fk_info[self._torso_name]["quat"]
            self._torso_ang_vel = fk_info[self._torso_name]["ang_vel"]

    def _update_odometry_state(self, odometry) -> tuple[np.ndarray, np.ndarray]:
        sequence = int(getattr(odometry, "sequence", 0))
        is_new_sample = sequence != self._last_odometry_sequence
        if is_new_sample:
            root_pos, root_quat = self._odometry_sensor_pose_to_root(
                np.asarray(odometry.position, dtype=np.float64),
                np.asarray(odometry.quaternion, dtype=np.float64),
            )
            position_mode = getattr(self.cfg_env.aimdk, "odometry_position_mode", "ABSOLUTE")
            if position_mode == "RELATIVE_START":
                if getattr(self, "_odometry_position_origin", None) is None:
                    self._odometry_position_origin = root_pos.copy()
                root_pos = root_pos - self._odometry_position_origin
            stamp = float(getattr(odometry, "stamp_sec", 0)) + float(getattr(odometry, "stamp_nanosec", 0)) * 1e-9
            if stamp <= 0.0:
                stamp = time.monotonic()

            if self._last_odometry_stamp is not None and self._last_odometry_root_pos is not None:
                dt = stamp - self._last_odometry_stamp
                if 1e-4 < dt <= self.cfg_env.aimdk.odometry_timeout:
                    velocity_world = (root_pos - self._last_odometry_root_pos) / dt
                    velocity_body = Rotation.from_quat(root_quat).inv().apply(velocity_world)
                    tau = self.cfg_env.aimdk.odometry_velocity_filter_time_constant
                    alpha = 1.0 if tau <= 0.0 else 1.0 - np.exp(-dt / tau)
                    self._filtered_base_lin_vel = (
                        (1.0 - alpha) * self._filtered_base_lin_vel + alpha * velocity_body
                    ).astype(np.float32)

            self._last_odometry_sequence = sequence
            self._last_odometry_stamp = stamp
            self._last_odometry_root_pos = root_pos
            self._last_odometry_root_quat = root_quat
            self._last_odometry_receipt_time = time.monotonic()

        if self._last_odometry_root_pos is None or self._last_odometry_root_quat is None:
            raise RuntimeError("No valid converted X2 odometry sample is available")

        root_pos = self._last_odometry_root_pos.copy()
        if self._last_odometry_receipt_time is not None:
            age = min(
                time.monotonic() - self._last_odometry_receipt_time,
                self.cfg_env.aimdk.odometry_timeout,
            )
            velocity_world = Rotation.from_quat(self._last_odometry_root_quat).apply(self._filtered_base_lin_vel)
            root_pos += velocity_world * age
        return root_pos.astype(np.float32), self._last_odometry_root_quat.astype(np.float32)

    def _odometry_sensor_pose_to_root(
        self,
        sensor_pos: np.ndarray,
        sensor_quat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert world->odometry-sensor pose into world->pelvis/root pose."""
        if sensor_pos.shape != (3,) or sensor_quat.shape != (4,):
            raise ValueError("Odometry position and quaternion must have shapes (3,) and (4,)")
        if not np.isfinite(sensor_pos).all() or not np.isfinite(sensor_quat).all():
            raise ValueError("Odometry pose must contain only finite values")

        torso_to_sensor_pos = np.asarray(
            self.cfg_env.aimdk.torso_to_odometry_sensor_position,
            dtype=np.float64,
        )
        torso_to_sensor_rot = Rotation.from_quat(self.cfg_env.aimdk.torso_to_odometry_sensor_quaternion)
        world_to_sensor_rot = Rotation.from_quat(sensor_quat)
        world_to_torso_rot = world_to_sensor_rot * torso_to_sensor_rot.inv()
        world_to_torso_pos = sensor_pos - world_to_torso_rot.apply(torso_to_sensor_pos)

        relative_fk = self.kinematics.forward(
            joint_pos=self._dof_pos,
            base_pos=np.zeros(3, dtype=np.float64),
            base_quat=np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        )
        pelvis_to_torso_pos = relative_fk[self._torso_name]["pos"]
        pelvis_to_torso_rot = Rotation.from_quat(relative_fk[self._torso_name]["quat"])
        world_to_pelvis_rot = world_to_torso_rot * pelvis_to_torso_rot.inv()
        world_to_pelvis_pos = world_to_torso_pos - world_to_pelvis_rot.apply(pelvis_to_torso_pos)
        return world_to_pelvis_pos, world_to_pelvis_rot.as_quat()

    def get_arm_observer_status(self):
        """Command diagnostics only; issued effort is not measured actuator torque."""
        observer = getattr(self, "_arm_observer", None)
        if observer is None:
            return {"mode": "OFF", "eligible": False}
        from robojudo.utils.x2_observer_reference import JOINT_NAMES

        return {
            "mode": observer.mode,
            "eligible": self._arm_observer_eligible,
            "joint_names": list(JOINT_NAMES),
            "estimated_disturbance_nm": observer.observer.estimated_disturbance_nm.tolist(),
            "proposed_nm": observer.proposed_nm.tolist(),
            "limited_nm": observer.limited_nm.tolist(),
            "issued_nm": observer.issued_nm.tolist(),
            "reset_generation": self._observer_reset_generation,
            "model_qualification": "simulation_only",
        }

    def reset_arm_observer(self):
        observer = getattr(self, "_arm_observer", None)
        if observer is not None:
            observer.reset()
            if self.aimdk is not None:
                self.aimdk.clear_arm_effort()
                self._observer_reset_generation = self.aimdk.get_safety_status().reset_generation

    def set_arm_observer_eligible(self, eligible):
        eligible = bool(eligible)
        if eligible != getattr(self, "_arm_observer_eligible", False):
            self.reset_arm_observer()
            self._arm_observer_eligible = eligible

    def _check_observer_safety(self, status):
        if status.state != "ACTIVE" or status.reset_generation != self._observer_reset_generation:
            self._arm_observer.reset()
        self._observer_reset_generation = status.reset_generation

    def _step_with_arm_observer(self, target, now):
        observer = self._arm_observer
        status = self.aimdk.get_safety_status()
        self._check_observer_safety(status)
        report = self.aimdk.get_state_freshness_report(
            self.cfg_env.aimdk.state_timeout, self.cfg_env.aimdk.odometry_timeout
        )
        sample_time = self._observer_feedback_time
        if (
            not self._arm_observer_eligible
            or observer.mode == "OFF"
            or status.state != "ACTIVE"
            or not report.required_streams_fresh
            or getattr(self, "_observer_sample_times", None) is None
            or sample_time is None
            or not np.isfinite(sample_time)
            or not 0 <= now - sample_time <= observer.max_dt
        ):
            self.reset_arm_observer()
            self.aimdk.step(target.tolist())
            return
        try:
            effort = observer.propose(
                target, self._dof_pos, self._dof_vel, now, sample_times=self._observer_sample_times
            )
            receipt = self.aimdk.step(target.tolist(), effort, self._observer_reset_generation)
            if receipt.reset_generation != self._observer_reset_generation:
                self.reset_arm_observer()
                return
            observer.acknowledge(receipt.arm_efforts, command_time=receipt.command_time_sec)
            if (
                observer.mode == "SHADOW"
                and now - getattr(self, "_last_shadow_log_time", float("-inf")) >= 1.0
                and logger.isEnabledFor(logging.INFO)
            ):
                self._last_shadow_log_time = now
                diagnostics = self.get_arm_observer_status()
                logger.info(
                    "X2 observer SHADOW (Nm; joint_names=%s): estimate=%s, proposed=%s, limited=%s, "
                    "issued=%s, reset_generation=%s",
                    diagnostics["joint_names"],
                    np.round(diagnostics["estimated_disturbance_nm"], 4).tolist(),
                    np.round(diagnostics["proposed_nm"], 4).tolist(),
                    np.round(diagnostics["limited_nm"], 4).tolist(),
                    diagnostics["issued_nm"],
                    diagnostics["reset_generation"],
                )
        except (TypeError, ValueError, RuntimeError, FloatingPointError):
            self.reset_arm_observer()
            raise

    def step(self, pd_target, hand_pose=None):
        if hand_pose is not None:
            logger.debug("AgiBotCppEnv ignores hand_pose until AimDK hand control is wired.")
        if self.enabled and self.aimdk is not None:
            try:
                pd_target = np.asarray(pd_target, dtype=np.float64)
                if pd_target.shape != (self.num_dofs,):
                    raise ValueError(f"PD target must have shape ({self.num_dofs},)")
            except (TypeError, ValueError):
                self.command_damping(self.cfg_env.aimdk.shutdown_damping)
                raise ValueError("AgiBotCppEnv received a malformed PD target; damping commands were sent") from None
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
            if getattr(self, "_arm_observer", None) is not None:
                self._step_with_arm_observer(clipped_target, now)
            else:
                self.aimdk.step(clipped_target.tolist())

    def command_passive(self):
        self.reset_arm_observer()
        if self.enabled and self.aimdk is not None:
            self.aimdk.set_passive()

    def command_damping(self, damping=5.0):
        if not np.isfinite(damping) or damping < 0.0:
            raise ValueError("Damping must be finite and non-negative")
        self.reset_arm_observer()
        if self.enabled and self.aimdk is not None:
            self.aimdk.set_damping(float(damping))

    def arm_position_control(self):
        self.reset_arm_observer()
        if self.enabled and self.aimdk is not None:
            self.aimdk.arm_position_control()

    def set_control_joint_names(self, joint_names):
        self.reset_arm_observer()
        joint_names = list(joint_names)
        if self.aimdk is not None:
            self.aimdk.set_control_joint_names(joint_names)
        self._control_joint_names = set(joint_names)

    def shutdown(self):
        self.reset_arm_observer()
        self.enabled = False
        if self.aimdk is not None:
            self.aimdk.shutdown()

    def set_gains(self, stiffness, damping):
        self.stiffness, self.damping = self._validate_gains(stiffness, damping)
        self.reset_arm_observer()
        observer = getattr(self, "_arm_observer", None)
        if observer is not None:
            observer.observer.kp = self.stiffness[observer.indices].copy()
            observer.observer.kd = self.damping[observer.indices].copy()
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
