import logging
import time

import numpy as np
from unitree_cpp import RobotState, SportState, UnitreeController  # type: ignore

from robojudo.environment import Environment, env_registry
from robojudo.environment.env_cfgs import UnitreeEnvCfg
from robojudo.tools.retarget import HandRetarget
from robojudo.utils.rotation import TransformAlignment

logger = logging.getLogger(__name__)


@env_registry.register
class UnitreeCppEnv(Environment):
    cfg_env: UnitreeEnvCfg

    def __init__(self, cfg_env: UnitreeEnvCfg, device="cpu"):
        self.enabled: bool = cfg_env.act
        self._control_joint_names: set[str] = set()
        self._last_clamp_log_time = 0.0
        super().__init__(cfg_env=cfg_env, device=device)
        self._motor_dof_count = cfg_env.motor_dof_count or self.num_dofs
        self._dof_idx = None if cfg_env.joint2motor_idx is None else np.asarray(cfg_env.joint2motor_idx, dtype=np.int32)
        self._validate_gains(self.stiffness, self.damping)
        self.RemoteControllerHandler = None

        cfg_unitree: UnitreeEnvCfg.UnitreeCfg = cfg_env.unitree

        cfg_unitree_dict: dict = cfg_unitree.to_dict()
        cfg_unitree_dict["num_dofs"] = self._motor_dof_count
        cfg_unitree_dict["stiffness"] = self._logical_to_motor(self.stiffness)
        cfg_unitree_dict["damping"] = self._logical_to_motor(self.damping)

        self.robot = cfg_unitree.robot
        self._odometry_type = cfg_env.odometry_type
        if self._odometry_type == "ZED":
            assert self.cfg_env.zed_cfg is not None, "zed_cfg must be set if odometry_type is 'ZED'"
            from robojudo.tools.zed_odometry import ZedOdometry

            self.zed_odometry = ZedOdometry(self.cfg_env.zed_cfg)
        elif self._odometry_type == "DUMMY":
            pass
        elif self._odometry_type == "UNITREE":
            pass

        self.hand_type = cfg_unitree.hand_type
        if self.hand_type == "Inspire":
            self.hand_retarget = HandRetarget(cfg_env.hand_retarget)
        elif self.hand_type == "Dex-3":
            self.hand_retarget = None  # TODO
        else:
            self.hand_retarget = None

        self.sport_state: SportState = None  # pyright: ignore[reportAttributeAccessIssue]
        self.robot_state: RobotState = None  # pyright: ignore[reportAttributeAccessIssue]

        self.unitree = UnitreeController(cfg_unitree_dict)
        if cfg_unitree.command_timeout > 0.0 or cfg_unitree.state_timeout > 0.0:
            required_methods = ("set_passive", "set_damping", "arm_position_control", "state_is_fresh")
            missing_methods = [name for name in required_methods if not hasattr(self.unitree, name)]
            if missing_methods:
                self.unitree.shutdown()
                raise RuntimeError(
                    "The installed unitree_cpp binding is missing real-deployment safety APIs "
                    f"{missing_methods}; rebuild it with `python submodule_install.py unitree_cpp`"
                )

        # born place alignment extra for h1 torso
        if self.robot == "h1":
            self.torso_align = TransformAlignment()

        # time.sleep(1)  # wait for unitree init
        self.self_check()

    def self_check(self):
        for _ in range(30):
            time.sleep(0.1)
            if self.unitree.self_check():
                logger.info("UnitreeCppEnv self check passed!")
                break
        if not self.unitree.self_check():
            logger.critical("UnitreeCppEnv self check failed!")
            if self.enabled:
                if hasattr(self.unitree, "set_damping"):
                    self.unitree.set_damping(self.cfg_env.unitree.shutdown_damping)
                else:
                    self.unitree.shutdown()
            raise RuntimeError("UnitreeCppEnv did not receive valid Unitree state")

    def reset(self):
        self.reset_alignment()

    def set_born_place(self, quat: np.ndarray | None = None, pos: np.ndarray | None = None):
        quat_ = self.base_quat if quat is None else quat
        pos_ = self.base_pos if pos is None else pos
        super().set_born_place(quat_, pos_)

        if self.robot == "h1":
            self.torso_align.set_base(quat=self.torso_quat)

        if self._odometry_type == "ZED":
            self.zed_odometry.set_zreo()

    def update(self):
        state_timeout = self.cfg_env.unitree.state_timeout
        if self.enabled and state_timeout > 0.0 and not self.unitree.state_is_fresh(state_timeout):
            self.command_damping(self.cfg_env.unitree.shutdown_damping)
            raise RuntimeError("UnitreeCppEnv low state became stale; damping commands were sent")

        # robot state
        self.robot_state = self.unitree.get_robot_state()
        self._dof_pos = self._motor_to_logical(self.robot_state.motor_state.q, dtype=np.float32)
        self._dof_vel = self._motor_to_logical(self.robot_state.motor_state.dq, dtype=np.float32)

        if self.robot == "g1":
            quat = np.array(self.robot_state.imu_state.quaternion, dtype=np.float32)[[1, 2, 3, 0]]
            ang_vel = np.array(self.robot_state.imu_state.gyroscope, dtype=np.float32)
            rpy = np.array(self.robot_state.imu_state.rpy, dtype=np.float32)

            if self.born_place_align:
                quat = self.base_align.align_quat(quat)

            self._base_quat = quat
            self._base_ang_vel = ang_vel
            self._base_rpy = rpy

        elif self.robot == "h1":
            raise NotImplementedError("H1 robot with unitree_cpp not supported yet.")

        # odometry
        if self._odometry_type == "ZED":
            self.zed_odometry.update()
            if self.zed_odometry.is_valid:
                # born place aligned in zed_odometry
                self._base_pos = self.zed_odometry.pos
                self._base_lin_vel = self.zed_odometry.lin_vel
        elif self._odometry_type == "DUMMY":
            self._base_pos = np.array([0.0, 0.0, 0.8])
            self._base_lin_vel = np.array([0.0, 0.0, 0.0])
        elif self._odometry_type == "UNITREE":
            self.sport_state = self.unitree.get_sport_state()
            base_pos = np.asarray(self.sport_state.position, dtype=np.float32)
            # Unitree reports this velocity along the robot axes. It is already
            # body-frame data; applying the separately sampled low-state IMU
            # orientation would rotate it a second time.
            self._base_lin_vel = np.asarray(self.sport_state.velocity, dtype=np.float32)
            self._base_pos = self.base_align.align_pos(base_pos) if self.born_place_align else base_pos

        # FK
        if self.update_with_fk:
            fk_info = self.fk()
            self._torso_pos = fk_info[self._torso_name]["pos"]
            if self.robot != "h1":
                self._torso_quat = fk_info[self._torso_name]["quat"]
                self._torso_ang_vel = fk_info[self._torso_name]["ang_vel"]

        # controller
        if self.RemoteControllerHandler:
            self.RemoteControllerHandler(self.robot_state.wireless_remote)

    def step(self, pd_target, hand_pose=None):
        assert len(pd_target) == self.num_dofs, "pd_target len should be num_dofs of env"
        positions = np.asarray(pd_target, dtype=np.float64)
        if not np.isfinite(positions).all():
            self.command_damping(self.cfg_env.unitree.shutdown_damping)
            raise FloatingPointError("UnitreeCppEnv received a non-finite PD target; damping commands were sent")

        clipped_positions = np.clip(positions, self.position_limits[:, 0], self.position_limits[:, 1])
        clipped_indices = np.flatnonzero(clipped_positions != positions)
        now = time.monotonic()
        if len(clipped_indices) and now - self._last_clamp_log_time >= 1.0:
            details = [
                f"{self.joint_names[i]}={positions[i]:.4f}->{clipped_positions[i]:.4f}"
                for i in clipped_indices
                if self.joint_names[i] in self._control_joint_names
            ]
            if details:
                logger.warning("Clamped Unitree PD targets: %s", ", ".join(details))
                self._last_clamp_log_time = now
        if self.enabled:
            self.unitree.step(self._logical_to_motor(clipped_positions).tolist())

        if hand_pose is not None:
            assert type(hand_pose) is np.ndarray, "hand_pose should be a numpy array"
            assert hand_pose.shape[0] == 2, "hand_pose should be of shape (2, -1)"
            if self.hand_retarget is not None:
                hand_pose = self.hand_retarget.from_pose_to_cmd(hand_pose)
                logger.debug(f"Hand pose retargeted: {hand_pose}")
            hand_pose = hand_pose.tolist()

            if self.enabled:
                self.unitree.step_hands(hand_pose[0], hand_pose[1])

    def command_passive(self):
        if self.enabled:
            self.unitree.set_passive()

    def command_damping(self, damping=5.0):
        if not np.isfinite(damping) or damping < 0.0:
            raise ValueError("Damping must be finite and non-negative")
        if self.enabled:
            self.unitree.set_damping(float(damping))

    def arm_position_control(self):
        if self.enabled:
            self.unitree.arm_position_control()

    def set_control_joint_names(self, joint_names):
        joint_names = list(joint_names)
        if joint_names != self.joint_names:
            raise ValueError("UnitreeCppEnv only supports controlling the complete environment joint layout")
        self._control_joint_names = set(joint_names)

    def shutdown(self):
        self.unitree.shutdown()
        self.enabled = False

    def set_gains(self, stiffness, damping):
        stiffness, damping = self._validate_gains(stiffness, damping)
        self.stiffness = stiffness
        self.damping = damping
        if not hasattr(self, "unitree"):  # TODO
            return
        if not self.enabled:
            return
        self.unitree.set_gains(
            self._logical_to_motor(stiffness).tolist(),
            self._logical_to_motor(damping).tolist(),
        )

    def _logical_to_motor(self, values) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (self.num_dofs,):
            raise ValueError(f"Logical joint data must have shape ({self.num_dofs},)")
        if self._dof_idx is None:
            return values.copy()
        motor_values = np.zeros(self._motor_dof_count, dtype=values.dtype)
        motor_values[self._dof_idx] = values
        return motor_values

    def _motor_to_logical(self, values, dtype=np.float64) -> np.ndarray:
        motor_values = np.asarray(values, dtype=dtype)
        if motor_values.shape != (self._motor_dof_count,):
            raise ValueError(f"Motor feedback must have shape ({self._motor_dof_count},)")
        if self._dof_idx is None:
            return motor_values.copy()
        return motor_values[self._dof_idx].copy()

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


if __name__ == "__main__":
    from robojudo.config.g1.env.g1_real_env_cfg import G1RealEnvCfg

    env = UnitreeCppEnv(cfg_env=G1RealEnvCfg())
    env.set_gains(
        stiffness=[kp * 0.0 for kp in env.stiffness],
        damping=[kd * 0.1 for kd in env.damping],
    )
    while 1:
        # env.step(np.zeros(29), np.ones((2, 7)) * -0)
        env.step(np.zeros(29), None)
        # if controller.remote_controller("A"):
        #     controller.shutdown()
        print(env.base_rpy)
        print(env.dof_pos)
        print(env.base_pos)
        env.update()
        # print(env.base_pos)
        time.sleep(0.1)
    # print("Exit")
    # from robojudo.controller import UnitreeCtrl
    # ctrl = UnitreeCtrl(env=env)

    # while True:
    #     env.update()
    #     state = ctrl.get_state()
    #     events = ctrl.get_events()
    #     print("State:", state)
    #     print("Events:", events)
    #     time.sleep(0.1)  # Simulate a control loop
