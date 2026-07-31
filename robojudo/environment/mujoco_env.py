import logging
import time

import mujoco
import mujoco_viewer
import numpy as np

from robojudo.environment import Environment, env_registry
from robojudo.environment.env_cfgs import MujocoEnvCfg
from robojudo.environment.utils.elastic_band import ElasticBand
from robojudo.environment.utils.mujoco_viz import MujocoVisualizer
from robojudo.environment.utils.odometry import (
    OdometryReplayProfile,
    SimulatedOdometry,
    root_pose_to_sensor,
    sensor_pose_to_root,
)
from robojudo.utils.util_func import quat_rotate_inverse_np, quatToEuler

logger = logging.getLogger(__name__)


@env_registry.register
class MujocoEnv(Environment):
    cfg_env: MujocoEnvCfg

    def __init__(self, cfg_env: MujocoEnvCfg, device="cpu"):
        super().__init__(cfg_env=cfg_env, device=device)
        self._control_mask = np.ones(self.num_dofs, dtype=bool)

        self.sim_duration = cfg_env.sim_duration
        self.sim_dt = cfg_env.sim_dt
        self.sim_decimation = cfg_env.sim_decimation
        self.control_dt = self.sim_dt * self.sim_decimation

        self.model = mujoco.MjModel.from_xml_path(cfg_env.xml)  # pyright: ignore[reportAttributeAccessIssue]
        self.model.opt.timestep = self.sim_dt
        self._dof_actuator_indices = self._resolve_actuator_indices(self.model, self.joint_names)
        self.data = mujoco.MjData(self.model)  # pyright: ignore[reportAttributeAccessIssue]
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)  # pyright: ignore[reportAttributeAccessIssue]
        mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
        self.elastic_band = (
            ElasticBand(cfg_env.elastic_band, self.model, self.data) if cfg_env.elastic_band is not None else None
        )

        self.viewer = None
        if not cfg_env.headless:
            self.viewer = mujoco_viewer.MujocoViewer(
                self.model,
                self.data,
                width=1200,
                height=900,
                hide_menus=True,
                diable_key_callbacks=True,
            )
            self.viewer.cam.distance = 3.0
            self.viewer.cam.elevation = -10.0
            self.viewer.cam.azimuth = 180.0
            # self.viewer._paused = True

        if cfg_env.visualize_extras and self.viewer is not None:
            self.visualizer = MujocoVisualizer(self.viewer, alignment=self.base_align)
        else:
            self.visualizer = None

        self.last_time = time.time()
        self.random_heading = cfg_env.random_heading
        self.initial_heading_degrees = cfg_env.initial_heading_degrees
        self.simulated_odometry = (
            SimulatedOdometry(cfg_env.simulated_odometry)
            if cfg_env.simulated_odometry is not None and cfg_env.simulated_odometry.enabled
            else None
        )

        self._apply_random_heading()

        self.update()  # get initial state

    @staticmethod
    def _resolve_actuator_indices(model, joint_names) -> np.ndarray:
        joint_to_actuator = {}
        for actuator_id in range(model.nu):
            if model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT:
                continue
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name not in joint_names:
                continue
            if joint_name in joint_to_actuator:
                raise ValueError(f"Multiple MuJoCo actuators found for joint '{joint_name}'")
            joint_to_actuator[joint_name] = actuator_id

        missing = [name for name in joint_names if name not in joint_to_actuator]
        if missing:
            raise ValueError(f"No MuJoCo actuator found for joints: {missing}")
        return np.asarray([joint_to_actuator[name] for name in joint_names], dtype=np.int32)

    def _apply_random_heading(self):
        """Rotate the root body by a random yaw if random_heading is enabled."""
        if self.initial_heading_degrees is not None:
            yaw = np.deg2rad(self.initial_heading_degrees)
        elif self.random_heading:
            yaw = np.random.uniform(0, 2 * np.pi)
        else:
            return
        c, s = np.cos(yaw / 2), np.sin(yaw / 2)
        q = self.data.qpos[3:7].copy()  # MuJoCo [w, x, y, z]
        # Pre-multiply by yaw rotation q_yaw=[c,0,0,s]: q_new = q_yaw ⊗ q
        self.data.qpos[3] = c * q[0] - s * q[3]
        self.data.qpos[4] = c * q[1] - s * q[2]
        self.data.qpos[5] = c * q[2] + s * q[1]
        self.data.qpos[6] = c * q[3] + s * q[0]

    def reborn(self, init_qpos=None, init_qvel=None):
        if init_qpos is not None:
            init_qpos = np.asarray(init_qpos, dtype=np.float64)
            if init_qpos.shape == (7,):
                self.data.qpos[0:7] = init_qpos
            elif init_qpos.shape == self.data.qpos.shape:
                self.data.qpos[:] = init_qpos
            else:
                raise ValueError(f"init_qpos shape {init_qpos.shape} must be (7,) or {self.data.qpos.shape}")
            if init_qvel is None:
                self.data.qvel[:] = 0.0
            else:
                init_qvel = np.asarray(init_qvel, dtype=np.float64)
                if init_qvel.shape != self.data.qvel.shape:
                    raise ValueError(f"init_qvel shape {init_qvel.shape} != {self.data.qvel.shape}")
                self.data.qvel[:] = init_qvel
            self.data.ctrl[:] = 0.0
        else:
            if self.model.nkey > 0:
                mujoco.mj_resetDataKeyframe(self.model, self.data, 0)  # pyright: ignore[reportAttributeAccessIssue]
            else:
                mujoco.mj_resetData(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
            self._apply_random_heading()
        mujoco.mj_forward(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
        if self.elastic_band is not None:
            self.elastic_band.reset()
        if self.simulated_odometry is not None:
            self.simulated_odometry.reset(float(self.data.time))

    def set_odometry_replay_profile(self, profile: OdometryReplayProfile | None):
        """Replace the simulated odometer while preserving its safety configuration."""
        cfg = self.cfg_env.simulated_odometry
        if cfg is None or not cfg.enabled:
            raise RuntimeError("Simulated odometry must be enabled before installing a replay profile")
        self.simulated_odometry = SimulatedOdometry(cfg, replay_profile=profile)
        self.simulated_odometry.reset(float(self.data.time))

    def reset(self):
        if self.elastic_band is not None:
            self.elastic_band.reset()
        if self.simulated_odometry is not None:
            self.simulated_odometry.reset(float(self.data.time))
        self.reset_alignment()

    def set_gains(self, stiffness, damping):
        assert len(stiffness) == self.num_dofs and len(damping) == self.num_dofs
        self.stiffness = np.asarray(stiffness)
        self.damping = np.asarray(damping)

    def set_control_joint_names(self, joint_names):
        unknown = [name for name in joint_names if name not in self.joint_names]
        if unknown:
            raise ValueError(f"Control joints missing from MuJoCo environment: {unknown}")
        selected = set(joint_names)
        self._control_mask = np.asarray([name in selected for name in self.joint_names], dtype=bool)

    def arm_position_control(self):
        return

    def _get_elastic_band(self) -> ElasticBand:
        if self.elastic_band is None:
            raise RuntimeError("ElasticBand is not configured for this MuJoCo environment")
        return self.elastic_band

    def toggle_elastic_band(self) -> bool:
        return self._get_elastic_band().toggle()

    def lower_elastic_band(self) -> float:
        return self._get_elastic_band().lower()

    def lift_elastic_band(self) -> float:
        return self._get_elastic_band().lift()

    def self_check(self):
        pass

    def set_born_place(self, quat: np.ndarray | None = None, pos: np.ndarray | None = None):
        quat_ = self.base_quat if quat is None else quat
        pos_ = self.base_pos if pos is None else pos
        super().set_born_place(quat_, pos_)

    def update(self, simple=False):  # TODO: clean sensors in xml
        """simple: only update dof pos & vel"""
        dof_pos = self.data.qpos.astype(np.float32)[-self.num_dofs :]
        dof_vel = self.data.qvel.astype(np.float32)[-self.num_dofs :]

        self._dof_pos = dof_pos.copy()
        self._dof_vel = dof_vel.copy()

        if simple:
            return

        raw_quat = self.data.qpos.astype(np.float32)[3:7][[1, 2, 3, 0]]
        quat = raw_quat.copy()
        ang_vel = self.data.qvel.astype(np.float32)[3:6]
        raw_base_pos = self.data.qpos.astype(np.float32)[:3]
        base_pos = raw_base_pos.copy()
        world_lin_vel = self.data.qvel.astype(np.float32)[0:3]

        if self.born_place_align:
            quat, base_pos = self.base_align.align_transform(quat, base_pos)

        # MuJoCo free-joint translation velocity is expressed in the raw world
        # frame. Convert it with the physical root orientation, not the
        # born-place-aligned orientation exposed to policies.
        lin_vel = quat_rotate_inverse_np(raw_quat, world_lin_vel)

        if self.simulated_odometry is not None:
            relative_fk = self.kinematics.forward(
                joint_pos=self._dof_pos,
                base_pos=np.zeros(3, dtype=np.float64),
                base_quat=np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            )
            root_to_torso_pos = np.asarray(relative_fk[self._torso_name]["pos"], dtype=np.float64)
            root_to_torso_quat = np.asarray(relative_fk[self._torso_name]["quat"], dtype=np.float64)
            odom_cfg = self.simulated_odometry.cfg
            sensor_pos, sensor_quat = root_pose_to_sensor(
                raw_base_pos,
                raw_quat,
                root_to_torso_pos,
                root_to_torso_quat,
                odom_cfg.torso_to_sensor_position,
                odom_cfg.torso_to_sensor_quaternion,
            )

            def convert_sensor_pose(position, quaternion):
                return sensor_pose_to_root(
                    position,
                    quaternion,
                    odom_cfg.torso_to_sensor_position,
                    odom_cfg.torso_to_sensor_quaternion,
                    root_to_torso_pos,
                    root_to_torso_quat,
                )

            estimate = self.simulated_odometry.update(
                float(self.data.time),
                sensor_pos,
                sensor_quat,
                convert_sensor_pose,
            )
            if estimate is not None:
                base_pos = estimate.position
                if self.born_place_align:
                    base_pos = self.base_align.align_pos(base_pos)
                lin_vel = estimate.linear_velocity_body
                if estimate.stale and odom_cfg.fail_on_stale:
                    raise RuntimeError(
                        f"Simulated odometry became stale ({estimate.age:.3f}s > {odom_cfg.timeout:.3f}s)"
                    )
        rpy = quatToEuler(quat)

        self._base_rpy = rpy.copy()
        self._base_quat = quat.copy()
        self._base_ang_vel = ang_vel.copy()

        self._base_pos = base_pos.copy()
        self._base_lin_vel = lin_vel.copy()

        if self.update_with_fk:
            fk_info = self.fk()
            self._fk_info = fk_info.copy()
            self._torso_ang_vel = fk_info[self._torso_name]["ang_vel"]
            self._torso_quat = fk_info[self._torso_name]["quat"]
            self._torso_pos = fk_info[self._torso_name]["pos"]

    def _render(self):
        if self.viewer is None:
            return
        self.viewer.cam.lookat = self.data.qpos.astype(np.float32)[:3]
        if self.viewer.is_alive:
            if self.elastic_band is not None:
                self.elastic_band.update_visualization(self.viewer)
            self.viewer.render()

    def _simulate_torque(self, torque_fn):
        self._render()
        for _ in range(self.sim_decimation):
            torque = np.asarray(torque_fn(), dtype=np.float64)
            torque = np.clip(torque, -self.torque_limits, self.torque_limits)
            self.data.ctrl[:] = 0.0
            self.data.ctrl[self._dof_actuator_indices] = torque
            if self.elastic_band is not None:
                self.elastic_band.apply()
            mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
            self.update(simple=True)
        self.update(simple=False)

    def command_passive(self):
        self._simulate_torque(lambda: np.zeros(self.num_dofs, dtype=np.float64))

    def command_damping(self, damping=5.0):
        if damping < 0.0:
            raise ValueError("Damping must be non-negative")
        self._simulate_torque(lambda: -self.dof_vel * damping)

    def step(self, pd_target, hand_pose=None):
        assert len(pd_target) == self.num_dofs, "pd_target len should be num_dofs of env"

        if hand_pose is not None:
            logger.info("Hand pose-->", hand_pose)
        pd_target = np.asarray(pd_target, dtype=np.float64)
        if getattr(getattr(self, "cfg_env", None), "clip_position_targets", False):
            pd_target = np.clip(pd_target, self.position_limits[:, 0], self.position_limits[:, 1])

        def pd_torque():
            torque = (pd_target - self.dof_pos) * self.stiffness - self.dof_vel * self.damping
            return np.where(self._control_mask, torque, 0.0)

        self._simulate_torque(pd_torque)

    def shutdown(self):
        if self.viewer is not None:
            self.viewer.close()

    @property
    def odometry_diagnostics(self) -> dict | None:
        if self.simulated_odometry is None:
            return None
        return self.simulated_odometry.diagnostics(float(self.data.time))


if __name__ == "__main__":
    from robojudo.config.g1.env.g1_mujuco_env_cfg import G1MujocoEnvCfg

    mujoco_env = MujocoEnv(cfg_env=G1MujocoEnvCfg())
    mujoco_env.viewer._paused = False

    while True:
        # mujoco_env.update()
        mujoco_env.step(np.zeros(mujoco_env.num_dofs))
        time.sleep(0.02)
