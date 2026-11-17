"""Shared simulation support for the selected bounded torque-integral benchmark."""

import math

import mujoco
import numpy as np
from box import Box

from robojudo.config.x2.env.x2_env_cfg import X2_ARM_JOINT_NAMES
from robojudo.config.x2.x2_cfg import x2_locomimic
from robojudo.environment import env_registry
from robojudo.environment.mujoco_env import MujocoEnv
from robojudo.pipeline.four_mode_pipeline import FALL_TILT_THRESHOLD_RAD
from robojudo.utils.bounded_integral import BoundedIntegralCorrection

POSTURE = dict(
    zip(
        X2_ARM_JOINT_NAMES,
        (
            -0.081,
            0.026,
            -0.049,
            -2.188,
            0.238,
            0.326,
            0.215,
            -0.081,
            -0.026,
            0.049,
            -2.188,
            -0.238,
            0.326,
            -0.215,
        ),
        strict=True,
    )
)


SIDES = ("left", "right")


class SimulationFailure(Exception):
    """A failed trial, retained in the evaluation results."""


def point_wrench(point, center_of_mass, force):
    """MuJoCo xfrc_applied stores world force followed by torque about body COM."""
    return np.concatenate((force, np.cross(point - center_of_mass, force)))


def force_point_metadata(xml):
    model = mujoco.MjModel.from_xml_path(str(xml))
    return {
        "force_point": "hand center: wrist_roll_link center of mass",
        "force_point_body_local_m": {
            side: model.body_ipos[model.body(f"{side}_wrist_roll_link").id].tolist() for side in SIDES
        },
    }


def torque_limit_metadata(env):
    """Identify configured limits without claiming physical-robot verification."""
    return {
        "source": "X2 locomomanipulation environment/policy DoF configuration",
        "hardware_limits_verified": False,
        "hardware_limit_kind": "unknown: peak, continuous and firmware-enforced limits have not been verified",
        "total_torque_clip": "PD plus additive effort, clipped at every simulation physics substep",
        "joints": {
            name: {
                "configured_total_limit_nm": float(env.torque_limits[index]),
                "xml_joint_actuator_force_range_nm": env.model.jnt_actfrcrange[env.model.joint(name).id].tolist(),
                "integral_bound_nm": float(env.integral.limits[list(POSTURE).index(name)]) if name in POSTURE else 0.0,
            }
            for index, name in enumerate(env.joint_names)
        },
    }


@env_registry.register
class X2IntegralEvaluationEnv(MujocoEnv):
    """Hand-center loads are separate from feedback-only arm torque correction."""

    def __init__(self, cfg_env, device="cpu"):
        self.measurement_start = None
        super().__init__(cfg_env, device)
        self.arm_indices = np.asarray([self.joint_names.index(name) for name in X2_ARM_JOINT_NAMES])
        self.ee_ids = [self.model.body(f"{side}_wrist_roll_link").id for side in SIDES]
        self.torso_id = self.model.body("torso_link").id
        self.pelvis_id = self.model.body("pelvis").id
        self.raw_effort = np.zeros(self.num_dofs)
        self.last_feedforward_effort = np.zeros(self.num_dofs)
        self.last_force_n = 0.0
        self.physics_steps = 0
        self.saturated_steps = 0
        self.joint_saturated_steps = np.zeros(self.num_dofs, dtype=int)
        self.control_saturated_steps = np.zeros(self.num_dofs, dtype=int)
        self.max_base_tilt = 0.0
        self.min_base_height = float("inf")
        self.integral = BoundedIntegralCorrection(
            self.stiffness[self.arm_indices], self.damping[self.arm_indices], self.torque_limits[self.arm_indices]
        )
        self.integral_enabled = False
        self.profile = "constant"
        self.force = np.zeros(3)
        self.command_vx = 0.0
        self.last_slew = np.zeros(len(self.arm_indices))
        self.last_slew_limits = np.full(len(self.arm_indices), self.integral.slew)
        self.max_effort_slew = 0.0
        self.max_effort_bound_excess = 0.0

    def _update_force_visualization(self):
        if self.viewer is None:
            return
        for side_index, (side, body_id) in enumerate(zip(SIDES, self.ee_ids, strict=True)):
            force = self.data.xfrc_applied[body_id, :3]
            magnitude = float(np.linalg.norm(force))
            visible = np.isfinite(force).all() and magnitude > 1e-6
            origin = self.force_points()[side_index]
            geom = mujoco.MjvGeom()
            # Arrows start at the force point: 1 N corresponds to 1 cm.
            tip = origin + 0.01 * force if visible else origin
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.008, origin, tip)
            self.viewer.add_marker(
                id=4100 + side_index,
                type=mujoco.mjtGeom.mjGEOM_ARROW,
                pos=geom.pos.copy(),
                mat=geom.mat.copy(),
                size=geom.size.copy(),
                rgba=np.asarray([1.0, 0.15 + 0.45 * side_index, 0.05, 0.9 if visible else 0.0]),
                label=f"{side}: {magnitude:.1f} N" if visible else "",
            )

    def _render(self):
        self._update_force_visualization()
        super()._render()

    def force_points(self):
        """World positions of the modeled hand centers (wrist-roll body COMs)."""
        return self.data.xipos[self.ee_ids].copy()

    def check_state(self):
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise SimulationFailure("nonfinite simulation state")
        rotation = self.data.xmat[self.pelvis_id].reshape(3, 3)
        tilt = math.acos(float(np.clip(rotation[2, 2], -1, 1)))
        height = float(self.data.xpos[self.pelvis_id, 2])
        self.max_base_tilt = max(self.max_base_tilt, tilt)
        self.min_base_height = min(self.min_base_height, height)
        if tilt >= FALL_TILT_THRESHOLD_RAD:
            raise SimulationFailure(f"base tilt {tilt:.3f} rad reached the pipeline fall threshold")
        if height < 0.30:
            raise SimulationFailure(f"pelvis height {height:.3f} m below 0.30 m")
        for warning in (
            mujoco.mjtWarning.mjWARN_BADQPOS,
            mujoco.mjtWarning.mjWARN_BADQVEL,
            mujoco.mjtWarning.mjWARN_BADQACC,
        ):
            if self.data.warning[warning].number:
                raise SimulationFailure(f"MuJoCo numerical warning: {warning.name}")

    def update_load(self):
        mujoco.mj_fwdPosition(self.model, self.data)
        self.check_state()
        for body_id in self.ee_ids:
            self.data.xfrc_applied[body_id] = point_wrench(
                self.data.xipos[body_id], self.data.xipos[body_id], self.force
            )

    def _simulate_torque(self, torque_fn):
        self.control_saturated_steps[:] = 0

        def loaded_torque():
            self.update_load()
            torque = torque_fn()
            self.physics_steps += 1
            saturated = np.abs(torque) > self.torque_limits
            self.saturated_steps += int(np.any(saturated))
            self.joint_saturated_steps += saturated
            self.control_saturated_steps += saturated
            return torque

        super()._simulate_torque(loaded_torque)
        mujoco.mj_fwdPosition(self.model, self.data)
        self.check_state()

    def reset_measurements(self):
        """Exclude supported preparation and policy warmup from measurement extrema."""
        self.physics_steps = self.saturated_steps = 0
        self.joint_saturated_steps[:] = 0
        self.max_base_tilt = 0.0
        self.min_base_height = float("inf")
        self.max_effort_slew = self.max_effort_bound_excess = 0.0

    def step(self, pd_target, hand_pose=None, *, feedforward_effort=None):
        if feedforward_effort is not None:
            raise ValueError("Evaluation owns integral effort")
        self.last_target = np.asarray(pd_target).copy()
        self.raw_effort[:] = 0
        previous = self.integral.effort.copy()
        if self.integral_enabled:
            position, velocity = self.correction_feedback()
            self.raw_effort[self.arm_indices] = self.integral.update(
                self.last_target[self.arm_indices],
                position,
                velocity,
                self.control_dt,
            )
        self.last_slew = (self.integral.effort - previous) / self.control_dt
        self.last_slew_limits = np.full(previous.shape, self.integral.slew)
        release_slew = getattr(self.integral, "release_slew", None)
        if release_slew is not None:
            desired = self.integral.desired_nm
            releasing = (previous != 0) & ((desired * previous <= 0) | (abs(desired) < abs(previous)))
            self.last_slew_limits = np.where(releasing, release_slew, self.integral.slew)
        self.max_effort_slew = max(self.max_effort_slew, float(np.max(np.abs(self.last_slew))))
        self.max_effort_bound_excess = max(
            self.max_effort_bound_excess, float(np.max(np.abs(self.integral.effort) - self.integral.limits))
        )
        MujocoEnv.step(self, pd_target, hand_pose, feedforward_effort=self.raw_effort)

    def correction_feedback(self):
        """Measured arm feedback; simulation studies may replace this channel."""
        return self.dof_pos[self.arm_indices], self.dof_vel[self.arm_indices]

    def clear_load(self):
        self.measurement_start = None
        self.integral_enabled = False
        self.integral.reset()
        self.force[:] = 0
        self.last_force_n = 0.0
        self.data.xfrc_applied[self.ee_ids] = 0
        self.raw_effort[:] = 0
        self.last_feedforward_effort[:] = 0
        self._update_force_visualization()

    def end_effector_positions(self):
        world = self.force_points()
        torso = (world - self.data.xpos[self.torso_id]) @ self.data.xmat[self.torso_id].reshape(3, 3)
        return world, torso


def evaluation_config(viewer=False):
    cfg = x2_locomimic()
    cfg.loco_policy = cfg.loco_policy.model_copy(deep=True, update={"policy_name": "policy_fine_tune"})
    cfg.device = "cpu"
    cfg.ctrl = []
    cfg.debug.log_obs = False
    cfg.record.enabled = False
    cfg.loco_policy.command_status_interval_s = 0.0
    cfg.env.headless = not viewer
    cfg.env.visualize_extras = False
    cfg.env.elastic_band.active = False
    cfg.env.env_type = "X2IntegralEvaluationEnv"
    cfg.env.random_heading = False
    cfg.env.feedforward_effort_limits = {
        name: float(cfg.env.dof.torque_limits[cfg.env.dof.joint_names.index(name)]) for name in POSTURE
    }
    cfg.joint_default_dof = cfg.joint_default_dof.model_copy(deep=True)
    for name, value in POSTURE.items():
        index = cfg.joint_default_dof.joint_names.index(name)
        cfg.joint_default_dof.default_pos[index] = value
    # The policy manager maintains this arm pose while the 15-action model controls legs/waist.
    cfg.upper_dof_pos_default = list(cfg.upper_dof_pos_default)
    for name, value in POSTURE.items():
        index = cfg.env.dof.joint_names.index(name) - (cfg.env.dof.num_dofs - cfg.upper_dof_num)
        cfg.upper_dof_pos_default[index] = value
    return cfg


def sample_row(env, elapsed):
    world, torso = env.end_effector_positions()
    errors = env.dof_pos[env.arm_indices] - env.last_target[env.arm_indices]
    rotation = env.data.xmat[env.pelvis_id].reshape(3, 3)
    row = {
        "elapsed_s": elapsed,
        "phase": "measurement",
        "force_per_hand_n": env.last_force_n,
        "base_height_m": float(env.data.xpos[env.pelvis_id, 2]),
        "base_tilt_rad": math.acos(float(np.clip(rotation[2, 2], -1, 1))),
        "arm_rms_error_rad": float(np.sqrt(np.mean(errors**2))),
        "saturated": int(np.any(env.last_saturation_mask)),
        "base_roll_rad": float(np.arctan2(rotation[2, 1], rotation[2, 2])),
        "base_pitch_rad": float(np.arcsin(np.clip(-rotation[2, 0], -1, 1))),
    }
    for side_index, side in enumerate(SIDES):
        for axis, label in enumerate("xyz"):
            row[f"{side}_world_{label}_m"] = float(world[side_index, axis])
            row[f"{side}_torso_{label}_m"] = float(torso[side_index, axis])
    for index in env.arm_indices:
        name = env.joint_names[index]
        row[f"{name}_target_rad"] = float(env.last_target[index])
        row[f"{name}_position_rad"] = float(env.dof_pos[index])
        row[f"{name}_velocity_rad_s"] = float(env.dof_vel[index])
        row[f"{name}_pd_nm"] = float(env.last_pd_torque[index])
        row[f"{name}_feedforward_nm"] = float(env.last_feedforward_effort[index])
        row[f"{name}_applied_nm"] = float(env.last_applied_torque[index])
    for index, name in enumerate(env.joint_names):
        row[f"{name}_saturation_fraction"] = float(env.control_saturated_steps[index] / env.sim_decimation)
    return row


def synthetic_velocity(vx):
    return Box(
        {
            "COMMANDS": [],
            "VELOCITY_SOURCE": "VelocityZmqCtrl",
            "POSTURE_SOURCE": None,
            "VelocityZmqCtrl": {
                "fresh": True,
                "linear_velocity": [vx, 0, 0],
                "angular_velocity": [0, 0, 0],
            },
        }
    )


def integral_sample(env, t):
    row = sample_row(env, t)
    row["command_vx_m_s"] = env.command_vx
    row["actual_vx_m_s"] = float(env.base_lin_vel[0])
    for axis, value in zip("xyz", env.force, strict=True):
        row[f"force_{axis}_n"] = float(value)
    for i, joint in enumerate(POSTURE):
        row[f"{joint}_correction_nm"] = float(env.integral.effort[i])
        if isinstance(env.integral, BoundedIntegralCorrection):
            row[f"{joint}_filtered_error_rad"] = float(env.integral.filtered_error[i])
            row[f"{joint}_integral_nm"] = float(env.integral.effort[i])
        else:
            row[f"{joint}_observer_nm"] = float(env.integral.effort[i])
            row[f"{joint}_disturbance_estimate_nm"] = float(env.integral.estimated_disturbance_nm[i])
            row[f"{joint}_velocity_prediction_rad_s"] = float(env.integral.velocity_prediction[i])
            row[f"{joint}_velocity_innovation_rad_s"] = float(env.integral.velocity_innovation[i])
        row[f"{joint}_effort_slew_nm_s"] = float(env.last_slew[i])
        row[f"{joint}_antiwindup"] = int(env.integral.antiwindup[i])
        row[f"{joint}_at_bound"] = int(abs(env.integral.effort[i]) >= env.integral.limits[i] - 1e-10)
        row[f"{joint}_at_slew_limit"] = int(abs(env.last_slew[i]) >= env.last_slew_limits[i] - 1e-10)
    return row


def rows_in(rows, start, end):
    return [r for r in rows if start <= r["elapsed_s"] < end]


def column(rows, key):
    return np.asarray([r[key] for r in rows])


def detrended_std(values):
    values = np.asarray(values)
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values))
    return float(np.std(values - np.polyval(np.polyfit(x, values, 1), x)))
