"""Retarget X2's calibration load points to G1 in the pelvis frame."""

import mujoco
import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation

# Express the local frames with +X along the fingers and +Y across the hand.
# G1's hand mesh uses local +X along the hand, while X2's uses local -Z.
G1_HAND_AXES = np.eye(3)
X2_HAND_AXES = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
G1_CARRY_OFFSET_M = np.array([-0.10, 0.0, -0.15])


def standing_state(xml, dof, arm_targets):
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    targets = dict(zip(dof.joint_names, dof.default_pos, strict=True)) | arm_targets
    for name, value in targets.items():
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_fwdPosition(model, data)
    return model, data


def hand_pose(data, body_name, local_axes):
    """Return the load point and hand orientation relative to the pelvis."""
    pelvis = data.body("pelvis")
    hand = data.body(body_name)
    world_to_pelvis = pelvis.xmat.reshape(3, 3).T
    position = world_to_pelvis @ (hand.xipos - pelvis.xpos)
    orientation = world_to_pelvis @ hand.xmat.reshape(3, 3) @ local_axes
    return position, orientation


def solve_arm(model, data, g1_dof, names, side, sign, target):
    ids = [model.joint(name).id for name in names]
    addresses = model.jnt_qposadr[ids]
    configured = np.asarray([g1_dof.position_limits[g1_dof.joint_names.index(name)] for name in names])
    lower = np.maximum(model.jnt_range[ids, 0], configured[:, 0])
    upper = np.minimum(model.jnt_range[ids, 1], configured[:, 1])
    target_point, target_orientation = target

    def error(q):
        data.qpos[addresses] = q
        mujoco.mj_fwdPosition(model, data)
        point, orientation = hand_pose(data, f"{side}_wrist_roll_rubber_hand", G1_HAND_AXES)
        angle = Rotation.from_matrix(target_orientation.T @ orientation).as_rotvec()
        return point - target_point, angle

    def residual(q):
        position, orientation = error(q)
        return np.concatenate((position, 0.01 * orientation))

    def orientation_cost(q):
        orientation = error(q)[1]
        return orientation @ orientation

    seeds = (data.qpos[addresses].copy(), [-0.6, sign, -0.25 * sign, 0, -sign])
    fits = [least_squares(residual, np.clip(seed, lower, upper), bounds=(lower, upper)) for seed in seeds]
    initial = min(fits, key=lambda fit: fit.cost)
    fit = minimize(
        orientation_cost,
        initial.x,
        method="SLSQP",
        bounds=list(zip(lower, upper, strict=True)),
        constraints=({"type": "eq", "fun": lambda q: error(q)[0]},),
        options={"ftol": 1e-12, "maxiter": 200},
    )
    position_error, _ = error(fit.x)
    if not fit.success or np.linalg.norm(position_error) > 1e-5:
        raise ValueError(f"X2 carry point has no verified G1 {side} arm solution: {fit.message}")
    return fit.x


def retarget_g1_carry(g1_xml, g1_dof, g1_joints, x2_xml, x2_dof, x2_targets):
    """Shift the X2 load points for G1 while minimizing wrist orientation error."""
    _, source = standing_state(x2_xml, x2_dof, x2_targets)
    model, data = standing_state(g1_xml, g1_dof, {})
    solutions = {}
    for side, sign in (("left", 1), ("right", -1)):
        names = [name for name in g1_joints if name.startswith(f"{side}_")]
        point, orientation = hand_pose(source, f"{side}_wrist_roll_link", X2_HAND_AXES)
        target = point + G1_CARRY_OFFSET_M, orientation
        angles = solve_arm(model, data, g1_dof, names, side, sign, target)
        solutions.update(zip(names, angles, strict=True))

    if any(
        contact.dist < 0 and model.geom_bodyid[contact.geom1] and model.geom_bodyid[contact.geom2]
        for contact in data.contact
    ):
        raise ValueError("Retargeted G1 carry pose has a self-collision")
    return np.asarray([solutions[name] for name in g1_joints])
