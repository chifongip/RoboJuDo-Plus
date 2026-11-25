"""G1 23-DOF stiff Locomanipulation arm parameters for MuJoCo."""

from copy import deepcopy

from robojudo.config.g1.env.g1_env_cfg import G1_23_ARM_JOINT_NAMES

JOINT_NAMES = G1_23_ARM_JOINT_NAMES

# MuJoCo additive-torque fit using the carry target in carry_pose.py.
_PARAMETERS = {
    "inertia": [
        0.08216920519903391,
        0.054029055565458575,
        0.02307514389447937,
        0.023523605813703843,
        0.0102,
        0.08223207430126483,
        0.054141546260217076,
        0.0231153526991415,
        0.023522338976306244,
        0.0102,
    ],
    "friction_viscosity": [
        52.615318037133186,
        45.23266122067566,
        21.161748940638486,
        20.57448312470401,
        8.89801138726859,
        52.56611323782564,
        45.292658710917905,
        21.164880628960617,
        20.567421925649406,
        8.89801359014609,
    ],
    "friction_limit": 0.1,
    "bandwidth_rad_s": 8.0,
}


def selected_observer_parameters():
    """Return a copy of the MuJoCo-fitted coefficients in JOINT_NAMES order."""
    return deepcopy(_PARAMETERS)
