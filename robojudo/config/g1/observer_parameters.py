"""G1 29-DOF stiff Locomanipulation arm parameters fitted in MuJoCo."""

from copy import deepcopy

from robojudo.config.g1.env.g1_env_cfg import G1_29_ARM_JOINT_NAMES

JOINT_NAMES = G1_29_ARM_JOINT_NAMES

# MuJoCo additive-torque fit using the current 29-DOF carry target.
_PARAMETERS = {
    "inertia": [
        0.12860527766360236,
        0.08197342060766341,
        0.042021806329721426,
        0.03591442046213406,
        0.01037,
        0.01,
        0.01,
        0.12842812374598464,
        0.08193883941324785,
        0.04205172550977063,
        0.03592922927115913,
        0.01037,
        0.01,
        0.01,
    ],
    "friction_viscosity": [
        54.397412283847785,
        45.566913270335945,
        24.466240880403888,
        22.920117692001146,
        9.050174705830033,
        11.102207997914238,
        10.08360479789641,
        54.3756049953329,
        45.56993876282425,
        24.469906029843813,
        22.90731944787912,
        9.0501308295424,
        11.102028936320897,
        10.083644323233575,
    ],
    "friction_limit": 0.1,
    "bandwidth_rad_s": 8.0,
}


def selected_observer_parameters():
    """Return a copy of the MuJoCo-fitted coefficients in JOINT_NAMES order."""
    return deepcopy(_PARAMETERS)
