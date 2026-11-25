"""G1 23-DOF stiff Locomanipulation arm parameters for MuJoCo."""

from copy import deepcopy

from robojudo.config.g1.env.g1_env_cfg import G1_23_ARM_JOINT_NAMES

JOINT_NAMES = G1_23_ARM_JOINT_NAMES

# MuJoCo additive-torque fit using the current 23-DOF carry target.
_PARAMETERS = {
    "inertia": [
        0.09186950253917056,
        0.061019161763885936,
        0.026431705651179672,
        0.02342746849523782,
        0.0102,
        0.0918514835622441,
        0.06106578483046442,
        0.026449785699018978,
        0.023426484888785452,
        0.0102,
    ],
    "friction_viscosity": [
        52.902388048386,
        45.217739673251415,
        21.16724895257324,
        20.570311618666157,
        8.898014352329428,
        52.86294285903987,
        45.21531807390772,
        21.169453138257435,
        20.563470711838786,
        8.898013126563626,
    ],
    "friction_limit": 0.1,
    "bandwidth_rad_s": 8.0,
}


def selected_observer_parameters():
    """Return a copy of the MuJoCo-fitted coefficients in JOINT_NAMES order."""
    return deepcopy(_PARAMETERS)
