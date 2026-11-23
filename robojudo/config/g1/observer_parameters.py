"""G1 23-DOF stiff Locomanipulation arm parameters for MuJoCo."""

from copy import deepcopy

from robojudo.config.g1.env.g1_env_cfg import G1_23_ARM_JOINT_NAMES

JOINT_NAMES = G1_23_ARM_JOINT_NAMES

# Fitted with scripts/evaluate_g1_observer.py --fit. Friction cap is from the MJCF.
_PARAMETERS = {
    "inertia": [
        0.269636,
        0.464476,
        0.08283,
        0.090581,
        0.007837,
        0.179305,
        0.29025,
        0.08283,
        0.088644,
        0.007625,
    ],
    "friction_limit": 0.1,
    "friction_viscosity": [
        0.819969,
        2.099862,
        7.033537,
        0.556919,
        7.14699,
        0.703229,
        2.298163,
        3.8369,
        1.154636,
        6.54996,
    ],
    "bandwidth_rad_s": 8.0,
}


def selected_observer_parameters():
    """Return fresh model coefficients in JOINT_NAMES order for simulation."""
    return deepcopy(_PARAMETERS)
