"""Selected X2 simulation observer reference; no robot/SDK integration.

The user accepted the measured carrying and sway tradeoffs on 2026-10-22.
Calibrated from standing additive-torque responses, then evaluated across
4/5/6 s preparation warmups with policy_fine_tune, hand-COM 20 N loads and
0.5 m/s commanded walking. These are nominal model coefficients, not verified
physical inertias or hardware torque ratings. The original integral remains
the baseline; correction bounds and slew are owned by the observer class.
"""

from copy import deepcopy

JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
)

_PARAMETERS = {
    "inertia": [
        0.15532404980256578,
        0.10078795654951321,
        0.07674159569739615,
        0.07182380882978089,
        0.03,
        0.03,
        0.03,
        0.1553018849773343,
        0.1012386139709888,
        0.07677501375357826,
        0.07193011707176135,
        0.03,
        0.03,
        0.03,
    ],
    "friction_limit": 0.3,
    "friction_viscosity": [
        106.80829994233635,
        228.63155219753165,
        29.852550817628547,
        44.32684035243929,
        28.85973128906408,
        31.97077929396376,
        32.10787293174743,
        106.63727466536166,
        228.8259861441458,
        29.86332093192248,
        44.346794114158094,
        28.860217878180926,
        31.968164529938882,
        32.102352113935815,
    ],
    "bandwidth_rad_s": 8.0,
    "method": "observer",
}


def selected_observer_parameters():
    """Return fresh fixed coefficients in JOINT_NAMES order for simulation only."""
    return deepcopy(_PARAMETERS)
