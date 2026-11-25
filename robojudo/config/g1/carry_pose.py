"""G1 arm targets for the carrying benchmark."""

from robojudo.config.g1.env.g1_env_cfg import G1_23_ARM_JOINT_NAMES, G1_29_ARM_JOINT_NAMES

G1_23_CARRY_POSE = dict(
    zip(
        G1_23_ARM_JOINT_NAMES,
        (
            -0.10538606454009122,
            0.3110875633300908,
            0.10263418707300465,
            -0.35178299386885914,
            -0.33279432595174296,
            -0.09901032254672848,
            -0.31164459060494387,
            -0.10147326142369337,
            -0.35069419705870486,
            0.33277267784391723,
        ),
        strict=True,
    )
)

# MuJoCo IK positions the 29-DOF wrist-yaw body's center of mass 5 cm forward
# of the 23-DOF wrist-hand body's center of mass, preserving its orientation in
# the pelvis frame.
G1_CARRY_POSE = dict(
    zip(
        G1_29_ARM_JOINT_NAMES,
        (
            -0.11419255959405335,
            0.27251687674442737,
            0.10410689202940233,
            -0.19539157984520328,
            -0.29943273717757873,
            -0.14033675393208384,
            -0.05876427125423291,
            -0.10820148059755065,
            -0.2735168993191418,
            -0.10306724654180412,
            -0.1963821038834506,
            0.29971813739005054,
            -0.13789937278443318,
            0.05800083238605073,
        ),
        strict=True,
    )
)
