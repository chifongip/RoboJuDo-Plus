"""X2 arm targets for the carrying benchmark."""

from robojudo.config.x2.env.x2_env_cfg import X2_ARM_JOINT_NAMES

X2_CARRY_POSE = dict(
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
