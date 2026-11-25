"""G1 23-DOF arm targets for the carrying benchmark."""

from robojudo.config.g1.env.g1_env_cfg import G1_23_ARM_JOINT_NAMES

G1_CARRY_POSE = dict(
    zip(
        G1_23_ARM_JOINT_NAMES,
        (
            0.14138370441992865,
            0.309611891932946,
            0.12363136242437246,
            -0.6125539252059866,
            -0.36206120905027156,
            0.14852794720373436,
            -0.3099796925970365,
            -0.12241566936009014,
            -0.610550312999851,
            0.36162135641089566,
        ),
        strict=True,
    )
)
