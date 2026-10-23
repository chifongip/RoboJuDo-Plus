"""Real X2 deployment preset; additive compensation remains opt-in."""

from pydantic import Field

from robojudo.config import cfg_registry
from robojudo.utils.x2_observer_reference import JOINT_NAMES

from .env.x2_real_env_cfg import X2ArmObserverCfg, X2RealEnvCfg
from .x2_cfg import x2_locomimic_real


def _observer_env() -> X2RealEnvCfg:
    env = x2_locomimic_real().env.model_copy(deep=True)
    limits_by_name = dict(zip(env.dof.joint_names, env.dof.torque_limits, strict=True))
    env.arm_observer = X2ArmObserverCfg(
        mode="OFF",
        # Reuse configured bounds by name, in observer order. Their presence does
        # not establish continuous hardware ratings or firmware clipping limits.
        actuator_limits_nm=[limits_by_name[name] for name in JOINT_NAMES],
        # Selected simulation settings for optional SHADOW evaluation. These are
        # not hardware-qualified ACTIVE limits; keep compensation OFF by default.
        correction_limits_nm=[min(8.0, 0.8 * limits_by_name[name]) for name in JOINT_NAMES],
        slew_nm_s=2.0,
    )
    return env


@cfg_registry.register
class x2_locomimic_real_observer(x2_locomimic_real):
    """Position-control deployment with inherited policies, ZMQ and safety gates.

    Edit _observer_env for observer commissioning before creating the pipeline.
    """

    env: X2RealEnvCfg = Field(default_factory=_observer_env)
