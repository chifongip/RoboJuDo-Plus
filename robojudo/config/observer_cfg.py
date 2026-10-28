"""Shared configuration for synchronous and timestamped joint observers."""

import math
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from robojudo.config import Config


class JointObserverCfg(Config):
    """Robot-independent observer settings; vectors follow joint_names order."""

    model_config = ConfigDict(extra="forbid")

    algorithm: Literal["linearized", "nonlinear"]
    # Nonlinear replay acceleration; auto uses NumPy when the optional extra is absent.
    backend: Literal["auto", "numpy", "numba"] = "auto"
    mode: Literal["OFF", "SHADOW", "ACTIVE"] = "OFF"
    joint_names: list[str]
    inertia: float | list[float]
    friction_limit: float | list[float] = 0.0
    friction_viscosity: float | list[float] = 0.0
    bandwidth_rad_s: float = Field(default=5.0, gt=0.0, allow_inf_nan=False)
    actuator_limits_nm: list[float] | None = None
    correction_limit_fraction: float = Field(default=0.8, gt=0.0, le=1.0, allow_inf_nan=False)
    # Zero disables the absolute Nm cap.
    correction_limit_nm: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    slew_nm_s: float = Field(gt=0.0, allow_inf_nan=False)

    release_slew_nm_s: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def check_parameters(self):
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("Observer joint names must be nonempty and unique")
        for field in ("inertia", "friction_limit", "friction_viscosity", "actuator_limits_nm"):
            value = getattr(self, field)
            if value is None:
                continue
            values = value if isinstance(value, list) else [value]
            if isinstance(value, list) and len(value) != len(self.joint_names):
                raise ValueError(f"Observer {field} must match joint_names")
            positive = field in ("inertia", "actuator_limits_nm")
            if any(not math.isfinite(v) or v < 0 or (positive and v == 0) for v in values):
                raise ValueError(f"Observer {field} contains invalid values")
        if self.algorithm == "linearized" and self.backend == "numba":
            raise ValueError("Numba backend supports only the nonlinear observer")
        if self.algorithm == "linearized" and self.release_slew_nm_s not in (None, self.slew_nm_s):
            raise ValueError("Linearized observer requires symmetric growth/release slew")
        return self

    def resolve_correction_limits(self, actuator_limits: list[float] | None = None) -> list[float]:
        """Resolve bounds in joint_names order using configured or environment actuator limits."""
        limits = self.actuator_limits_nm if self.actuator_limits_nm is not None else actuator_limits
        if limits is None:
            raise ValueError("Fraction-based correction limits require actuator limits")
        if len(limits) != len(self.joint_names) or any(not math.isfinite(v) or v <= 0 for v in limits):
            raise ValueError("Actuator limits must be finite positive values matching joint_names")
        corrections = [self.correction_limit_fraction * value for value in limits]
        if self.correction_limit_nm > 0:
            return [min(self.correction_limit_nm, value) for value in corrections]
        return corrections

    def validate_hardware(self):
        """Hardware ACTIVE must never inherit actuator limits from a simulator."""
        if self.mode == "ACTIVE" and self.actuator_limits_nm is None:
            raise ValueError("ACTIVE hardware observer requires explicit actuator limits")
