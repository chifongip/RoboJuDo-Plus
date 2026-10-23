"""Shared configuration for synchronous and timestamped joint observers."""

import math
from typing import Literal

from pydantic import Field, model_validator

from robojudo.config import Config


class JointObserverCfg(Config):
    """Robot-independent observer settings; vectors follow joint_names order."""

    algorithm: Literal["linearized", "nonlinear"]
    mode: Literal["OFF", "SHADOW", "ACTIVE"] = "OFF"
    joint_names: list[str]
    inertia: float | list[float]
    friction_limit: float | list[float] = 0.0
    friction_viscosity: float | list[float] = 0.0
    bandwidth_rad_s: float = Field(default=5.0, gt=0.0, allow_inf_nan=False)
    actuator_limits_nm: list[float] | None = None
    correction_limits_nm: float | list[float]
    slew_nm_s: float = Field(gt=0.0, allow_inf_nan=False)

    release_slew_nm_s: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def check_parameters(self):
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("Observer joint names must be nonempty and unique")
        for field in ("inertia", "friction_limit", "friction_viscosity", "actuator_limits_nm", "correction_limits_nm"):
            value = getattr(self, field)
            if value is None:
                continue
            values = value if isinstance(value, list) else [value]
            if isinstance(value, list) and len(value) != len(self.joint_names):
                raise ValueError(f"Observer {field} must match joint_names")
            positive = field in ("inertia", "actuator_limits_nm", "correction_limits_nm")
            if any(not math.isfinite(v) or v < 0 or (positive and v == 0) for v in values):
                raise ValueError(f"Observer {field} contains invalid values")
        if self.algorithm == "linearized" and self.release_slew_nm_s not in (None, self.slew_nm_s):
            raise ValueError("Linearized observer requires symmetric growth/release slew")
        if self.actuator_limits_nm is not None:
            corrections = self.correction_limits_nm
            if not isinstance(corrections, list):
                corrections = [corrections] * len(self.joint_names)
            if any(c > a for c, a in zip(corrections, self.actuator_limits_nm, strict=True)):
                raise ValueError("Correction limits must not exceed actuator limits")
        return self

    def validate_hardware(self):
        """Hardware ACTIVE must never inherit actuator limits from a simulator."""
        if self.mode == "ACTIVE" and self.actuator_limits_nm is None:
            raise ValueError("ACTIVE hardware observer requires explicit actuator limits")
