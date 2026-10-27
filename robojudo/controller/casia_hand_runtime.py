"""Lazy RoboJuDo adapter for the optional dual CASIA Hand-M SDK."""

from __future__ import annotations

CASIA_JOINT_SUFFIXES = (
    "thumb_proximal",
    "thumb_intermediate",
    "index_proximal",
    "middle_proximal",
    "ring_proximal",
    "pinky_proximal",
    "index_intermediate",
    "middle_intermediate",
    "ring_intermediate",
    "pinky_intermediate",
)
CASIA_LEFT_JOINT_NAMES = tuple(f"left_{suffix}" for suffix in CASIA_JOINT_SUFFIXES)
CASIA_RIGHT_JOINT_NAMES = tuple(f"right_{suffix}" for suffix in CASIA_JOINT_SUFFIXES)
CASIA_JOINT_NAMES = (*CASIA_LEFT_JOINT_NAMES, *CASIA_RIGHT_JOINT_NAMES)


class CasiaHandRuntime:
    """Construct the SDK runtime only when a CASIA hardware config is selected.

    Keeping the vendor import inside ``__new__`` lets all other RoboJuDo
    controllers and configurations import without the optional native module.
    """

    def __new__(cls, cfg, **kwargs):
        try:
            import casiahand_sdk
        except ImportError as exc:
            raise RuntimeError(
                "CASIA Hand control requires casiahand_sdk; run "
                "`python submodule_install.py casiahand_sdk` in the active environment"
            ) from exc

        if tuple(casiahand_sdk.CASIA_LEFT_JOINT_NAMES) != CASIA_LEFT_JOINT_NAMES or tuple(
            casiahand_sdk.CASIA_RIGHT_JOINT_NAMES
        ) != CASIA_RIGHT_JOINT_NAMES:
            raise RuntimeError("RoboJuDo and casiahand_sdk use different CASIA physical joint schemas")
        return casiahand_sdk.CasiaHandRuntime(cfg, **kwargs)


__all__ = [
    "CASIA_JOINT_NAMES",
    "CASIA_LEFT_JOINT_NAMES",
    "CASIA_RIGHT_JOINT_NAMES",
    "CasiaHandRuntime",
]
