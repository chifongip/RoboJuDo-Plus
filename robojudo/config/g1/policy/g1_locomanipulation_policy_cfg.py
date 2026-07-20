from typing import ClassVar, Literal, TypeVar

from pydantic import model_validator

from robojudo.config import ASSETS_DIR
from robojudo.policy.policy_cfgs import PolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig

from ..env.g1_env_cfg import G1_23_DOF_INDICES, G1_29DoF

_T = TypeVar("_T")


def _select_23(values: list[_T]) -> list[_T]:
    return [values[index] for index in G1_23_DOF_INDICES]


_G1_29_DOF = G1_29DoF()
G1_LOCOMANIPULATION_29_JOINT_NAMES = _G1_29_DOF.joint_names
G1_LOCOMANIPULATION_23_JOINT_NAMES = _select_23(G1_LOCOMANIPULATION_29_JOINT_NAMES)
G1_LOCOMANIPULATION_23_ACTION_JOINT_NAMES = G1_LOCOMANIPULATION_23_JOINT_NAMES[:13]
G1_LOCOMANIPULATION_29_ACTION_JOINT_NAMES = G1_LOCOMANIPULATION_29_JOINT_NAMES[:15]

G1_LOCOMANIPULATION_29_DEFAULT_POS = [
    *[-0.1, 0.0, 0.0, 0.3, -0.2, 0.0],
    *[-0.1, 0.0, 0.0, 0.3, -0.2, 0.0],
    *[0.0, 0.0, 0.0],
    *[0.35, 0.18, 0.0, 0.87, 0.0, 0.0, 0.0],
    *[0.35, -0.18, 0.0, 0.87, 0.0, 0.0, 0.0],
]
G1_LOCOMANIPULATION_23_DEFAULT_POS = _select_23(G1_LOCOMANIPULATION_29_DEFAULT_POS)

_LOWER_STIFF_STIFFNESS = [100.0, 100.0, 100.0, 200.0, 20.0, 20.0] * 2
_LOWER_STIFF_DAMPING = [2.5, 2.5, 2.5, 5.0, 0.2, 0.1] * 2
_LOWER_DEFAULT_STIFFNESS = [
    40.17923863450712,
    99.09842777666111,
    40.17923863450712,
    99.09842777666111,
    28.50124619574858,
    28.50124619574858,
] * 2
_LOWER_DEFAULT_DAMPING = [
    2.557889775413375,
    6.308801853496639,
    2.557889775413375,
    6.308801853496639,
    1.814445686584846,
    1.814445686584846,
] * 2

G1_LOCOMANIPULATION_PD_GAIN_PRESETS_23: dict[str, tuple[list[float], list[float]]] = {
    "stiff": (
        [*_LOWER_STIFF_STIFFNESS, 200.0, *([60.0] * 10)],
        [*_LOWER_STIFF_DAMPING, 5.0, *([1.5] * 10)],
    ),
    "default": (
        [*_LOWER_DEFAULT_STIFFNESS, 40.17923863450712, *([14.25062309787429] * 10)],
        [*_LOWER_DEFAULT_DAMPING, 2.557889775413375, *([0.907222843292423] * 10)],
    ),
}
G1_LOCOMANIPULATION_PD_GAIN_PRESETS_29: dict[str, tuple[list[float], list[float]]] = {
    "stiff": (
        [*_LOWER_STIFF_STIFFNESS, 200.0, 1200.0, 1200.0, *([60.0] * 14)],
        [*_LOWER_STIFF_DAMPING, 5.0, 5.0, 5.0, *([1.5] * 14)],
    ),
}

G1_LOCOMANIPULATION_ACTION_SCALE_PRESETS_23: dict[str, list[float]] = {
    "stiff": [
        *[0.22, 0.3475, 0.22, 0.17375, 0.625, 0.625],
        *[0.22, 0.3475, 0.22, 0.17375, 0.625, 0.625],
        0.11,
    ],
    "default": [
        *[
            0.5475464629911068,
            0.35066146637882434,
            0.5475464629911068,
            0.35066146637882434,
            0.43857731392336724,
            0.43857731392336724,
        ],
        *[
            0.5475464629911068,
            0.35066146637882434,
            0.5475464629911068,
            0.35066146637882434,
            0.43857731392336724,
            0.43857731392336724,
        ],
        0.5475464629911068,
    ],
}
G1_LOCOMANIPULATION_ACTION_SCALE_PRESETS_29: dict[str, list[float]] = {
    "stiff": [
        *G1_LOCOMANIPULATION_ACTION_SCALE_PRESETS_23["stiff"],
        0.010416666666666666,
        0.010416666666666666,
    ],
}

G1_LOCOMANIPULATION_29_TORQUE_LIMITS = [
    *[88.0, 139.0, 88.0, 139.0, 50.0, 50.0],
    *[88.0, 139.0, 88.0, 139.0, 50.0, 50.0],
    *[88.0, 50.0, 50.0],
    *[25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0],
    *[25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0],
]
G1_LOCOMANIPULATION_23_TORQUE_LIMITS = _select_23(G1_LOCOMANIPULATION_29_TORQUE_LIMITS)

_G1_POSITION_LIMITS = dict(zip(_G1_29_DOF.joint_names, _G1_29_DOF.position_limits, strict=True))


class G1Locomanipulation23ObsDoF(DoFConfig):
    joint_names: list[str] = G1_LOCOMANIPULATION_23_JOINT_NAMES
    default_pos: list[float] | None = G1_LOCOMANIPULATION_23_DEFAULT_POS
    stiffness: list[float] | None = G1_LOCOMANIPULATION_PD_GAIN_PRESETS_23["stiff"][0]
    damping: list[float] | None = G1_LOCOMANIPULATION_PD_GAIN_PRESETS_23["stiff"][1]
    torque_limits: list[float] | None = G1_LOCOMANIPULATION_23_TORQUE_LIMITS
    position_limits: list[list[float]] | None = [_G1_POSITION_LIMITS[name] for name in joint_names]

    PD_GAIN_PRESETS: ClassVar[dict[str, tuple[list[float], list[float]]]] = (
        G1_LOCOMANIPULATION_PD_GAIN_PRESETS_23
    )

    @classmethod
    def from_preset(cls, preset: str) -> "G1Locomanipulation23ObsDoF":
        stiffness, damping = cls.PD_GAIN_PRESETS[preset]
        return cls(stiffness=stiffness, damping=damping)


class G1Locomanipulation29ObsDoF(DoFConfig):
    joint_names: list[str] = G1_LOCOMANIPULATION_29_JOINT_NAMES
    default_pos: list[float] | None = G1_LOCOMANIPULATION_29_DEFAULT_POS
    stiffness: list[float] | None = G1_LOCOMANIPULATION_PD_GAIN_PRESETS_29["stiff"][0]
    damping: list[float] | None = G1_LOCOMANIPULATION_PD_GAIN_PRESETS_29["stiff"][1]
    torque_limits: list[float] | None = G1_LOCOMANIPULATION_29_TORQUE_LIMITS
    position_limits: list[list[float]] | None = [_G1_POSITION_LIMITS[name] for name in joint_names]

    PD_GAIN_PRESETS: ClassVar[dict[str, tuple[list[float], list[float]]]] = (
        G1_LOCOMANIPULATION_PD_GAIN_PRESETS_29
    )

    @classmethod
    def from_preset(cls, preset: str) -> "G1Locomanipulation29ObsDoF":
        stiffness, damping = cls.PD_GAIN_PRESETS[preset]
        return cls(stiffness=stiffness, damping=damping)


class G1Locomanipulation23ActionDoF(G1Locomanipulation23ObsDoF):
    _subset: bool = True
    _subset_joint_names: list[str] | None = G1_LOCOMANIPULATION_23_ACTION_JOINT_NAMES


class G1Locomanipulation29ActionDoF(G1Locomanipulation29ObsDoF):
    _subset: bool = True
    _subset_joint_names: list[str] | None = G1_LOCOMANIPULATION_29_ACTION_JOINT_NAMES


class G1LocomanipulationPolicyCfg(PolicyCfg):
    policy_type: str = "G1LocomanipulationPolicy"
    robot: str = "g1"
    policy_name: str
    pd_gain_preset: str
    disable_autoload: bool = True
    freq: int = 50

    action_scale: float = 1.0
    action_clip: float | None = 100.0
    action_beta: float = 1.0
    history_length: int = 5
    action_scales: list[float]

    gait_period: float = 0.6
    standing_command_threshold: float = 0.1
    command_decay: float = 0.95
    height_step: float = 0.02
    waist_yaw_step: float = 0.1
    commands_map: list[list[float]] = [
        [-0.5, 0.0, 1.0],
        [0.5, 0.0, -0.5],
        [1.0, 0.0, -1.0],
        [0.5, 0.76, 0.78],
        [-1.5708, 0.0, 1.5708],
    ]

    @property
    def policy_file(self) -> str:
        return (ASSETS_DIR / f"models/g1/locomanipulation/{self.policy_name}.onnx").as_posix()


class G1Locomanipulation23PolicyCfg(G1LocomanipulationPolicyCfg):
    policy_name: str = "policy_23dof_stiff"
    pd_gain_preset: Literal["default", "stiff"] = "stiff"
    obs_dof: DoFConfig = G1Locomanipulation23ObsDoF()
    action_dof: DoFConfig = G1Locomanipulation23ActionDoF()
    num_obs: int = 360
    action_scales: list[float] = G1_LOCOMANIPULATION_ACTION_SCALE_PRESETS_23["stiff"]
    history_obs_dims: dict[str, int] = {
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "command": 3,
        "base_height_command": 1,
        "waist_yaw_command": 1,
        "phase": 2,
        "joint_pos": 23,
        "joint_vel": 23,
        "actions": 13,
    }

    @model_validator(mode="after")
    def apply_preset(self):
        self.obs_dof = G1Locomanipulation23ObsDoF.from_preset(self.pd_gain_preset)
        self.action_dof = G1Locomanipulation23ActionDoF.from_preset(self.pd_gain_preset)
        self.action_scales = G1_LOCOMANIPULATION_ACTION_SCALE_PRESETS_23[self.pd_gain_preset]
        return self


class G1Locomanipulation29PolicyCfg(G1LocomanipulationPolicyCfg):
    policy_name: str = "policy_29dof_stiff"
    pd_gain_preset: Literal["stiff"] = "stiff"
    obs_dof: DoFConfig = G1Locomanipulation29ObsDoF()
    action_dof: DoFConfig = G1Locomanipulation29ActionDoF()
    num_obs: int = 430
    action_scales: list[float] = G1_LOCOMANIPULATION_ACTION_SCALE_PRESETS_29["stiff"]
    history_obs_dims: dict[str, int] = {
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "command": 3,
        "base_height_command": 1,
        "waist_yaw_command": 1,
        "phase": 2,
        "joint_pos": 29,
        "joint_vel": 29,
        "actions": 15,
    }

    @model_validator(mode="after")
    def apply_preset(self):
        self.obs_dof = G1Locomanipulation29ObsDoF.from_preset(self.pd_gain_preset)
        self.action_dof = G1Locomanipulation29ActionDoF.from_preset(self.pd_gain_preset)
        self.action_scales = G1_LOCOMANIPULATION_ACTION_SCALE_PRESETS_29[self.pd_gain_preset]
        return self
