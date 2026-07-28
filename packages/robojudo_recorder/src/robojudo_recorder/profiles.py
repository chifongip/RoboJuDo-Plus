from dataclasses import dataclass

from .protocol import LOCOMOTION_COMMAND_NAMES, ControlSample


@dataclass(frozen=True)
class NamedJointProfile:
    """Runtime-negotiated robot schema shared by X2, G1 variants, and hands."""

    robot_type: str
    joint_names: tuple[str, ...]

    @classmethod
    def from_sample(cls, sample: ControlSample) -> "NamedJointProfile":
        return cls(robot_type=sample.robot_type, joint_names=sample.joint_names)

    @property
    def state_names(self) -> list[str]:
        return [f"{name}.pos" for name in self.joint_names]

    @property
    def action_names(self) -> list[str]:
        return [*[f"{name}.pos" for name in self.joint_names], *LOCOMOTION_COMMAND_NAMES]

    def validate(self, sample: ControlSample):
        if sample.robot_type != self.robot_type or sample.joint_names != self.joint_names:
            raise ValueError(
                "recording profile changed: "
                f"expected {self.robot_type}/{self.joint_names}, got {sample.robot_type}/{sample.joint_names}"
            )
