from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LOCOMOTION_COMMAND_NAMES = ["base.velocity.x", "base.velocity.y", "base.yaw_rate", "base.height"]


@dataclass(frozen=True)
class ControlSample:
    episode_id: int
    task: str
    robot_type: str
    source_timestamp_ns: int
    receive_timestamp_ns: int
    joint_names: tuple[str, ...]
    joint_positions: np.ndarray
    joint_position_commands: np.ndarray
    velocity_height_command: np.ndarray
    # Offline DAgger label is optional so existing teleop samples remain valid.
    dagger: dict | None = None

    @classmethod
    def from_message(cls, message: dict, receive_timestamp_ns: int) -> ControlSample:
        joint_names = tuple(message["joint_names"])
        joint_positions = np.asarray(message["joint_positions"], dtype=np.float32)
        joint_commands = np.asarray(message["joint_position_commands"], dtype=np.float32)
        locomotion_command = np.asarray(message["velocity_height_command"], dtype=np.float32)
        if len(set(joint_names)) != len(joint_names):
            raise ValueError("joint_names must be unique")
        expected = (len(joint_names),)
        if joint_positions.shape != expected or joint_commands.shape != expected:
            raise ValueError(f"joint position vectors must have shape {expected}")
        if locomotion_command.shape != (4,):
            raise ValueError("velocity_height_command must have shape (4,)")
        for value in (joint_positions, joint_commands, locomotion_command):
            if not np.isfinite(value).all():
                raise ValueError("control sample contains non-finite values")
        # Offline DAgger validates source/session labels before raw persistence;
        # arbitrary truthy strings must never be accepted as expert labels.
        dagger = message.get("dagger")
        if dagger is not None:
            if not isinstance(dagger, dict):
                raise ValueError("dagger metadata must be an object")
            if not isinstance(dagger.get("expert_intervention"), bool):
                raise ValueError("dagger expert_intervention must be boolean")
            if not isinstance(dagger.get("expert_applied"), bool):
                raise ValueError("dagger expert_applied must be boolean")
            session = dagger.get("intervention_session")
            if isinstance(session, bool) or not isinstance(session, int) or session < 0:
                raise ValueError("dagger intervention_session must be non-negative")
            source = dagger.get("action_source")
            if source not in {"policy", "expert"}:
                raise ValueError("dagger action_source must be policy or expert")
            if dagger["expert_applied"] != (source == "expert"):
                raise ValueError("dagger expert_applied and action_source disagree")
            dagger = dict(dagger)
        return cls(
            episode_id=int(message["episode_id"]),
            task=str(message["task"]),
            robot_type=str(message["robot_type"]),
            source_timestamp_ns=int(message["timestamp_ns"]),
            receive_timestamp_ns=receive_timestamp_ns,
            joint_names=joint_names,
            joint_positions=joint_positions,
            joint_position_commands=joint_commands,
            velocity_height_command=locomotion_command,
            dagger=dagger,
        )

    def timestamp_ns(self, clock: str) -> int:
        return self.source_timestamp_ns if clock == "source" else self.receive_timestamp_ns

    @property
    def state_names(self) -> list[str]:
        return [f"{name}.pos" for name in self.joint_names]

    @property
    def action_names(self) -> list[str]:
        return [*[f"{name}.pos" for name in self.joint_names], *LOCOMOTION_COMMAND_NAMES]

    @property
    def action(self) -> np.ndarray:
        return np.concatenate((self.joint_position_commands, self.velocity_height_command)).astype(np.float32)
