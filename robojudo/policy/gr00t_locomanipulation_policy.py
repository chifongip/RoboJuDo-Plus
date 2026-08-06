import numpy as np


class Gr00tLocomanipulationPolicyMixin:
    """Route Locomanipulation commands between manual control and GR00T policy.

    This mixin must precede a robot-specific Locomanipulation policy in the
    MRO. It only replaces the five-element high-level command source; the base
    policy still builds observations and runs lower-body ONNX inference.

    Command modes:
    - Takeover disabled: delegate to the base policy's joystick/keyboard path.
    - Takeover enabled and stream fresh: use GR00T ``[vx, vy, yaw, height]``.
    - Takeover enabled and stream stale: zero velocity and hold last height.

    Upper-body targets are handled by the pipeline mixin, not by this class.
    """

    def reset(self):
        super().reset()
        self._gr00t_takeover_was_enabled = False

    def _get_commands(self, ctrl_data) -> np.ndarray:
        stream = ctrl_data.get("Gr00tZmqCtrl", {})
        takeover_enabled = bool(stream.get("takeover_enabled", False))
        if not takeover_enabled:
            # Clear the last VLA velocity once before restoring manual input.
            if getattr(self, "_gr00t_takeover_was_enabled", False):
                self.current_vel_cmd[:] = 0.0
            self._gr00t_takeover_was_enabled = False
            # The next class in the MRO owns joystick/keyboard command parsing.
            return super()._get_commands(ctrl_data)

        self._gr00t_takeover_was_enabled = True
        commands = self.cmd.copy()
        command = stream.get("locomotion_command")
        external_active = bool(stream.get("fresh", False))

        if external_active:
            command = np.asarray(command, dtype=np.float32)
            if command.shape != (4,) or not np.isfinite(command).all():
                raise ValueError("active GR00T locomotion_command must be a finite vector with shape (4,)")
            for index in range(4):
                commands[index] = self._clip_command(command[index], self.commands_map[index])
            self.current_vel_cmd[:] = commands[:3]
            self._target_height = float(commands[3])
        else:
            # Do not fall back to joystick on a timeout while takeover remains enabled.
            self.current_vel_cmd[:] = 0.0
            commands[:3] = 0.0
            commands[3] = self.cmd[3]

        # GR00T does not output waist yaw; keep the trained default command.
        commands[4] = self._command_defaults[4]
        self._target_waist_yaw = float(commands[4])
        self.cmd[:] = commands
        print(
            f"\rvel=({commands[0]:+.1f}, {commands[1]:+.1f}, "
            f"{commands[2]:+.1f}) h={commands[3]:.3f} wy={commands[4]:+.2f}",
            end="",
            flush=True,
        )
        return commands
