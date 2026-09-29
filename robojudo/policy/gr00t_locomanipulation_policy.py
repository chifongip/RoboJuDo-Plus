import numpy as np


class Gr00tLocomanipulationPolicyMixin:
    """Route Locomanipulation commands between manual control and GR00T policy.

    This mixin must precede a robot-specific Locomanipulation policy in the
    MRO. It only replaces the five-element high-level command source; the base
    policy still builds observations and runs lower-body ONNX inference.

    Command modes:
    - Takeover disabled: delegate to the base policy's joystick/keyboard path.
    - Takeover enabled and stream fresh: use GR00T ``[vx, vy, yaw, height]``.
    - Offline DAgger expert applied: use joystick ``[vx, vy, yaw, height]``
      together with the expert arm/hand targets selected by the controller.
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
        if bool(stream.get("expert_applied", False)):
            # Offline DAgger treats the human correction as one composite
            # action: dex-teleop supplies both arms and both hands, while the
            # local joystick supplies all four locomotion command dimensions.
            # Calling the base path preserves its deadzone, velocity decay and
            # D-pad height smoothing instead of duplicating that mapping here.
            commands = np.asarray(super()._get_commands(ctrl_data), dtype=np.float32).copy()
            if commands.shape != (5,) or not np.isfinite(commands).all():
                raise ValueError("expert locomotion path must produce a finite five-element command")
            # Waist remains at the trained default and is not part of DAgger.
            commands[4] = self._command_defaults[4]
            self._target_waist_yaw = float(commands[4])
            self.cmd[:] = commands
            stream["locomotion_command"] = commands[:4].copy()
            stream["locomotion_source"] = "expert"
            return commands

        commands = self.cmd.copy()
        command = stream.get("locomotion_command")
        # Outside an applied expert correction, locomotion remains tied to the
        # GR00T policy stream and stops when that stream becomes stale.
        external_active = bool(stream.get("policy_fresh", stream.get("fresh", False)))

        if external_active:
            command = np.asarray(command, dtype=np.float32)
            if command.shape != (4,) or not np.isfinite(command).all():
                raise ValueError("active GR00T locomotion_command must be a finite vector with shape (4,)")
            for index in range(4):
                commands[index] = self._clip_command(command[index], self.commands_map[index])
            self.current_vel_cmd[:] = commands[:3]
            self._target_height = float(commands[3])
            stream["locomotion_source"] = "policy"
        else:
            # Do not fall back to joystick on a timeout while takeover remains enabled.
            self.current_vel_cmd[:] = 0.0
            commands[:3] = 0.0
            commands[3] = self.cmd[3]
            stream["locomotion_source"] = "hold"

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
