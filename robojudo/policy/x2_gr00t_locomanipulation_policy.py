import numpy as np

from robojudo.config.x2.policy.x2_gr00t_locomanipulation_policy_cfg import (
    X2Gr00tLocomanipulationPolicyCfg,
)
from robojudo.policy import policy_registry
from robojudo.policy.x2_locomanipulation_policy import X2LocomanipulationPolicy


@policy_registry.register
class X2Gr00tLocomanipulationPolicy(X2LocomanipulationPolicy):
    """Use GR00T velocity and height outputs instead of manual axes."""

    cfg_policy: X2Gr00tLocomanipulationPolicyCfg

    def _get_commands(self, ctrl_data) -> np.ndarray:
        commands = self.cmd.copy()
        stream = ctrl_data.get("Gr00tZmqCtrl", {})
        command = stream.get("locomotion_command")
        external_active = bool(stream.get("takeover_enabled", False) and stream.get("fresh", False))

        if external_active:
            command = np.asarray(command, dtype=np.float32)
            if command.shape != (4,) or not np.isfinite(command).all():
                raise ValueError("active GR00T locomotion_command must be a finite vector with shape (4,)")
            for index in range(4):
                commands[index] = self._clip_command(command[index], self.commands_map[index])
            self.current_vel_cmd[:] = commands[:3]
            self._target_height = float(commands[3])
        else:
            self.current_vel_cmd[:] = 0.0
            commands[:3] = 0.0
            commands[3] = self.cmd[3]

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
