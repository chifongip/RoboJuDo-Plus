import numpy as np

from robojudo.controller.velocity_source import get_pressed_velocity_keys, get_selected_velocity_source
from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import SmoothPolicyCfg
from robojudo.policy.utils.velocity_command import clip_velocity, get_fresh_zmq_velocity
from robojudo.utils.util_func import command_remap, quatToEuler


@policy_registry.register
class SmoothPolicy(Policy):
    cfg_policy: SmoothPolicyCfg

    def __init__(self, cfg_policy, device):
        super().__init__(cfg_policy=cfg_policy, device=device)

        self.scales_ang_vel = self.cfg_policy.obs_scales.ang_vel
        self.scales_dof_vel = self.cfg_policy.obs_scales.dof_vel
        # self.scales_lin_vel = self.cfg_policy.obs_scales.lin_vel

        self.n_priv_latent = 4 + 1 + (self.num_dofs) * 2 + 3
        self._init_history(np.zeros(self.history_obs_size))

        self.cycle_time = self.cfg_policy.cycle_time
        self.commands_map = self.cfg_policy.commands_map

        self.reset()

    def reset(self):
        self.timestep: int = 0

    def post_step_callback(self, commands=None):
        self.timestep += 1

    def _get_phase(self):
        phase = self.timestep * self.dt / self.cycle_time
        return phase

    def _get_commands(self, ctrl_data):
        commands = np.array(self.commands_map)[:, 1].copy()  # default commands
        selected = get_selected_velocity_source(ctrl_data)
        if selected == "VelocityZmqCtrl":
            velocity = get_fresh_zmq_velocity(ctrl_data[selected])
            commands[:3] = 0.0 if velocity is None else clip_velocity(velocity, self.commands_map[:3])
        elif selected in ["JoystickCtrl", "RosJoystickCtrl", "UnitreeCtrl"]:
            axes = ctrl_data[selected]["axes"]
            lx, ly, rx = axes["LeftX"], axes["LeftY"], axes["RightX"]
            commands[0] = command_remap(ly, self.commands_map[0])
            commands[1] = command_remap(lx, self.commands_map[1])
            commands[2] = command_remap(rx, self.commands_map[2])
        elif selected == "KeyboardCtrl":
            pressed = get_pressed_velocity_keys(ctrl_data[selected])
            values = [
                1.5 * (("w" in pressed) - ("s" in pressed)),
                1.5 * (("d" in pressed) - ("a" in pressed)),
                1.5 * (("e" in pressed) - ("q" in pressed)),
            ]
            for index, value in enumerate(values):
                commands[index] = command_remap(value, self.commands_map[index])
        elif selected is None:
            commands[:3] = 0.0
        return commands

    def get_observation(self, env_data, ctrl_data):
        base_quat = env_data.base_quat
        rpy = quatToEuler(base_quat)
        rp = rpy[:2]

        commands = self._get_commands(ctrl_data)
        print(f"Commands: {commands}")
        phase = self._get_phase()

        sin_pos = [np.sin(2 * np.pi * phase)]
        cos_pos = [np.cos(2 * np.pi * phase)]

        obs_prop = np.concatenate(
            [
                sin_pos,
                cos_pos,
                commands,
                env_data.base_ang_vel * self.scales_ang_vel,
                rp,
                env_data.dof_pos - self.default_dof_pos,  # 19
                env_data.dof_vel * self.scales_dof_vel,  # 19
                self.last_action,  # 19
            ]
        )

        assert obs_prop.shape[0] == self.history_obs_size
        obs_hist = np.array(self.history_buf).flatten()

        priv_latent = np.zeros(self.n_priv_latent)

        self.history_buf.append(obs_prop)

        obs = np.concatenate((obs_prop, priv_latent, obs_hist))

        extras = {
            "phase": phase,
            "commands": commands,
        }
        return obs, extras
