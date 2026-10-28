# X2 observer in interactive simulation

Both simulation and hardware use the [shared observer config and runtime](joint_observer.md).
This preset selects the same `nonlinear` algorithm and parameters as
`x2_locomimic_real_observer`, with compensation ACTIVE in simulation and OFF by
default in the real preset.

Run from the repository root in the normal RoboJuDo environment, with the same
X2 assets and policy checkpoints required by `x2_locomimic`:

```bash
python scripts/run_pipeline.py -c x2_locomimic_observer
```

This preset retains the keyboard, joystick, and ZMQ controls of `x2_locomimic`.
Press `i` to enter joint-default preparation, wait for interpolation to finish,
then press `j` for RL control. Press `]` to select locomotion if another policy
is selected. Joystick equivalents are Y, X, and Back. Use the existing velocity
and upper-body controls as usual. The elastic-band controls remain `7` (lower),
`8` (lift), and `9` (toggle).

Compensation is ACTIVE only during locomotion in `RL_DEFAULT`, after policy
interpolation finishes. Mimic, recovery, passive, damping, and joint-default
control disable it. Mode/policy changes and simulation resets clear its history;
the first observer sample after reset issues zero correction.

The observer updates at 50 Hz using simulation time and the final arm targets.
Its arm correction is held across four 5 ms physics substeps. It uses the shared
simulation reference model (8 rad/s bandwidth), per-joint correction limits of
`0.8 × actuator limit`, 2 Nm/s growth and 16 Nm/s release toward zero.
Opposite-sign correction grows on a later update at 2 Nm/s. Only the 14 arm joints
receive compensation; total PD plus correction still passes through MuJoCo's
torque limits. These settings are simulation parameters.

Set `observer.correction_limit_fraction` (default `0.8`) to choose the fraction of
each actuator's torque limit. `observer.correction_limit_nm=0.0` disables the absolute
cap; a positive value such as `8.0` caps each correction at that many Nm. Rebuild the
observer after changing these settings. The preset's feedforward command bounds
use the actuator limits, so the observer's fraction and cap determine its correction bounds.

INFO logs use the same compact single-line format as the real environment:
mode, disturbance estimate, issued correction, and bound/limiter counts. They
appear once per simulated second while eligible. Programmatic diagnostics
are available through `pipeline.env.get_observer_status()`; `issued_nm` is
admitted feedforward effort, not measured actuator torque after total clipping.

To evaluate SHADOW (estimation with zero applied correction) or OFF, change
`observer=JointObserverCfg(algorithm="nonlinear", mode="ACTIVE", ...)` in the `x2_locomimic_observer` preset in
`robojudo/config/x2/x2_locomimic_observer_cfg.py`, then restart the pipeline.
There is no runtime mode-switch control. The original `x2_locomimic` preset has
no observer, and the real-robot observer preset remains OFF by default.

Headless verification:

```bash
python -m unittest discover -s tests -p 'test_x2_observer_mujoco_env.py'
```

## Other robots

`ObserverMujocoEnv` is robot-independent. Use `ObserverMujocoEnvCfg` from
`robojudo.environment.env_cfgs`, or combine it with a robot's MuJoCo config as
the X2 preset does. Import `JointObserverCfg` from `robojudo.config.observer_cfg`.
Supply `observer=JointObserverCfg(...)` with an explicit `algorithm` and the selected
`joint_names`, nominal `inertia`, optional friction parameters, `bandwidth_rad_s`,
and `slew_nm_s`. Correction bounds default to `correction_limit_fraction=0.8` with
`correction_limit_nm=0.0` (no absolute cap). A positive cap applies to every selected
joint. Scalar model parameters apply to every selected joint; vectors follow the
configured joint order.
Optional `actuator_limits_nm` overrides the selected environment torque limits.
Configure `feedforward_effort_limits` for those joints as well; omitted joints
have zero admitted correction.

The generic environment enables a configured ACTIVE/SHADOW observer on ordinary
position-control steps. Pipelines can call `set_observer_eligible(False)` to
disable compensation and clear history, and `reset_observer()` to clear history
without changing eligibility. Passive and damping commands always clear history
and apply zero feedforward. X2-specific locomotion gating is implemented by its
pipeline, outside the generic environment. The real environment uses the same
configuration type and observer methods; its transport and safety checks remain
in the AimDK environment integration. The historical linearized algorithm remains
available through `algorithm="linearized"` with symmetric growth/release slew.
