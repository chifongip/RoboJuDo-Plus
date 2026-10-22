# X2 bounded torque-integral simulation

The retained method is fixed-gain bounded torque-integral correction. It uses
joint targets, measured positions and velocities, and elapsed control time.
It has no force measurement, payload estimate, placement event, leakage,
motion-dependent gain or automatic release rule.

This is simulation-only. AimDK, the C++ wrapper and real-robot configuration are
unchanged. The current wrapper sends zero additive effort. The settings below
are experimental reference values; they are not selected deployment gains or a
proof of stability for the proprietary low-level controller.

The selected item is the conventional controller structure, not new deployment
gains. It corresponds to the earlier fixed-gain experiment; the historical
15.6 mm result belonged to a removed experimental controller. Cleanup does not
claim improved unloading performance.

## Retained implementation

- `robojudo/utils/bounded_integral.py`: the single `BoundedIntegralCorrection` controller.
- `scripts/x2_integral_sim.py`: shared simulation, hand-center force arrows and telemetry.
- `scripts/evaluate_x2_transitions.py`: matched PD/integral pickup/place benchmark.
- `scripts/plot_x2_transitions.py`: force, hand displacement, balance and all-joint plots.
- `tests/test_x2_integral.py`, `tests/test_x2_integral_sim.py` and
  `tests/test_x2_transitions.py`: controller, physics and trajectory regressions.

The rejected leaky/asymmetric/motion-gated controllers, privileged compensation
and held-effort diagnostics, parameter-search tools, position-response
identification experiment, duplicate benchmarks and their dedicated reports
were removed during cleanup. Saved historical logs remain available; their
controller hashes identify previous implementations and their results must not
be attributed to the retained controller.

## Control law

For joint error `e = target - position`, the unconstrained controller is

$$
\begin{aligned}
\tau_f \dot{e}_f &= e - e_f, \\
\dot{z} &= K_i e_f, \\
\tau &= K_p e - K_d \dot{q} + z.
\end{aligned}
$$

The implementation applies a continuous soft integration deadband to filtered
error. It bounds the stored correction itself and limits each update, avoiding
an additional hidden integral state behind clipping. Conditional anti-windup
blocks updates that would push the modeled PD-plus-integral torque farther into
actuator saturation. Invalid feedback resets the controller and ends the trial.

Default simulation settings are:

| Parameter | Value |
| --- | --- |
| Integral gain | `Ki = 0.5 * Kp/s` |
| Error filter time constant | 0.25 s |
| Integration deadband | 0.2 degrees |
| Per-joint additive torque bound | `min(8 Nm, 0.8 * simulated actuator limit)` |
| Additive torque slew | 2 Nm/s |

The deadband is an integration noise band, not a joint-accuracy requirement.
The deployment reference-speed limiter acts on position targets separately;
integration continues during reference motion. Correction is updated at the
50 Hz high-level cadence and held through the physics substeps. Only the 14 arm
joints receive additive effort; the locomotion policy continues controlling the
legs and waist.

Conventional integral feedback and actuator anti-windup have an established
basis; see Åström and Murray's [PID chapter](https://www.cds.caltech.edu/~murray/FBS/PID_Control.html)
and Åström and Rundqwist's [Integrator Windup and How to Avoid It](https://doi.org/10.23919/ACC.1989.4790464).
These references do not establish stability of the sampled, filtered and bounded
X2 loop. The remaining design task is to characterize the additive-torque
response, then choose gains/filtering with sampling, delay and coupling in view.

## Run and view

From the repository root:

The benchmark uses the checkpoint at
`assets/models/x2/locomanipulation/policy_fine_tune.onnx`. It selects that policy
locally, without changing the global `x2_locomimic` policy setting. The checkpoint
is included in the preceding model/config commit.

```bash
conda activate robojudo-plus
PYTHONPATH="" python scripts/evaluate_x2_transitions.py \
  --repetitions 1 --jobs 3 --output logs/x2_transitions/my-evaluation
```

This runs six 105-second trials: PD and integral control for unloaded,
constant-load and sway profiles. Omitting `--repetitions` runs three repetitions.
No ROS node, network controller or hardware command is created. Output
directories must be new. Fixed coefficients can be changed explicitly with
`--integral-gain-scale`, `--error-filter-tau` and `--error-deadband-deg`.

For a viewer demonstration:

```bash
PYTHONPATH="" python scripts/evaluate_x2_transitions.py \
  --profile constant --repetitions 1 --viewer \
  --output logs/x2_transitions/my-viewer
```

The viewer requires `--jobs 1`. A loaded-only demonstration has no unloaded
reference, so load-induced displacement comparisons are marked unavailable.

The benchmark moves from the native default posture to grasp and the specified
carrying posture, applies 20 N downward per hand at the modeled hand center
(`*_wrist_roll_link` COM), walks forward with a 0.5 m/s command, stops, lowers and
unloads, then returns to default. Pickup and placement are gradual. Sway adds
0.8 Hz fore-aft and vertical components of up to 5 N while loaded. Initial
preparation uses support, which is removed before policy warmup and measurement.
The actual deployment upper-body smoothing and reference-speed limiter are used
in process without sockets. The locomotion model remains `policy_fine_tune`.

## Results

`summary.json` records controller/model hashes, coefficients, posture and force
geometry, actual walking speed, completion status, bounds and saturation.
Each phase includes absolute mean and 95th-percentile errors for every arm
joint. Traces record issued/desired posture, actual position, torque correction,
external force and body motion at 50 Hz. Obsolete freeze and motion-weight
telemetry is no longer logged.

Comparisons use each method's matching unloaded trajectory to distinguish
planned motion from load-induced hand displacement. The separate post-unloading
peak covers seconds 79–82. Carrying and body-motion measurements accompany that
peak. Effort residuals and baseline slopes are diagnostics, not recovery
pass/fail requirements. Reports have no angular accuracy or unloading-clearance
cutoff. Falls, invalid state and numerical failures still terminate trials.

Plot with any environment containing NumPy and Matplotlib:

```bash
python scripts/plot_x2_transitions.py logs/x2_transitions/my-evaluation
```

Each profile gets `comparison.png` and `posture_tracking.png`; the latter shows
all 14 joints. Force arrows originate at the hand center and scale at 1 cm/N.

## Verification

```bash
PYTHONPATH="" python -m unittest discover -s tests -p 'test_x2_integral*.py'
PYTHONPATH="" python -m unittest discover -s tests -p 'test_x2_transitions.py'
```
