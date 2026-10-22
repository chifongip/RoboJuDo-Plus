# X2 arm torque compensation in simulation

The selected simulation method is a bounded model-based disturbance observer.
The original fixed-gain bounded integral controller remains the comparison
baseline. Both add correction torque to the existing PD controller and use only
joint targets, measured positions/velocities, elapsed control time and their own
internal state. No force sensor, payload truth, placement signal or task phase
enters either controller.

The user accepted the observer's carrying-accuracy tradeoff and chose to keep it
after reviewing the sway/balance and saturation measurements. This selection is
for simulation. SDK code, real-robot configuration and hardware commands are
unchanged.

## Retained files

| File | Purpose |
| --- | --- |
| `robojudo/utils/disturbance_observer.py` | Bounded model-based observer |
| `robojudo/utils/bounded_integral.py` | Original integral baseline |
| `scripts/x2_observer_reference.py` | Selected fixed coefficients and joint order |
| `scripts/x2_integral_sim.py` | Shared simulation, hand-center loads and telemetry |
| `scripts/evaluate_x2_transitions.py` | PD/integral/observer benchmark |
| `scripts/plot_x2_transitions.py` | Comparison and all-14-joint plots |

The four matching test modules cover the controllers, simulation, selected
coefficients and transition/recovery measurements. Identification, parameter
search, rejected candidate runners and their dedicated tests have been removed.
Their execution sources and historical results remain in the experiment logs.

## Controller assumptions

The integral baseline filters `target - position`, applies a continuous soft
integration deadband, and accumulates bounded correction torque. Its settings
remain `Ki = 0.5*Kp/s`, 0.25 s error filtering and a 0.2-degree integration deadband.
The deadband is a noise band, not an accuracy requirement. Conditional anti-windup
blocks updates that push modeled total actuator torque farther into saturation.
See Åström and Murray's [PID chapter](https://www.cds.caltech.edu/~murray/FBS/PID_Control.html)
and Åström and Rundqwist's [Integrator Windup and How to Avoid It](https://doi.org/10.23919/ACC.1989.4790464).

The observer predicts the next joint velocity from a nominal local dynamics
model, including configured PD, previously issued correction and dissipative
friction. A velocity residual gives a disturbance-torque estimate:

```
innovation = measured_velocity - predicted_velocity
estimate += (1 - exp(-bandwidth*dt)) * innovation / nominal_input_gain
correction = bounded_and_slew_limited(-estimate)
```

The predictor uses four implicit substeps per control interval. The selected
bandwidth is **8 rad/s**. All 14 nominal inertia/viscosity coefficients are fixed
in `x2_observer_reference.py`; there is no task-dependent scheduling. Nominal
friction magnitude is 0.3 Nm. The observer retains nominal friction instead of
intentionally canceling its damping. Its estimate is bounded, and conditional
anti-windup freezes outward effort/estimate updates during modeled saturation.

With an exact locally affine predictor and constant disturbance, the unbounded
estimation error contracts by `exp(-bandwidth*dt)`. This does not establish
stability of the nonlinear walking robot. The model was fitted to additive-torque
responses and validated on held-out standing pulses. Walking coupling and wrist
sliding dynamics are less certain; fitted nominal parameters are not verified
physical inertias. Earlier position-command identification did not establish the
torque-loop model. The observer follows the nominal-model residual approach in
Chen et al.'s [disturbance-observer overview](https://doi.org/10.1109/TIE.2015.2478397).

Both controllers use the same correction bound, `min(8 Nm, 0.8*configured limit)`,
and **2 Nm/s** slew limit. Correction is updated at 50 Hz and held through four
5 ms physics steps. Only the 14 arm joints receive additive effort; the existing
locomotion policy controls the legs and waist.

## Run and plot

The benchmark always uses `policy_fine_tune.onnx`, the specified carrying posture,
a hand-body center-of-mass force application point, 20 N downward per hand under
constant load, and a 0.5 m/s walking command. Sway adds the original 0.8 Hz force
variation. Every trial lasts 105 s and includes reach, pickup/lift, carrying,
walking, lowering/unloading, recovery and return to the default posture.

```bash
conda activate robojudo-plus
PYTHONPATH="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/evaluate_x2_transitions.py \
  --repetitions 3 --warmup-seconds 4 5 6 --jobs 3 \
  --output logs/x2_transitions/my-observer-comparison
python scripts/plot_x2_transitions.py logs/x2_transitions/my-observer-comparison
```

The default comparison runs all three methods on unloaded, constant-load and
sway profiles: 27 trials for three repetitions. `--repetitions 1` runs nine trials
with the default 5 s warmup. Output directories must be new. Repeated
`--condition pd_only|integral|observer` and `--profile unloaded|constant|sway`
arguments restrict the run. Loaded displacement comparisons require matching
unloaded trials. For example, `--condition observer --repetitions 1 --viewer
--jobs 1` displays the selected method without creating a network controller.
Plotting requires Matplotlib in the Python environment used for that command.

Outputs include raw CSV traces, per-trial JSON, `summary.json`, `summary.csv`,
`joint_errors.csv`, and comparison/all-joint PNGs after plotting. All 14 joints'
RMS, mean absolute, p95 and maximum absolute errors are reported by phase, with
per-joint saturation and correction activity. Body oscillation, peak tilt, height,
achieved walking speed, slew and bound excess remain in the summaries.
`comparisons` contains the integral-versus-PD measurements;
`observer_comparisons` contains the observer-versus-PD measurements. Method names
remain explicit so historical results cannot be attributed to another controller.

## Validated results and accepted tradeoffs

The selected observer completed nine 105-second trials across 4/5/6 s preparation
warmups, with no falls, numerical failures or policy mode exits. The repeated 5 s
traces match the original selected candidate byte for byte. Across warmups:

| Method / load | Walking RMS (deg) | Unload peak (mm) | Recovery to PD envelope (s) | Walk roll / pitch std (deg) | Saturated physics (%) |
| --- | ---: | ---: | ---: | --- | ---: |
| Integral / constant | 0.2476–0.2492 | 29.147–29.160 | 5.90 | 1.1132–1.1140 / 0.3013–0.3020 | 0–0.0571 |
| Observer / constant | 0.3062–0.3081 | 0.360–0.361 | 0.02* | 1.1089–1.1132 / 0.2972–0.2978 | 0–0.0571 |
| Integral / sway | 0.8821–0.8844 | 28.543–29.424 | 6.24–6.92 | 1.1489–1.1876 / 1.1392–1.1671 | 5.4571–7.5619 |
| Observer / sway | 0.8756–0.8923 | 0.764–1.320 | 0.02* | 1.1475–1.1679 / 1.1207–1.1240 | 5.0905–5.5286 |

Walking RMS uses 55–60 s, roll/pitch standard deviation is detrended over
45–60 s, and unloading peak is the worse hand's 3D torso-frame displacement
relative to the same method's unloaded trajectory during 79–82 s. Saturation is
the fraction of physics steps with any actuator clipped over the complete trial.

Recovery means entry into the matching PD trial's measured 79–89 s displacement
envelope, remaining inside for the rest of the hold with at least 1 s observed.
`*` The observer is already inside that envelope at the first 20 ms sample;
this does **not** establish 20 ms physical settling or zero displacement. Across
all validation trials, no hand crosses below its matching unloaded vertical
reference during that hold. This is observed behavior, not a no-overshoot guarantee.

The user explicitly accepted the following tradeoffs; the original strict
baseline-range diagnostics were not retroactively marked as passing:

- Constant-load walking RMS increases from about 0.248 to 0.308 degrees.
  Aggregate RMS does not bound every joint/instant below one degree. Constant-load
  walking maxima are 0.7558–0.7563 degrees, versus 0.5204–0.5249 for integral.
  Sway walking maxima are 3.10–3.17 degrees, versus 3.00–3.04; one sway acceleration
  transient reaches 6.37 degrees, versus 4.84 in its paired integral trial.
- Whole-trial sway tilt peaks span 15.37–20.31 degrees for the observer and
  14.74–28.85 for integral. The worst observed peak improves, but the paired
  5 s warmup peak increases from 14.74 to 20.31 degrees.
- In the 6 s warmup sway carry-standing phase, pitch standard deviation rises
  from 1.633 to 1.907 degrees, peak tilt from 14.873 to 15.971 degrees, and minimum
  height decreases by 5.72 mm. Left wrist-pitch saturation occupies 0.75% of that
  phase's physics steps, versus zero for integral. Overall sway saturation is
  lower in each matched trial; this does not erase the local increase.
- Walking correction ripple increases. Constant-load left-elbow detrended
  correction standard deviation rises from 0.000127 to 0.14715 Nm; under sway,
  from 0.08476 to 0.35889 Nm. All correction bounds and slew limits still hold.

Configured arm torque limits per side are `[36, 36, 24, 24, 24, 2.2, 2.2]` Nm.
These are simulation policy limits, not verified physical peak/continuous or
firmware ratings. The XML wrist-pitch/roll ranges are ±4.8 Nm; the configured
±2.2 Nm limits are stricter. PD plus correction is clipped every physics step.
No hardware deployment qualification or global walking-stability proof is claimed.

## Evidence and verification

The full measured comparison, all-joint errors, plots, source snapshots and user
selection are retained at:

- `logs/x2_transitions/torque-id-baselines-20261022`
- `logs/x2_transitions/observer-validation-20261022/experiment_report.md`
- `logs/x2_transitions/observer-validation-20261022/decision.json`

Identification and rejected gain/controller experiments are historical evidence,
not the retained implementation. Removed leaky/transition-aware methods and their
results must not be attributed to the current integral or observer controller.
The cleanup source archive is `logs/x2_transitions/observer-cleanup-20261022`.

Run the retained hardware-free checks with:

```bash
python -m unittest discover -s tests -p 'test_x2_integral*.py'
python -m unittest discover -s tests -p test_x2_disturbance_observer.py
python -m unittest discover -s tests -p test_x2_transitions.py
```
