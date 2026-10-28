# X2 arm torque compensation in simulation

The selected simulation method is a bounded model-based disturbance observer.
The original fixed-gain bounded integral controller remains the comparison
baseline. Both add correction torque to the existing PD controller and use only
joint targets, measured positions/velocities, elapsed control time and their own
internal state. No force sensor, payload truth, placement signal or task phase
enters either controller.

The user accepted the observer's carrying-accuracy tradeoff and chose to keep it
after reviewing the sway/balance and saturation measurements. This selection is
for simulation. SDK code, real-robot configuration and hardware commands were
unchanged during these studies. The optional deployment integration described
below does not extend their hardware qualification.

## Retained files

| File | Purpose |
| --- | --- |
| `robojudo/utils/observers/linearized.py` | Bounded model-based observer |
| `robojudo/utils/bounded_integral.py` | Original integral baseline |
| `robojudo/config/x2/observer_parameters.py` | Selected fixed coefficients and joint order |
| `scripts/x2_integral_sim.py` | Shared simulation, hand-center loads and telemetry |
| `scripts/evaluate_x2_transitions.py` | PD/integral/observer benchmark |
| `scripts/plot_x2_transitions.py` | Comparison and all-14-joint plots |

Matching tests cover the controllers, feedback channel, payload mechanics, simulation
and recovery measurements. The retained robustness tools are:

| File | Purpose |
| --- | --- |
| `scripts/evaluate_x2_robustness.py` | Original 594-trial stress matrix and shared execution/provenance helpers |
| `scripts/evaluate_x2_compliant_robustness.py` | Full 792-trial compliant-payload matrix |
| `scripts/x2_feedback_perturbations.py` | Matched feedback imperfections and plant uncertainty cases |
| `scripts/x2_robustness_sim.py` | Isolated feedback injection and physics-cadence telemetry |
| `scripts/x2_robustness_analysis.py` | All-joint, balance, saturation and recovery measurements |
| `scripts/x2_robustness_report.py` | Original stress-matrix reporting |
| `scripts/x2_compliant_payload.py` | Shared three-axis compliant-payload mechanics |

Identification, parameter-search, rejected-controller and one-off study runners
and their dedicated tests are archived under the historical experiment logs.
The observer has no retained experimental damping or phase-dependent intervention.

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
in `robojudo/config/x2/observer_parameters.py`; there is no task-dependent scheduling. Nominal
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

The reported comparisons used the same correction bound, `min(8 Nm, 0.8*configured limit)`,
for both controllers. The observer now defaults to `0.8*configured limit`; the integral
baseline retains the 8 Nm cap. Both use a **2 Nm/s** slew limit. Correction is updated
at 50 Hz and held through four 5 ms physics steps. Only the 14 arm joints receive
additive effort; the existing locomotion policy controls the legs and waist.

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

The default comparison runs PD-only, integral, linearized (`observer`) and
`nonlinear_observer` methods on unloaded, constant-load and compliant-payload
profiles: 36 trials for three repetitions. `--repetitions 1` runs twelve trials
with the default 5 s warmup. Output directories must be new. Repeated
`--condition pd_only|integral|observer|nonlinear_observer` and
`--profile unloaded|constant|compliant_payload|sway`
arguments restrict the run. Loaded displacement comparisons require matching
unloaded trials. For example, `--condition observer --repetitions 1 --viewer
--jobs 1` displays the selected method without creating a network controller.
Plotting requires Matplotlib in the Python environment used for that command.

The nonlinear condition uses the same frozen 8 rad/s model and correction bounds,
with 2 Nm/s growth and 16 Nm/s release. All methods use the synchronous correction
core in this benchmark, under identical final-target filtering and total-torque
clipping; native command transport is not emulated. Generate a compact comparison
with `python scripts/report_x2_transition_methods.py <output-directory>`.
The [four-method transition comparison](x2_transition_methods_comparison.md)
records the 36-trial rerun with compliant load replacing prescribed sway.

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

## Finite robustness study

The simulation-only robustness runner preserves the selected observer at 8 rad/s:

```bash
PYTHONPATH="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/evaluate_x2_robustness.py --jobs 8 \
  --output logs/x2_transitions/observer-robustness-20261022
python scripts/plot_x2_transitions.py \
  logs/x2_transitions/observer-robustness-20261022 --physics --case N0 --case X2 --case P2
```

There are 22 cases and 594 primary trials, with all three controllers, all three
load profiles and paired seeds/warmups `(1,4 s), (2,5 s), (3,6 s)`. The runner
requires exact nominal trace reproduction before perturbed trials and separately
checks serial/parallel equivalence. Each simulation has isolated state and random
streams. BLAS/OpenMP/ONNX are limited to one thread per process. XML compilation
is limited to two simultaneous initializations, and its asset cache is released
after construction to limit memory without changing the plant. On this host,
16-worker startup attempts terminated abruptly under resource pressure; the
completed study used eight workers, with approximately 2.1–2.4 GiB live memory
per worker. Eight is now the runner default; `--jobs` remains configurable.

`x2_feedback_perturbations.py` defines the fixed matrix: position noise at
0.01/0.10 degrees standard deviation; velocity noise at 0.005/0.05 rad/s;
50 ms colored correlated noise at 0.03 degrees/0.015 rad/s; fixed feedback delays
of 5/20/40 ms; jitter over 10–30 ms; 40/80 ms dropout bursts; plant inertia
multipliers 0.7/1.3, armature 0.5/1.5 and friction loss 0.5/1.5; two combined
cases and two shared arm-PD/correction cases. The resolved matrix is archived in
`manifest.json`, and `--case-file` accepts an explicit JSON list of configurations
including an unchanged N0. These ranges are exploratory assumptions: archived
identification supports local-model uncertainty, but supplies no physical
confidence interval, measured sensor-noise spectrum or transport distribution.

Acquisition runs at 200 Hz. Noise is added before packet delivery; white noise
is independent, while colored noise has a 50 ms AR(1) correlation time,
within-arm correlation 0.5 and position/velocity correlation 0.5. Arms are
independent. The delivery queue discards out-of-order old packets. Dropped
acquisitions are discarded; the last delivered noisy packet is held, without
extrapolation or clean-state substitution. Correction continues at 50 Hz using
elapsed control time. No freshness flags, phases or load truth enter correction.
Except for P1/P2, perturbations affect only additive-correction feedback; the
existing PD servo and locomotion sensing remain unchanged. P1/P2 also perturb
arm PD at its existing physics cadence. No actuator delay is added.

The reference nominal observer model remains fixed while uncertainty changes
only the actual MuJoCo plant. Link masses, centers of mass and geometry remain
fixed. Fitted observer friction viscosity is not treated as a plant parameter.
For relevant background see [Chen et al.](https://doi.org/10.1109/TIE.2015.2478397)
and [Hespanha et al.](https://web.ece.ucsb.edu/~hespanha/published/ncs_v15p.pdf).

Physics telemetry is logged at 5 ms into compressed, chunked NPZ files, with
clean evaluation state separate from delivered controller measurements.
The legacy CSV retains 20 ms samples and PD-envelope recovery semantics.
The new recovery threshold is the matched unloaded hand's p99 Euclidean
excursion around its median XYZ position over 20–25 s, without detrending.
This is an unloaded **motion** envelope, not an isolated sensor-noise estimate.
Drift and degenerate envelopes are reported. Settlement after unloading at
79 s requires remaining inside through 89 s with at least 1 s remaining.
First-sample membership is not a physical settling-time measurement.
Peak time, signed extrema, rebound, terminal residual and late residual RMS
are reported alongside censored settling outcomes.

Each case has legacy and physics summaries, all-14-joint error/saturation/ripple
metrics and matched hand comparisons. The study root contains paired changes,
failures, plots, an experiment report, nominal/equivalence checks and source,
configuration, seed, model and XML provenance. Historical logs are preserved.
Use `--analyze-only --output <existing-study>` to regenerate analysis; this
rewrites derived reports only. No hardware qualification or continuous operating
envelope is inferred from these finite, exploratory simulation cases.

### Robustness results and qualification (2026-10-22)

The finite study is complete: 594 primary trials, 586 completed and eight stopped
at the existing simulation fall-tilt threshold. The observer failed two sway
trials: friction loss ×0.5 and correlated colored measurement noise, both with
the seed-3/6 s preparation pairing. Integral failed six sway trials across
friction, feedback-delay and shared-PD cases. PD-only completed all trials but
retained much larger carrying errors. These counts are not failure probabilities.

The observer's 27 nominal reference traces reproduce historical shared fields
exactly. The new 5 ms constant-load unloading peak is 0.370–0.372 mm, including
the release state; worst-hand entry that remains inside its measured unloaded
motion band is 0.490–0.495 s. The legacy 20 ms PD-envelope metric remains intact.
Under high velocity noise, carrying accuracy, unloading displacement and ripple
worsen. When arm PD also receives delayed feedback, all methods develop severe
oscillatory regulation/saturation; wide unloaded motion bands can make immediate
band entry uninformative about physical settling.

The selected observer coefficients remain unchanged. Previously accepted nominal
tradeoffs do not constitute acceptance of these new failures or stress costs.
Results describe the sampled points, not a continuous operating envelope or
hardware qualification. Physical torque ratings remain unverified.

The concise findings and complete evidence are in
`logs/x2_transitions/observer-robustness-20261022/findings.md` and that directory's
matrix report, all-joint CSVs, physics summaries, plots and source/hash audits.

### Archived diagnostic and rejected-controller studies

Same-state interventions showed that reducing correction authority prevented the
two original sway falls, with worse arm accuracy; they did not isolate the full
mechanical failure mechanism. A 4 rad/s observer and an added-damping candidate
were rejected after nominal sway falls. Their runners, intervention hooks,
optional damping and dedicated tests are not retained implementation. Sources
and unfavorable results remain in:

- `logs/x2_transitions/observer-causality-20261022-final`
- `logs/x2_transitions/observer-bandwidth4-20261022`
- `logs/x2_transitions/observer-damping05-20261022`

### Shared compliant payload in three dimensions

`compliant_payload` is a separate profile; the original `sway` stress test is
unchanged. One translation-only mass moves relative to the midpoint of the
hand centers. It represents a compliant grip that stays held, with equal hand
reactions. Object rotation, unequal load sharing, collisions and slipping are
outside this approximation.

The fixed exploratory parameters are mass `40/9.81 kg`, total Cartesian
stiffness `[100, 100, 800] N/m`, and damping ratio `0.25` per axis, with
`C = 2 * 0.25 * sqrt(m*K)`. The full-load equilibrium is 50 mm below the hand
midpoint and gives 20 N downward per hand. Initial displacement `[5,0,1] mm`
and lateral velocity `0.005*sqrt(Ky/m) m/s` excite all three directions; a
second condition reverses both signs.

For midpoint position/velocity `a, va` and payload state `p, v`, the force on
the object is `F = K*(a-p) + C*(va-v)`. During the existing pickup/unloading
schedule, load fraction `lambda` scales both the object grip force and the
equal-and-opposite hand reactions. A virtual support supplies
`-(1-lambda)*m*g`. At full load, `m*pdd = m*g + F`, and each hand receives
`-F/2`. Actual dynamic forces are logged without clipping.

Payload state advances once per robot physics step using semi-implicit Euler;
it is detached and caught by virtual support after unloading. Support work,
changing-coupling work, damping/catch dissipation and numerical energy residual
are logged explicitly. Payload truth is evaluation-only and never enters the
controller. MuJoCo shows a sphere and compliant-connection markers.

```bash
PYTHONPATH="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/evaluate_x2_transitions.py --profile compliant_payload \
  --condition observer --repetitions 1 --warmup-seconds 5 --viewer --jobs 1 \
  --output logs/x2_transitions/my-compliant-view
```

The archived 39-trial study had 18 exact historical reference replays, 18 compliant
trials and three smaller-timestep comparisons. Its one-off runner is archived;
the benchmark above retains the payload profile and MuJoCo visualization.
Controllers are PD-only, retained integral and the selected 8 rad/s observer.
Primary physics/telemetry is 5 ms; the numerical checks use 2.5 ms physics with separately
derived 5 ms analysis traces. Controller updates stay at 50 Hz throughout.
Payload parameters and relative initial conditions are matched; realized forces
respond to each controller's trajectory and therefore need not match. The model
parameters are not hardware measurements or a full grasp/contact model.

All 39 runs completed without falls or numerical failures. Across six primary
compliant trials, the observer's 55–60 s aggregate walking RMS was
0.253–0.255°, versus integral 0.210–0.211° and PD-only approximately 5.234°.
Observer unloading peak was 0.376–0.378 mm and motion-band recovery
0.510–0.515 s; the integral retained lower carrying RMS but unloaded by
29.270–29.285 mm. These aggregate errors do not bound every joint/transient.

Smaller-timestep outcomes were unchanged, but full-run world payload positions
differed by 4.1–13.7 cm. Active relative payload motion differed by up to
1.63 mm, so whole-trajectory numerical convergence is not established. Fine
unloading/recovery is not qualified because the study did not include matched
fine-step unloaded baselines. Forces are recorded at the start of each physics
interval with an explicit evaluation timestamp; payload positions/velocities
are recorded at the end. Full results and provenance:
`logs/x2_transitions/compliant-payload-20261022/findings.md`.

### Full compliant-payload robustness study

```bash
PYTHONPATH="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/evaluate_x2_compliant_robustness.py --jobs 8 \
  --output logs/x2_transitions/my-compliant-robustness
```

This runs 22 perturbation cases × three controllers × four load conditions ×
three paired seed/warmup settings: 792 trials. Conditions are unloaded, constant
load and both compliant-payload initial motion directions. The original prescribed
sway is available in the benchmark and original robustness runner. All 36 nominal
traces must match the archived 39-trial study before perturbed acquisition starts.
The runner archives sources/configurations and writes progress, per-trial legacy
and 5 ms traces, per-case physics summaries, all-joint tables and trace hashes.
Output directories must be new. Acquisition completion and analysis completion
are recorded separately in `execution.json` and `completion.json`.

The completed study acquired 792 trials in 109.8 minutes with eight workers:
791 completed and one observer trial fell under shared-PD combined impairment P2
(positive payload motion, seed 2, at 71 s). All 36 nominal checks were exact;
16,625,200 physics samples contained no invalid numeric values. All correction
bounds and slew limits held. The previous F1/C1 sway falls did not recur with the
compliant payload, whose fore–aft forces are much smaller and motion-dependent.
This does not establish that their original failure mechanism was fixed.

Nominal observer walking RMS was 0.253–0.255° and unloading peak 0.376–0.378 mm.
High velocity noise increased these to 0.544–0.563° and 4.295–6.373 mm. The
integral retained lower nominal carrying RMS. Shared-PD cases produced severe
oscillation/clipping across methods, including completed trials; the P2 fall
cannot be attributed to the observer alone. Noisy unloaded motion enlarges
recovery envelopes, so first-sample membership does not establish fast settling.
Timestep convergence remains unresolved and physical torque ratings unverified.

Results, all 792 trial comparisons and audit:
`logs/x2_transitions/compliant-robustness-20261022/findings.md`,
`review_overview.csv` and `review_audit.json` in that directory.

The final cleanup preserves removed sources and hashes in
`logs/x2_transitions/observer-final-cleanup-20261022`. Historical logs remain intact.

## Optional deployment integration

The [shared observer API](joint_observer.md) now separates algorithm selection,
robot parameters and environment integration. Historical measurements above
remain the validated linearized simulation reference.

The real X2 environment now supports `observer.mode` values `OFF`, `SHADOW`
and `ACTIVE`; the deployment observer preset defaults to `OFF` and base hardware presets use `observer=None`. Eligibility additionally requires
`RL_DEFAULT` with an active `X2LocomanipulationPolicy` or its GR00T subclass.
Loco-mimic pipelines permit only idle locomanipulation: interpolation, mimic,
recovery and joint-default preparation remain uncompensated. Arm target ownership
and the existing upper-body filtering are unchanged.

`SHADOW` is opt-in and always transmits zero additive effort. Its predictor uses
zero issued correction, even when its proposed correction is nonzero. It uses the
frozen simulation reference and configured simulation torque limits unless
explicit limits are supplied. `env.get_observer_status()` exposes named,
reference-order proposed, headroom-limited and issued command vectors. These are
command diagnostics, not measured torque or a counterfactual compensated rollout.

`ACTIVE` requires explicit `actuator_limits_nm` (14 positive finite values in the
reference's left-seven/right-seven order) and positive finite `slew_nm_s`.
Correction limits use `correction_limit_fraction * actuator_limit`, optionally
capped by a positive `correction_limit_nm`; zero disables the absolute cap.
The fraction defaults to 0.8 and must lie in `(0, 1]`, so correction limits cannot
exceed actuator limits. There are no hardware-qualified defaults, and simulation
limits are never substituted for missing active limits. Configure the mode before environment construction;
changing the source configuration in place does not reconfigure the running
observer. Runtime eligibility disable clears correction immediately.

The observer runs after final target clipping. Stable per-joint native receive
timestamps determine which samples are new and how much observed time elapsed.
Repeated samples freeze adaptation; Python monotonic elapsed time bounds command
slew only. Timestamped accepted targets and corrections are replayed over each
joint's observed interval, including command changes during inference delays.
Intervals lacking complete command history initialize feedback without adaptation.
Reset clears both sample and command history. Native timestamps share one clock
domain; they do not require Python and C++ clock epochs to match. Per-joint receive ages are captured atomically with the
native joint snapshot; newer backend packets cannot make an older Python cache
appear fresh. These are receive ages, not verified physical acquisition ages. It leaves PD targets and gains unchanged. Correction is
bounded and slew-limited, then restricted to modeled PD torque headroom. If PD
already exceeds a configured limit, only correction reducing saturation is
permitted. Safety removal can reduce correction faster than the normal slew
limit. Accepted effort is reconciled into the next predictor; the nonlinear observer allows estimate
unwinding under downstream limiting while blocking outward adaptation. The predictor models total actuator
clipping, but cannot verify proprietary servo clipping between host updates.
The selected nominal model remains simulation-derived, including in active mode.

The adapter queues positions and arm effort atomically under its existing command
mutex. Position-only calls clear effort. Non-arm effort remains zero. HOLD,
damping, passive operation, re-arm, gain/control selection changes and shutdown
clear buffered effort. Before deselecting arms or entering re-arm's idle state,
the adapter explicitly publishes zero effort to the previous arm selection,
preserving its current PD/HOLD/damping behavior. Loco-mimic interpolation clears
correction in the same cycle that it starts. The runner explicitly shuts down X2
on normal exit, interruption, preparation failure and loop errors, before other
cleanup. A reset generation prevents a correction computed before a
watchdog transition from being accepted afterward, including transient holds.
The environment also resets on mode/policy changes, stale feedback, excessive
elapsed time, invalid samples and rejected submissions. Re-entry starts with zero
correction. Existing watchdog deadlines, recovery requirements and damping
latching remain in force.

### Firmware evidence and remaining deployment blockers

The integration request explicitly confirmed that the target SDK combines PD and
nonzero additive effort correctly. This is user-provided verification, not a
firmware-version qualification performed by this change.

The inspected vendor SDK message and bundled AimDK X2 1.0.0 Joint Control docs
(`third_party/aimdk/docs/cn/_build/html/en/dev/Interface/control_mod/joint_control.html`)
define position, velocity, effort in Nm, stiffness and damping. They do not
specify a combined PD-plus-effort equation or identify firmware versions that
implement it. The [vendor Joint Control documentation](https://x2-aimdk.agibot.com/en/latest/Interface/control_mod/joint_control.html)
was also checked; direct retrieval timed out. Field presence alone is not evidence
of firmware addition semantics.

Before deployment, the remaining blockers are a versioned firmware/support or
verification reference; verified arm peak/continuous and firmware torque limits;
qualification of the nominal observer model, correction bounds and slew on the
actual platform; measured feedback/transport timing; and resolution or explicit
qualification of the documented robustness failures and walking-stability limits.
Supplying numeric configuration does not resolve these blockers. No hardware
connection, deployment process or robot command is part of this integration.
Historical studies and rejected-controller archives are unchanged.

### Deployment-path regression coverage

`tests/test_x2_observer_closed_loop.py` drives a headless MuJoCo plant through the
real deployment environment and observer wrapper with a fake native adapter. It
covers OFF/SHADOW/ACTIVE, arm-only routing, saturation, downstream correction
limiting/reconciliation, eligibility changes, stale snapshots and zero-effort
re-entry. The native regression suite captures the actual publishing loop through
an in-process sink, without constructing actuator publishers, and checks HOLD,
recovery, reset generations, re-arm, deselection and latched damping messages.
These tests supplement the historical studies; they do not requalify the frozen
simulation results or establish hardware stability.

Trial summaries now record physics/controller timesteps. Analysis checks timestamp
coverage at the recorded cadence, including 2.5 ms refinement. Historical summaries
without timestep metadata infer cadence from trace timestamps.
