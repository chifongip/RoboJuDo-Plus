# X2 ACTIVE observer response review

Historical report: source locations and API names below refer to the reviewed
implementation. See [shared joint observers](joint_observer.md) for the current
configuration, algorithms and runtime after the behavior-preserving refactor.

Reviewed 2026-10-23 following successful real-world compensation and reported
overshoot on load removal. The strongest code-level explanation is the symmetric
2 Nm/s output slew limit. Two additional mechanisms deserve correction before
aggressive tuning: finite velocity residuals crossing the friction-model regime,
and estimate unwinding being frozen during torque limiting.

The original review left observer control logic and configuration unchanged. The supplied path
identifies the preset, whose checked-in mode is OFF; it does not provide an ACTIVE
unloading recording. The results below establish reproducible mechanisms, not
their relative contribution to the reported hardware transient.

The deployment implementation now addresses these mechanisms. See
[deployment release validation](x2_observer_release_validation.md) for the changes,
abrupt-unloading comparisons, selected release rate and measured tradeoffs. The
historical simulation reference and its coefficients remain unchanged; numerical
examples below describe the original deployment implementation.

## 1. Output slew dominates unloading response

Locations: `robojudo/config/x2/x2_locomimic_observer_cfg.py:23` and
`robojudo/utils/disturbance_observer.py:147`.

The same rate limits compensation growth and removal. At nominal 50 Hz it allows
only 0.04 Nm change per command. With no intervening safety/headroom removal,
returning a correction of magnitude U to zero requires at least U / 2 seconds:

| Initial correction | Minimum time to zero |
| --- | --- |
| 1 Nm | 0.5 s |
| 4 Nm | 2 s |
| 8 Nm, shoulder/elbow preset bound | 4 s |
| 1.76 Nm, wrist pitch/roll preset bound | 0.88 s |

The fixed estimator bandwidth is 8 rad/s, supplied by
`robojudo/utils/x2_observer_reference.py` through `X2ArmObserver.__init__`.
For a locally affine, exact predictor without constraints, its time constant is
0.125 s and 90% response takes 0.288 s. Increasing that bandwidth cannot remove
the output slew bottleneck. Publishing the held correction every 2 ms also does
not change the 50 Hz observer calculation or its Nm/s limit.

### Reproduction through the timestamped ACTIVE wrapper

An offline synthetic plant uses the frozen local model, preset arm gains and
bounds, zero fixed targets, and a -4 Nm left shoulder pitch disturbance from
1 to 7 s. It runs `X2ArmObserver.propose` and accepted-command acknowledgement
every 20 ms. Plant integration uses the same four implicit substeps as the model.

| Bandwidth (rad/s) | Symmetric slew (Nm/s) | Time after removal to <=0.4 Nm | Positive joint peak after removal |
| --- | --- | --- | --- |
| 8 | 2 | 1.82 s | 4.44 degrees |
| 16 | 2 | 1.80 s | 4.44 degrees |
| 8 | 8 | 0.46 s | 3.74 degrees |
| 16 | 8 | 0.46 s | 3.74 degrees |
| 8 | 20 | 0.20 s | 2.82 degrees |

These are mechanism-isolation experiments, without gravity, coupling, transport
delay, feedback quantization or model mismatch. They are not MuJoCo whole-robot
trials or predictions of hardware angles. Faster output decay materially reduces
the residual-torque duration; it does not eliminate the initial mechanical
transient. The faster symmetric candidates also change loading response and can
briefly reverse correction, so these numbers are not proposed deployment values.

## 2. The local input gain can exaggerate a finite unloading residual

Locations: `robojudo/utils/disturbance_observer.py:97` and `:136`.

The model switches between a high-viscosity region near rest and capped friction
while sliding. The update divides the measured velocity innovation by the
derivative evaluated along the old predicted trajectory. That derivative is
appropriate for small changes within a regime; a sudden removal can cross into
a different regime with substantially different torque-to-velocity gain.

An isolated left shoulder pitch example starts exactly at rest, with +4 Nm
issued correction and -4 Nm estimated/actual disturbance. After actual load
removal, using the same model as the plant:

| Quantity | Value |
| --- | --- |
| Predicted endpoint velocity before detecting release | 0 rad/s |
| Local velocity gain | 0.009014 rad/s per Nm |
| Actual unloaded endpoint velocity at 20 ms | 0.366704 rad/s |
| Innovation divided by local gain | 40.68 Nm |
| Estimate after the 8 rad/s filtered update | +2.015 Nm, previously -4 Nm |

The physical disturbance changed by only +4 Nm. The estimate has already crossed
zero on the first new sample; the long positive torque tail is therefore not
evidence that the estimator always detects removal slowly. A faster symmetric
slew or higher bandwidth can expose the resulting erroneous reverse correction.
The existing small-disturbance contraction test uses 0.15 Nm and does not cover
this finite regime-crossing example.

Candidate improvement: invert the bounded nonlinear interval prediction for an
instantaneous disturbance estimate, then low-pass that estimate. A safeguarded
scalar solve can be vectorized over joints, but must replay the actual accepted
command segments for each receive interval. A cheaper candidate is a bounded
innovation/trust-region update. Compare both for finite steps and reversals;
merely increasing inertia or replacing the input gain would also change noise
sensitivity and the calibrated dynamics.

## 3. Limiting can prevent disturbance estimates from unwinding

Locations: `robojudo/utils/disturbance_observer.py:152` and `:62`, with ACTIVE
reconciliation at `robojudo/utils/x2_arm_observer.py:159`.

The internal anti-windup decision uses the output increment's direction, then
freezes the entire estimate. Following a headroom reduction, issued effort can
be below the compensation implied by an old estimate. Even when that estimate
is moving toward zero, the slew-limited output can still request an outward
increment. Saturation then freezes the useful estimate reduction.

Separately, `acknowledge_effort` restores the entire previous estimate whenever
issued effort differs from proposed effort, regardless of the estimate update's
direction. In an isolated synthetic example, unloading feedback changes an
estimate from -4 to -3.409 Nm and proposes +3.96 Nm. PD requires 35 Nm, leaving
1 Nm headroom under the 36 Nm bound. Acknowledging +1 Nm restores the estimate
to -4 Nm. A constrained local-model rollout retained -4 Nm for 50 consecutive
unloaded samples while issuing zero compensation.

This is a conditional failure mode, not proof that the reported deployment
reached a torque bound. The general fix should distinguish estimation from
command authority: always predict from the acknowledged effort, and allow
bounded estimate changes that unwind blocked compensation. Freeze outward
adaptation selectively rather than rolling back every limited update. Preserve
SHADOW semantics and all generation, freshness, lifecycle and headroom checks.

## Recommended implementation order

1. Add separate growth and release slew settings in the deployment configuration,
   retaining current behavior as the default and the frozen simulation reference.
   Apply a faster rate only while reducing existing compensation magnitude. If
   the desired correction changes sign, release to zero first and build opposite
   torque at the growth rate. Do not classify release by numeric increment sign:
   both positive and negative compensation must work symmetrically. Choose rates
   from measured correction magnitudes and the desired removal time; for example,
   clearing 4 Nm in 0.25 s requires at least 16 Nm/s before estimator delay.
   The 8/20 Nm/s experiments above are diagnostic comparisons only.
2. Fix directional estimate unwinding at both internal saturation and downstream
   acknowledgement. Retain the actual issued correction as the predictor input.
3. Address the nonlinear finite-residual update before increasing bandwidth or
   permitting rapid reverse torque. Include large steps near rest, friction
   transitions, negative compensation and genuine load reversals.
4. Record ACTIVE transients at every observer update: final target, q/dq, native
   receive times and sample age at use, velocity prediction/innovation/input gain,
   estimate, desired/proposed/headroom-limited/issued effort, slew occupancy,
   anti-windup, reset generation, and loop/inference duration. Current status
   exposes estimate/proposed/limited/issued vectors but omits several of these;
   periodic INFO output now includes the same compact summary of estimates,
   issued corrections and bound/limiting counts in ACTIVE and SHADOW.

Measure carrying error, both directions of release peak, time to remove torque,
settling, torque ripple and saturation occupancy together. Compare standing and
walking, dropped/repeated samples, bounded latency and downstream clipping.
Reduced unloading displacement alone is insufficient: previous whole-robot
studies documented unfavorable sway/noise/delay behavior, including rejected
bandwidth and extra-damping candidates; see `x2_upper_body_integral.md`.

## Secondary timing and feedback checks

`FourModePipelineMixin.step` snapshots state before policy inference; the observer
uses that cached arm feedback after inference. Command-history replay accounts
for accepted commands over receive intervals, but cannot observe a release that
occurred after its latest sample. Measure snapshot-to-command age before moving
the observer to a faster loop. A later optimization is an atomic fresh arm
snapshot immediately before observer evaluation; a native observer at a higher
rate would additionally need held final targets/gains, actual issued effort and
the same safety generation/ownership rules. Increasing publisher frequency alone
does not provide either improvement.

External target filtering has EMA alpha 0.95, a nominal time constant of about
0.39 s at 50 Hz, plus a target speed cap. It can delay commanded placement motion,
but it does not filter compensation and cannot explain the constant-target
unloading example above. Compare against the final target before changing it.

Earlier local OFF-mode feedback recordings, reviewed in
`logs/collection-20261023T024526Z/review/findings.md`, showed discrete velocity
levels and nonzero constant hold velocities. Their applicability to this ACTIVE
run is unknown. At the isolated rest-state gain above, a 0.02442 rad/s velocity
code change would alter the filtered estimate by about 0.40 Nm in one sample.
Check the velocity field against quantization-aware position evolution before
raising bandwidth. Avoid masking it with an additional slow velocity filter.
The general bandwidth/noise tradeoff is also described in
[Chen et al.'s DOB overview](https://doi.org/10.1109/TIE.2015.2478397); velocity
filtering adds bandwidth constraints, as discussed by
[Sariyildiz and Ohnishi](https://ieeexplore.ieee.org/document/6822593/).

## Local evidence and validation

Local, git-ignored artifacts are in `logs/x2_observer_response_review/`:
`reproduce.py`, `results.json`, five response CSVs, `plot.py` and `response.png`.
Reproduce the numerical results from the repository root:

```bash
python logs/x2_observer_response_review/reproduce.py
```

The plot script additionally requires matplotlib. All experiments use Python
observers only, without constructing the native adapter or connecting to a robot.

The existing suites passed, 39 tests total:

```bash
python -m unittest discover -s tests -p test_x2_disturbance_observer.py
python -m unittest discover -s tests -p test_x2_arm_observer.py
python -m unittest discover -s tests -p test_x2_locomimic_observer_cfg.py
python -m unittest discover -s tests -p test_x2_observer_closed_loop.py
```

The final suite exercises the deployment path through a fake adapter and a
headless MuJoCo plant. These suites cover existing contracts; the isolated
unloading and estimate-freeze reproductions intentionally expose behavior not
asserted by their current tests.
