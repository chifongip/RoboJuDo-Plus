# Shared joint observers

Simulation and hardware use the same `JointObserverCfg` and `JointObserver`
runtime. Robot parameters and environment safety are supplied separately.
Algorithm selection is explicit and independent of the environment.

| Component | Location | Responsibility |
| --- | --- | --- |
| `JointObserverCfg` | `robojudo/config/observer_cfg.py` | Algorithm, joint order, model, mode, bounds and slew settings |
| `JointObserver` | `robojudo/utils/observers/runtime.py` | State selection, feedback intervals, accepted-command history, headroom and diagnostics |
| `LinearizedDisturbanceObserver` | `robojudo/utils/observers/linearized.py` | Historical local-gain estimation, symmetric slew and original anti-windup |
| `NonlinearDisturbanceObserver` | `robojudo/utils/observers/nonlinear.py` | Bounded finite-residual estimation, separate release slew and directional unwinding |
| X2 parameter preset | `robojudo/config/x2/observer_parameters.py` | Frozen X2 model coefficients and joint order |
| Environment integration | MuJoCo or AimDK environment | Acquire feedback, issue corrections, acknowledge admitted commands, and enforce safety |

Both algorithms share the nominal dynamics in `observers/model.py`; neither
inherits the other's estimator or limiting behavior. The core/runtime import no
robot-specific configuration or SDK.

## Configuration

Import the shared type from its defining module:

```python
from robojudo.config.observer_cfg import JointObserverCfg

# Synthetic example only: these parameters are not calibrated for a robot.
observer = JointObserverCfg(
    algorithm="nonlinear",
    mode="SHADOW",
    joint_names=["shoulder", "elbow"],
    inertia=[0.1, 0.05],
    friction_limit=0.3,
    friction_viscosity=[40.0, 30.0],
    bandwidth_rad_s=8.0,
    correction_limits_nm=[1.0, 1.0],
    slew_nm_s=2.0,
    release_slew_nm_s=16.0,
)
```

Model parameters and correction bounds accept scalars or vectors. Vectors follow
`joint_names`, which also defines correction and status order. Actuator bounds,
when supplied, are a vector in that same order. All configuration is validated
again when constructing the runtime.

`algorithm` is required. For `linearized`, release slew must be omitted or equal
to growth slew. For `nonlinear`, omission uses the growth rate; a larger release
rate applies toward zero only, with opposite torque growing on a later update
at the growth rate. OFF and SHADOW always return zero issued correction; SHADOW
still estimates using acknowledged zero effort. Changing a source configuration
does not reconfigure a running observer: rebuild it and reset its history.

The registered X2 presets use the same nonlinear algorithm and parameters:

| Preset | Algorithm | Mode | Growth / release (Nm/s) |
| --- | --- | --- | --- |
| `x2_locomimic_observer` | nonlinear | ACTIVE | 2 / 16 |
| `x2_locomimic_real_observer` | nonlinear | OFF | 2 / 16 |

Base hardware presets use `observer=None` to disable the feature. A configured
OFF observer also disables it. Simulation may use its model's actuator bounds
when `actuator_limits_nm` is omitted. The hardware boundary calls
`validate_hardware()` before constructing any native controller: ACTIVE requires
explicit actuator bounds, and correction bounds and growth slew are required
configuration fields. Configured bounds do not establish hardware qualification.

## Runtime and environment contracts

```python
from robojudo.utils.observers import JointObserver

runtime = JointObserver(
    observer,
    environment_joint_names,
    stiffness,
    damping,
    max_dt=0.1,
    actuator_limits=environment_torque_limits,
    control_dt=0.02,
)
effort = runtime.propose(target, position, velocity, now)
runtime.acknowledge(accepted_effort)
```

`target`, `position`, `velocity`, environment gains and fallback actuator limits
use the full environment joint order. `propose()` returns a NumPy vector in
configured observer order. `acknowledge()` accepts that same order and must
contain the correction actually admitted by the command boundary.

Synchronous MuJoCo callers use simulation time and acknowledge admitted effort
before advancing physics. Timestamped callers additionally pass
`sample_times=receive_timestamps` to `propose()` and
`command_time=receipt_time` to `acknowledge()`. Receive timestamps use the full
environment order and share the command-receipt clock domain; that clock need
not share an epoch with `now`. Repeated samples cannot adapt estimates. New
samples replay accepted target/effort segments over each joint's receive interval.
Intervals without sufficient history initialize without adaptation. Reset clears
both sample and command history. Environments choose the allowed interval:
MuJoCo uses five control periods; AimDK uses the smaller command/state timeout.

AimDK additionally restricts observers to exactly its configured arm-joint set
and maps corrections, bounds and receipts into/out of native arm order. These
restrictions live in the environment, not the generic runtime. Existing native
freshness, generation, ownership, watchdog and lifecycle checks remain in place.
The current X2 pipeline's policy eligibility rules are unchanged; a generic core
does not automatically enable compensation for other robot/policy combinations.

Both environments expose `reset_observer()`, `set_observer_eligible(bool)` and
`get_observer_status()`. Status includes `algorithm`, configured joint names,
estimate, proposed/headroom-limited/issued corrections, prediction, innovation
and limiting masks. Nonlinear-only diagnostics are present for that algorithm.
Returned arrays are copied lists. Issued correction is a command diagnostic,
not measured actuator torque.

## Migration

This is a clean API break; old imports and aliases are removed:

| Previous API | Replacement |
| --- | --- |
| `X2ArmObserverCfg` | Shared `JointObserverCfg` with explicit joint names, model and algorithm |
| `env.arm_observer` | `env.observer`; disabled base hardware configs use `None` |
| `X2ArmObserver` and old utility modules | `JointObserver` from `robojudo.utils.observers` |
| `BoundedDisturbanceObserver` | `LinearizedDisturbanceObserver` |
| `DeploymentDisturbanceObserver` | `NonlinearDisturbanceObserver` |
| Arm-specific observer environment methods | `reset_observer`, `set_observer_eligible`, `get_observer_status` |
| Utility/script X2 reference exports | `robojudo.config.x2.observer_parameters` |

The removed `arm_observer` configuration key raises a migration error. Update
external configuration overrides and scripts along with imports; parameters are
no longer selected implicitly by an X2 wrapper. Another robot can provide its
own parameter preset to the same config/runtime, but still needs model
calibration and a compatible environment integration for hardware use.

## Refactor validation

The fixture `tests/fixtures/observer_f0ba949.npz` was captured from commit
`f0ba949` before modifying the implementation. Its deterministic replay records
proposed, limited and issued corrections, estimates, predictions, innovations and
anti-windup flags through resets, repeated timestamps and clipped receipts.
The shared runtime matches every value **bit-for-bit**, including nonlinear
SHADOW operation. `test_observer_parity.py` locks that reference.

Standing and walking headless MuJoCo comparisons reran seed 1 for PD-only,
linearized and nonlinear controllers, with nominal feedback, delayed/quantized
feedback plus release-time dropout, and downstream clipping. All **36 paired
loaded/unloaded traces** match the committed validation data exactly in every
recorded field except execution time. No new tuning was performed. Results and
the comparison manifest are in `logs/x2_observer_restructure/` (git-ignored).

```bash
python -m unittest discover -s tests -p 'test*observer*.py'
python scripts/evaluate_x2_observer_release.py --plant mujoco --rates 16 \
  --cases nominal impaired clipped --seeds 1 --load -20 \
  --output logs/x2_observer_restructure/standing
python scripts/evaluate_x2_observer_release.py --plant mujoco --walking --rates 16 \
  --cases nominal impaired clipped --seeds 1 --load -20 \
  --output logs/x2_observer_restructure/walking
```

The observer suites cover both algorithms with arbitrary names, counts and
joint order, plus timing, command reconciliation and hardware safety boundaries.
Permanent fake-SDK tests exercise the full AimDK environment constructor for both
algorithms in OFF, SHADOW and ACTIVE, including shuffled joint order and distinct
per-joint correction bounds. They also verify native command ordering, clipped
receipt reconciliation, and rejection of invalid ACTIVE configurations before
the native controller is constructed.

All **69 observer tests** and Ruff checks pass. The broader suite reports
the same four unrelated errors as before: missing optional `robojudo_recorder`
and missing velocity-source priorities in three GR00T presets; one test is skipped.
The full suite's ZeroMQ tests use loopback sockets, not robot connections.
Historical response-review and release-validation reports retain the names used
when those experiments were conducted.
