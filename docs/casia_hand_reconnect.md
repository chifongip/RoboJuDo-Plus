# CasiaHand hot-unplug recovery

`g1_23_gr00t_locomanipulation_stiff_real` enables automatic CasiaHand reconnection.
The shared offline DAgger configuration inherits this setting. Other CasiaHand
configurations and standalone SDK users retain the existing behavior unless
`CasiaHandCfg(auto_reconnect=True)` / `CasiaHandConfig(auto_reconnect=True)` is selected.

## Rebuild before running

The change includes the native C++ extension. In the same Python environment used
for `scripts/run_pipeline.py`, rebuild the editable SDK installation from the
RoboJuDo repository root:

```bash
python -m pip install --no-build-isolation -e third_party/casiahand_sdk
```

This requires the SDK's existing C++ build dependencies, `scikit-build-core` and
`pybind11`. Reinstalling is necessary even with an editable installation: Python
source changes are immediate, but native extension changes are not. An older
extension produces an explicit rebuild error when automatic reconnection is enabled.
The GR00T command endpoint does not need to change. On hosts that benefit from
CPU affinity, prefix the existing pipeline command with `taskset -c 0-7,10-31`.
The verification commands below use this same CPU mask; adjust it for other hosts.

## Runtime behavior

- The pipeline may start while the hands are unpowered or USB is absent. Hardware
  discovery, initialization and retries run in the CasiaHand worker, outside the
  robot control loop. Construction waits up to `startup_timeout_s` for the first
  connection or failed attempt, allowing present hands to qualify before camera
  startup. Missing hardware still falls back to background retries.
- When dual-hand feedback is older than `joint_state_timeout_s` (default 0.25 s),
  GR00T actions become unavailable immediately. In X/RL mode, the arms return to
  `upper_body_default_pose` using the existing joint velocity limit. The pipeline
  publishes takeover disabled and restores its existing manual locomotion path.
- After `reconnect_timeout_s` (default 1 s) without successful dual-hand feedback,
  the worker closes the old SDK and serial port. Native worker failures trigger
  the same recovery. Failed attempts are retried after `reconnect_interval_s`
  (default 1 s); there is no retry limit.
- Each attempt gives serial initialization and subsequent feedback qualification
  separate `startup_timeout_s` budgets (default 5 s each). Three distinct recent
  dual-hand samples qualify the connection. OS scheduling and serial cleanup can
  add time to those budgets. One working hand alone cannot qualify the connection.
  Startup timeouts report valid/rejected sample counts and the latest sample age.
- On recovery, a new GR00T control session invalidates old arm, hand and locomotion
  actions. If X and Start takeover are still enabled, control resumes only after
  a command matching the new session arrives. Hands initially approach new targets
  from measured joint positions at at most 1 rad/s; command expiry stops this ramp.
- Pressing B or disabling Start during an outage cancels automatic action takeover.
  Reconnection still restores feedback. The usual online B/disable hand-zero command
  is preserved; zero commands requested while offline are discarded, not replayed.
- Exit interrupts retry waits and joins the worker before returning. No replacement
  SDK is opened before the previous instance has released its serial port.

The worker pins the first adapter's unique USB serial number, or its physical USB
port when a unique serial number is unavailable. It resolves the new tty node
following unplug/replug, so changing `ttyCH341USB0` to `ttyCH341USB1` is supported.
Without a serial number, reconnect to the same physical USB port. If the configured
node is absent on the first attempt, exactly one WCH USB serial candidate is needed
for initial binding; otherwise the worker waits and reports ambiguity. Explicit
`/dev/serial/by-id/` or `/dev/serial/by-path/` paths must appear before initial binding.
No udev rules are installed or changed automatically.

`get_data()` retains the existing joint/action fields and adds `connected`,
`connection_state` (`CONNECTING`, `CONNECTED`, `RECONNECTING`, `STOPPED`),
`connection_generation`, `reconnect_attempts`, `last_error`, and `joint_state_age_s`.
`connected` describes an initialized, qualified instance; use `joint_state_fresh`
to determine whether feedback is currently usable. Recoverable transport failures
return status rather than terminating the pipeline. Unexpected programming errors
or failures to release the old instance remain fatal.

## Verification

Hardware-free SDK and integration tests:

```bash
python -m unittest discover -s third_party/casiahand_sdk/tests
python -m unittest discover -s tests -p 'test_casia_reconnect.py'
python -m unittest discover -s tests -p 'test_gr00t_observation_stream.py'
python -m unittest discover -s tests -p 'test_upper_body_speed_limit.py'
python -m unittest discover -s tests -p 'test_offline_dagger.py'
```

The native deadline/cleanup test uses a pseudo-terminal with no robot attached:

```bash
cmake -S third_party/casiahand_sdk -B /tmp/casia-native-tests -DCASIAHAND_BUILD_PYTHON=OFF -DCASIAHAND_BUILD_TESTS=ON
cmake --build /tmp/casia-native-tests
ctest --test-dir /tmp/casia-native-tests --output-on-failure
```

On a supported, supervised robot setup, verify power-only loss, USB-only loss,
and simultaneous loss while X and Start takeover are active. Check that the arms
return to the configured default pose, the process remains running, and replugging
restores fresh feedback with a new generation/session before new actions run.
Repeat with B or Start disabled during the outage, and with shutdown while waiting
for hardware. Observe recovery time and verify tty renumbering on the same USB port.
These hardware scenarios are not covered by the fake-device tests.
