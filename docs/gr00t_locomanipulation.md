# X2 GR00T locomanipulation control

The GR00T deployment adapter produces absolute X2 arm targets and one locomotion command per action-horizon step:

```python
{
    "positions": {"left_shoulder_pitch_joint": 0.1, "...": 0.0},
    "locomotion_command": np.array([vx, vy, yaw_rate, height], dtype=np.float32),
}
```

RoboJuDo keeps this autonomous path separate from the existing teleoperation controller and policy:

- `Gr00tZmqCtrl` atomically receives all 14 arm positions and the four-element locomotion command.
- `X2Gr00tLocomanipulationPolicy` feeds the GR00T velocity and height into the existing X2 lower-body ONNX policy.
- `X2Gr00tLocomanipulationPipeline` applies one takeover gate to both command groups and rate-limits arm targets.
- Existing `UpperBodyZmqCtrl`, `X2LocomanipulationPolicy`, and `X2LocomanipulationPipeline` behavior is unchanged.

## Publish commands

Bind the GR00T robot-side publisher to port 8559 and send the complete command in one JSON message. The sequence number
is optional, but recommended for rejecting reordered action-horizon steps while the stream is active.

```python
context = zmq.Context.instance()
publisher = context.socket(zmq.PUB)
publisher.bind("tcp://*:8559")
time.sleep(0.5)

sequence = 0
for command in commands:
    publisher.send_json(
        {
            "sequence": sequence,
            "positions": command["positions"],
            "locomotion_command": command["locomotion_command"].tolist(),
        }
    )
    sequence += 1
    time.sleep(1.0 / 30.0)
```

All configured joints are required in every message. Invalid positions, a command other than
`[vx, vy, yaw_rate, height]`, non-finite values, or a replayed sequence reject the complete message without refreshing
the stream timeout. A sequence restart is accepted after the stream has timed out, allowing a restarted publisher to
recover.

The policy output is already in the command convention recorded by RoboJuDo, so it is not passed through joystick
axis remapping. RoboJuDo clips it to the X2 training limits before lower-body inference.

## Run in simulation

```bash
conda activate robop
python scripts/run_pipeline.py -c x2_gr00t_locomanipulation
```

Select `JOINT_DEFAULT`, wait for interpolation to complete, then enter `RL_DEFAULT`. Enable the GR00T takeover with
the existing upper-body toggle (`Start` or `L` on the configured simulation joystick, or `t` on the keyboard).

## Run on X2

```bash
conda activate robop
python scripts/run_pipeline.py -c x2_gr00t_locomanipulation_real
```

On the real config, the joystick remains responsible for mode transitions, damping/shutdown, takeover, and recording.
Its axes do not supply velocity or height to `X2Gr00tLocomanipulationPolicy`.

GR00T commands are applied only while all of the following are true:

```text
RL_DEFAULT
AND upper-body takeover enabled
AND GR00T stream fresh
AND upper-body control available
```

When the stream becomes stale, `vx`, `vy`, and yaw rate immediately become zero while height holds its last valid
value. Arm targets return toward their configured defaults through the pipeline's explicit joint-velocity limiter.
The default limit is 4 rad/s, or 0.08 rad per 50 Hz control step.

## Action-horizon scheduling

Eight commands at 30 Hz cover approximately 267 ms. Request the next GR00T action chunk while the current chunk is
still executing. A synchronous request after consuming the complete chunk can leave a gap longer than the default
250 ms controller timeout. A double-buffered producer avoids periodic stop/recovery transitions.

## Recording

Recording needs no GR00T-specific writer. The pipeline reports the clipped locomotion command actually passed into
the lower-body policy, and the existing recorder saves it together with measured arm positions, final rate-limited arm
targets, and camera RGB frames.
