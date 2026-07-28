# Upper-body LeRobot v3 recording

RoboJuDo-Plus publishes the state and final commands used by its control loop to a separate recorder service. The service
owns the camera and writes a LeRobot v3 dataset, so video encoding and dataset dependencies cannot block robot control.

## Dataset features

Each robot negotiates a named-joint schema when the first sample arrives. Agibot X2, Unitree G1 29-DoF, and G1 23-DoF therefore use
the same recorder implementation but produce separate, fixed-schema datasets.

- `observation.state`: actual positions of all joints controlled by upper-body teleoperation.
- `observation.images.<camera name>`: RGB frames from the configured camera.
- `action`: final upper-body position targets followed by **`vx`, `vy`, yaw rate, and height commands**.

Adding hand joints to `UpperBodyZmqCtrlCfg.joint_names` automatically adds their positions and final targets to the
negotiated schema. Do not change the joint list while appending episodes to one dataset.

## Install

Install the recorder in its own environment:

```bash
pip install -e "packages/robojudo_recorder[realsense]"
```

Use `[opencv]` instead of `[realsense]` for a V4L2/OpenCV camera. A ZMQ RGB camera only needs the base package; JPEG
frames additionally require `[opencv]`.

## Run

Create a recorder configuration from `packages/robojudo_recorder/recorder.example.yaml`, then start the service:

```bash
robojudo-recorder --config packages/robojudo_recorder/recorder.example.yaml
```

Start a supported locomanipulation pipeline with recording enabled:

```bash
python scripts/run_pipeline.py -c x2_locomanipulation_real --record --record-task "pick up the red cup"
```

The pipeline binds `tcp://*:8560` by default and the recorder connects to `tcp://127.0.0.1:8560`. Use
`--record-endpoint` or the pipeline `record.endpoint` config to change the publisher endpoint.

Recording is controlled from the joystick after entering `RL_DEFAULT` and enabling upper-body takeover:

- Xbox/standard joystick: hold `LB+RB`, then press `Start` to start/end and `Back` to pause/resume.
- Unitree remote: hold `L1+R1`, then press `Start` to start/end and `Select` to pause/resume.

Start waits for the first fresh upper-body command. End saves the episode; pause/resume keeps one logical episode and
does not write frames during the pause. Disabling takeover, leaving `RL_DEFAULT`, switching away from the locomotion
policy, or exiting the pipeline automatically stops and saves an active episode. Stale teleop frames are skipped.

Recorder messages use a bounded, non-blocking ZMQ queue. If the recorder is absent or cannot keep up, RoboJuDo drops
recording samples and logs a counter instead of delaying the control loop.

Set `dataset.resume: true` to append episodes after restarting the recorder. Resume validates FPS, robot type, joint
names, action names, and camera shape before opening a new episode; it will not silently mix incompatible schemas.

## Camera backends

Camera creation is registry based. Built-in types are `realsense`, `opencv`, and `zmq`. A new backend implements
`CameraSource` and registers itself with `register_camera("name")`; dataset and pipeline code do not need changes.
For direct cameras, configured camera FPS must equal dataset FPS.

The ZMQ backend expects multipart messages containing a JSON header and image payload:

```text
[ {"sequence": 42, "timestamp_ns": 123456789}, image bytes ]
```

Set `encoding: raw_rgb` for contiguous `uint8[H,W,3]` bytes or `encoding: jpeg` for JPEG payloads.
The ZMQ backend uses local receive time by default. Set `timestamp_mode: source` only when the camera publisher and
recorder clocks are synchronized.

For processes on one machine, set `sync.clock: source`. Across machines, set it to `receive` unless the hosts share a
PTP/chrony-synchronized clock. Frames without a sufficiently recent control sample are dropped.
