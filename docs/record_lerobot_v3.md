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
frames additionally require `[opencv]`. The ROS 2 backend uses `[ros2]` for compressed-image decoding; `rclpy` and
`sensor_msgs` must come from the sourced ROS 2 environment.

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

Camera creation is registry based. Built-in types are `realsense`, `opencv`, `ros2`, and `zmq`. A new backend implements
`CameraSource` and registers itself with `register_camera("name")`; dataset and pipeline code do not need changes.
For direct cameras, configured camera FPS must equal dataset FPS.

### ROS 2 CompressedImage

Use `type: ros2` to subscribe directly to a `sensor_msgs/msg/CompressedImage` topic:

```yaml
camera:
  type: ros2
  name: head_rgb
  topic: /aima/hal/sensor/stereo_head_front_right/rgb_image/compressed
  qos_reliability: best_effort
  qos_depth: 1
  ros_python_executable: /usr/bin/python3
  fps: 30
  # width: 640
  # height: 480
```

The complete example is `packages/robojudo_recorder/recorder.ros2.example.yaml`. The recorder may run under Python
3.11 while ROS 2 Humble uses Python 3.10: this backend launches `ros_python_executable` as a small subscriber process
and transfers compressed bytes to the recorder over a loopback ZMQ socket. The ROS Python only needs `rclpy`,
`sensor_msgs`, and `pyzmq`; OpenCV decoding and dataset writing remain in the recorder process.

`best_effort` with depth 1 matches the usual ROS sensor-data QoS and keeps only the latest frame; set
`qos_reliability: reliable` when the publisher uses reliable delivery. Width and height are optional and inferred from
the first frame; when configured, every decoded frame is validated against them. Set camera FPS to the topic
publication rate; it must equal `dataset.fps`. Frames are timestamped with local monotonic receive time, so use
`sync.clock: receive`. Set `ROS_DOMAIN_ID` before starting the recorder when the publisher is in a non-default domain.

The ZMQ backend expects multipart messages containing a JSON header and image payload:

```text
[ {"sequence": 42, "timestamp_ns": 123456789}, image bytes ]
```

Set `encoding: raw_rgb` for contiguous `uint8[H,W,3]` bytes or `encoding: jpeg` for JPEG payloads.
The ZMQ backend uses local receive time by default. Set `timestamp_mode: source` only when the camera publisher and
recorder clocks are synchronized.

For a ZMQ camera on the same machine, `sync.clock: source` can preserve source timestamps. Across machines, use
`sync.clock: receive` unless the hosts share a PTP/chrony-synchronized clock. Frames without a sufficiently recent
control sample are dropped.
