# Upper-body LeRobot v3 recording

RoboJuDo-Plus records data in two stages so camera decoding and H.264 encoding cannot reduce real-time capture throughput.

```text
RoboJuDo controls ─┐
                   ├─> robojudo-recorder ─> timestamped raw episode
camera payloads ───┘

timestamped raw episode ─> robojudo-finalize ─> synchronized LeRobot v3 dataset
```

## Dataset features

The first control sample establishes a named-joint schema. X2, G1 29-DoF, and G1 23-DoF therefore share the recorder
implementation while producing separate fixed-schema datasets.

- `observation.state`: measured positions of the joints controlled by upper-body teleoperation.
- `observation.images.<camera name>`: synchronized RGB video.
- `action`: final joint position targets followed by `vx`, `vy`, yaw rate, and height command.

The dedicated `UpperBodyHandZmqCtrlCfg` path appends the 24 measured OmniHand active joints and their final clipped
commands to the arm schema. Hand joints remain separate from the X2/AimDK environment joint list; arm-only controllers
and pipelines do not import or initialize the hand runtime.

The symmetric `UpperBodyCasiaHandZmqCtrlCfg` path appends the 20 physical CASIA Hand-M motor positions and commands
to a G1 arm schema. It accepts only dex teleop's 10-joint-per-hand `sim2real` layout; the 14-joint-per-hand MuJoCo
layout is deliberately rejected.

## Stage 1: real-time capture

Install and start the standalone service:

```bash
pip install -e "packages/robojudo_recorder[ros2]"
robojudo-recorder --config packages/robojudo_recorder/recorder.ros2.example.yaml
```

Then start a supported pipeline:

```bash
python scripts/run_pipeline.py \
  -c x2_locomanipulation_real \
  --record \
  --record-task "pick up the red cup"
```

For the X2 OmniHand preset, install the vendor wheel once in the deployment environment:

```bash
python submodule_install.py omnihand_sdk
sudo bash third_party/omnihand_sdk/linux/x64/setup_udev.sh  # use linux/aarch64 on ARM
```

Log out and back in after installing the udev rules. RoboJuDo then owns the OmniHand SDK and HCAN adapters directly;
do not start the standalone `omnihand_zmq_server.py`. Start dex teleop with its synchronized arm-and-hand stream, then
run the real pipeline above:

```bash
python teleop/robot_control/vr_arm_hand_teleop.py \
  --backend real \
  --robot x2 \
  --hand omnihand \
  --sync-frame-enable-zmq
```

RoboJuDo subscribes to the atomic 14-arm-plus-24-hand frames on dex teleop port 8560; the old per-hand ports are not
used. Use a new dataset root/repository for this 38-joint schema; it cannot be appended to an existing arm-only dataset.

For G1 with dual CASIA Hand-M hardware, install the source-built SDK in the deployment environment:

```bash
python submodule_install.py casiahand_sdk
```

Make sure the active user can access the configured serial device (default `/dev/ttyUSB0`). Start dex teleop with only
the synchronized stream; the legacy ports 5555/5556 are unnecessary because RoboJuDo owns the CASIA SDK directly:

```bash
python teleop/robot_control/vr_arm_hand_teleop.py \
  --backend real \
  --robot g1_23 \
  --hand casia \
  --no-casia-enable-zmq \
  --sync-frame-enable-zmq
```

Then run the matching real pipeline, for example:

```bash
python scripts/run_pipeline.py \
  -c g1_23_casia_locomanipulation_stiff_real \
  --record \
  --record-task "pick up the red cup"
```

The available CASIA presets are `g1_23_casia_locomanipulation_default_real`,
`g1_23_casia_locomanipulation_stiff_real`, and `g1_29_casia_locomanipulation_stiff_real`. They record 10 or 14 arm
joints followed by the same 20 physical hand motors in both state and action. Do not start a standalone CASIA ZMQ
receiver or hardware server alongside RoboJuDo.

The recorder service binds `tcp://*:8560` by default and the pipeline connects to `tcp://127.0.0.1:8560`. Recording
controls are documented in [the recorder README](../packages/robojudo_recorder/README.md).

The real-time stage writes each control sample immediately to `controls.jsonl`, including source and receive timestamps,
measured state, final action, and locomotion command. Camera frames are written independently per camera with both
timestamps, sequence number, encoding, and shape. JPEG/PNG payloads are preserved byte-for-byte; raw RGB sources are
JPEG-compressed but never H.264-encoded. No Parquet or MP4 is produced during capture.

Confirmed episodes are atomically moved from `dataset.raw_root/.pending` to `dataset.raw_root/episodes`. Discarded
episodes are deleted. A crash leaves the current directory under `.pending` for inspection instead of exposing it as a
complete episode.

For ROS 2, prefer `sensor_msgs/msg/CompressedImage` at high resolution:

```yaml
camera:
  type: ros2
  name: head_rgb
  topic: /camera/color/image_raw/compressed
  message_type: compressed
  qos_reliability: reliable
  qos_depth: 1
  ros_python_executable: /usr/bin/python3
  fps: 30
```

The helper process forwards the original compressed payload over loopback ZMQ. Raw `sensor_msgs/msg/Image` is also
supported, but requires RGB conversion and JPEG compression in the live path. Camera FPS may differ from dataset FPS.

## Stage 2: offline finalize

After collection, convert every committed raw episode:

```bash
robojudo-finalize --config packages/robojudo_recorder/recorder.ros2.example.yaml
```

To process selected raw directories, repeat `--episode`:

```bash
robojudo-finalize --config recorder.yaml \
  --episode capture_1787018367000000000_episode_3
```

Finalization uses `sync.clock` (`receive` by default) and creates a uniform `dataset.fps` time grid. It chooses the
nearest frame from every camera within `sync.max_camera_delta_ms`, linearly interpolates measured state between adjacent
control samples, and zero-order-holds the preceding action. It does not extrapolate outside control boundaries. Only
then does it decode images, encode H.264, and write LeRobot v3 Parquet and metadata.

Each raw episode receives `finalize_report.json` with:

- raw control and per-camera frame counts/FPS;
- source sequence gaps;
- requested, written, camera-dropped, and control-dropped target slots;
- unique, duplicated, and unused camera-frame counts;
- camera delta and control age mean/p95/max;
- control samples over `sync.max_control_age_ms`.

Finalization is idempotent: a successful raw episode is skipped when its report and output Parquet still exist. Set
`dataset.resume: true` when appending multiple runs to one output dataset; the writer validates the existing schema.

## Clock selection

Use `sync.clock: receive` when components run on different machines or source clocks are not guaranteed to share an
epoch. Use `source` only when camera and control publishers are in the same synchronized PTP/chrony clock domain. Both
timestamps are retained in the raw files, so a raw capture can be re-finalized with a corrected clock policy.

## Output layout

```text
<dataset.raw_root>/episodes/capture_.../
├── manifest.json
├── controls.jsonl
├── finalize_report.json
└── cameras/<name>/
    ├── frames.jsonl
    └── 00000000.jpg

<dataset.root>/
├── data/chunk-000/file-000.parquet
├── videos/observation.images.<name>/chunk-000/file-000.mp4
└── meta/
```

Run tests with:

```bash
python -m unittest discover -s packages/robojudo_recorder/tests -v
```
