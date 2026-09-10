# X2 and G1 23-DoF GR00T locomanipulation control

The GR00T deployment adapter produces absolute robot arm targets and one locomotion command per action-horizon step:

```python
{
    "positions": {"left_shoulder_pitch_joint": 0.1, "...": 0.0},
    "locomotion_command": np.array([vx, vy, yaw_rate, height], dtype=np.float32),
}
```

RoboJuDo keeps this autonomous path separate from the existing teleoperation controller and policy:

- `Gr00tZmqCtrl` publishes camera/joint observations and atomically receives arm/locomotion commands.
- Robot-specific GR00T policy classes feed velocity and height into the existing lower-body ONNX policies.
- A shared GR00T pipeline mixin applies one takeover gate to both command groups and rate-limits arm targets.
- Existing `UpperBodyZmqCtrl` and standard X2/G1 Locomanipulation policy and pipeline behavior is unchanged.

The X2 profile requires 14 arm joints. The G1 23-DoF profile requires 10 arm joints: five per arm, without the wrist
yaw and wrist pitch joints that are absent from the logical 23-DoF layout.

Install the shared camera backends before starting a GR00T pipeline. G1's default RealSense path needs
`pip install -e "packages/robojudo_recorder[realsense,opencv]"`; X2's ROS2 path needs the `ros2` extra and the system
ROS helper dependencies documented in `packages/robojudo_recorder/README.md`.

## Transport
`robojudo/controller/gr00t_zmq_ctrl.py`

`Gr00tZmqCtrl` owns both directions of the deployment transport:

```text
RoboJuDo PUB tcp://*:8561  -> deploy SUB    JPEG + measured upper joints + task
RoboJuDo SUB deploy:8559   <- deploy PUB    upper targets + velocity/height command
```

### Threading and data flow

The two controller flows have separate responsibilities:

```text
50 Hz pipeline/control thread:
env_data.dof_pos
    -> Gr00tZmqCtrl.get_data_with_hook()
    -> update latest thread-safe upper-joint snapshot
    -> non-blockingly receive the latest GR00T command
    -> return ctrl_data to the pipeline

GR00T observation worker thread:
independent camera capture threads
    -> per-camera latest-only JPEG workers
    -> timestamp matching with a maximum camera skew
    + latest upper-joint snapshot
    + task
    -> one atomic msgpack/JPEG multipart observation PUB :8561
```

The camera worker does not read the robot environment directly. Joint state is sampled by the control thread, then
shared with the worker through the locked latest snapshot. Conversely, the worker only publishes observations; robot
PD targets are still computed and applied synchronously by the pipeline control thread.

The observation header is msgpack. A legacy single-camera configuration publishes protocol v1 as `[header, jpeg]`.
A multi-camera configuration publishes protocol v2 as `[header, jpeg_0, ...]`, with payload order defined by
`header["image_keys"]`. All image parts are sent by one `send_multipart()` call. Camera capture and per-camera JPEG
encoding run outside the 50 Hz control thread, and bounded queues discard stale frames instead of accumulating deploy
latency. `max_camera_skew_ms` rejects a candidate bundle when its earliest and latest image timestamps are too far
apart.

G1 uses recorder camera backends; both X2 simulation and X2 real use the ROS2 compressed-image topic configured in
`x2_cfg.py`. Keep using `Gr00tZmqCtrlCfg.camera` for one camera, or set `Gr00tZmqCtrlCfg.cameras` for multiple cameras.
For example, configure three RealSense devices by serial number in `g1_vla_cfg.py` (replace the placeholders with the
serials reported by `rs-enumerate-devices -s`):

```python
cameras=[
    Gr00tCameraCfg(
        type="realsense",
        name="head_rgb",
        image_key="ego_view",
        options={"serial_number": "HEAD_SERIAL", "width": 640, "height": 480, "fps": 30},
    ),
    Gr00tCameraCfg(
        type="realsense",
        name="left_wrist_rgb",
        image_key="left_wrist_view",
        options={"serial_number": "LEFT_WRIST_SERIAL", "width": 640, "height": 480, "fps": 30},
    ),
    Gr00tCameraCfg(
        type="realsense",
        name="right_wrist_rgb",
        image_key="right_wrist_view",
        options={"serial_number": "RIGHT_WRIST_SERIAL", "width": 640, "height": 480, "fps": 30},
    ),
],
camera_pending_capacity=2,
camera_encoder_queue_capacity=2,
camera_poll_timeout_ms=2,
max_camera_skew_ms=50,
```

The protocol v2 header additionally reports `image_shapes`, per-camera source/receive timestamps and sequences, and
the selected bundle's `camera_skew_ns` for runtime diagnostics.

Command messages remain JSON, and every message must contain all configured joints. Invalid positions,
a command other than
`[vx, vy, yaw_rate, height]`, non-finite values, or a replayed sequence reject the complete message without refreshing
the stream timeout. A sequence restart is accepted after the stream has timed out, allowing a restarted publisher to
recover.

The policy output is already in the command convention recorded by RoboJuDo, so it is not passed through joystick
axis remapping. RoboJuDo clips it to the selected robot policy's training limits before lower-body inference.

## Command source routing
`robojudo/policy/gr00t_locomanipulation_policy.py`

`Gr00tLocomanipulationPolicyMixin` only replaces the five-element high-level command source. It does not run GR00T
inference, generate upper-body targets, or replace the robot-specific Locomanipulation ONNX policy. The base policy
continues to build observations and infer lower-body joint targets.

```text
takeover disabled
    -> call the base Locomanipulation joystick/keyboard command path

takeover enabled + GR00T stream fresh
    -> use clipped GR00T [vx, vy, yaw_rate, height]
    -> keep waist yaw at the trained default

takeover enabled + GR00T stream stale
    -> set vx, vy, and yaw_rate to zero
    -> hold the last valid height
    -> do not silently fall back to joystick
```

When takeover changes from enabled to disabled, the mixin clears the previous VLA velocity once before delegating to
the manual command path. This prevents a centered joystick from inheriting and gradually decaying the last VLA motion.
Upper-body joint targets follow the same takeover state but are applied separately by
`Gr00tLocomanipulationPipelineMixin`.

## Run X2

```bash
conda activate robop
python scripts/run_pipeline.py -c x2_gr00t_locomanipulation
```

Select `JOINT_DEFAULT`, wait for interpolation to complete, then enter `RL_DEFAULT`. Enable the GR00T takeover with
the existing upper-body toggle (`Start` or `L` on the configured simulation joystick, or `t` on the keyboard).

```bash
conda activate robop
python scripts/run_pipeline.py -c x2_gr00t_locomanipulation_real
```

On the real config, the joystick remains responsible for mode transitions, damping/shutdown, takeover, and recording.
Its axes supply velocity and height while takeover is disabled; fresh GR00T commands replace them during takeover.

## Run G1 23-DoF

Select the configuration matching the lower-body model and PD gains used during data collection/deployment:

```bash
conda activate robop

# Real G1 with Unitree remote
python scripts/run_pipeline.py -c g1_23_gr00t_locomanipulation_stiff_real
```

Use `--gr00t-task "pick up the red cup"` to override the language instruction in the selected config. When recording,
`--record-task` also becomes the GR00T task unless `--gr00t-task` is supplied explicitly.
When deploy runs on another host, pass
`--gr00t-command-endpoint tcp://<deploy-ip>:8559`; the default command endpoint is localhost.

The real G1 configurations use the existing Unitree remote mode transitions and safety shutdown. Press `Start` to
toggle the shared GR00T arm/base takeover after entering `RL_DEFAULT`. The G1 command ranges come from its recorded
Locomanipulation model, including the `[0.5, 0.78]` base-height range and `0.76` default height.

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

Actively disabling upper-body takeover restores the standard Locomanipulation joystick velocity/height controls and
rate-limits the arms back to their defaults. On CASIA-equipped G1 configurations, the same disable edge also commands
both hands to their zero/default pose before leaving the hardware command gate closed. A stream timeout while takeover
remains enabled does not fall back to the joystick; it keeps zero velocity until the operator explicitly disables
takeover.

## Action-horizon scheduling

Run the double-buffered deploy client from Isaac-GR00T:

```bash
uv run python examples/RoboJuDo/run_robojudo_client.py \
  --profile g1_23dof \
  --camera-layout mulcam \
  --robot-endpoint tcp://<robot-ip>:8561 \
  --policy-host <policy-server-ip> \
  --policy-port 5555 \
  --command-endpoint tcp://*:8559 \
  --execution-mode rtc \
  --execution-horizon 8 \
  --rtc-prefix-schedule exp \
  --rtc-max-guidance-weight 10
```

Omit `--camera-layout mulcam` for a legacy single-camera deployment. Use `--profile x2` for X2. Eight commands at 30
Hz cover approximately 267 ms. The client receives observations and
requests the next action chunk in background threads while its command loop continues publishing at 30 Hz.

## Recording

The deployment observation stream does not depend on recording. If a VLA rollout also needs to be saved, use
`packages/robojudo_recorder/recorder.gr00t.example.yaml` as an optional second subscriber. It extracts only RGB from
port 8561; measured joints and final executed actions still come from the separate 8560 record samples produced by
`--record`. Without `--record` and an active episode, no dataset frames are written.
