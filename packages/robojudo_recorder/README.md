# RoboJuDo Recorder

`robojudo-recorder` 是一个独立于机器人控制循环的 LeRobot v3 数据录制服务。它接收 RoboJuDo pipeline
发布的上半身状态和控制命令，将控制样本与相机帧配对，并按 episode 写入 Parquet 和 H.264 视频。

本模块不在运行时依赖 `third_party/lerobot`。LeRobot v3 的目录、字段和元数据由本模块独立实现，方便在
X2、Unitree G1 29-DoF、G1 23-DoF 以及后续灵巧手配置之间复用。

## 录制内容

每一帧包含：

| LeRobot feature | 内容 |
| --- | --- |
| `observation.state` | 上半身 teleop 所控制关节的实际位置 |
| `observation.images.<camera_name>` | 相机 RGB 图像，保存为 H.264 MP4 |
| `action` | 上半身最终关节位置目标，后接 `vx`、`vy`、yaw rate 和 height command |

关节字段使用名称而不是固定索引。首次收到控制样本时，recorder 根据 `robot_type` 和 `joint_names` 建立
schema。把灵巧手关节加入 RoboJuDo 的 `UpperBodyZmqCtrlCfg.joint_names` 后，状态和 action schema 会自动
包含这些关节。

## 架构

```text
GR00T deployment（始终存在）:
camera backend ─> Gr00tZmqCtrl observation stream ─> GR00T deploy

可选 VLA rollout recording:
Gr00tZmqCtrl observation stream ── RGB ─────────────┐
RoboJuDo pipeline ── measured state/action samples ─┴─> recorder service ─> LeRobot v3 dataset
```

普通 teleop 录制时，相机读取、图像解码、视频编码和 Parquet 写盘都在 recorder 进程中完成。GR00T
模式下，相机采集和 observation 编码由 controller 的后台线程完成，recorder 仍负责视频编码和数据写盘；
两种模式都不会在机器人控制线程执行阻塞式相机 I/O。

GR00T 配置下，相机由 `Gr00tZmqCtrl` 的后台线程读取，并发布包含 RGB、实测关节位置和 task 的 deployment
observation stream。这个 stream 属于 GR00T 推理闭环，不依赖 recorder，也不需要 `--record`。只有在额外
录制 VLA rollout 时，recorder 才可以作为可选的第二个 subscriber 复用其中的 RGB payload。

## 安装

基础安装：

```bash
conda activate robop
pip install -e packages/robojudo_recorder
```

根据相机类型安装对应 extra：

```bash
# V4L2/USB/OpenCV camera
pip install -e "packages/robojudo_recorder[opencv]"

# Intel RealSense
pip install -e "packages/robojudo_recorder[realsense]"

# ROS 2 CompressedImage 或 raw Image；OpenCV 用于压缩图像解码
pip install -e "packages/robojudo_recorder[ros2]"

# 测试和 lint
pip install -e "packages/robojudo_recorder[dev]"

# 仅在采集完成后需要手动上传时安装
pip install -e "packages/robojudo_recorder[hub]"

# GR00T + G1 RealSense observation publisher（相机采集和 JPEG 编码）
pip install -e "packages/robojudo_recorder[realsense,opencv]"
```

## 快速开始：ROS 2 相机

ROS 2 Humble 通常使用系统 Python 3.10，而 `robop` 使用 Python 3.11。两者的二进制扩展不兼容，因此
recorder 不会在 Python 3.11 中直接导入 `rclpy`：

(使用ROS2 camera务必确保本机系统ubuntu22，并且已安装ROS2 Humble)

```text
/usr/bin/python3 (ROS 2 helper)
    └─ subscribe sensor_msgs/msg/CompressedImage 或 sensor_msgs/msg/Image
        └─ image bytes + metadata over loopback ZMQ
            └─ robop Python 3.11 recorder
```

ROS helper 由 recorder 自动启动，不需要手动运行。系统 ROS Python 需要能够导入 `rclpy`、
`sensor_msgs` 和 `zmq`：

```bash
/usr/bin/python3 -c "import rclpy, zmq; from sensor_msgs.msg import CompressedImage, Image; print('ROS bridge ready')"
```

使用现成的配置：

```bash
conda activate robop
source /opt/ros/humble/setup.bash

robojudo-recorder \
  --config packages/robojudo_recorder/recorder.ros2.example.yaml
```

Use `ros2 topic info -v /aima/hal/sensor/stereo_head_front_right/rgb_image/compressed` to check the Reliability of your camera stream is RELIABLE or BEST-EFFORT

关键相机配置如下：

```yaml
camera:
  type: ros2
  name: head_rgb
  topic: /aima/hal/sensor/rgbd_head_front/rgb_image
  message_type: raw
  qos_reliability: reliable
  qos_depth: 1
  ros_python_executable: /usr/bin/python3
  fps: 30
  # width: 640
  # height: 480
```

`message_type` 可设为 `raw`（`sensor_msgs/msg/Image`）或 `compressed`
（`sensor_msgs/msg/CompressedImage`），省略时默认 `compressed`，兼容已有配置。raw 模式支持
`rgb8`、`bgr8`、`rgba8`、`bgra8` 和 `mono8`，输出统一转换为 RGB。`width` 和 `height` 可以省略，
recorder 会从第一帧推断；指定后，每帧都会进行尺寸校验。ROS 图像使用
本机 monotonic 接收时间，因此 `sync.clock` 应设置为 `receive`。

如果 ROS publisher 使用非默认 domain，启动前设置：

```bash
export ROS_DOMAIN_ID=<domain_id>
```

## 启动 RoboJuDo pipeline

建议先启动 recorder，再在另一个终端启动 pipeline，避免连接建立前丢弃控制样本：

```bash
conda activate robop
python scripts/run_pipeline.py \
  -c g1_23_locomanipulation_locomimic \
  --record \
  --record-task "pick up the red cup"
```

默认连接关系：

```text
pipeline bind:   tcp://*:8560
recorder connect: tcp://127.0.0.1:8560
```

pipeline 在另一台机器时，将 recorder 配置中的 `control_endpoint` 改成：

```yaml
control_endpoint: tcp://<pipeline-ip>:8560
```

pipeline 端可以用 `--record-endpoint` 修改 bind endpoint。

### 可选：录制 VLA rollout 时复用 deployment RGB

仅运行 GR00T deploy 时不需要启动 recorder，也不需要下面的配置。当用户还希望把 VLA rollout 保存为
LeRobot dataset 时，recorder 不应再次打开 RealSense 或 ROS2 topic，而应作为旁路消费者复用 deployment
observation 中的 RGB：

```bash
conda activate robop
robojudo-recorder --config packages/robojudo_recorder/recorder.gr00t.example.yaml
```

默认连接关系：

```text
Gr00tZmqCtrl observation PUB: tcp://*:8561
GR00T deploy observation SUB: tcp://<robot-ip>:8561
recorder camera SUB:          tcp://127.0.0.1:8561
```

`camera.type: zmq` 会从 GR00T header 自动读取 JPEG encoding 和图像尺寸，不需要重复填写 `width`、
`height`。这个 camera backend 只从 8561 提取 RGB；它不会使用 observation header 中的 joint positions
作为 dataset state。dataset 的实测 joints 和最终执行 action 仍来自启用 `--record` 后的 8560 control
sample，并按 timestamp 与 RGB 配对。没有 `--record` 和有效录制 episode 时，recorder 不会写入数据。

## 手柄录制控制

开始录制前必须：

1. 进入 `RL_DEFAULT`。
2. 启用上半身 takeover。
3. 确认上半身 teleop stream 是 fresh 状态。

默认按键：

| 控制器 | 开始 / 结束并保存 | 暂停 / 继续 |
| --- | --- | --- |
| Xbox/标准 joystick | 按住 `LB+RB`，按 `Start` | 按住 `LB+RB`，按 `Back` |
| Unitree remote | 按住 `L1+R1`，按 `Start` | 按住 `L1+R1`，按 `Select` |

暂停期间不会写入控制样本或相机帧，但恢复后仍属于同一个 episode。关闭上半身 takeover、离开
`RL_DEFAULT`、切换离开 locomotion policy 或退出 pipeline，会自动结束并保存当前 episode。

仅仅启动 recorder 不会创建数据。第一次收到有效录制样本时才会初始化 dataset writer；结束一个包含有效
帧的 episode 后，Parquet、视频和 episode metadata 才会完整写入。

## 手动上传 Hugging Face

recorder 运行期间不会连接 Hugging Face，也不会因为结束 episode 而上传数据。采集完成并检查本地
`dataset.root` 后，再显式执行上传命令：

```bash
conda activate robop
pip install -e "packages/robojudo_recorder[hub]"
robojudo-upload-dataset record_data/x2_move_box_center \\
  --repo-id Breeze-park/x2_move_box_center
```

该命令会创建（或复用）dataset repository，并上传整个本地 dataset 目录。私有仓库可增加
`--private`；访问凭据由 `huggingface-cli login` 或 `HF_TOKEN` 提供。上传失败不会影响已经保存的本地
数据，也不会阻塞录制控制循环。

## 配置说明

完整配置结构：

```yaml
control_endpoint: tcp://127.0.0.1:8560

dataset:
  root: record_data/g1_23_upper_body
  repo_id: local/g1_23_upper_body
  fps: 30
  codec: libx264
  resume: false

camera:
  type: ros2
  name: head_rgb
  topic: /camera/rgb
  message_type: raw
  qos_reliability: reliable
  qos_depth: 1
  ros_python_executable: /usr/bin/python3
  fps: 30

sync:
  clock: receive
  max_control_age_ms: 50
  poll_timeout_ms: 10
```

| 字段 | 含义 |
| --- | --- |
| `control_endpoint` | recorder 连接的 RoboJuDo control sample endpoint |
| `dataset.root` | 本地 dataset 输出目录 |
| `dataset.repo_id` | 写入 LeRobot metadata 的 dataset 标识 |
| `dataset.fps` | dataset 和视频 FPS |
| `dataset.codec` | PyAV/FFmpeg encoder，例如 `libx264` |
| `dataset.resume` | 是否在已有兼容 dataset 后追加 episode |
| `camera.type` / `cameras[].type` | `opencv`、`realsense`、`ros2` 或 `zmq` |
| `camera.name` / `cameras[].name` | LeRobot image feature 名称的一部分，同一配置内必须唯一 |
| `sync.clock` | control sample 使用 `source` 或 `receive` timestamp |
| `sync.max_control_age_ms` | 相机帧允许匹配的最大控制样本年龄 |
| `sync.poll_timeout_ms` | 每次等待相机帧的最长时间 |
| `sync.pending_frame_capacity` | 首个控制样本或等待匹配期间最多缓存的相机帧数 |

单相机配置继续使用 `camera:`。同时采集多个相机时改用 `cameras:`：

```yaml
cameras:
  - type: ros2
    name: head_rgb
    topic: /aima/hal/sensor/stereo_head_front_right/rgb_image/compressed
    node_name: robojudo_recorder_head_rgb
    qos_reliability: reliable
    qos_depth: 1
    ros_python_executable: /usr/bin/python3
    fps: 30
  - type: ros2
    name: wrist_rgb
    topic: /aima/hal/sensor/right_wrist/rgb_image/compressed
    node_name: robojudo_recorder_wrist_rgb
    qos_reliability: reliable
    qos_depth: 1
    ros_python_executable: /usr/bin/python3
    fps: 30
```

第一项是主相机，其 timestamp 用于匹配 control sample。每轮只有在所有相机都返回新 sequence 时才写入一条
dataset frame，因此所有视频的帧数与 Parquet 行数保持一致；某台相机缺帧时整组跳过。不能同时配置
`camera:` 和 `cameras:`。如果相机配置中存在 `fps`，它必须等于 `dataset.fps`。

## 相机 backend

### ROS 2

支持 `sensor_msgs/msg/CompressedImage`，包括 OpenCV 能解码的 JPEG/PNG 数据。推荐 sensor-data QoS：

```yaml
qos_reliability: reliable
qos_depth: 1
```

如果收不到图像，使用下面的命令检查 publisher QoS，并让 recorder 配置与其一致：

```bash
ros2 topic info -v /aima/hal/sensor/stereo_head_front_right/rgb_image/compressed
```

当前 ROS backend 不订阅原始 `sensor_msgs/msg/Image`。

### OpenCV

```yaml
camera:
  type: opencv
  name: head_rgb
  device: 0
  width: 640
  height: 480
  fps: 30
```

`device` 可以是 V4L2 index 或 OpenCV 支持的设备路径。

### RealSense

```yaml
camera:
  type: realsense
  name: head_rgb
  serial_number: ""
  width: 640
  height: 480
  fps: 30
```

`serial_number` 为空时使用第一个可用设备；多相机环境中应明确指定序列号。

### ZMQ

```yaml
camera:
  type: zmq
  name: head_rgb
  endpoint: tcp://127.0.0.1:8561
  encoding: auto
  timestamp_mode: receive
  fps: 30
```

publisher 需要发送 multipart message：

```text
[JSON header, image bytes]
```

header 支持 JSON 或 msgpack，至少包含 `sequence` 和 `timestamp_ns`。`encoding` 可设为 `auto`、
`raw_rgb` 或 `jpeg`；`auto` 从 header 读取 encoding 和 shape。跨机器时推荐
`timestamp_mode: receive`；只有 camera publisher 与 recorder 确认共享同一时钟域时才使用 source timestamp。

## 同步策略

每个相机帧会匹配同一 episode 中时间不晚于该帧的最新控制样本。如果样本年龄超过
`sync.max_control_age_ms`，该图像帧会被丢弃。相同控制样本不会重复写入多张图像。

一般建议：

- recorder、pipeline 和 ROS helper 在同一机器：`sync.clock: receive`。
- pipeline 与 recorder 跨机器：`sync.clock: receive`。
- 自定义 ZMQ camera 使用 source timestamp：只有所有生产者时钟通过 PTP/chrony 同步后才使用。

## 输出目录

输出符合 LeRobot v3 的主要结构：

```text
<dataset.root>/
├── data/chunk-000/file-000.parquet
├── videos/observation.images.head_rgb/chunk-000/file-000.mp4
├── videos/observation.images.wrist_rgb/chunk-000/file-000.mp4
└── meta/
    ├── info.json
    ├── stats.json
    ├── tasks.parquet
    └── episodes/chunk-000/file-000.parquet
```

每个 episode 使用独立的 data Parquet 和视频文件。超过 1000 个 episode 后会自动进入下一个 chunk。

### 继续已有 dataset

```yaml
dataset:
  resume: true
```

追加前会校验 LeRobot 版本、FPS、robot type、关节名称、action 名称，以及完整的 camera 名称和 shape。
任一 schema 不一致都会拒绝追加，避免在同一 dataset 中混入不兼容数据。

## 常见问题

### 如何确认 recorder 正在实际录制

Recorder 终端会按生命周期输出以下日志：

```text
Camera backend connected: name=head_rgb type=ros2
Camera stream ready: head_rgb=(1552, 2064, 3)
Episode 1 armed: pick up the box
Episode 1 recording first synchronized frame from cameras: head_rgb
Episode 1 recording progress: 150 frames (5.0 s dataset time)
Episode 1 saved: 300 frames, 0 stale frames dropped, root=record_data/example
```

`Camera backend connected` 只说明 backend/helper 已启动；必须出现 `Camera stream ready` 才表示已收到图像。
如果持续没有图像，会每 5 秒输出 `Waiting for camera frames: ...`。episode 结束时没有任何可配对帧，则明确输出
`was not saved because no synchronized camera/control frames were recorded`。

### `Recorder unavailable or saturated; dropped ... samples`

pipeline 没有 recorder 连接，或 recorder 消费速度不足。确认 recorder 已启动且双方 endpoint 一致。持续从
第一帧开始增长通常表示 recorder service 尚未连接，而不是相机本身失败。

### `ROS 2 camera helper exited with status ...`

ROS Python helper 启动失败。先验证系统 Python：

```bash
/usr/bin/python3 -c "import rclpy, zmq; from sensor_msgs.msg import CompressedImage"
```

同时确认已经 source 正确的 ROS setup，并检查 `ros_python_executable` 和 `ROS_DOMAIN_ID`。

### recorder 已连接但没有输出文件

确认已经进入 `RL_DEFAULT`、启用上半身 takeover，并用 joystick 开始录制。没有 fresh teleop/control sample
时，相机帧不会单独写入 dataset。

### ROS topic 存在但没有图像

检查 topic message type、QoS reliability、ROS domain 和 namespace。该 backend 要求
`sensor_msgs/msg/CompressedImage`，不是 `sensor_msgs/msg/Image`。

### `dataset root is not empty`

选择新的 `dataset.root`，或者在确认 schema 相同后设置 `dataset.resume: true`。

### encoder 不可用

如果 `libx264` 不可用，检查当前 PyAV/FFmpeg build 提供的 encoder，或者修改 `dataset.codec`。

## 测试

```bash
conda activate robop
python -m unittest discover -s packages/robojudo_recorder/tests -v
```

ROS backend 测试通过 mock 验证 helper 启动、ZMQ transport、CompressedImage 解码、RGB 转换和尺寸校验；
真实机器人部署前仍应使用目标 ROS topic 做一次短 episode 验证。

## 扩展新的相机

实现 `CameraSource` 的 `shape`、`connect()`、`read()` 和 `close()`，然后注册 factory：

```python
from robojudo_recorder.cameras import register_camera
from robojudo_recorder.cameras.base import CameraSource


@register_camera("my_camera")
class MyCameraSource(CameraSource):
    ...
```

最后在 `robojudo_recorder/cameras/__init__.py` 中导入 backend 模块。dataset writer 和 RoboJuDo pipeline 不需要
针对具体相机修改。
