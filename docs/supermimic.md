# SuperMimic G1 部署

支持 29DoF / 23DoF 仿真，直接复用 `MujocoEnv`。策略在 `supermimic_policy.py` 中用
`torch.load(..., weights_only=True)` 加载 checkpoint，重建推理网络并严格加载 state dict。
运行时不需要 `supermimic_mjlab`、mjlab、世界模型、critic 或训练 replay。

## 输入与权重

| 策略 | checkpoint | 实际推理输入 |
| --- | --- | --- |
| 29DoF motion tracking | `supermimic_actor`，或 `supermimic_online` 完整训练断点 | `10 × 99` 本体历史 + `2680` 参考特征 |
| 23DoF motion tracking | `supermimic_actor`，或 `supermimic_online` 完整训练断点 | `10 × 75` 本体历史 + `2560` 参考特征 |
| 29DoF velocity tracking | `supermimic_velocity_actor` 独立推理导出 | `10 × 93` 本体历史 + `(vx, vy, wz)` |
| 23DoF velocity tracking | `supermimic_velocity_actor` 独立推理导出 | `10 × 75` 本体历史 + `(vx, vy, wz)` |

29DoF motion 使用 `full_body_268`：10 帧、14 个 body 的参考特征，共 2680 维；
本体观测包含 `motion_anchor_ori_b`，每帧 99 维。要求 `relative_orientation=False`。
每帧顺序为相对关节位置、关节速度、root 局部角速度、投影重力、6D torso 朝向 anchor、上一动作。
Anchor 采用 `[R00, R01, R10, R11, R20, R21]` 排列；参考特征中的 6D 朝向采用按列拼接，二者不能混用。

23DoF motion 使用 `full_body_256_v1`，10 帧共 2560 维参考特征。
本次权重未启用 `motion_anchor_ori_b`，本体观测每帧 75 维；配置必须匹配权重的 anchor 开关。
关节、网络、动作和 NPZ 均为原生 23DoF，不通过丢弃 29DoF 动作适配。
参考中的腕部跟踪 body 为 `left_wrist_roll_rubber_hand` / `right_wrist_roll_rubber_hand`。

Velocity 配置的 `reference_dim` 描述训练阶段 motion 参考接口：29DoF 配置为 730，
本次上传的 23DoF checkpoint 为 2560。速度 actor 本身只消费本体历史和速度指令，
不需要参考动作、motion anchor 或 phase；也兼容训练元数据保留 `reference_dim=2680` 的 29DoF 速度导出。
必须使用不含 anchor、匹配关节数的速度权重，不能直接使用 motion decoder 代替。
`vx/vy` 为 yaw-only heading 坐标系速度（m/s），`wz` 为世界角速度 z 分量（rad/s）。
指令在网络内除以 checkpoint 保存的 command scale，外部不重复归一化。

两类策略都保留 checkpoint 中的观测归一化统计量。历史按时间从旧到新排列，首次用当前观测填满；
上一动作保存裁剪后的归一化动作。默认关闭动作裁剪和额外平滑，与原推理保持一致。
关节顺序、默认姿态来自 checkpoint；配置中的 PD 和逐关节 action scale 按名称映射到该顺序。

## 启动

```bash
conda activate robop

taskset -c 0-7,10-31 python scripts/run_pipeline.py -c g1_supermimic_velocity

taskset -c 0-7,10-31 python scripts/run_pipeline.py -c g1_supermimic_motion
```

在 `robojudo/config/g1/g1_supermimic_cfg.py` 中通过 policy 的 `motion_file` 指定单个转换后的 SuperMimic NPZ。
文件需要包含 50 Hz、`wxyz`、匹配机器人关节和跟踪 body 的动作数据，加载时会检查形状和有限值。
动作文件可以位于任意目录、使用任意文件名；新动作的跟踪效果需要实际仿真验证。
Checkpoint 和动作路径统一在 `robojudo/config/g1/g1_supermimic_cfg.py` 配置：
`_SUPERMIMIC_DIR` 指向权重所在目录，`_MOTION_CHECKPOINT`、`_VELOCITY_CHECKPOINT`、
`_MOTION23_CHECKPOINT`、`_VELOCITY23_CHECKPOINT` 分别指定 29DoF motion、29DoF velocity、
23DoF motion 和 23DoF velocity 权重。
默认 23DoF 权重使用 `assets/models/g1/supermimic/g1_23dof/` 下的
`policy_best.pt` 和 `velocity_policy_last.pt`；
29DoF 权重和动作文件默认使用当前仓库同级 `supermimic_mjlab` 下的对应文件。
`_MOTION_FILE`、`_MOTION23_FILE` 分别指定 29DoF 和 23DoF 动作文件。
每个入口通过 policy 的 `checkpoint=`、`motion_file=` 显式选择权重和动作。

速度入口默认用键盘：`W/S` 前后、`A/D` 左右、`Q/E` 左右转；松开按键恢复零指令。
键盘默认幅值为 `(0.5, 0.3, 0.5)`，指令绝对值上限为 `(1.0, 0.5, 1.0)`；
可在 policy config 的 `keyboard_command` 和 `command_limits` 修改。
Policy 也支持现有手柄和 `VelocityZmqCtrl`，使用现有速度源仲裁；ZMQ 输入为物理单位。

启用 anchor 的 motion 在首帧将参考 torso 的 yaw 对齐当前机器人 torso，后续保持该偏移。
未来参考包含当前帧起的 10 帧，结尾复制最后一帧。播放结束通知现有 pipeline 执行退出过渡；
需要循环播放时设置 `loop_motion=True`。重置会清空历史和参考朝向对齐。
现有键盘 `Ctrl+Shift+|` 可重启 motion，`Ctrl+Shift+>` 可退出；Escape 的退出行为沿用原 pipeline。

## Loco / Mimic 切换

```bash
taskset -c 0-7,10-31 python scripts/run_pipeline.py -c g1_supermimic_velocity_locomimic
```

默认进入 velocity tracking；`W/S`、`A/D`、`Q/E` 控制速度。
释放 `[` 切换到 motion tracking，释放 `]` 返回 velocity tracking。
Motion 播放结束后自动返回 loco；切换过渡期间再次切换会被忽略。
切入 motion 时重置播放状态和参考朝向对齐，返回 loco 时重置其历史。
两类策略均控制全身 29DoF，不启用上身姿态覆盖。

配置中的 `mimic_policies` 可加入多个 motion 配置；在 loco 状态下使用 `;` / `'` 选择下一项 / 上一项。
每个 mimic 配置通过 `checkpoint` 指定权重、`motion_file` 指定动作。

## 23DoF Loco / Mimic

默认动作文件为 `assets/motions/g1/supermimic/g1_23dof/General_A5_-_Pick_Up_Box_stageii.npz`，
由配置中的 `_MOTION23_FILE` 统一指定。

两种策略均使用 SuperMimic，loco 加载 `velocity_policy_last.pt`，mimic 加载 `policy_best.pt`：

```bash
taskset -c 0-7,10-31 python scripts/run_pipeline.py -c g1_23_supermimic_velocity_locomimic
```

默认进入速度控制；释放 `[` 切入 motion，释放 `]` 返回速度控制，动作结束自动返回 loco。
两种策略均控制完整 23 个关节，不启用上身姿态覆盖。

独立 velocity tracking：

```bash
taskset -c 0-7,10-31 python scripts/run_pipeline.py -c g1_23_supermimic_velocity
```

独立 motion tracking：

```bash
taskset -c 0-7,10-31 python scripts/run_pipeline.py -c g1_23_supermimic_motion
```

现有 Locomanipulation 策略与 23DoF SuperMimic 切换：

```bash
taskset -c 0-7,10-31 python scripts/run_pipeline.py -c g1_23_maniloco_supermimic
```

按键与 29DoF 切换入口一致；修改配置中的 `motion_file` 选择要跟踪的动作。
默认 loco 为仓库现有 `G1Locomanipulation23PolicyCfg`，使用
`assets/models/g1/locomanipulation/policy_23dof_stiff.onnx` 及配套的 `stiff` PD 预设。
它观察全部 23 个关节，输出 12 个腿部关节加腰部 yaw 的 13 维动作；
双臂的其余 10 个关节由现有切换框架保持/插值，腰部输出保留给 loco 策略。
进入 mimic 后全部 23 个关节由 SuperMimic 控制，PD 和默认姿态随策略切换。

可以在 `g1_23_maniloco_supermimic.loco_policy` 替换其他兼容该 23DoF 机器人的策略配置。
如果替换成控制全身的 loco，需要同时设置 `upper_dof_num=0`、`upper_dof_pos_default=None`、
`upper_dof_override_indices=[]`，以保留其全身输出。
这两个 23DoF 仿真配置直接继承 `RlLocoMimicPipelineCfg`，各自独立配置环境、控制器、策略与上身覆盖，
无需通过 29DoF 配置修改参数。

## 23DoF 真机配置

真机入口分别为 `g1_23_supermimic_velocity_locomimic_real` 和 `g1_23_maniloco_supermimic_real`，
继承各自的 23DoF 仿真配置，checkpoint 和 motion 路径沿用同一文件中的设置。
前者由 SuperMimic 控制完整 23 个关节；后者使用 `G1Locomanipulation23PolicyCfg` 的 stiff 预设作为 loco，
在 loco 状态下支持双臂 ZMQ 控制，在 mimic 状态下由 SuperMimic 接管全身。

两者复用现有 `G1LocomanipulationLocoMimicPipeline` 的四模式与切换逻辑，
以及 `G1_23RealEnvCfg` 的逻辑 23DoF → Unitree 29 槽电机映射。
配置保留 `do_safety_check=True`、0.1 秒指令/状态超时和 5.0 的退出阻尼。
在各自真机配置的 `env.unitree.net_if` 中填写机器人网卡名称，两个入口需分别配置。

启动后处于 PASSIVE 模式，按遥控器 `Y` 执行 1.5 秒默认姿态插值，完成后按 `X` 才能进入 RL loco。
`A` 返回 PASSIVE，`B` 进入 DAMPING；`Start` 切入 mimic，`Select` 返回 loco，
`R1` / `L1` 选择下一项 / 上一项 mimic，`L1+R1+A` 退出。
Maniloco 入口另外用 `L2` 开关双臂 ZMQ 控制；切入 mimic 时自动关闭该覆盖。
真机运行前需完成模型和动作的仿真验证，并由现场人员支撑机器人完成默认姿态过渡。

## 仿真配置

仿真入口使用原有 `MujocoEnv` 的显式 PD 力矩控制，控制 50 Hz、物理 200 Hz。
PD、力矩上限和 `0.25 × effort / kp` 动作缩放来自训练预设；
其余动力学和接触设置沿用现有机器人 XML 与环境实现。
CPU MuJoCo 与训练端 Warp 后端仍需分别评估，接口一致不代表轨迹逐位一致。

加载时严格检查 checkpoint 与配置的关节集合和参考 schema，拒绝 23DoF / 29DoF 交叉加载。
