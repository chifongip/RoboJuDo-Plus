## Offline DAgger 数据与接管审计

  ### 1. 已修复：expert segment 不再制造“假的 episode 结束”

  RLInf 保存完整成功 episode，然后只把满足以下条件的 action chunk 起点暴露给训练器：

  从 t 开始的所有非 padding action
  都必须满足 intervene_flag = true

  也就是：

  valid(t) = all(expert_applied[t : t + H])

  其中 H 是模型 action horizon。RLInf 明确按 chunk 过滤，而不是仅过滤单帧。RLInf HG-DAgger 文档

  旧 finalizer 会把每个 expert intervention 切成一个新的 LeRobot episode：

  policy | expert expert expert | policy
           └── synthetic episode ──┘

  这样 intervention 结束附近的样本会被训练框架视为“自然 episode 结束”，未来 action 不足时通常复制最后一个 action/padding。原本应该因为后
  续进入 policy 而被拒绝的 chunk，可能变成：

  expert_8, expert_9, expert_9, expert_9, ...

  这是目前最大的 action-horizon 问题。expert_min_frames=2 也与真实 H 无关。

  当前实现：

  - raw 层继续保存完整 rollout；
  - finalized 层同样保留完整真实 episode；
  - 每帧保留 `expert_applied` 等 DAgger 标签；
  - Select 释放不再触发 `save_episode()`，因此不会产生伪 terminal。

  训练侧仍需要针对模型的 action horizon 生成 `valid(t)`；这是后续训练采样器的职责，不由 finalizer
  通过裁切 episode 代替。

  相关当前代码：/home/breeze/Desktop/workplace/Humanoid/RoboJuDo-Plus/packages/robojudo_recorder/src/robojudo_recorder/finalize.py:209

  ### 2. 已修复：最新 expert frame 无效时立即撤销旧 action

  当前收到：

  expert_valid = false

  时，controller 现在会在确认 frame ID 属于当前 session 且更新后，立即清除上一个 expert target，保留新的
  `expert_frame_id` 用于诊断，并在下一控制 tick 将 `expert_applied=false`。随后按现有安全逻辑回到 policy；如果
  policy 也不 fresh，则保持现有 fail-closed 行为。

  因此可能出现：

  frame 100: IK valid   -> q100
  frame 101: IK invalid -> 仍执行并记录 q100
  frame 102: IK invalid -> 仍可能执行 q100

  ### 3. observation/action 时间对齐策略

  Offline DAgger 不强制引入严格 observation ID join。它与普通数据使用相同的固定 FPS 重采样语义：相机按
  阈值选择，state 插值，action 使用当时生效的零阶保持关节目标。raw 层仍保留时间戳和来源信息用于诊断。


  ### 4. 已修复：最终数据保留 DAgger 标签

  label-aware LeRobot schema 现在逐帧写入：

  - 哪个 intervention session；
  - 对应哪个 expert frame；
  - Select 到真正 expert applied 的延迟；
  - 是否重复使用同一 expert action；
  - 数据来自第几轮 DAgger、哪个 policy checkpoint。

  expert_intervention
  expert_applied
  action_source
  intervention_session
  expert_frame_id

  为避免破坏已有无标签 LeRobot dataset 的固定 schema，offline DAgger YAML 默认写入独立的 `_dagger`
  dataset；训练时再与基础 dataset 聚合。

  ## P1：影响数据质量和后续扩展

  ### 5. 没有 success/failure、abort 和任务结果过滤

  RLInf 默认可配置：

  online_lerobot:
    only_success: true

  只有完整成功且自然终止的 rollout 进入训练 archive；失败 episode 可以丢弃或独立保存。RLInf episode 处理

  当前只能根据 recorder start/pause/stop 推断数据是否可用，没有明确：

  episode_success
  episode_failure
  operator_abort
  natural_terminal

  这意味着失败后的无效接管、任务已经不可恢复时的动作，也可能被加入原数据集。

  对于 offline DAgger，可以不强制只留成功，但必须记录 outcome，训练时再选择策略。

  ### 6. 已修复：action_source=expert 覆盖完整训练 action

  expert 接管现在原子覆盖：

  - 双臂关节目标；
  - 双手关节目标；
  - 本机 joystick 生成的四维 `[vx, vy, yaw_rate, height]`。

  dex-teleop 的 arm/hand payload 与 joystick intervention gate 任一无效时，`expert_applied=false`，所有维度一起
  回到 policy/hold。腰部 command 固定为训练默认值，不进入 recorder 的四维 locomotion action。

  ### 7. 无法验证 policy action horizon 的 handback 是否新鲜

  RLInf 的 smooth_intervene 在连续接管跨越 action chunk 时跳过 policy 推理，释放后重新进入正常 inference，从而清晰定义 chunk 边界。
  smooth intervention 说明

  当前 policy stream 在 expert 接管期间继续运行，这并不一定有问题，甚至可能让 policy 持续看到 expert 驱动后的最新状态。但当前没有记录：

  policy_chunk_id
  policy_horizon_index
  policy_source_observation_sequence
  policy_generated_timestamp_ns

  所以无法判断释放 Select 时恢复的是：

  - 基于最新 expert 后状态生成的动作；
  - 还是接管前/接管中的旧 horizon 元素。

  这对 offline expert 数据不是阻塞项，但对 rollout 质量和以后 online DAgger 是明显缺口。

  ### 8. 缺少正式的数据聚合策略

  当前配置直接：

  resume: true
  root: 原始数据集路径

  这只是物理追加，还没有处理：

  - DAgger round 和 policy checkpoint provenance；
  - 原始 demonstration 与 correction 的采样比例；
  - expert corrections 数量过少导致被原数据淹没；
  - 重算 normalization statistics；
  - 重复 episode/frame；
  - OpenPI 和 GR00T 不同 action horizon 下的 valid-start 数量。

  因此“把 expert data 加进旧 dataset 一起训练”在算法概念上没错，但工程上还需要一个 aggregation manifest 和训练采样策略。否则 correction
  占比可能太低，模型几乎学不到。

  ## P2：可以等 online DAgger 再做

  这些是 RLInf 已有、但本阶段可以明确不实现的：

  - actor 在线训练；
  - rolling LeRobot dataset；
    -异步接收 episode；
    -训练 readiness gate；
    -模型权重同步；
    -训练/采集并发；
    -最近 N 个 expert-valid 样本窗口；
    -在线 loss、gradient norm、logical sample 指标。

  这些不是当前 offline 阶段的缺陷，而是有意缩小范围。

  ## 推荐修改顺序

  1. 修复 invalid expert frame 仍复用旧 action。
  2. 不再把 intervention release 直接伪造成 episode terminal。
  3. 按模型 action horizon 构造 all-expert chunk-start mask。
  4. 补齐 expert frame、observation sequence、state/action 时间戳。
  5. 真正丢弃超龄 control/camera 样本。
  6. 保留完整 rollout、DAgger 标签和 outcome；另生成训练 view。
  7. 最后再补 dataset aggregation、采样比例和 normalization 工具。

  做到前五项后，当前系统就可以作为比较可靠的 offline DAgger pipeline 使用；Actor、在线训练和权重同步可以继续留到下一阶段。
