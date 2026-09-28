## P0：建议 offline DAgger 使用前修复

  ### 1. 当前 expert segment 会制造“假的 episode 结束”

  RLInf 保存完整成功 episode，然后只把满足以下条件的 action chunk 起点暴露给训练器：

  从 t 开始的所有非 padding action
  都必须满足 intervene_flag = true

  也就是：

  valid(t) = all(expert_applied[t : t + H])

  其中 H 是模型 action horizon。RLInf 明确按 chunk 过滤，而不是仅过滤单帧。RLInf HG-DAgger 文档

  当前 finalizer 则把每个 expert intervention 切成一个新的 LeRobot episode：

  policy | expert expert expert | policy
           └── synthetic episode ──┘

  这样 intervention 结束附近的样本会被训练框架视为“自然 episode 结束”，未来 action 不足时通常复制最后一个 action/padding。原本应该因为后
  续进入 policy 而被拒绝的 chunk，可能变成：

  expert_8, expert_9, expert_9, expert_9, ...

  这是目前最大的 action-horizon 问题。expert_min_frames=2 也与真实 H 无关。

  建议：

  - raw 层继续保存完整 rollout；
  - finalized 层也保留完整 episode 和 expert_applied；
  - 为每种模型/action horizon 生成 valid chunk-start index；
  - 或者如果必须导出 expert-only synthetic episode，至少裁掉 intervention 末尾 H-1 个训练起点，而不能把释放 Select 当自然终止。

  相关当前代码：/home/breeze/Desktop/workplace/Humanoid/RoboJuDo-Plus/packages/robojudo_recorder/src/robojudo_recorder/finalize.py:209

  ### 2. 最新 expert frame 无效时，旧 action 仍可能继续生效

  当前收到：

  expert_valid = false

  时不会立即清除上一个有效 expert target。旧目标会继续保持 expert_applied=true，直到 expert_timeout_s 到期。

  因此可能出现：

  frame 100: IK valid   -> q100
  frame 101: IK invalid -> 仍执行并记录 q100
  frame 102: IK invalid -> 仍可能执行 q100

  注释写的是“invalid IK frames must not keep an old action fresh”，但实现只是“不刷新时间”，没有立即撤销旧 action。

  建议在收到当前 session、更新 frame ID、但 expert_valid=false 的消息时：

  - 更新 latest_expert_frame_id；
  - 立即将当前 candidate 标记无效；
  - expert_applied=false；
  - 根据安全策略选择 hold 或 policy，而不是继续把旧目标标成新 expert 数据。

  相关代码：/home/breeze/Desktop/workplace/Humanoid/RoboJuDo-Plus/robojudo/controller/gr00t_zmq_ctrl.py:728

  ### 3. observation/action 还没有严格的因果配对

  RLInf 的环境接口天然保存：

  obs_t -> 实际执行 action_t -> next_obs
  RLInf 的保证主要来自“同一个 env.step 调用内记录实际执行动作”，而不是事后靠时间戳同步。记录的是同一次 step 中实际选择执行的动作，不需要事后在两个异步 ZMQ 流之间猜测 action 来源。

  当前 recorder 是：

  读取 state
  -> 选择并发送 action
  -> post-step 生成一个 timestamp
  -> finalizer 用 nearest camera 配对


  ### 4. 当前最终数据没有保留 DAgger 标签

  四个字段只存在 raw 数据中。finalizer 用它们过滤后，写入 LeRobot 的只有 state/action/image。

  结果是最终数据无法审计：

  - 哪个 intervention session；
  - 对应哪个 expert frame；
  - Select 到真正 expert applied 的延迟；
  - 是否重复使用同一 expert action；
  - 数据来自第几轮 DAgger、哪个 policy checkpoint。

  RLInf 保留完整 episode 和 intervene_flag，训练阶段再建立逻辑样本索引。当前也应至少在最终 archive 中保留：

  expert_intervention
  expert_applied
  action_source
  intervention_session
  expert_frame_id

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

  ### 6. action_source=expert 实际是混合来源

  当前 expert 只覆盖：

  - 双臂关节目标；
  - 手部目标。

  但 locomotion command 仍来自 policy，最终 action 又把四维 locomotion command 一起写入：

  [arm joints, hand joints, policy locomotion]

  因此整帧标为 action_source=expert 不等于所有 action 维度都是 expert。

  RLInf 双臂也允许部分接管：只接管左臂时，右臂仍保留 policy action，但其文档明确把它定义为组合 action。RLInf 双臂 DAgger

  建议将来源细化为：

  arm_source
  hand_source
  locomotion_source

  或者：

  expert_action_mask[action_dim]

  当前阶段如果明确只学习 upper-body，可以暂时接受混合 action，但不能把全维度都解释成 expert label。

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