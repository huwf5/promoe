# Scheduler-Aware Eviction 与 Decoder Warmup Overlap 设计文档

- 日期：2026-05-13
- 范围：在前三个任务已实现的 deterministic initial cache、encoder/decoder metadata、global cache 基础上，增加 scheduler-aware eviction、phase-aware fetch scheduler、decoder hot expert warmup overlap。
- 不在范围：实现 encoder predictor、重写多 stream/multi-inflight I/O、替换现有 `ExpertStatus` 并发状态机、引入 SIDA 完整多索引策略的所有 ablation 变体。

## 1. 背景

当前 `src` 已经具备：

- `ModuleMeta::is_encoder_layer()` / `is_decoder_layer()`。
- `per_layer_cache=false` 的 global cache。
- 每次 `generate` 前 deterministic reset，并通过 manual 或 hot expert plan 同步加载初始 cache。
- `initial_hot_expert_file` 可由 Python 侧解析 hot expert JSON，并生成 `initial_expert_plan` 给 C++。

现有不足是：cache policy 仍然是普通 LRU/FIFO/NN/MIN，不知道 encoder expert 在 encoder forward 结束后已经不再需要；scheduler 仍是 precise queue 优先、再按 layer id 扫 prefetch queue；encoder 阶段的 I/O 空闲时间不能用来预热 decoder hot expert。

本设计借鉴 SIDA 的 `SchedulerAwareEvictionPolicy` 与三阶段 I/O，但保持实现贴合当前 C++ 架构：不重写线程模型，不引入新的 active urgent 状态集合，继续依赖现有 `ExpertStatus` 保证物理 buffer 安全。

## 2. 目标

1. 新增 `cache_policy=scheduler_aware`，区分 encoder/decoder expert。
2. encoder expert 在明确不需要后标记为 reclaimable，并优先作为 victim。
3. 普通 demand / predictor prefetch 可在 reclaimable 不足时继续走 encoder-first victim 兜底。
4. `decoder_warmup_overlap` 只能使用 reclaimable encoder slot；没有 reclaimable 时跳过，不驱逐普通 encoder 或 decoder。
5. Fetch scheduler 增加 phase：
   - encoder phase：precise 最高优先级，普通 encoder/near-future prefetch 次之，普通队列空时才做 decoder warmup overlap。
   - decoder warmup overlap：作为 encoder phase 的空闲填充，不抢普通 prefetch。
   - decoder predictor phase：真实 decoder forward 开始后，清空 warmup overlap 队列，继续使用现有 decoder predictor prefetch。
6. decoder warmup overlap 的 hot expert 来源复用 `initial_hot_expert_file`。
7. decoder warmup plan 排除 initial cache 中已有 expert，并按每层第 1 热、第 2 热、第 3 热的 depth-interleave 顺序排列。

## 3. Scheduler-Aware Eviction

新增 `CachePolicySchedulerAware`，注册名：

```text
cache_policy = "scheduler_aware"
```

第一版使用简化索引：

```text
global_lru
encoder_lru
decoder_lru
reclaimable_encoder
```

`access_on_miss()` 与 `access_on_hit()` 均更新 LRU。encoder expert 进入 `encoder_lru`，decoder expert 进入 `decoder_lru`，所有 expert 同时进入 `global_lru`。`evict()` 从所有索引移除 victim。

新增策略信号：

```cpp
mark_reclaimable(ExpertHandler* expert);
mark_layer_reclaimable(int layer_idx);
mark_layer_reclaimable_except(int layer_idx, const std::unordered_set<int>& needed_eids);
has_reclaimable_encoder() const;
```

victim 优先级：

```text
普通 demand / predictor prefetch:
  1. reclaimable_encoder 中仍在 cache 的 expert
  2. encoder_lru 中仍在 cache 的 expert
  3. decoder_lru 中仍在 cache 的 expert
  4. global_lru 兜底

decoder_warmup_overlap:
  1. reclaimable_encoder 中仍在 cache 的 expert
  2. 没有 victim，返回 nullptr，跳过 overlap
```

`reclaimable` 是调度语义，不代表立即覆盖 buffer。实际驱逐和等待仍走现有 `CacheMngr::miss()` / `ExpertStatus` 逻辑：

- `kReady` victim 可立即复用。
- `kLaunching` / `kUsing` victim 由现有 wait lambda 等待。
- `kFetching` victim 按现有逻辑取消部分 fetch。

## 4. Reclaimable 标记时机

使用现有 hook 事件，不新增 Python forward 逻辑。

### `report_one_layer(layer, experts)`

当 `layer` 是 encoder layer 时，router 已经给出本层真实需要的 expert 集合。此时：

```text
mark_layer_reclaimable_except(layer, needed_eids)
```

即本层当前 cache 中、但不在 `needed_eids` 中的 encoder expert 立刻变为 reclaimable。

### `one_expert_done(layer, expert)`

当 `layer` 是 encoder layer 时，该 expert 真实执行已经结束：

```text
mark_reclaimable(layer, expert)
```

### `one_moe_layer_done(layer)`

当 encoder layer 完成后，该层后续不会再执行：

```text
mark_layer_reclaimable(layer)
```

decoder layer 不因为这些事件变 reclaimable。

## 5. Phase-Aware Scheduler

新增 scheduler phase：

```cpp
enum FetchSchedulerPhase {
  kEncoderPhase,
  kDecoderPredictorPhase,
};
```

`decoder_warmup_overlap` 不作为长期 phase 存储，而是 encoder phase 中普通队列为空时的一类 task。

phase 切换：

```text
reset_and_load_initial_cache()
  -> scheduler.start_generation(...)
  -> phase = kEncoderPhase
  -> rebuild decoder warmup queue

advance_actual_layer(generation, layer)
  -> if layer is decoder:
       phase = kDecoderPredictorPhase
       clear decoder warmup queue
```

出队优先级：

```text
1. precise_job_queue
2. normal prefetch queue，按 current/near-future layer 优先
3. encoder phase 且 normal queue 为空时，尝试 decoder warmup overlap
4. idle
```

普通 prefetch queue 仍使用现有 `per_layer_job_queues`，但 `pop_next_task()` 不再简单从 layer 0 扫到 `num_layer-1`。它根据 `current_layer` 计算优先级：

```text
当前层 highest
未来层按距离升序
过去层最低
未知 current_layer 时按 layer id 升序
```

这为未来 encoder predictor 留好位置：encoder predictor 只要继续提交普通 `PrefetchLayerTask`，自然优先于 decoder warmup overlap。

## 6. Decoder Warmup Overlap Plan

计划来源复用 `initial_hot_expert_file`，不新增单独 hot 文件参数。

Python 侧在 `inject_model()` 中，如果启用 decoder warmup overlap 且提供 `initial_hot_expert_file`：

1. 读取同一个 hot expert JSON。
2. 提取 decoder hot experts。
3. 排除已经在 initial cache plan 中的 `(layer, expert)`。
4. 按 depth-interleave 排序：

```text
decoder layer 0 rank 0
decoder layer 1 rank 0
...
decoder layer N rank 0
decoder layer 0 rank 1
decoder layer 1 rank 1
...
```

5. 格式化为 C++ 已熟悉的字符串：

```text
decoder_warmup_expert_plan = "6:3,7:1,8:9,..."
```

C++ scheduler 在 generation reset 后解析该 plan，填入 `decoder_warmup_queue`。运行时提交规则：

- 如果 expert 已在 cache，跳过。
- 如果 policy 没有 reclaimable encoder，暂停本轮 overlap。
- 如果 scheduler 已进入 decoder predictor phase，清空剩余 warmup queue。
- 如果 `CacheMngr::miss(..., kDecoderWarmupOverlap)` 返回 skip，则不产生 CopyTask。

## 7. 配置

新增配置：

```text
cache_policy = "scheduler_aware"
enable_decoder_warmup_overlap = true/false
decoder_warmup_expert_plan = "layer:expert,..."
```

Python CLI 可暴露：

```text
--enable_decoder_warmup_overlap True
```

`decoder_warmup_expert_plan` 由 Python 内部从 `initial_hot_expert_file` 生成，不要求用户手写。

## 8. 指标与调试

第一版至少输出或可通过日志/NVTX 观察：

- reclaimable 标记次数。
- scheduler-aware eviction victim 类型：reclaimable encoder / encoder / decoder / global fallback。
- decoder warmup overlap submit 次数。
- decoder warmup overlap skip 原因：
  - no reclaimable encoder
  - already in cache
  - decoder phase started
  - stale generation

这些指标用于确认策略行为，不作为功能正确性的唯一依据。

## 9. 验收标准

1. `cache_policy=scheduler_aware` 可启动，并在 global cache 下运行。
2. encoder layer router 后，本层未使用但在 cache 中的 encoder expert 被标记 reclaimable。
3. encoder expert done 后，该 expert 被标记 reclaimable。
4. encoder layer done 后，该层 cache 中 encoder expert 都被标记 reclaimable。
5. 普通 miss 优先逐出 reclaimable encoder，再逐出 encoder，最后才逐出 decoder。
6. decoder warmup overlap 没有 reclaimable encoder 时跳过，不逐出 decoder 或普通 encoder。
7. decoder warmup plan 复用 `initial_hot_expert_file`，排除 initial cache 已有 expert，并按 depth-interleave 排序。
8. 进入 decoder predictor phase 后，warmup overlap queue 被清空，现有 decoder predictor prefetch 继续工作。
9. 不启用新配置时，现有 LRU/FIFO/NN/MIN 行为保持不变。
