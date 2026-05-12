# Deterministic Initial Cache 设计文档

- 日期：2026-05-13
- 范围：为 `src` 的线上 cache/prefetch 路径增加 deterministic generate-start 初始化、encoder/decoder stage metadata、global cache 下的手写 initial cache plan。
- 不在范围：SchedulerAware eviction、三阶段 I/O、decoder warmup overlap、hot expert 文件、显式 expert id 列表。

## 1. 背景

当前 `src` 初始化 GPU cache 时只分配 cache line，`prefetched_experts` 初始为空。运行时由 demand miss 和 predictor prefetch 按需填充 cache。这个行为适合普通运行，但不利于实验复现：连续两次 `generate` 可能从不同 cache 内容或残留 policy 状态开始。

本设计把每次 `generate` 开始前的状态固定下来：先清理上一轮运行态，再按手写配置同步加载一组固定 expert。这样后续任务可以在同一个初始 cache 状态上比较 eviction 和 I/O 策略。

## 2. 目标

1. 每次 `generate` 都从同一个 deterministic cache session 开始。
2. 初始 cache 不是空的，而是由手写 plan 填满。
3. 第一版使用 global cache，即 `per_layer_cache=false`。
4. C++ 层能判断 MoE layer 属于 encoder 还是 decoder。
5. initial plan 可以包含 encoder 和 decoder expert。
6. plan 大小必须等于 global cache size；不补齐、不截断。
7. 第一版不引入 `active_urgent` 集合，实际驱逐安全继续交给现有 `ExpertStatus` 和 `CacheMngr::miss` 状态机。

## 3. Generate 生命周期

Python 注入层在模型上包住 `generate`。每次调用 `model.generate(...)` 时，先执行：

```text
reset_and_load_initial_cache()
  -> clear scheduler queues and advance generation
  -> clear cache contents and expert runtime state
  -> reset cache policy state
  -> build manual initial cache plan
  -> synchronously load planned experts
  -> verify all planned experts are ready
  -> call original generate
```

选择 Python 侧入口是因为 C++ hook 通常在 layer/logits forward 时才被触发；那时已经晚于“forward 前 cache 必须 ready”的要求。

## 4. Stage Metadata

`ModuleMeta` 增加 encoder/decoder MoE 层数量：

```cpp
int num_encoder_moe_layer = 0;
int num_decoder_moe_layer = 0;
```

并提供 helper：

```cpp
int first_decoder_layer() const;
bool is_encoder_layer(int layer_idx) const;
bool is_decoder_layer(int layer_idx) const;
```

Switch adapter 写入：

```text
num_encoder_moe_layer = num_encoder_sparse_layers
num_decoder_moe_layer = num_decoder_sparse_layers
```

层编号保持现状：

```text
encoder global layer: 0 ... E-1
decoder global layer: E ... E+D-1
```

默认 adapter 写入：

```text
num_encoder_moe_layer = 0
num_decoder_moe_layer = num_moe_layer
```

这让非 Switch 模型继续按 general MoE 路径运行，不触发 encoder-first 特殊语义。

## 5. Manual Initial Plan

第一版只支持手写 layer budget 和 sequential expert 顺序：

```text
initial_cache_policy = "manual"
initial_layer_budgets = "0:40,1:10,12:8"
initial_expert_order = "sequential"
reset_cache_on_generate_start = true
initial_cache_ready_barrier = true
per_layer_cache = false
```

计划生成规则：

```text
0:40  -> (layer 0, expert 0..39)
1:10  -> (layer 1, expert 0..9)
12:8  -> (layer 12, expert 0..7)
```

校验规则：

- layer id 必须满足 `0 <= layer < num_layer`。
- budget 必须满足 `0 <= budget <= num_expert`。
- 同一个 layer 不能重复配置。
- `sum(initial_layer_budgets) == cache_size`，否则报错。
- plan 可以包含 decoder layer。
- 第一版不支持 hot expert 文件。
- 第一版不支持显式 expert id 列表。

如果 plan size 小于 cache size，错误信息必须包含：

```text
cache_size
plan_size
missing
initial_layer_budgets
```

如果 plan size 大于 cache size，错误信息必须包含：

```text
cache_size
plan_size
overflow
initial_layer_budgets
```

## 6. Cache 与状态语义

初始化后：

- initial encoder expert 是 `Ready`，进入当前 cache policy 的普通状态，不是 reclaimable。
- initial decoder expert 是 `Ready`，进入当前 cache policy 的普通状态，不是 reclaimable。
- 第一版不引入 `active_urgent` 集合。

后续任务如果把 encoder expert 标记为 reclaimable，该标记只表示“优先 victim 候选”。实际是否能覆盖 GPU buffer，仍由现有 `CacheMngr::miss` 状态机决定：

- `kReady` victim 可以立刻切到 `kIdle` 并复用 buffer。
- `kLaunching` / `kUsing` victim 通过现有 wait lambda 等待回到可复用状态。
- `kFetching` victim 按现有逻辑取消部分 fetch 并回到 idle。

## 7. 兼容性

默认配置保持现有行为：

- `reset_cache_on_generate_start=false`
- `initial_cache_policy=""`
- `initial_layer_budgets=""`
- `per_layer_cache` 默认值不因本设计改变

只有显式打开 deterministic initial cache 时，才要求 `per_layer_cache=false` 和 manual plan 填满 cache。

## 8. 验收标准

1. 同一配置连续两次 `generate`，初始化后的 cache expert 集合完全一致。
2. `initial_layer_budgets` 总数小于 cache size 时启动或 generate 前报错。
3. `initial_layer_budgets` 总数大于 cache size 时启动或 generate 前报错。
4. Switch 模型的 `ModuleMeta` 能正确区分 encoder 和 decoder MoE layer。
5. manual plan 可以同时包含 encoder 和 decoder layer。
6. 不启用新配置时，现有示例和测试保持原行为。
