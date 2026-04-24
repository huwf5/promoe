# Switch Trace Export 设计文档

- 日期：2026-04-24
- 范围：为 HuggingFace Switch Transformers（以 `switch-base-128` 为代表）产出与 `deps/sparse-llm-cache-scripts/train-predict-model/train_predict_model.py` **数据流等价** 的训练契约，使既有训练脚本 **零改动** 即可分别训练 decoder predictor 与 encoder predictor。
- 不在范围：修改 `train_predict_model.py`、修改 `src/` 线上 predictor、实现 encoder predictor 的线上加载路径。

---

## 1. 背景与现状

- **原流程（DeepSeek-V2-Lite，decoder-only）**：`llama.cpp --trace-logits` 产出 5 个原始 `.pt`（`moe_layer_gate_scores.pt` 等）→ `convert_llama_trace_to_train_format.py` 转成 7 个 `decode_stage_*.pt` 契约 → `train_predict_model.py` 训练。契约细节见 `deps/sparse-llm-cache-scripts/train-predict-model/TRACE_DATA_FORMAT.md`。
- **目标模型 Switch-base-128（HF）**：encoder-decoder，`V=128`，`K=1`（top-1 路由），encoder / decoder 各 `L_moe=6` 个稀疏层（`sparse_step=2`，即 `block.{1,3,5,7,9,11}`），router 含 `router_jitter_noise=0.01`（采集时需关）。
- **数据集**：`deps/sparse-llm-cache-scripts/dataset/chatgpt-prompts/prompt_list.txt`（360 条 prompt）。
- **下游训练契约**：`Trace.unpack_from_dir` 读目录中固定命名的 `.pt` 文件，推导 `num_expert = max(expert_selection) + 1`、`num_moe_layer = expert_selection.shape[1]`、`per_token_expert = expert_selection.shape[2]`。

## 2. 目标与等价性定义

本设计采用 **"数据流等价（B 级）"**：

1. **契约等价**：产出目录中的 `.pt` 文件命名、形状、dtype、轴顺序与 `Trace.unpack_from_dir` 完全一致，只是 `(V, K, L_moe) = (128, 1, 6)` 随 Switch 变化。
2. **阶段语义对齐**：
   - Switch encoder 一次性处理整 prompt ≡ llama prefill → 落盘到 `encoder/` 子目录。
   - Switch decoder 自回归 generate ≡ llama decode → 落盘到 `decoder/` 子目录。
3. **验证手段**：契约层断言 + `train_predict_model.py` 端到端冒烟训练 + 同 seed 可复现。

明确地 **不追求**：与 DeepSeek trace 的数值可比性；与 llama 五文件中间格式的 byte-level 一致。

## 3. 关键决策摘要

| 决策点 | 选择 | 理由简述 |
|---|---|---|
| 等价层级 | 契约 + 数据流（B） | encoder=prefill / decoder=decode 自然对齐；无冗余中间格式 |
| 采集方式 | `model.generate`（greedy, `max_new_tokens=64`） | 与线上 predictor 真实推理路径一致 |
| Hook 挂点 | `SwitchTransformersSparseMLP.register_forward_hook` | 直接拿到 `(hidden, router_logits)`，无需深入 router 内部 |
| 输入特征 | `router_logits`（末维 V=128） | 与 `convert_llama_trace_to_train_format.py --moe-input gate` 默认一致 |
| gate 标签 | `router_logits` | 同上 |
| freq 标签 | `softmax(router_logits)` | Switch top-1 下最自然的软标签；行和 ≈ 1 |
| expert_selection | `argmax(router_logits)`，形状 `[N, L_moe, 1]` | K=1 |
| 落盘 | `out/decoder/` 与 `out/encoder/` 两独立子目录 | 训练脚本零改动；encoder/decoder 完全解耦 |
| 精度 | 全程 float32 | 避免 softmax 数值飘移；Switch-base 小，不需 fp16 |
| 随机性 | `torch.manual_seed` + `transformers.set_seed` + 显式 `router.jitter_noise=0` + greedy | 同 seed byte-level 可复现 |

## 4. 架构与模块切分

新增目录：`deps/sparse-llm-cache-scripts/switch-trace-export/`

```
switch-trace-export/
├── switch_trace_export.py   # 入口脚本
└── README.md                # 使用说明
```

新增 Shell 封装：`deps/sparse-llm-cache-scripts/switch_trace.sh`（对标既有 `trace.sh`）。

脚本内部 4 个阶段，模块职责单一、可独立测试：

```
prompt_list.txt
    │
    ▼
 SwitchRunner        # 加载 HF 模型、关 jitter_noise、安装 hook、batched generate
    │
    ▼
 TraceAccumulator    # 为 encoder / decoder 各维护一套张量缓冲，按 prompt append
    │
    ▼
 ContractWriter      # 按 Trace 契约把缓冲写到 out/decoder/ 与 out/encoder/
    │
    ▼
 Verifier            # 契约层断言 + 可选 端到端冒烟训练
```

### 4.1 模块接口约定

- **`SwitchRunner`**
  - 输入：模型路径、prompt 列表、`max_new_tokens`、batch_size、device、seed
  - 输出：每条 prompt 的 `PromptRecord { encoder_per_layer: list[6] of (logits[S, V]), decoder_per_layer: list[6] of list-of-step-logits, encoder_token_ids, decoder_token_ids }`
  - 关键行为：给每个 `SwitchTransformersSparseMLP` 安装 hook，hook 内按当前正在跑的 stage（encoder / decoder）与 sparse_layer_id 写入"当前 prompt 的 buffer"
- **`TraceAccumulator`**
  - 维护 6 个 per-stage 的 per-layer 列表，按 prompt 顺序 append
  - 同时维护 `token_ids / seq_id_of_token / token_idx_in_seq` 三路元信息
- **`ContractWriter`**
  - 把 list 拼成 `[N, L_moe, V]` / `[N, L_moe, 1]` / `[N]` 并 `torch.save`
  - `metadata.json` 写 `{stage, num_expert, num_moe_layer, per_token_expert, N, seed, max_new_tokens}`，非契约，仅调试
- **`Verifier`**
  - 给一个目录（`out/decoder` 或 `out/encoder`），跑 §7 的所有断言；任一失败抛 `AssertionError`

## 5. Hook 细节

### 5.1 挂点

每个 `SwitchTransformersSparseMLP` 的 `forward` 返回 `(hidden_states, router_logits)`。对所有这样的子模块 `register_forward_hook`：

```python
def sparse_mlp_hook(module, inputs, output):
    _, router_logits = output                    # [B, S, V=128]
    stage = module._promoe_stage                 # "encoder" | "decoder"
    layer_id = module._promoe_sparse_layer_id    # 0..5
    runner.record(stage, layer_id, router_logits)
```

### 5.2 稀疏层编号映射

配置里 `encoder_sparse_step = decoder_sparse_step = 2`、总层 12：稀疏层为 `block.{1, 3, 5, 7, 9, 11}`。**契约里的 `num_moe_layer = 6`，稀疏层在契约中以出现顺序编号 0..5**（不保留绝对 block id）。这与 `train_predict_model.py` 使用的"第 i 个 MoE 层"语义一致。绝对层 id 写入 `metadata.json` 仅作调试参考。

### 5.3 encoder / decoder 识别

安装 hook 时遍历 `model.encoder.block` 与 `model.decoder.block`，在子模块上分别设置 `_promoe_stage` 与 `_promoe_sparse_layer_id` 属性；hook 内直接读。不依赖 `module.training` 或其他易变信号。

### 5.4 每步 1 行的保证

- **encoder**：一条 prompt 一次 forward，`S = prompt_len`，hook 输出 `[1, S, V]`。按 `attention_mask` 剔除 pad 后，按顺序记录 `S_valid` 行。
- **decoder**：`model.generate` 每步 `S = 1`（past_key_values 开启），hook 每步记录 1 行。EOS 提前停止则该 prompt decoder 侧行数少于 `max_new_tokens`。
- **batch 内多 prompt**：按 batch 维逐行拆开；hook 拿到 `[B, S, V]`，用每个样本自己的 attention_mask 与生成长度裁切。

## 6. 落盘规格

```
out_dir/
├── decoder/
│   ├── expert_selection.pt                              # [N_dec, 6, 1]   int64
│   ├── decode_stage_moe_layer_logits_per_token.pt       # [N_dec, 6, 128] float32  = router_logits
│   ├── decode_stage_moe_layer_gate_logits_per_token.pt  # [N_dec, 6, 128] float32  = router_logits（同上张量）
│   ├── decode_stage_expert_freq_per_token.pt            # [N_dec, 6, 128] float32  = softmax(router_logits)
│   ├── decode_stage_token_ids_per_token.pt              # [N_dec]          int64
│   ├── decode_stage_seq_id_of_token.pt                  # [N_dec]          int64
│   ├── decode_stage_token_idx_in_seq.pt                 # [N_dec]          int64
│   └── metadata.json
├── encoder/
│   └── <与 decoder/ 同名 7 个 .pt + metadata.json>      # 文件名沿用 decode_stage_*
└── run_log.json                                         # seed / 耗时 / N_enc / N_dec / EOS 分布等
```

**关于 encoder 子目录沿用 `decode_stage_*` 前缀**：这里 `decode_stage_` 是训练契约的命名常量，不代表阶段语义；真实阶段由子目录名承载。这样 `train_predict_model.py --logits_path out/encoder` 无需任何改动即可训练 encoder predictor。`metadata.json` 显式记录 `"stage": "encoder"`。

**元信息语义**：

- `decoder/`：`seq_id = prompt 索引 0..359`；`token_idx_in_seq = 该 prompt 已生成到第几个 token（0 起）`；`token_ids = 该步 decoder 生成的 token id`。
- `encoder/`：`seq_id = prompt 索引`；`token_idx_in_seq = 该 token 在 prompt 内的绝对位置`；`token_ids = 该位置的输入 token id`（pad 位置不进入契约）。

**无尾部全零问题**：按真实长度 append；不预分配 `E`。

## 7. 验证（等价性的操作化定义）

### 7.1 契约层断言（`Verifier.verify(path)`）

1. `Trace().unpack_from_dir(path)` 不抛
2. `trace.prepare_tensors()` 后：`num_expert == 128`、`num_moe_layer == 6`、`per_token_expert == 1`
3. 所有形如 `[N, ...]` 的张量第 0 维一致
4. `decode_stage_expert_freq_per_token.sum(dim=-1)` 与全 1 张量差 `<1e-5`
5. `expert_selection[:, :, 0].max() < 128` 且 `>= 0`
6. 对每个 `seq_id`，`token_idx_in_seq` 在该序列内严格从 0 递增（步长 1）
7. 所有浮点张量 dtype 为 `torch.float32`；所有 int 张量为 `torch.int64`

### 7.2 端到端冒烟训练（可选 `--smoke-train` 时运行）

```bash
python train_predict_model.py \
  --logits_path out/decoder \
  --predict_model_path /tmp/smoke_dec \
  --predict_output freq --model_type split \
  --window 3 --hidden_size 256 --n_layer 2 \
  --batch_size 128 --lr 0.001 --threshold 0.01
```

`encoder/` 同样再跑一次。两次都需：脚本无报错跑完 + test loss 下降 + `metas.json` 写出。

### 7.3 可复现性

同 seed 跑两次，产物目录内所有 `.pt` 用 `torch.equal` 全相等。

## 8. 确定性 & 性能要点

- `model.eval()`；`torch.set_grad_enabled(False)`
- 遍历所有 `SwitchTransformersTop1Router`，显式 `router.jitter_noise = 0.0`
- `torch.manual_seed(seed)`、`transformers.set_seed(seed)`、`np.random.seed(seed)`
- `generate(do_sample=False)`（greedy）
- 全程 float32
- batched generate（默认 `batch_size=8`），通过 tokenizer 的 `padding="longest"` + `attention_mask`；hook 内用 attention_mask 滤 pad

## 9. CLI 规约

脚本：`deps/sparse-llm-cache-scripts/switch-trace-export/switch_trace_export.py`

```bash
python switch_trace_export.py \
  --model-path /mnt/huwf5/promoe/deps/sparse-llm-cache-scripts/huggingface-modules/modules/transformers_modules/google/switch-base-128 \
  --prompt-file /mnt/huwf5/promoe/deps/sparse-llm-cache-scripts/dataset/chatgpt-prompts/prompt_list.txt \
  --output-dir  /mnt/huwf5/promoe/deps/moe-traces/switch-base-128-chatgpt-prompts \
  --max-new-tokens 64 \
  --batch-size 8 \
  --device cuda:0 \
  --seed 42 \
  --verify \
  [--smoke-train]    # 可选，跑端到端冒烟训练
```

Shell 封装：`deps/sparse-llm-cache-scripts/switch_trace.sh`，默认参数与 CLI 示例一致，保持与既有 `trace.sh` 风格（env 覆盖）。

## 10. 测试策略

- **单元级**：
  - `SwitchRunner` 小模型/小 prompt（1 条 × `max_new_tokens=2`）的 buffer 内容断言（形状、stage 区分、layer_id 排列）
  - `TraceAccumulator` 对 2 条 prompt 拼接后元信息正确（`seq_id`、`token_idx_in_seq` 连贯）
  - `ContractWriter` 写出后 `Trace.unpack_from_dir` 能读
- **集成级**：
  - 小规模（5 条 prompt × `max_new_tokens=4`）跑全流程 + `Verifier` 全绿
  - `--smoke-train` 对 decoder 与 encoder 子目录各跑 1 次
- **可复现性**：`seed=42` 跑两次，`torch.equal` 比对所有 `.pt`

## 11. 已知边界与非目标

- 本设计只管数据生成。encoder predictor 的线上加载/调度属于 `src/` 改造，不在本 spec 范围。
- 未来若其他 Switch 规模（`switch-base-8/16/32/64/256`、`switch-large-128`）接入，只是 `V` 与 `num_layers` 不同，本脚本参数化即可，不改架构。
- `model.generate` 采样策略扩展（beam / sampling）属于后续增强；MVP 仅支持 greedy，以保证可复现性。

## 12. 下游一键命令示例

```bash
# 采集
./deps/sparse-llm-cache-scripts/switch_trace.sh

# 训 decoder predictor
python deps/sparse-llm-cache-scripts/train-predict-model/train_predict_model.py \
  --logits_path /mnt/huwf5/promoe/deps/moe-traces/switch-base-128-chatgpt-prompts/decoder \
  --predict_model_path /mnt/huwf5/promoe/deps/sparse-llm-cache-scripts/moe-predict-models/switch-base-128/decoder \
  --predict_output freq --model_type split --window 3

# 训 encoder predictor（未来线上支持后启用）
python deps/sparse-llm-cache-scripts/train-predict-model/train_predict_model.py \
  --logits_path /mnt/huwf5/promoe/deps/moe-traces/switch-base-128-chatgpt-prompts/encoder \
  --predict_model_path /mnt/huwf5/promoe/deps/sparse-llm-cache-scripts/moe-predict-models/switch-base-128/encoder \
  --predict_output freq --model_type split --window 3
```
