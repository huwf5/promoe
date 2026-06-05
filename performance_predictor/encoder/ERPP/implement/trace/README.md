# ERPP Trace Exporter

本目录实现 ERPP encoder 训练数据的独立 trace 工具。

## dtype 策略

本地 `switch-base-128/config.json` 中：

```text
torch_dtype = float32
router_dtype = float32
```

因此 exporter 默认：

```text
model forward dtype: float32
layer0_attn_out.pt: float32
router_logits.pt: float32
router_probs.pt: float32
expert_weights.pt: float32
expert_selection.pt: int64
input_ids.pt / attention_mask.pt / seq_ids.pt: int64
```

这和原模型默认保持一致。只有显式传：

```bash
--storage-dtype float16
```

或：

```bash
--storage-dtype bfloat16
```

才会压缩浮点 trace 的落盘 dtype。

## 脚本

```text
export_erpp_encoder_trace.py
verify_erpp_trace.py
inspect_erpp_trace.py
```

`export_erpp_encoder_trace.py` 只跑 encoder forward，采集：

```text
layer0_attn_out.pt       [S, T, H]
router_logits.pt         [S, L, T, E]
router_probs.pt          [S, L, T, E]
expert_selection.pt      [S, L, T, K]
expert_weights.pt        [S, L, T, K]
```

## 示例命令

```bash
python performance_predictor/encoder/ERPP/implement/trace/export_erpp_encoder_trace.py \
  --model-path deps/sparse-llm-cache-scripts/huggingface-modules/modules/transformers_modules/google/switch-base-128 \
  --train-prompt-file /path/to/train_prompts.txt \
  --validation-prompt-file /path/to/validation_prompts.txt \
  --output-dir performance_predictor/encoder/ERPP/data/traces/switch-base-128-erpp \
  --max-input-tokens 512 \
  --batch-size 8 \
  --device cuda:0 \
  --storage-dtype float32 \
  --verify \
  --print-status
```

验证已有 trace：

```bash
python performance_predictor/encoder/ERPP/implement/trace/verify_erpp_trace.py \
  --trace-dir performance_predictor/encoder/ERPP/data/traces/switch-base-128-erpp
```

查看一个 token 的 router 信息：

```bash
python performance_predictor/encoder/ERPP/implement/trace/inspect_erpp_trace.py \
  --trace-dir performance_predictor/encoder/ERPP/data/traces/switch-base-128-erpp \
  --split validation \
  --sample 0 \
  --token 0 \
  --topk 5
```

## Hook 点

- `model.encoder.block[0].layer[0]`: `SwitchTransformersLayerSelfAttention`，采集 layer0 attention 后 hidden。
- encoder `SwitchTransformersSparseMLP`: 采集每个 sparse MoE/router 层的 router logits 和 expert index。

metadata 会记录：

```text
canonical_input = encoder.block.0.layer.0.self_attention.output_hidden_states_after_residual
router_layer_to_model_block = [1, 3, 5, 7, 9, 11]
router_jitter_noise = config.router_jitter_noise
router_jitter_active = false
```

说明：exporter 使用 `model.eval()` 复现推理行为。HF Switch 的 router jitter 只在 `self.training` 为 true 时生效，因此 eval 下 jitter 本来就是 inactive。exporter 不再主动修改 `jitter_noise`，只记录 config 原值。

## 注意

实际模型文件约 29GB，导出会加载完整模型。默认不要在 CPU 上跑大规模 prompt；建议先用 1-2 条 prompt 做 smoke trace，再跑完整数据。
