# swtich model 256

## base info
模型位置： 
/mnt/huwf5/promoe/deps/sparse-llm-cache-scripts/huggingface-modules/modules/transformers_modules/google/switch-basedfsdaf 256


数据位置：
/mnt/huwf5/promoe/deps/sparse-llm-cache-scripts/dataset/mmlu/professional_law

## encoder predictor

## Encoder ERPP trace wrapper

`experiment/scripts/trace/run_encoder_erpp_trace.py` wraps the existing ERPP exporter:

```text
performance_predictor/encoder/ERPP/implement/trace/export_erpp_encoder_trace.py
```

It keeps the original ERPP trace files and schema intact, and only adapts paths and defaults for `experiment/`.

Default input:

```text
experiment/datasets/<dataset>/<task>/test/prompt_list.txt|pt
experiment/datasets/<dataset>/<task>/validation/prompt_list.txt|pt
```

Default output:

```text
experiment/traces/<model>-<dataset>-<task>/encoder_erpp_trace/
```

Default runtime口径 matches the single-sample small-demo path:

```text
--padding longest
--batch-size 1
--max-input-tokens 512
```

Generated split files remain compatible with the ERPP trace format:

```text
attention_mask.pt
expert_selection.pt
expert_weights.pt
input_ids.pt
layer0_attn_out.pt
prompt_texts.jsonl
router_logits.pt
router_probs.pt
seq_ids.pt
```

Storage dtype defaults to `auto`, which resolves `config.torch_dtype` from the model `config.json` before calling the ERPP exporter. You can still override it with `--storage-dtype float32|float16|bfloat16`.

Model placement defaults to `--model-device-map single-auto`, which uses only the GPU selected by `--device` plus CPU offload. The default GPU budget is based on current free GPU memory, not total memory, with a small reserve to avoid checkpoint dispatch OOM. If the selected GPU is already full, the model may be placed mostly on CPU and will be slow. Use `--model-device-map auto` only if you explicitly want Transformers to spread the model over all visible devices. Use `--model-device-map single` only if the whole model fits on `--device`.

Model load dtype is controlled by `--model-torch-dtype`. It defaults to `auto` and follows the model config, so it does not lower precision by default. This is separate from trace storage dtype.

Training code must filter padding tokens with `attention_mask.pt`; the wrapper also records this rule in `metadata.json`:

```python
valid_tokens = attention_mask.bool()      # [S, T]
valid_tokens = valid_tokens.unsqueeze(1)  # [S, 1, T], broadcast over encoder router layers
```

Example:

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
  experiment/scripts/trace/run_encoder_erpp_trace.py \
  --model-path experiment/models/google/switch-base-128 \
  --dataset mmlu \
  --task-name professional_law \
  --device cuda:0 \
  --storage-dtype auto \
  --model-device-map single-auto
```

Use `--dry-run` to print the resolved underlying ERPP exporter command without loading the model.

