# Encoder Predictor Sparse Cache Trace

This directory contains trace tools for encoder predictor training data.

The exporter uses the same sparse cache entry point as `examples/small-demo/transformers-app.py`:

```python
sparse_llm_cache.utils.hack_transformers(...)
```

It is intentionally different from pure HuggingFace CPU/GPU offload exporters. Expert parameters may be stored on CPU while idle, but selected experts are staged to GPU before expert forward. The trace is meant to avoid CPU expert computation while still supporting memory-constrained runs.

## Default Trace口径

- `padding=longest`
- `batch_size=1`
- `max_input_tokens=512`
- `max_new_tokens=1`
- `model_torch_dtype=auto`
- `device=cuda:0`
- `num_predict_expert_per_layer=0`
- `reorder_experts=False`
- `early_preempt=False`
- `predict_input_mode=no_predict`

The exporter writes tensors padded at save time when needed. Training code must use `attention_mask.pt` to ignore positions where `attention_mask == 0`.

## Export

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
  experiment/scripts/trace/encoder_predictor/export_sparse_cache_encoder_trace.py \
  --model-path experiment/models/google/switch-base-128 \
  --dataset mmlu \
  --task-name professional_law \
  --device cuda:0 \
  --print-status
```

Default output:

```text
experiment/traces/switch-base-128-mmlu-professional_law/encoder_predictor_sparse_cache_trace/
```

## Smoke Test

Use this before a full trace run:

```bash
printf 'What is the answer?\n' > /tmp/encoder_predictor_trace_train_prompts.txt
printf 'What is the answer?\n' > /tmp/encoder_predictor_trace_validation_prompts.txt

/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
  experiment/scripts/trace/encoder_predictor/export_sparse_cache_encoder_trace.py \
  --model-path experiment/models/google/switch-base-128 \
  --dataset mmlu \
  --task-name professional_law \
  --train-prompt-file /tmp/encoder_predictor_trace_train_prompts.txt \
  --validation-prompt-file /tmp/encoder_predictor_trace_validation_prompts.txt \
  --output-dir /tmp/encoder_predictor_sparse_cache_trace_smoke \
  --device cuda:0 \
  --batch-size 1 \
  --max-input-tokens 512 \
  --max-new-tokens 1 \
  --gpu-mem-limit-gb 12 \
  --cache-rate 0.375 \
  --cache-policy lru \
  --per-layer-cache False \
  --print-status
```

The exporter calls `model.generate(...)` directly. Trace tensors are collected by hooks on encoder modules; the script does not call `_prefetch_mngr.reset_and_load_initial_cache()` or `_sparse_cache_old_generate()`. The sparse cache runtime still performs on-demand expert staging through hooks installed by `sparse_llm_cache.utils.hack_transformers(...)`.


## NLLB-MoE 54B Export

NLLB-MoE uses top2 routing. The exporter records post-capacity sparse router probabilities, `expert_selection.pt` with `K=2`, `expert_weights.pt`, and `expert_selection_mask.pt`. Use the sparse cache path and keep `cache_rate=0.01` for this model.

```bash
PYTHONPATH=/mnt/huwf5/promoe/src:/mnt/huwf5/promoe/deps/transformers/src /mnt/huwf5/conda-envs/promoe-moe-cache/bin/python   experiment/scripts/trace/encoder_predictor/export_sparse_cache_encoder_trace.py   --model-path /mnt/huwf5/promoe/experiment/models/facebook/nllb-moe-54b   --dataset mmlu   --task-name professional_law   --device cuda:0   --batch-size 1   --max-input-tokens 512   --max-new-tokens 1   --cache-rate 0.01   --cache-policy lru   --per-layer-cache False   --print-status
```

For a one-sample smoke run, pass `--train-prompt-file`, `--validation-prompt-file`, and `--output-dir` exactly as in the Switch smoke test, but keep the NLLB `--model-path` and `--cache-rate 0.01`.

Train SRC-SimpleNN on the NLLB trace with the top2 objective:

```bash
PYTHONPATH=/mnt/huwf5/promoe/src:/mnt/huwf5/promoe/deps/transformers/src /mnt/huwf5/conda-envs/promoe-moe-cache/bin/python   experiment/scripts/train/encoder_predictor_src_simplenn_token_hard_ce.py   --trace-dir experiment/traces/nllb-moe-54b-mmlu-professional_law/encoder_predictor_sparse_cache_trace   --loss-type multi_label_bce   --device cuda:0
```

`--loss-type auto` also selects `multi_label_bce` when the trace metadata has `model_type=nllb-moe`; passing the flag explicitly makes the training intent clear. By default, positive BCE terms use `expert_weights`; add `--no-use-expert-weights-in-loss` when you want both routed experts to be equal hard positives.

## Compare

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
  experiment/scripts/trace/encoder_predictor/compare_encoder_traces.py \
  --left performance_predictor/encoder/ERPP/data/traces/switch-base-128-mmlu-professional_law-erpp \
  --right experiment/traces/switch-base-128-mmlu-professional_law/encoder_predictor_sparse_cache_trace
```

The comparison report separates real-token differences from padding-token differences by using `attention_mask.pt`.
