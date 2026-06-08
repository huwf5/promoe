# Sparse Cache Encoder Predictor Trace Spec

## Scope

This directory owns the encoder predictor trace exporter that runs through the repository's sparse expert cache path.

Allowed changes:

- Create or modify files only under `experiment/scripts/trace/encoder_predictor/`.
- Reuse existing runtime modules from `src/sparse_llm_cache` by importing them.
- Reuse local Transformers sources under `deps/transformers/src` by adding them to `sys.path`.

Forbidden changes:

- Do not modify `src/`.
- Do not modify `deps/`.
- Do not modify existing files outside `experiment/scripts/trace/encoder_predictor/`.
- Do not create a git worktree.
- Do not commit git changes.

Runtime environment:

- Python interpreter: `/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python`
- Working directory: `/mnt/huwf5/promoe`
- Network is not required. The exporter must support offline local model paths.

## Goal

Export encoder predictor training traces using the same sparse cache execution path as `examples/small-demo/transformers-app.py`, while ensuring expert computation happens on GPU after expert weights are staged to GPU by `sparse_llm_cache`.

This exporter is not a pure HuggingFace baseline exporter. It must call `sparse_llm_cache.utils.hack_transformers(...)` before `from_pretrained(...)`, so model loading and expert execution use the repository's cache manager, expert parameter indirection, and on-demand expert fetch path.

## Execution Semantics

The exporter must:

- Load Switch Transformer models through patched Transformers after calling `sparse_llm_cache.utils.hack_transformers(...)`.
- Use `device_map=0` by default, matching `examples/small-demo/transformers-app.py`.
- Use model dtype `auto` by default, matching the model config/checkpoint rather than forcing lower precision.
- Keep non-expert computation on the selected CUDA device.
- Let `sparse_llm_cache` manage expert parameter staging:
  - expert parameters may be stored on CPU when idle;
  - selected experts are moved or mapped into GPU cache before expert forward;
  - expert forward must run with CUDA hidden states and CUDA expert parameters.
- Avoid Accelerate CPU offload for normal module computation.
- Default to the small-demo input口径:
  - `--padding longest`
  - `--batch-size 1`
  - `--max-input-tokens 512`
- Use `model.generate(...)` by default, matching the simple small-demo model invocation.
- Do not call sparse cache internals such as `_prefetch_mngr.reset_and_load_initial_cache()` or `_sparse_cache_old_generate()` from the exporter.
- Capture encoder data from hooks on the encoder pass. Decode output length is only used to trigger the runtime path; encoder trace fields must not depend on generated text content.

Default sparse cache configuration for trace generation:

- `cache_rate`: configurable, default `0.375`
- `cache_policy`: configurable, default `lru`
- `per_layer_cache`: configurable, default `False`
- `num_predict_expert_per_layer`: fixed `0`
- `reorder_experts`: fixed `False`
- `early_preempt`: fixed `False`
- `chunk_prefetch`: fixed `False`
- `predict_input_mode`: fixed `no_predict`
- `pin_memory`: enabled internally

The fixed `num_predict_expert_per_layer=0` means no predictor prefetch is used. On-demand expert fetch remains active through the sparse cache runtime.

## Inputs

Required CLI arguments:

- `--model-path`: local model path, for example `experiment/models/google/switch-base-128`
- `--dataset`: dataset name, for example `mmlu`
- `--task-name`: task name, for example `professional_law`

Optional CLI arguments:

- `--train-split`: default `test`
- `--validation-split`: default `validation`
- `--train-prompt-file`: explicit train prompt file override
- `--validation-prompt-file`: explicit validation prompt file override
- `--output-dir`: explicit output directory override
- `--device`: default `cuda:0`
- `--batch-size`: default `1`
- `--max-input-tokens`: default `512`
- `--padding`: choices `longest` or `max_length`, default `longest`
- `--max-new-tokens`: default `1`
- `--storage-dtype`: choices `float32`, `float16`, `bfloat16`, default `float32`
- `--model-torch-dtype`: choices `auto`, `float32`, `float16`, `bfloat16`, default `auto`
- `--seed`: default `42`
- `--cache-rate`: default `0.375`
- `--cache-policy`: default `lru`
- `--per-layer-cache`: default `False`
- `--assert-gpu-expert-forward`: default enabled
- `--no-verify`: disables final structural verification
- `--print-status`: prints per-batch progress

Default prompt file resolution:

```text
experiment/datasets/<dataset>/<task-name>/<train-split>/prompt_list.txt
experiment/datasets/<dataset>/<task-name>/<validation-split>/prompt_list.txt
```

Default output directory:

```text
experiment/traces/<model-name>-<dataset>-<task-name>/encoder_predictor_sparse_cache_trace/
```

For `experiment/models/google/switch-base-128`, `<model-name>` is `switch-base-128`.

## Outputs

Each split directory must contain:

```text
<output-dir>/
  metadata.json
  run_log.json
  train/
    attention_mask.pt
    expert_selection.pt
    expert_weights.pt
    input_ids.pt
    layer0_attn_out.pt
    prompt_texts.jsonl
    router_logits.pt
    router_probs.pt
    seq_ids.pt
  validation/
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

Tensor contracts:

- `input_ids.pt`: `torch.int64`, shape `[S, T]`
- `attention_mask.pt`: `torch.int64`, shape `[S, T]`
- `layer0_attn_out.pt`: floating dtype selected by `--storage-dtype`, shape `[S, T, H]`
- `router_logits.pt`: floating dtype selected by `--storage-dtype`, shape `[S, L, T, E]`
- `router_probs.pt`: floating dtype selected by `--storage-dtype`, shape `[S, L, T, E]`
- `expert_selection.pt`: `torch.int64`, shape `[S, L, T, K]`
- `expert_weights.pt`: floating dtype selected by `--storage-dtype`, shape `[S, L, T, K]`
- `seq_ids.pt`: `torch.int64`, shape `[S]`
- `prompt_texts.jsonl`: one JSON object per sample: `{"seq_id": int, "text": str}`

For Switch top-1 routing, `K == 1`.

Storage padding rule:

- The model forward must use the requested tokenizer padding.
- If split batches produce different sequence lengths, tensors are padded only at save time to the split maximum length.
- Save-time padding uses zero values.
- `attention_mask.pt` is the source of truth for real tokens.
- Training must ignore positions where `attention_mask == 0`.

## Metadata

`metadata.json` must record:

- command-line arguments
- resolved repo root
- resolved model path
- resolved prompt file paths
- output directory
- sparse cache config passed to `hack_transformers`
- model dtype request and resolved dtype
- device
- padding
- batch size
- max input tokens
- max new tokens
- model config fields:
  - `model_type`
  - `hidden_size`
  - `num_experts`
  - `num_selected_experts`
  - `num_sparse_encoder_layers`
  - `num_sparse_decoder_layers`
  - `encoder_sparse_step`
  - `decoder_sparse_step`
- router layer mapping:
  - encoder sparse layer index
  - encoder block id
  - global sparse layer id if available from the patched module metadata
- split summaries:
  - sample count
  - tensor shapes
  - true token lengths from `attention_mask`
- verification result when verification is enabled

`run_log.json` must record:

- start timestamp
- end timestamp
- elapsed seconds
- Python executable
- CUDA availability
- selected device
- CUDA device name if available
- cache config
- per-split sample counts

## Verification

The exporter must verify:

- all required output files exist;
- tensor shapes align within each split;
- `router_logits.shape[:3] == expert_selection.shape[:3]`;
- `expert_selection.shape[-1] == num_selected_experts`;
- expert ids are in `[0, num_experts)`;
- `input_ids`, `attention_mask`, and `seq_ids` are integer tensors;
- floating tensors are floating tensors;
- prompt JSONL line count equals sample count;
- if `--assert-gpu-expert-forward` is enabled, every observed expert forward has CUDA input and CUDA parameters on the selected device.

Verification must not require modifying `src/` or `deps/`.

## Comparison Utility

This directory should also provide a local comparison utility for trace diagnosis.

The utility must compare two trace directories split by split and report:

- exact equality for `input_ids.pt`, `attention_mask.pt`, `seq_ids.pt`, and `prompt_texts.jsonl`;
- total, valid-token, and padding-token diffs for `expert_selection.pt`;
- valid-token and padding-token float max/mean diffs for `layer0_attn_out.pt`, `router_logits.pt`, `router_probs.pt`, and `expert_weights.pt`;
- per-layer valid-token expert diffs;
- per-layer valid-token router logit max/mean diffs.

The comparison utility must use `attention_mask.pt` to separate real-token differences from save-time padding differences.

## Acceptance Criteria

The implementation is acceptable when:

- all created/modified files are under `experiment/scripts/trace/encoder_predictor/`;
- the exporter can be invoked with the documented conda Python;
- it imports and uses `sparse_llm_cache.utils.hack_transformers(...)`;
- it does not use `device_map="auto"` or Accelerate CPU offload;
- it writes the full ERPP-style encoder predictor trace file set;
- it records metadata proving the sparse cache execution configuration;
- local unit tests in this directory pass;
- dry-run argument/path tests pass without loading the full model;
- on a CUDA machine with enough runtime access, a real run can export the MMLU professional_law trace;
- the comparison utility can compare the new trace with the old ERPP trace and separate padding differences from valid-token differences.

## Reference Commands

Export:

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
  experiment/scripts/trace/encoder_predictor/export_sparse_cache_encoder_trace.py \
  --model-path experiment/models/google/switch-base-128 \
  --dataset mmlu \
  --task-name professional_law \
  --device cuda:0 \
  --padding longest \
  --batch-size 1 \
  --max-input-tokens 512 \
  --max-new-tokens 1 \
  --print-status
```

Compare:

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
  experiment/scripts/trace/encoder_predictor/compare_encoder_traces.py \
  --left performance_predictor/encoder/ERPP/data/traces/switch-base-128-mmlu-professional_law-erpp \
  --right experiment/traces/switch-base-128-mmlu-professional_law/encoder_predictor_sparse_cache_trace
```
