# Promoe LSP Baseline

This directory stores Promoe baseline scripts and per-run artifacts for LSP benchmark comparison.

## Run Layout

Each run is stored under:

```text
experiment/baseline/LSP/promoe/runs/<dataset>/<task>/<split>/<gpu_profile>/<model>/<run_id>/
```

Example:

```text
experiment/baseline/LSP/promoe/runs/mmlu/professional_law/validation/gpu4gb/google_switch-base-128/20260616_120000/
```

Run artifacts:

- `command.txt`: command and environment used to launch the run.
- `config.env`: normalized run configuration.
- `env.txt`: git, Python, and GPU environment snapshot.
- `run.log`: stdout/stderr from direct `examples/small-demo/transformers-app.py` execution.
- `samples.csv`: one row per measured request, for later CDF export.
- `summary.json`: machine-readable run summary.
- `summary.tsv`: one-row tabular run summary.

## Run Promoe Baselines

`run_all.sh` is the main entry point. It does not call `performance/run_switch_mmlu_validation.sh`; it keeps the experiment parameters in this directory and directly launches `examples/small-demo/transformers-app.py`.

```bash
experiment/baseline/LSP/promoe/scripts/run_all.sh
```

Useful overrides:

```bash
GPU_CONFIGS="gpu4gb gpu8gb" \
MODELS="switch-base-128 switch-base-256 switch-large-128" \
GPU_ID=0 \
BACKEND_MODE=base \
MAX_NEW_TOKENS=32 \
PROMOE_BENCHMARK_WARMUP=4 \
experiment/baseline/LSP/promoe/scripts/run_all.sh
```

Parameter groups inside the script:

- Common runtime: `PYTHON_BIN`, `BATCH_SIZE`, `MAX_NEW_TOKENS`, `PROMOE_BENCHMARK_WARMUP`, `SAMPLE_INDICES`.
- Task: `DATASET_NAME`, `MMLU_TASK`, `SPLIT`.
- GPU profiles: `GPU_CONFIGS`, `GPU_ID`, profile GPU memory, and per-profile/per-model `CACHE_RATE`. GPU-related values are applied in `apply_gpu_model_overrides()`.
- Model registry: `MODEL_ID`, `INITIAL_HOT_EXPERT_FILE`, `PREDICTOR_ROOT`, `ERPP_ENCODER_MODEL_PATH`. Model entries should not set GPU memory or cache ratio.
- Promoe method: `BACKEND_MODE`, `CACHE_RATE`, `CACHE_POLICY`, `NUM_PRED`, `REORDER`, `PREEMPT`, `INITIAL_CACHE_POLICY`.
- Output metadata: `RUN_ROOT`, `RUN_ID`, `PROMOE_BENCHMARK_*`.


GPU/model cache defaults are intentionally explicit. Missing GPU/model pairs fail fast instead of guessing. Current known mapping:

```text
switch-base-128: 4GB=0.125, 8GB=0.25, 12GB/24GB/48GB=0.375
```

Add new mappings in `apply_gpu_model_overrides()` when cache ratios for other models are decided.

`BATCH_SIZE` is intentionally restricted to `1` so `samples.csv` remains one row per request. Built-in GPU profiles are `default`, `gpu4gb`, `gpu8gb`, `gpu12gb`, `gpu24gb`, and `gpu48gb`. A global `CACHE_RATE=...` override still wins over profile/model defaults when you need a one-off run.

## `samples.csv`

Required fields:

```text
run_id,baseline,backend_mode,model_id,gpu_profile,gpu_id,gpu_mem_gb,cache_rate,dataset,task,split,sample_idx,batch_idx,prompt_tokens,new_tokens,input_ms,cache_init_ms,ttft_ms,tpot_ms,e2e_ms,decode_tokens_per_second_excl_first,e2e_tokens_per_second,gen_forward_steps,warmup,is_valid
```

Metric definitions:

- `ttft_ms`: `generate()` start to first generated-token root forward completion.
- `tpot_ms`: average decode token time excluding first token.
- `e2e_ms`: `ttft_ms + tpot_ms * max(new_tokens - 1, 0)`.
- `decode_tokens_per_second_excl_first`: per-request decode throughput excluding first token.
- `e2e_tokens_per_second`: per-request generated-token throughput over `e2e_ms`.
- `is_valid`: `1` for summary/CDF samples, `0` for warmup or excluded samples.

## `summary.tsv`

The summary keeps Promoe benchmark-compatible top-level fields:

```text
benchmark_ttft_ms_avg
benchmark_decode_tpot_ms_avg
benchmark_decode_tokens_per_second_excl_first
benchmark_e2e_tokens_per_second
benchmark_e2e_ms_avg
```

It also includes p50/p90/p95/p99 for `ttft_ms`, `tpot_ms`, and `e2e_ms`.
