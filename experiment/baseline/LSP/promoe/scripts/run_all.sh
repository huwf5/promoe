#!/usr/bin/env bash
# Run Promoe LSP baseline experiments directly from the experiment tree.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
PROMOE_ROOT="${ROOT}/experiment/baseline/LSP/promoe"
DEMO_PY="${ROOT}/examples/small-demo/transformers-app.py"
COLLECT_SUMMARY="${PROMOE_ROOT}/scripts/collect_summary.py"

# Common runtime parameters.
PYTHON_BIN="${PYTHON_BIN:-/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python}"
GPU_ID_OVERRIDE="${GPU_ID:-}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
GPU_MEM_GB_OVERRIDE="${GPU_MEM_GB:-}"
PROMOE_BENCHMARK_WARMUP="${PROMOE_BENCHMARK_WARMUP:-4}"
SAMPLE_INDICES="${SAMPLE_INDICES:-}"
PROMOE_DRY_RUN="${PROMOE_DRY_RUN:-0}"

# Task parameters.
DATASET_NAME="${DATASET_NAME:-mmlu}"
MMLU_TASK="${MMLU_TASK:-professional_law}"
SPLIT="${SPLIT:-validation}"

# Promoe method parameters. These match the existing performance script defaults.
BACKEND_MODE="${BACKEND_MODE:-${MODE:-ours}}"
CACHE_RATE_OVERRIDE="${CACHE_RATE:-}"
PER_LAYER_CACHE="${PER_LAYER_CACHE:-False}"
DETERMINISTIC_INIT="${DETERMINISTIC_INIT:-1}"
INITIAL_CACHE_POLICY="${INITIAL_CACHE_POLICY:-hot_encoder_balanced_coverage}"
CACHE_STAGE_OCCUPANCY_LOG="${CACHE_STAGE_OCCUPANCY_LOG:-0}"
SPARSE_CACHE_LOG_PREFETCH_DECISION="${SPARSE_CACHE_LOG_PREFETCH_DECISION:-0}"
ERPP_ENCODER_DIAGNOSTICS="${ERPP_ENCODER_DIAGNOSTICS:-0}"
ENABLE_DECODER_PHASE_WARMUP_OVERLAP="${ENABLE_DECODER_PHASE_WARMUP_OVERLAP:-False}"
ENABLE_ERPP_ENCODER_PREFETCH="${ENABLE_ERPP_ENCODER_PREFETCH:-False}"
INITIAL_HOT_EXPERT_FILE_OVERRIDE="${INITIAL_HOT_EXPERT_FILE:-}"
PREDICTOR_ROOT_OVERRIDE="${PREDICTOR_ROOT:-}"
ERPP_ENCODER_MODEL_PATH_OVERRIDE="${ERPP_ENCODER_MODEL_PATH:-}"
MODEL_REVISION_OVERRIDE="${MODEL_REVISION:-main}"

# GPU/model sweep parameters.
GPU_CONFIGS="${GPU_CONFIGS:-default}"
MODELS="${MODELS:-switch-base-128}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${PROMOE_ROOT}/runs}"

if [[ "${BATCH_SIZE}" != "1" ]]; then
  echo "error: BATCH_SIZE must be 1 so samples.csv remains one row per request, got: ${BATCH_SIZE}" >&2
  exit 1
fi
if [[ ! -f "${DEMO_PY}" ]]; then
  echo "error: transformers app not found: ${DEMO_PY}" >&2
  exit 1
fi
if [[ ! -f "${COLLECT_SUMMARY}" ]]; then
  echo "error: summary collector not found: ${COLLECT_SUMMARY}" >&2
  exit 1
fi
if [[ ! -x "${PYTHON_BIN}" && "${PROMOE_DRY_RUN}" != "1" ]]; then
  echo "error: python not executable: ${PYTHON_BIN}" >&2
  exit 1
fi


gpu_config() {
  local profile="$1"
  GPU_PROFILE="${profile}"
  GPU_PROFILE_KEY="${profile//\//_}"
  GPU_ID="${GPU_ID_OVERRIDE:-0}"
  GPU_MEM_PROFILE_GB=""

  case "${profile}" in
    default)
      ;;
    gpu4gb)
      GPU_ID="${GPU4GB_GPU_ID:-${GPU_ID_OVERRIDE:-0}}"
      GPU_MEM_PROFILE_GB="4"
      ;;
    gpu8gb)
      GPU_ID="${GPU8GB_GPU_ID:-${GPU_ID_OVERRIDE:-0}}"
      GPU_MEM_PROFILE_GB="8"
      ;;
    gpu12gb)
      GPU_ID="${GPU12GB_GPU_ID:-${GPU_ID_OVERRIDE:-0}}"
      GPU_MEM_PROFILE_GB="12"
      ;;
    gpu24gb)
      GPU_ID="${GPU24GB_GPU_ID:-${GPU_ID_OVERRIDE:-0}}"
      GPU_MEM_PROFILE_GB="24"
      ;;
    gpu48gb)
      GPU_ID="${GPU48GB_GPU_ID:-${GPU_ID_OVERRIDE:-0}}"
      GPU_MEM_PROFILE_GB="48"
      ;;
    *)
      echo "error: unknown GPU profile: ${profile}" >&2
      exit 1
      ;;
  esac
}

model_config() {
  local alias="$1"
  MODEL_ALIAS="${alias}"
  MODEL_REVISION="${MODEL_REVISION_OVERRIDE}"
  GPU_MEM_GB=""
  PREDICTOR_ROOT=""
  ERPP_ENCODER_MODEL_PATH=""

  case "${alias}" in
    switch-base-128)
      MODEL_ID="google/switch-base-128"
# /mnt/huwf5/promoe/experiment/traces/switch-base-128-mmlu-professional_law-test/hot_experts/switch-base-128.test.json
      INITIAL_HOT_EXPERT_FILE="${ROOT}/experiment/traces/switch-base-128-mmlu-professional_law-test/hot_experts/switch-base-128.test.json"
      PREDICTOR_ROOT="${ROOT}/deps/sparse-llm-cache-scripts/moe-predict-models/switch-base-128-mmlu-professional_law-test-train-validation-val/sep/decoder"
      ERPP_ENCODER_MODEL_PATH="${ROOT}/experiment/models/predictors/encoder_expert_prefetch/mmlu-professional_law/switch-base-128/sparse-cache-b1-longest-v1/ble/noisyor-from-src-simplenn-token-hardce-h384-l1-drop0p5-lr1e4-bs2-seed0-validtrim/encoder_predictor_ble.ts"
      ;;
    switch-base-256)
      MODEL_ID="google/switch-base-256"
# /mnt/huwf5/promoe/experiment/traces/switch-base-256-mmlu-professional_law-test/hot_experts/switch-base-256.test.json
      INITIAL_HOT_EXPERT_FILE="${ROOT}/experiment/traces/switch-base-256-mmlu-professional_law-test/hot_experts/switch-base-256.test.json"
      PREDICTOR_ROOT="${ROOT}/deps/sparse-llm-cache-scripts/moe-predict-models/switch-base-256-mmlu-professional_law-test-train-validation-val/sep/decoder"
      ERPP_ENCODER_MODEL_PATH="${ROOT}/experiment/models/predictors/encoder_expert_prefetch/mmlu-professional_law/switch-base-256/sparse-cache-b1-longest-v1/ble/noisyor-from-src-simplenn-token-hardce-h384-l1-drop0p5-lr1e4-bs512-seed0-validtrim/encoder_predictor_ble.ts"
      ;;
    switch-large-128)
      MODEL_ID="google/switch-large-128"
# /mnt/huwf5/promoe/experiment/traces/switch-large-128-mmlu-professional_law-test/hot_experts/switch-large-128.test.json
      INITIAL_HOT_EXPERT_FILE="${ROOT}/experiment/traces/switch-large-128-mmlu-professional_law-test/hot_experts/switch-large-128.test.json"
      PREDICTOR_ROOT="${ROOT}/deps/sparse-llm-cache-scripts/moe-predict-models/switch-large-128-mmlu-professional_law-test-train-validation-val/sep/decoder"
      ERPP_ENCODER_MODEL_PATH="${ROOT}/experiment/models/predictors/encoder_expert_prefetch/mmlu-professional_law/switch-large-128/sparse-cache-b1-longest-v1/ble/noisyor-from-src-simplenn-token-hardce-h384-l1-drop0p5-lr1e4-bs512-seed0-validtrim/encoder_predictor_ble.ts"
      ;;
    *)
      echo "error: unknown model alias: ${alias}" >&2
      exit 1
      ;;
  esac

  apply_gpu_model_overrides
  if [[ -n "${INITIAL_HOT_EXPERT_FILE_OVERRIDE}" ]]; then
    INITIAL_HOT_EXPERT_FILE="${INITIAL_HOT_EXPERT_FILE_OVERRIDE}"
  fi
  if [[ -n "${PREDICTOR_ROOT_OVERRIDE}" ]]; then
    PREDICTOR_ROOT="${PREDICTOR_ROOT_OVERRIDE}"
  fi
  if [[ -n "${ERPP_ENCODER_MODEL_PATH_OVERRIDE}" ]]; then
    ERPP_ENCODER_MODEL_PATH="${ERPP_ENCODER_MODEL_PATH_OVERRIDE}"
  fi
  MODEL_KEY="${MODEL_ID//\//_}"
}


apply_gpu_model_overrides() {
  case "${GPU_PROFILE}" in
    default)
      GPU_MEM_GB="4"
      ;;
    gpu4gb|gpu8gb|gpu12gb|gpu24gb|gpu48gb)
      GPU_MEM_GB="${GPU_MEM_PROFILE_GB}"
      ;;
    *)
      echo "error: no GPU memory mapping for GPU profile: ${GPU_PROFILE}" >&2
      exit 1
      ;;
  esac

  # Known mapping. Add new model/GPU pairs here only after measuring or deciding the cache ratio.
  # switch-base-128: 4GB(0.125), 8GB(0.25), 12GB+(0.375)
  case "${GPU_PROFILE}:${MODEL_ALIAS}" in
    default:switch-base-128)
      CACHE_RATE="0.125"
      ;;
    gpu4gb:switch-base-128)
      CACHE_RATE="0.125"
      ;;
    gpu8gb:switch-base-128)
      CACHE_RATE="0.25"
      ;;
    gpu12gb:switch-base-128|gpu24gb:switch-base-128|gpu48gb:switch-base-128)
      CACHE_RATE="0.375"
      ;;
    *)
      echo "error: no CACHE_RATE mapping for GPU/model pair: ${GPU_PROFILE}/${MODEL_ALIAS}" >&2
      exit 1
      ;;
  esac

  if [[ -n "${GPU_MEM_GB_OVERRIDE}" ]]; then
    GPU_MEM_GB="${GPU_MEM_GB_OVERRIDE}"
  fi
  if [[ -n "${CACHE_RATE_OVERRIDE}" ]]; then
    CACHE_RATE="${CACHE_RATE_OVERRIDE}"
  fi
}

mode_config() {
  case "${BACKEND_MODE}" in
    base)
      NUM_PRED="${NUM_PRED:-0}"
      REORDER="False"
      PREEMPT="False"
      CACHE_POLICY="${CACHE_POLICY:-lru}"
      ENABLE_DECODER_WARMUP_OVERLAP="${ENABLE_DECODER_WARMUP_OVERLAP:-False}"
      ;;
    ours)
      NUM_PRED="${NUM_PRED:-6}"
      REORDER="True"
      PREEMPT="True"
      CACHE_POLICY="${CACHE_POLICY:-lru}"
      ENABLE_DECODER_WARMUP_OVERLAP="${ENABLE_DECODER_WARMUP_OVERLAP:-False}"
      ;;
    overlap)
      NUM_PRED="${NUM_PRED:-6}"
      REORDER="${REORDER:-True}"
      PREEMPT="${PREEMPT:-True}"
      CACHE_POLICY="${CACHE_POLICY:-scheduler_aware}"
      ENABLE_DECODER_WARMUP_OVERLAP="${ENABLE_DECODER_WARMUP_OVERLAP:-True}"
      ;;
    *)
      echo "error: BACKEND_MODE must be base, ours, or overlap, got: ${BACKEND_MODE}" >&2
      exit 1
      ;;
  esac
}

validate_runtime_inputs() {
  local prompt_file="${ROOT}/deps/sparse-llm-cache-scripts/dataset/mmlu/${MMLU_TASK}/${SPLIT}/prompt_list.pt"
  if [[ ! -f "${prompt_file}" ]]; then
    echo "error: prompt file not found: ${prompt_file}" >&2
    exit 1
  fi
  if [[ "${DETERMINISTIC_INIT}" == "1" || "${DETERMINISTIC_INIT}" == "true" || "${DETERMINISTIC_INIT}" == "True" ]]; then
    if [[ ! -f "${INITIAL_HOT_EXPERT_FILE}" ]]; then
      echo "error: hot expert file not found: ${INITIAL_HOT_EXPERT_FILE}" >&2
      exit 1
    fi
  fi
  if [[ "${ENABLE_ERPP_ENCODER_PREFETCH}" == "1" || "${ENABLE_ERPP_ENCODER_PREFETCH}" == "true" || "${ENABLE_ERPP_ENCODER_PREFETCH}" == "True" ]]; then
    if [[ ! -f "${ERPP_ENCODER_MODEL_PATH}" ]]; then
      echo "error: ERPP encoder model not found: ${ERPP_ENCODER_MODEL_PATH}" >&2
      exit 1
    fi
  fi
}

write_run_metadata() {
  cat > "${RUN_DIR}/config.env" <<EOF
BASELINE=promoe
BACKEND_MODE=${BACKEND_MODE}
MODEL_ALIAS=${MODEL_ALIAS}
MODEL_ID=${MODEL_ID}
MODEL_REVISION=${MODEL_REVISION}
DATASET_NAME=${DATASET_NAME}
MMLU_TASK=${MMLU_TASK}
SPLIT=${SPLIT}
RUN_ID=${RUN_ID}
RUN_DIR=${RUN_DIR}
PYTHON_BIN=${PYTHON_BIN}
GPU_PROFILE=${GPU_PROFILE}
GPU_ID=${GPU_ID}
GPU_MEM_GB=${GPU_MEM_GB}
BATCH_SIZE=${BATCH_SIZE}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS}
PROMOE_BENCHMARK_WARMUP=${PROMOE_BENCHMARK_WARMUP}
SAMPLE_INDICES=${SAMPLE_INDICES}
CACHE_RATE=${CACHE_RATE}
CACHE_POLICY=${CACHE_POLICY}
PER_LAYER_CACHE=${PER_LAYER_CACHE}
NUM_PRED=${NUM_PRED}
REORDER=${REORDER}
PREEMPT=${PREEMPT}
DETERMINISTIC_INIT=${DETERMINISTIC_INIT}
INITIAL_CACHE_POLICY=${INITIAL_CACHE_POLICY}
INITIAL_HOT_EXPERT_FILE=${INITIAL_HOT_EXPERT_FILE}
PREDICTOR_ROOT=${PREDICTOR_ROOT}
ENABLE_DECODER_WARMUP_OVERLAP=${ENABLE_DECODER_WARMUP_OVERLAP}
ENABLE_DECODER_PHASE_WARMUP_OVERLAP=${ENABLE_DECODER_PHASE_WARMUP_OVERLAP}
ENABLE_ERPP_ENCODER_PREFETCH=${ENABLE_ERPP_ENCODER_PREFETCH}
ERPP_ENCODER_MODEL_PATH=${ERPP_ENCODER_MODEL_PATH}
EOF

  {
    echo "cwd=${ROOT}"
    echo "git_commit=$(git -C "${ROOT}" rev-parse HEAD 2>/dev/null || true)"
    echo "git_status_short_begin"
    git -C "${ROOT}" status --short 2>/dev/null || true
    echo "git_status_short_end"
    echo "python=${PYTHON_BIN}"
    "${PYTHON_BIN}" --version 2>&1 || true
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
  } > "${RUN_DIR}/env.txt"
}

build_args() {
  COMMON_ARGS=(
    --model_id "${MODEL_ID}"
    --model_revision "${MODEL_REVISION}"
    --dataset "${SPLIT}"
    --batch_size "${BATCH_SIZE}"
    --max_num_batch "${MAX_NUM_BATCH}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --num_predict_expert_per_layer "${NUM_PRED}"
    --cache_rate "${CACHE_RATE}"
    --cache_policy "${CACHE_POLICY}"
    --per_layer_cache "${PER_LAYER_CACHE}"
    --reorder_experts "${REORDER}"
    --early_preempt "${PREEMPT}"
    --predict_input_mode moe_layer_logits
    --layer_predict_interval 1
    --layer_predict_max_window 3
    --layer_predict_use_last_output True
    --predictor_model_path "${PREDICTOR_ROOT}"
    --gpu_mem_limit_gb "${GPU_MEM_GB}"
  )

  if [[ "${DETERMINISTIC_INIT}" == "1" || "${DETERMINISTIC_INIT}" == "true" || "${DETERMINISTIC_INIT}" == "True" ]]; then
    COMMON_ARGS+=(--initial_cache_policy "${INITIAL_CACHE_POLICY}")
    COMMON_ARGS+=(--initial_hot_expert_file "${INITIAL_HOT_EXPERT_FILE}")
  fi

  COMMON_ARGS+=(--enable_decoder_warmup_overlap "${ENABLE_DECODER_WARMUP_OVERLAP}")
  if [[ "${ENABLE_ERPP_ENCODER_PREFETCH}" == "1" || "${ENABLE_ERPP_ENCODER_PREFETCH}" == "true" || "${ENABLE_ERPP_ENCODER_PREFETCH}" == "True" ]]; then
    COMMON_ARGS+=(
      --enable_erpp_encoder_prefetch True
      --erpp_encoder_model_path "${ERPP_ENCODER_MODEL_PATH}"
      --erpp_encoder_budgets "${ERPP_ENCODER_BUDGETS:-dynamic_noisy_or_sum}"
      --erpp_encoder_layers "${ERPP_ENCODER_LAYERS:-1,2,3,4,5}"
      --enable_erpp_encoder_jit_refill "${ENABLE_ERPP_ENCODER_JIT_REFILL:-True}"
      --erpp_encoder_jit_refill_window "${ERPP_ENCODER_JIT_REFILL_WINDOW:-1}"
      --erpp_encoder_jit_refill_floor_mode "${ERPP_ENCODER_JIT_REFILL_FLOOR_MODE:-budget}"
      --erpp_encoder_jit_refill_floor_value "${ERPP_ENCODER_JIT_REFILL_FLOOR_VALUE:--1}"
      --erpp_encoder_jit_refill_low_watermark_ratio "${ERPP_ENCODER_JIT_REFILL_LOW_WATERMARK_RATIO:-0.90}"
      --erpp_encoder_jit_refill_layers "${ERPP_ENCODER_JIT_REFILL_LAYERS:-1,2,3,4,5}"
      --erpp_encoder_jit_refill_per_idle "${ERPP_ENCODER_JIT_REFILL_PER_IDLE:--1}"
      --erpp_encoder_expert_copy_us "${ERPP_ENCODER_EXPERT_COPY_US:--1}"
      --erpp_encoder_expert_compute_us "${ERPP_ENCODER_EXPERT_COMPUTE_US:--1}"
      --enable_erpp_encoder_jit_topk_cover "${ENABLE_ERPP_ENCODER_JIT_TOPK_COVER:-True}"
    )
  else
    COMMON_ARGS+=(--enable_erpp_encoder_prefetch False)
  fi
}

write_command() {
  {
    echo "cd \"${ROOT}\""
    printf 'CUDA_VISIBLE_DEVICES=%q ' "${GPU_ID}"
    printf 'PYTHONPATH=%q ' "${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
    printf 'PROMOE_BENCHMARK_OUTPUT_DIR=%q ' "${RUN_DIR}"
    printf 'PROMOE_BENCHMARK_RUN_ID=%q ' "${RUN_ID}"
    printf 'PROMOE_BENCHMARK_BASELINE=%q ' "promoe"
    printf 'PROMOE_BENCHMARK_BACKEND_MODE=%q ' "${BACKEND_MODE}"
    printf 'PROMOE_BENCHMARK_TASK=%q ' "${MMLU_TASK}"
    printf 'PROMOE_BENCHMARK_WARMUP=%q ' "${PROMOE_BENCHMARK_WARMUP}"
    printf 'PROMOE_BENCHMARK_GPU_PROFILE=%q ' "${GPU_PROFILE}"
    printf 'PROMOE_BENCHMARK_GPU_ID=%q ' "${GPU_ID}"
    printf 'PROMOE_BENCHMARK_GPU_MEM_GB=%q ' "${GPU_MEM_GB}"
    if [[ -n "${SAMPLE_INDICES}" ]]; then
      printf 'PROMOE_SAMPLE_INDICES=%q ' "${SAMPLE_INDICES}"
    fi
    printf '%q %q' "${PYTHON_BIN}" "${DEMO_PY}"
    printf ' %q' "${COMMON_ARGS[@]}"
    echo
  } > "${RUN_DIR}/command.txt"
}

run_one_model() {
  local profile="$1"
  local alias="$2"
  gpu_config "${profile}"
  model_config "${alias}"
  mode_config

  if [[ -n "${SAMPLE_INDICES}" ]]; then
    export PROMOE_SAMPLE_INDICES="${SAMPLE_INDICES}"
    local sample_count=0
    IFS=',' read -ra sample_specs <<< "${SAMPLE_INDICES}"
    for raw in "${sample_specs[@]}"; do
      if [[ "${raw}" == *-* ]]; then
        local start="${raw%-*}"
        local stop="${raw#*-}"
        sample_count=$((sample_count + stop - start + 1))
      elif [[ -n "${raw}" ]]; then
        sample_count=$((sample_count + 1))
      fi
    done
    MAX_NUM_BATCH="${MAX_NUM_BATCH:-${sample_count}}"
  else
    unset PROMOE_SAMPLE_INDICES || true
    MAX_NUM_BATCH="${MAX_NUM_BATCH:-999999}"
  fi

  RUN_DIR="${RUN_ROOT}/${DATASET_NAME}/${MMLU_TASK}/${SPLIT}/${GPU_PROFILE_KEY}/${MODEL_KEY}/${RUN_ID}"
  mkdir -p "${RUN_DIR}"

  if [[ "${PROMOE_DRY_RUN}" != "1" ]]; then
    validate_runtime_inputs
  fi
  write_run_metadata
  build_args
  write_command

  if [[ "${PROMOE_DRY_RUN}" == "1" ]]; then
    echo "1" > "${RUN_DIR}/DRY_RUN"
    echo "dry-run: ${RUN_DIR}"
    return
  fi

  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
  export PROMOE_BENCHMARK_OUTPUT_DIR="${RUN_DIR}"
  export PROMOE_BENCHMARK_RUN_ID="${RUN_ID}"
  export PROMOE_BENCHMARK_BASELINE="promoe"
  export PROMOE_BENCHMARK_BACKEND_MODE="${BACKEND_MODE}"
  export PROMOE_BENCHMARK_TASK="${MMLU_TASK}"
  export PROMOE_BENCHMARK_WARMUP
  export PROMOE_BENCHMARK_GPU_PROFILE="${GPU_PROFILE}"
  export PROMOE_BENCHMARK_GPU_ID="${GPU_ID}"
  export PROMOE_BENCHMARK_GPU_MEM_GB="${GPU_MEM_GB}"
  export SPARSE_CACHE_LOG_STAGE_OCCUPANCY="${CACHE_STAGE_OCCUPANCY_LOG}"
  export SPARSE_CACHE_LOG_PREFETCH_DECISION
  export SPARSE_CACHE_LOG_ERPP_ENCODER_DIAGNOSTICS="${ERPP_ENCODER_DIAGNOSTICS}"

  "${PYTHON_BIN}" "${DEMO_PY}" "${COMMON_ARGS[@]}" 2>&1 | tee "${RUN_DIR}/run.log"
  "${PYTHON_BIN}" "${COLLECT_SUMMARY}" --run-dir "${RUN_DIR}"
}

for gpu_profile in ${GPU_CONFIGS}; do
  for model_alias in ${MODELS}; do
    run_one_model "${gpu_profile}" "${model_alias}"
  done
done
