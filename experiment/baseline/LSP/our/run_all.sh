#!/usr/bin/env bash
# Run our improved Promoe LSP experiments directly from the experiment tree.
# cd /mnt/huwf5/promoe
#
# Example:
#   GPU_ID=0 \
#   GPU_CONFIGS="gpu4gb gpu8gb gpu12gb gpu16gb gpu24gb gpu40gb gpu48gb" \
#   MODELS="switch-base-128 switch-base-256 switch-large-128 nllb" \
#   experiment/baseline/LSP/our/run_all.sh
#
# Smoke test:
#   PROMOE_DRY_RUN=1 GPU_CONFIGS=gpu4gb MODELS=switch-base-128 experiment/baseline/LSP/our/run_all.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
OUR_ROOT="${ROOT}/experiment/baseline/LSP/our"
PROMOE_ROOT="${ROOT}/experiment/baseline/LSP/promoe"
DEMO_PY="${ROOT}/examples/small-demo/transformers-app.py"
COLLECT_SUMMARY="${PROMOE_ROOT}/scripts/collect_summary.py"

PYTHON_BIN="${PYTHON_BIN:-/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
PROMOE_BENCHMARK_WARMUP="${PROMOE_BENCHMARK_WARMUP:-4}"
SAMPLE_INDICES="${SAMPLE_INDICES:-}"
PROMOE_DRY_RUN="${PROMOE_DRY_RUN:-0}"

DATASET_NAME="${DATASET_NAME:-mmlu}"
MMLU_TASK="${MMLU_TASK:-professional_law}"
SPLIT="${SPLIT:-validation}"

BACKEND_MODE="${BACKEND_MODE:-${MODE:-overlap}}"
PER_LAYER_CACHE="${PER_LAYER_CACHE:-False}"
DETERMINISTIC_INIT="${DETERMINISTIC_INIT:-1}"
INITIAL_CACHE_POLICY="${INITIAL_CACHE_POLICY:-hot_encoder_balanced_coverage}"
CACHE_STAGE_OCCUPANCY_LOG="${CACHE_STAGE_OCCUPANCY_LOG:-0}"
SPARSE_CACHE_LOG_PREFETCH_DECISION="${SPARSE_CACHE_LOG_PREFETCH_DECISION:-0}"
ERPP_ENCODER_DIAGNOSTICS="${ERPP_ENCODER_DIAGNOSTICS:-0}"
SPARSE_CACHE_LOG_ERPP_ENCODER_DIAGNOSTICS="${SPARSE_CACHE_LOG_ERPP_ENCODER_DIAGNOSTICS:-0}"
SPARSE_CACHE_LOG_RUNTIME_EXPERT_SET="${SPARSE_CACHE_LOG_RUNTIME_EXPERT_SET:-0}"
ENABLE_DECODER_PHASE_WARMUP_OVERLAP="${ENABLE_DECODER_PHASE_WARMUP_OVERLAP:-False}"
ENABLE_ERPP_ENCODER_PREFETCH="${ENABLE_ERPP_ENCODER_PREFETCH:-True}"
MODEL_REVISION="${MODEL_REVISION:-main}"

ERPP_ENCODER_BUDGETS="${ERPP_ENCODER_BUDGETS:-dynamic_noisy_or_sum}"
#TODO: not first layer
ERPP_ENCODER_LAYERS="${ERPP_ENCODER_LAYERS:-non_first}"
ENABLE_ERPP_ENCODER_JIT_REFILL="${ENABLE_ERPP_ENCODER_JIT_REFILL:-True}"
# TODO
ERPP_ENCODER_JIT_REFILL_WINDOW="${ERPP_ENCODER_JIT_REFILL_WINDOW:-5}"
ERPP_ENCODER_JIT_REFILL_FLOOR_MODE="${ERPP_ENCODER_JIT_REFILL_FLOOR_MODE:-budget}"
ERPP_ENCODER_JIT_REFILL_FLOOR_VALUE="${ERPP_ENCODER_JIT_REFILL_FLOOR_VALUE:--1}"
ERPP_ENCODER_JIT_REFILL_LOW_WATERMARK_RATIO="${ERPP_ENCODER_JIT_REFILL_LOW_WATERMARK_RATIO:-0.90}"
ERPP_ENCODER_JIT_REFILL_LAYERS="${ERPP_ENCODER_JIT_REFILL_LAYERS:-auto}"
ERPP_ENCODER_JIT_REFILL_PER_IDLE="${ERPP_ENCODER_JIT_REFILL_PER_IDLE:-2}"
ENABLE_ERPP_ENCODER_JIT_TOPK_COVER="${ENABLE_ERPP_ENCODER_JIT_TOPK_COVER:-False}"
# ENABLE_ERPP_ENCODER_JIT_TOPK_COVER="${ENABLE_ERPP_ENCODER_JIT_TOPK_COVER:-True}"

GPU_CONFIGS="${GPU_CONFIGS:-gpu4gb }"
# GPU_CONFIGS="${GPU_CONFIGS:-gpu4gb gpu8gb gpu12gb gpu16gb gpu24gb }"
# GPU_CONFIGS="${GPU_CONFIGS:-gpu4gb gpu8gb gpu12gb gpu16gb gpu24gb gpu40gb gpu48gb}"
# MODELS="${MODELS:-switch-base-128 switch-base-256 switch-large-128 nllb}"
MODELS="${MODELS:-switch-base-128 }"
# MODELS="${MODELS:-switch-base-128 switch-base-256 }"
# MODELS="${MODELS:-switch-base-256 switch-large-128 nllb}"
RUN_ROOT="${RUN_ROOT:-${OUR_ROOT}/runs}"

sanitize_id() {
  local value="$1"
  value="${value,,}"
  value="${value// /_}"
  value="${value//\//_}"
  value="${value//[^a-z0-9._-]/_}"
  value="$(printf '%s' "${value}" | sed -E 's/_+/_/g; s/^_+//; s/_+$//')"
  printf '%s' "${value:-unknown}"
}

detect_gpu_name() {
  local query_gpu="${GPU_ID%%,*}"
  local name=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    name="$(nvidia-smi --id="${query_gpu}" --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true)"
    if [[ -z "${name}" ]]; then
      name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sed -n "$((query_gpu + 1))p" || true)"
    fi
  fi
  printf '%s' "${name:-unknown_gpu}"
}

HOST_ID="$(sanitize_id "$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo unknown_host)")"
GPU_NAME_RAW="$(detect_gpu_name)"
GPU_NAME_ID="$(sanitize_id "${GPU_NAME_RAW}")"
RUN_ID="${RUN_ID:-our_${HOST_ID}_${GPU_NAME_ID}_$(date +%Y%m%d_%H%M%S)}"

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
  GPU_MEM_PROFILE_GB=""

  case "${profile}" in
    default) ;;
    gpu4gb) GPU_ID="${GPU4GB_GPU_ID:-${GPU_ID}}"; GPU_MEM_PROFILE_GB="4" ;;
    gpu8gb) GPU_ID="${GPU8GB_GPU_ID:-${GPU_ID}}"; GPU_MEM_PROFILE_GB="8" ;;
    gpu12gb) GPU_ID="${GPU12GB_GPU_ID:-${GPU_ID}}"; GPU_MEM_PROFILE_GB="12" ;;
    gpu16gb) GPU_ID="${GPU16GB_GPU_ID:-${GPU_ID}}"; GPU_MEM_PROFILE_GB="16" ;;
    gpu24gb) GPU_ID="${GPU24GB_GPU_ID:-${GPU_ID}}"; GPU_MEM_PROFILE_GB="24" ;;
    gpu40gb) GPU_ID="${GPU40GB_GPU_ID:-${GPU_ID}}"; GPU_MEM_PROFILE_GB="40" ;;
    gpu48gb) GPU_ID="${GPU48GB_GPU_ID:-${GPU_ID}}"; GPU_MEM_PROFILE_GB="48" ;;
    *) echo "error: unknown GPU profile: ${profile}" >&2; exit 1 ;;
  esac
}

model_hot_expert_file() {
  local alias="$1"
  echo "${ROOT}/experiment/traces/${alias}-mmlu-${MMLU_TASK}-test/hot_experts/${alias}.test.json"
}

model_config() {
  local alias="$1"
  MODEL_ALIAS="${alias}"
  PREDICTOR_ROOT=""
  ERPP_ENCODER_MODEL_PATH=""

  case "${alias}" in
    switch-base-128)
      MODEL_ID="google/switch-base-128"
      INITIAL_HOT_EXPERT_FILE="$(model_hot_expert_file switch-base-128)"
      PREDICTOR_ROOT="${ROOT}/deps/sparse-llm-cache-scripts/moe-predict-models/switch-base-128-mmlu-professional_law-test-train-validation-val/sep/decoder_sparse-cache-b1-longest-v1"
      ERPP_ENCODER_MODEL_PATH="${ROOT}/experiment/models/predictors/encoder_expert_prefetch/mmlu-professional_law/switch-base-128/sparse-cache-b1-longest-v1/ble/noisyor-from-src-simplenn-token-hardce-h384-l1-drop0p5-lr1e4-bs2-seed0-validtrim/encoder_predictor_ble.ts"
      ;;
    switch-base-256)
      MODEL_ID="google/switch-base-256"
      INITIAL_HOT_EXPERT_FILE="$(model_hot_expert_file switch-base-256)"
      PREDICTOR_ROOT="${ROOT}/deps/sparse-llm-cache-scripts/moe-predict-models/switch-base-256-mmlu-professional_law-test-train-validation-val/sep/decoder_sparse-cache-b1-longest-v1"
      ERPP_ENCODER_MODEL_PATH="${ROOT}/experiment/models/predictors/encoder_expert_prefetch/mmlu-professional_law/switch-base-256/sparse-cache-b1-longest-v1/ble/noisyor-from-src-simplenn-token-hardce-h384-l1-drop0p5-lr1e4-bs512-seed0-validtrim/encoder_predictor_ble.ts"
      ;;
    switch-large-128)
      MODEL_ID="google/switch-large-128"
      INITIAL_HOT_EXPERT_FILE="$(model_hot_expert_file switch-large-128)"
      PREDICTOR_ROOT="${ROOT}/deps/sparse-llm-cache-scripts/moe-predict-models/switch-large-128-mmlu-professional_law-test-train-validation-val/sep/decoder_sparse-cache-b1-longest-v1"
      ERPP_ENCODER_MODEL_PATH="${ROOT}/experiment/models/predictors/encoder_expert_prefetch/mmlu-professional_law/switch-large-128/sparse-cache-b1-longest-v1/ble/noisyor-from-src-simplenn-token-hardce-h384-l1-drop0p5-lr1e4-bs512-seed0-validtrim/encoder_predictor_ble.ts"
      ;;
    nllb|nllb-moe-54b)
      MODEL_ID="facebook/nllb-moe-54b"
      INITIAL_HOT_EXPERT_FILE="${ROOT}/experiment/traces/nllb-moe-54b-mmlu-${MMLU_TASK}-test/hot_experts/nllb-moe-54b.test.json"
      PREDICTOR_ROOT="${ROOT}/deps/sparse-llm-cache-scripts/moe-predict-models/nllb-moe-54b-mmlu-professional_law-test-train-validation-val/sep/decoder_sparse-cache-b1-longest-v1"
      ERPP_ENCODER_MODEL_PATH="${ROOT}/experiment/models/predictors/encoder_expert_prefetch/mmlu-professional_law/nllb-moe-54b/sparse-cache-b1-longest-v1/ble/noisyor-from-src-simplenn-token-bce-equal-top2-tokcnt0p004-h384-l1-drop0p5-lr1e4-bs512-seed0-validtrim/encoder_predictor_ble.ts"
      ;;
    *) echo "error: unknown model alias: ${alias}" >&2; exit 1 ;;
  esac

  ERPP_ENCODER_MODEL_PATH="${ERPP_ENCODER_MODEL_PATH_OVERRIDE:-${ERPP_ENCODER_MODEL_PATH}}"
  resolve_gpu_model_settings
  MODEL_KEY="${MODEL_ID//\//_}"
}

resolve_gpu_model_settings() {
  case "${GPU_PROFILE}" in
    default) GPU_MEM_GB="4" ;;
    gpu4gb|gpu8gb|gpu12gb|gpu16gb|gpu24gb|gpu40gb|gpu48gb) GPU_MEM_GB="${GPU_MEM_PROFILE_GB}" ;;
    *) echo "error: no GPU memory mapping for GPU profile: ${GPU_PROFILE}" >&2; exit 1 ;;
  esac

  case "${GPU_PROFILE}:${MODEL_ALIAS}" in
    default:switch-base-128|gpu4gb:switch-base-128) CACHE_RATE="0.125" ;;
    gpu8gb:switch-base-128) CACHE_RATE="0.25" ;;
    gpu12gb:switch-base-128) CACHE_RATE="0.375" ;;
    gpu16gb:switch-base-128) CACHE_RATE="0.5" ;;
    gpu24gb:switch-base-128) CACHE_RATE="0.8" ;;
    gpu40gb:switch-base-128) CACHE_RATE="1.0" ;;
    gpu48gb:switch-base-128) CACHE_RATE="1.0" ;;
    default:switch-base-256|gpu4gb:switch-base-256) CACHE_RATE="0.05" ;;
    gpu8gb:switch-base-256) CACHE_RATE="0.13" ;;
    gpu12gb:switch-base-256) CACHE_RATE="0.2" ;;
    gpu16gb:switch-base-256) CACHE_RATE="0.25" ;;
    gpu24gb:switch-base-256) CACHE_RATE="0.405" ;;
    gpu40gb:switch-base-256) CACHE_RATE="0.7" ;;
    gpu48gb:switch-base-256) CACHE_RATE="0.85" ;;
    default:switch-large-128|gpu4gb:switch-large-128) CACHE_RATE="0.07" ;;
    gpu8gb:switch-large-128) CACHE_RATE="0.15" ;;
    gpu12gb:switch-large-128) CACHE_RATE="0.225" ;;
    gpu16gb:switch-large-128) CACHE_RATE="0.30" ;;
    gpu24gb:switch-large-128) CACHE_RATE="0.45" ;;
    gpu40gb:switch-large-128) CACHE_RATE="0.8" ;;
    gpu48gb:switch-large-128) CACHE_RATE="0.95" ;;
    default:nllb|default:nllb-moe-54b|gpu4gb:nllb|gpu4gb:nllb-moe-54b|gpu8gb:nllb|gpu8gb:nllb-moe-54b) CACHE_RATE="skip" ;;
    gpu12gb:nllb|gpu12gb:nllb-moe-54b) CACHE_RATE="0.01" ;;
    gpu16gb:nllb|gpu16gb:nllb-moe-54b) CACHE_RATE="0.02" ;;
    gpu24gb:nllb|gpu24gb:nllb-moe-54b) CACHE_RATE="0.0625" ;;
    gpu40gb:nllb|gpu40gb:nllb-moe-54b) CACHE_RATE="0.145" ;;
    gpu48gb:nllb|gpu48gb:nllb-moe-54b) CACHE_RATE="0.2" ;;
    *) echo "error: no CACHE_RATE mapping for GPU/model pair: ${GPU_PROFILE}/${MODEL_ALIAS}" >&2; exit 1 ;;
  esac

  CACHE_RATE="${CACHE_RATE_OVERRIDE:-${CACHE_RATE}}"
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
    *) echo "error: BACKEND_MODE must be base, ours, or overlap, got: ${BACKEND_MODE}" >&2; exit 1 ;;
  esac

  if [[ "${ENABLE_ERPP_ENCODER_PREFETCH}" == "1" || "${ENABLE_ERPP_ENCODER_PREFETCH}" == "true" || "${ENABLE_ERPP_ENCODER_PREFETCH}" == "True" ]]; then
    CACHE_POLICY="scheduler_aware"
    PER_LAYER_CACHE="False"
  fi
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
  cat > "${RUN_DIR}/config.env" <<CONFIG_EOF
BASELINE=our
BACKEND_MODE=${BACKEND_MODE}
MODEL_ALIAS=${MODEL_ALIAS}
MODEL_ID=${MODEL_ID}
MODEL_REVISION=${MODEL_REVISION}
DATASET_NAME=${DATASET_NAME}
MMLU_TASK=${MMLU_TASK}
SPLIT=${SPLIT}
RUN_ID=${RUN_ID}
RUN_DIR=${RUN_DIR}
HOST_ID=${HOST_ID}
GPU_NAME=${GPU_NAME_RAW}
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
ERPP_ENCODER_BUDGETS=${ERPP_ENCODER_BUDGETS}
ERPP_ENCODER_LAYERS=${ERPP_ENCODER_LAYERS}
ENABLE_ERPP_ENCODER_JIT_REFILL=${ENABLE_ERPP_ENCODER_JIT_REFILL}
ERPP_ENCODER_JIT_REFILL_WINDOW=${ERPP_ENCODER_JIT_REFILL_WINDOW}
ERPP_ENCODER_JIT_REFILL_FLOOR_MODE=${ERPP_ENCODER_JIT_REFILL_FLOOR_MODE}
ERPP_ENCODER_JIT_REFILL_FLOOR_VALUE=${ERPP_ENCODER_JIT_REFILL_FLOOR_VALUE}
ERPP_ENCODER_JIT_REFILL_LOW_WATERMARK_RATIO=${ERPP_ENCODER_JIT_REFILL_LOW_WATERMARK_RATIO}
ERPP_ENCODER_JIT_REFILL_LAYERS=${ERPP_ENCODER_JIT_REFILL_LAYERS}
ERPP_ENCODER_JIT_REFILL_PER_IDLE=${ERPP_ENCODER_JIT_REFILL_PER_IDLE}
ENABLE_ERPP_ENCODER_JIT_TOPK_COVER=${ENABLE_ERPP_ENCODER_JIT_TOPK_COVER}
CONFIG_EOF

  {
    echo "cwd=${ROOT}"
    echo "git_commit=$(git -C "${ROOT}" rev-parse HEAD 2>/dev/null || true)"
    echo "git_status_short_begin"
    git -C "${ROOT}" status --short 2>/dev/null || true
    echo "git_status_short_end"
    echo "host=$(hostname 2>/dev/null || true)"
    echo "gpu_name=${GPU_NAME_RAW}"
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
      --erpp_encoder_budgets "${ERPP_ENCODER_BUDGETS}"
      --erpp_encoder_layers "${ERPP_ENCODER_LAYERS}"
      --enable_erpp_encoder_jit_refill "${ENABLE_ERPP_ENCODER_JIT_REFILL}"
      --erpp_encoder_jit_refill_window "${ERPP_ENCODER_JIT_REFILL_WINDOW}"
      --erpp_encoder_jit_refill_floor_mode "${ERPP_ENCODER_JIT_REFILL_FLOOR_MODE}"
      --erpp_encoder_jit_refill_floor_value "${ERPP_ENCODER_JIT_REFILL_FLOOR_VALUE}"
      --erpp_encoder_jit_refill_low_watermark_ratio "${ERPP_ENCODER_JIT_REFILL_LOW_WATERMARK_RATIO}"
      --erpp_encoder_jit_refill_layers "${ERPP_ENCODER_JIT_REFILL_LAYERS}"
      --erpp_encoder_jit_refill_per_idle "${ERPP_ENCODER_JIT_REFILL_PER_IDLE}"
      --enable_erpp_encoder_jit_topk_cover "${ENABLE_ERPP_ENCODER_JIT_TOPK_COVER}"
    )
  else
    COMMON_ARGS+=(
      --enable_erpp_encoder_prefetch False
      --enable_erpp_encoder_jit_refill False
    )
  fi
}

write_command() {
  {
    echo "cd \"${ROOT}\""
    printf 'CUDA_VISIBLE_DEVICES=%q ' "${GPU_ID}"
    printf 'PYTHONPATH=%q ' "${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
    printf 'PROMOE_BENCHMARK_OUTPUT_DIR=%q ' "${RUN_DIR}"
    printf 'PROMOE_BENCHMARK_RUN_ID=%q ' "${RUN_ID}"
    printf 'PROMOE_BENCHMARK_BASELINE=%q ' "our"
    printf 'PROMOE_BENCHMARK_BACKEND_MODE=%q ' "${BACKEND_MODE}"
    printf 'PROMOE_BENCHMARK_TASK=%q ' "${MMLU_TASK}"
    printf 'PROMOE_BENCHMARK_WARMUP=%q ' "${PROMOE_BENCHMARK_WARMUP}"
    printf 'PROMOE_BENCHMARK_GPU_PROFILE=%q ' "${GPU_PROFILE}"
    printf 'PROMOE_BENCHMARK_GPU_ID=%q ' "${GPU_ID}"
    printf 'PROMOE_BENCHMARK_GPU_MEM_GB=%q ' "${GPU_MEM_GB}"
    printf 'PROMOE_BENCHMARK_HOST=%q ' "${HOST_ID}"
    printf 'PROMOE_BENCHMARK_GPU_NAME=%q ' "${GPU_NAME_RAW}"
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
  if [[ "${CACHE_RATE}" == "skip" ]]; then
    echo "skip: GPU profile ${GPU_PROFILE} is not configured to run ${MODEL_ALIAS}"
    return
  fi
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
  export PROMOE_BENCHMARK_BASELINE="our"
  export PROMOE_BENCHMARK_BACKEND_MODE="${BACKEND_MODE}"
  export PROMOE_BENCHMARK_TASK="${MMLU_TASK}"
  export PROMOE_BENCHMARK_WARMUP
  export PROMOE_BENCHMARK_GPU_PROFILE="${GPU_PROFILE}"
  export PROMOE_BENCHMARK_GPU_ID="${GPU_ID}"
  export PROMOE_BENCHMARK_GPU_MEM_GB="${GPU_MEM_GB}"
  export PROMOE_BENCHMARK_HOST="${HOST_ID}"
  export PROMOE_BENCHMARK_GPU_NAME="${GPU_NAME_RAW}"
  export SPARSE_CACHE_LOG_STAGE_OCCUPANCY="${CACHE_STAGE_OCCUPANCY_LOG}"
  export SPARSE_CACHE_LOG_PREFETCH_DECISION
  export SPARSE_CACHE_LOG_ERPP_ENCODER_DIAGNOSTICS
  export SPARSE_CACHE_LOG_RUNTIME_EXPERT_SET

  "${PYTHON_BIN}" "${DEMO_PY}" "${COMMON_ARGS[@]}" 2>&1 | tee "${RUN_DIR}/run.log"
  "${PYTHON_BIN}" "${COLLECT_SUMMARY}" --run-dir "${RUN_DIR}"
}

for gpu_profile in ${GPU_CONFIGS}; do
  for model_alias in ${MODELS}; do
    run_one_model "${gpu_profile}" "${model_alias}"
  done
done
