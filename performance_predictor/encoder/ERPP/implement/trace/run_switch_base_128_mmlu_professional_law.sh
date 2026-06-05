#!/usr/bin/env bash
set -euo pipefail

# Run from repo root, or this script will cd to repo root automatically.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${REPO_ROOT}"

MODEL_PATH="${MODEL_PATH:-deps/sparse-llm-cache-scripts/huggingface-modules/modules/transformers_modules/google/switch-base-128}"
DATASET_DIR="${DATASET_DIR:-deps/sparse-llm-cache-scripts/dataset/mmlu/professional_law}"
TRAIN_PROMPT_FILE="${TRAIN_PROMPT_FILE:-${DATASET_DIR}/test/prompt_list.txt}"
VALIDATION_PROMPT_FILE="${VALIDATION_PROMPT_FILE:-${DATASET_DIR}/validation/prompt_list.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-performance_predictor/encoder/ERPP/data/traces/switch-base-128-mmlu-professional_law-erpp}"

DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-512}"
STORAGE_DTYPE="${STORAGE_DTYPE:-float32}"
SEED="${SEED:-42}"
PADDING="${PADDING:-max_length}"

mkdir -p "$(dirname "${OUTPUT_DIR}")"

echo "[ERPP] repo root: ${REPO_ROOT}"
echo "[ERPP] model: ${MODEL_PATH}"
echo "[ERPP] train prompts: ${TRAIN_PROMPT_FILE}"
echo "[ERPP] validation prompts: ${VALIDATION_PROMPT_FILE}"
echo "[ERPP] output: ${OUTPUT_DIR}"
echo "[ERPP] device=${DEVICE} batch_size=${BATCH_SIZE} max_input_tokens=${MAX_INPUT_TOKENS} storage_dtype=${STORAGE_DTYPE}"

python3 performance_predictor/encoder/ERPP/implement/trace/export_erpp_encoder_trace.py \
  --model-path "${MODEL_PATH}" \
  --train-prompt-file "${TRAIN_PROMPT_FILE}" \
  --validation-prompt-file "${VALIDATION_PROMPT_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-input-tokens "${MAX_INPUT_TOKENS}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --padding "${PADDING}" \
  --storage-dtype "${STORAGE_DTYPE}" \
  --verify \
  --print-status

python3 performance_predictor/encoder/ERPP/implement/trace/inspect_erpp_trace.py \
  --trace-dir "${OUTPUT_DIR}" \
  --split validation \
  --sample 0 \
  --token 0 \
  --topk 5

echo "[ERPP] done. Summary files:"
echo "  ${OUTPUT_DIR}/metadata.json"
echo "  ${OUTPUT_DIR}/run_log.json"
find "${OUTPUT_DIR}" -maxdepth 2 -type f -printf '%p %s bytes\n'
