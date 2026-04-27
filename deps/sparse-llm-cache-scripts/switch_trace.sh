#!/usr/bin/env bash
#
# 采集 HuggingFace Switch Transformers 的 router trace，产出与 train_predict_model.py
# 数据流等价的 encoder/、decoder/ 契约目录（见 switch-trace-export/README.md）。
#
# 可通过环境变量覆盖（示例）：
#   MODEL_PATH=/path/to/switch-base-128 OUTPUT_DIR=/tmp/switch-traces ./switch_trace.sh
#   DEVICE=cpu BATCH_SIZE=2 MAX_NEW_TOKENS=32 ./switch_trace.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/huggingface-modules/modules/transformers_modules/google/switch-base-128}"
PROMPT_FILE="${PROMPT_FILE:-${SCRIPT_DIR}/dataset/chatgpt-prompts/prompt_list.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${DEPS_ROOT}/moe-traces/switch-base-128-chatgpt-prompts}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-42}"
EXTRA_FLAGS=("--verify")

[[ -d "${MODEL_PATH}" ]] || { echo "Error: model dir not found: ${MODEL_PATH}" >&2; exit 1; }
[[ -f "${PROMPT_FILE}" ]] || { echo "Error: prompt file not found: ${PROMPT_FILE}" >&2; exit 1; }

echo "Using: MODEL_PATH=${MODEL_PATH}"
echo "       PROMPT_FILE=${PROMPT_FILE}"
echo "       OUTPUT_DIR=${OUTPUT_DIR}"
echo "       MAX_NEW_TOKENS=${MAX_NEW_TOKENS} BATCH_SIZE=${BATCH_SIZE} DEVICE=${DEVICE} SEED=${SEED}"

python3 "${SCRIPT_DIR}/switch-trace-export/switch_trace_export.py" \
    --model-path "${MODEL_PATH}" \
    --prompt-file "${PROMPT_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    "${EXTRA_FLAGS[@]}"
