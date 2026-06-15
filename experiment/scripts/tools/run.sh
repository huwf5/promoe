#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python}"
# MODEL_DIR="${MODEL_DIR:-/mnt/huwf5/promoe/experiment/models/google/switch-large-128}"
MODEL_DIR="${MODEL_DIR:-/mnt/huwf5/promoe/experiment/models/facebook/nllb-moe-54b}"
CACHE_RATES="${CACHE_RATES:-0,0.001,0.01,0.025,0.05,0.1,0.125,0.25,0.375,0.5,0.75,1.0}"
GPU_SIZES_GB="${GPU_SIZES_GB:-4,8,12,24,48,80}"
OUT_DIR="${OUT_DIR:-/mnt/huwf5/promoe/experiment/scripts/tools/output}"
MODE="${MODE:-calibrate}"
BASE_RUNTIME_GB="${BASE_RUNTIME_GB:-10}"

cd /mnt/huwf5/promoe

case "${MODE}" in
  calibrate)
    # Real lower-bound path: load the model, measure non-expert GPU baseline,
    # then add cache_slots * max_single_expert_bytes for each cache_rate.
    # No safety margin is added unless you pass it directly to the Python tool.
    "${PYTHON_BIN}" experiment/scripts/tools/estimate_moe_gpu_curve.py \
      --model_id "${MODEL_DIR}" \
      --cache_rates "${CACHE_RATES}" \
      --gpu_sizes_gb "${GPU_SIZES_GB}" \
      --calibrate_gpu_baseline \
      --output_dir "${OUT_DIR}"
    ;;

  manual)
    # Fast path: use a known non-cache baseline instead of measuring it.
    # Example: MODE=manual BASE_RUNTIME_GB=8 experiment/scripts/tools/run.sh
    "${PYTHON_BIN}" experiment/scripts/tools/estimate_moe_gpu_curve.py \
      --model_id "${MODEL_DIR}" \
      --cache_rates "${CACHE_RATES}" \
      --gpu_sizes_gb "${GPU_SIZES_GB}" \
      --base_runtime_gb "${BASE_RUNTIME_GB}" \
      --output_dir "${OUT_DIR}"
    ;;

  *)
    echo "Unsupported MODE=${MODE}. Use MODE=calibrate or MODE=manual." >&2
    exit 2
    ;;
esac
