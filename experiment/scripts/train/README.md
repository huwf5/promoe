# Encoder Predictor Training

This directory contains experiment-owned training entry points for encoder predictor models.

## SIDA-GRU-SA Hard CE

Default command:

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
  experiment/scripts/train/encoder_predictor_sida_gru_sa_hard_ce.py
```

Smoke command:

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
  experiment/scripts/train/encoder_predictor_sida_gru_sa_hard_ce.py \
  --epochs 1 \
  --max-train-batches 1 \
  --max-eval-batches 1 \
  --output-dir /tmp/encoder-predictor-sida-gru-sa-hard-ce-smoke
```

The script reads sparse-cache encoder traces from:

```text
experiment/traces/switch-base-128-mmlu-professional_law/encoder_predictor_sparse_cache_trace
```

It trains only the `sida-gru-sa-hard-ce` model. Padding is ignored by expanding
`attention_mask` from `[B,T]` to `[B,L,T]`; loss and validation metrics are
computed only where `attention_mask == 1`.
## Prefetch Report

After training, generate top-1 through top-num-expert prefetch curves, budget-gap
statistics, and oracle-count accuracy outputs with:

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
  experiment/scripts/train/evaluate_encoder_predictor_prefetch.py \
  --model-dir experiment/models/predictors/encoder_expert_prefetch/mmlu-professional_law/switch-base-128/sparse-cache-b1-longest-v1/blte/sida-gru-sa-hardce-h256-rnnl2-drop0-lr1e4-bs2-seed0-validtrim \
  --output-dir experiment/models/predictors/encoder_expert_prefetch/mmlu-professional_law/switch-base-128/sparse-cache-b1-longest-v1/reports/ble-noisyor-report
```

The report evaluates each sample after trimming to valid tokens, matching the
batch=1 no-padding runtime use case. It writes `curves.csv/json`,
`budget_gap.csv`, `oracle_count_metrics.csv`, `precision_curve.png`,
`recall_curve.png`, `budget_gap_curve.png`, and `fixed_vs_oracle_accuracy.png`
when matplotlib is available.

## SRC-SimpleNN Token Hard CE h384-l1

Default command matching the original `src-simplenn-token-hard-ce-h384-l1` settings:

```bash
CUDA_VISIBLE_DEVICES=0 \
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
  experiment/scripts/train/encoder_predictor_src_simplenn_token_hard_ce.py \
  --model-path experiment/models/google/switch-base-128 \
  --dataset mmlu \
  --task-name professional_law \
  --epochs 80 \
  --batch-size 2 \
  --lr 1e-4 \
  --hidden-dim 384 \
  --src-layers 1 \
  --dropout 0.5 \
  --weight-decay 0.0 \
  --device auto \
  --early-stop \
  --early-stop-window 8 \
  --early-stop-threshold 0.003
```

The script keeps the original SRC SimpleNN token structure and hard-CE objective.
It loads only `layer0_attn_out.pt`, `attention_mask.pt`, and
`expert_selection.pt`. Each sample is trimmed to valid tokens before model
forward so padding is not trained or evaluated.

Single-model report:

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
  experiment/scripts/train/evaluate_encoder_predictor_prefetch.py \
  --model-dir experiment/models/predictors/encoder_expert_prefetch/mmlu-professional_law/switch-base-128/sparse-cache-b1-longest-v1/blte/src-simplenn-token-hardce-h384-l1-drop0p5-lr1e4-bs2-seed0-validtrim \
  --output-dir experiment/models/predictors/encoder_expert_prefetch/mmlu-professional_law/switch-base-128/sparse-cache-b1-longest-v1/reports/ble-noisyor-report \
  --device auto \
  --aggregator noisy_or
```

Original versus experiment-trained comparison on the same sparse-cache trace:

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
  experiment/scripts/train/evaluate_encoder_predictor_prefetch.py \
  --model-dirs \
    performance_predictor/encoder/ERPP/implement/model/src-simplenn-token-hard-ce-h384-l1 \
    experiment/models/predictors/encoder_expert_prefetch/mmlu-professional_law/switch-base-128/sparse-cache-b1-longest-v1/blte/src-simplenn-token-hardce-h384-l1-drop0p5-lr1e4-bs2-seed0-validtrim \
  --labels \
    original-src-simplenn-token-hard-ce-h384-l1 \
    experiment-src-simplenn-token-hardce-h384-l1 \
  --trace-dir experiment/traces/switch-base-128-mmlu-professional_law/encoder_predictor_sparse_cache_trace \
  --output-dir experiment/models/predictors/encoder_expert_prefetch/mmlu-professional_law/switch-base-128/sparse-cache-b1-longest-v1/reports/compare-ble-noisyor-src-original-vs-experiment \
  --device auto \
  --aggregator noisy_or
```

The report writes global and per-layer fixed-budget top1-through-top-num-expert
curves, fixed budget gap, noisy-or sum dynamic budget gap, dynamic-budget
prefetch accuracy, oracle-count metrics, and plots when matplotlib is available.

## TorchScript Export

After training a BLTE predictor, export both hidden-only TorchScript artifacts in one run:

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
  experiment/scripts/train/export_encoder_predictor_torchscript.py \
  --blte-dir experiment/models/predictors/encoder_expert_prefetch/mmlu-professional_law/switch-base-256/sparse-cache-b1-longest-v1/blte/src-simplenn-token-hardce-h384-l1-drop0p5-lr1e4-bs512-seed0-validtrim
```

Default outputs:

```text
<blte-dir>/encoder_predictor_blte.ts
<ble-dir>/encoder_predictor_ble.ts
```

The BLTE artifact exposes `forward(hidden) -> [B,L,T,E]`. The BLE artifact exposes
`forward(hidden) -> [B,L,E]` and applies softmax plus noisy-or over all provided
tokens. No runtime attention mask is required; the exported contract assumes the
input hidden tensor contains no padding tokens.


## Predictor Taxonomy

New training and report outputs use the BLTE/BLE taxonomy:

```text
experiment/models/predictors/
  encoder_expert_prefetch/
    mmlu-professional_law/
      switch-base-128/
        sparse-cache-b1-longest-v1/
          blte/<run_name>/
          ble/<ble_view_name>/
          reports/<report_name>/
```

The old `experiment/models/encoder_predictor/...` layout has been migrated into
this taxonomy and the old directory has been removed. Historical reports copied
from the old layout live under `reports/legacy-from-*`. New training, BLE
derivation, report generation, and comparisons should use only the taxonomy
paths shown above.
