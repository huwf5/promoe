#!/usr/bin/env bash
# =============================================================================
# Encoder Expert Prefetch 实验工作流
# =============================================================================
#
# 本脚本描述从「统计热点 expert」到「训练 encoder predictor、导出 TorchScript
# 并评估预取质量」的完整流水线。三步顺序执行，后一步依赖前一步产物。
#
# 默认 workload:
#   model   = switch-base-128 / switch-base-256 / switch-large-128
#   dataset = mmlu
#   task    = professional_law
#
# 推荐 Python 环境:
#   /mnt/huwf5/conda-envs/promoe-moe-cache/bin/python
#
# -----------------------------------------------------------------------------
# Step 1. export hot experts
# -----------------------------------------------------------------------------
# 目的:
#   在真实推理路径上 hook router，统计每个 expert 被 dispatch 的次数，
#   得到该 workload 下的「热点 expert」快照，供 cache 预热 / 容量规划参考。
#
# 脚本:
#   experiment/scripts/trace/export_hot_experts.py
#
# 输入:
#   experiment/datasets/<dataset>/<task>/<split>/prompt_list.pt
#
# 输出:
#   experiment/traces/<model>-<dataset>-<task>-<split>/hot_experts/<model>.<split>.json
#
# 口径:
#   - small-demo 输入形状：动态 padding，--enc-pad-to 仅作截断上限
#   - 统计 router dispatch mask 中非零条目（capacity drop 后的真实命中）
#   - --router-topk auto 读取 config.num_selected_experts
#
# 示例 (switch-base-128):
#   python3 experiment/scripts/trace/export_hot_experts.py \
#     --model-path experiment/models/google/switch-base-128 \
#     --dataset mmlu \
#     --task-name professional_law \
#     --split test \
#     --router-topk auto
#
# 大模型 GPU 放不下的示例 (switch-base-256, 单卡 + CPU offload):
#   /mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
#     experiment/scripts/trace/export_hot_experts.py \
#     --model-path experiment/models/google/switch-base-256 \
#     --dataset mmlu \
#     --task-name professional_law \
#     --split test \
#     --router-topk auto \
#     --torch-dtype auto \
#     --device-map auto \
#     --device cuda:1 \
#     --max-cpu-memory 256GiB \
#     --offload-folder /tmp/promoe-hf-offload/switch-base-256
#
# -----------------------------------------------------------------------------
# Step 2. get encoder hook data for encoder_predictor
# -----------------------------------------------------------------------------
# 目的:
#   通过 sparse cache 运行时 hook encoder，导出 predictor 训练所需的 trace：
#   layer0 attention 输出（predictor 输入特征）+ expert_selection（标签）。
#
# 脚本 (推荐):
#   experiment/scripts/trace/encoder_predictor/export_sparse_cache_encoder_trace.py
#
# 说明:
#   - 与 small-demo 相同的 sparse cache 入口 (hack_transformers)
#   - 默认 padding=longest, batch_size=1, max_input_tokens=512
#   - test split → train/，validation split → validation/
#   - --storage-dtype 默认 auto：从 model.config.torch_dtype 推断保存 dtype
#   - 存储时可能 pad 到 split 内最大 token 长度；训练必须用 attention_mask.pt 过滤
#
# 输入:
#   experiment/datasets/<dataset>/<task>/test/prompt_list.txt
#   experiment/datasets/<dataset>/<task>/validation/prompt_list.txt
#
# 输出:
#   experiment/traces/<model>-<dataset>-<task>/encoder_predictor_sparse_cache_trace/
#     train/   layer0_attn_out.pt, attention_mask.pt, expert_selection.pt, ...
#     validation/  (同上)
#     metadata.json
#
# 关键文件:
#   layer0_attn_out.pt   [S, T, H]  predictor 输入
#   attention_mask.pt    [S, T]     1=真实 token, 0=存储 padding
#   expert_selection.pt  [S, L, T, K]  router 选中的 expert 索引
#
# 示例:
#   /mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
#     experiment/scripts/trace/encoder_predictor/export_sparse_cache_encoder_trace.py \
#     --model-path experiment/models/google/switch-base-128 \
#     --dataset mmlu \
#     --task-name professional_law \
#     --device cuda:0 \
#     --storage-dtype auto \
#     --model-torch-dtype auto \
#     --print-status
#
# 旧版 ERPP trace（已弃用，仅作历史对照）:
#   experiment/scripts/trace/run_encoder_erpp_trace.py
#   → experiment/traces/.../encoder_erpp_trace/
#
# -----------------------------------------------------------------------------
# Step 3. train_encoder_predictor + export TorchScript + prefetch 评估报告
# -----------------------------------------------------------------------------
# 目的:
#   用 Step 2 的 trace 训练 encoder predictor（BLTE 模型），导出 hidden-only
#   TorchScript（BLTE + BLE）供 C++ runtime 使用，并在 validation split 上
#   生成预取质量报告。
#
# 训练脚本 (任选其一):
#   experiment/scripts/train/encoder_predictor_src_simplenn_token_hard_ce.py
#   experiment/scripts/train/encoder_predictor_sida_gru_sa_hard_ce.py
#
# 训练输出 (BLTE taxonomy):
#   experiment/models/predictors/encoder_expert_prefetch/
#     <workload>/<base_model>/sparse-cache-b1-longest-v1/blte/<run_name>/
#
# 训练示例 (SRC-SimpleNN token hard-CE, 对齐原 h384-l1 配置):
#   只需要指定 model + dataset + task；脚本会自动使用 Step 2 默认 trace，
#   并输出到对应 predictor taxonomy 的 BLTE 目录。
#
#   CUDA_VISIBLE_DEVICES=0 \
#   /mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
#     experiment/scripts/train/encoder_predictor_src_simplenn_token_hard_ce.py \
#     --model-path experiment/models/google/switch-base-128 \
#     --dataset mmlu \
#     --task-name professional_law \
#     --epochs 80 \
#     --batch-size 512 \
#     --lr 1e-4 \
#     --hidden-dim 384 \
#     --src-layers 1 \
#     --dropout 0.5 \
#     --device auto \
#     --early-stop \
#     --early-stop-window 8 \
#     --early-stop-threshold 0.003
#
# ---- 训练后：导出 TorchScript (hidden-only BLTE + BLE) ----
# 目的:
#   将 Step 3 训练得到的 BLTE checkpoint 导出为 C++ runtime 可用的 TorchScript：
#   token 级 BLTE logits，以及经 noisy_or 聚合成层内 expert score 的 BLE 模型。
#   运行时仅需 hidden states，不需要 attention mask（batch=1、无 storage padding 口径）。
#
# 脚本:
#   experiment/scripts/train/export_encoder_predictor_torchscript.py
#
# 输入:
#   experiment/models/predictors/encoder_expert_prefetch/
#     <workload>/<base_model>/sparse-cache-b1-longest-v1/blte/<run_name>/
#       config.json, best_model.pt (或 model.pt)
#
# 输出:
#   blte/<run_name>/encoder_predictor_blte.ts      forward(hidden) -> [B,L,T,E]
#   blte/<run_name>/export_manifest.json
#   ble/noisyor-from-<run_name>/encoder_predictor_ble.ts   forward(hidden) -> [B,L,E]
#   ble/noisyor-from-<run_name>/export_manifest.json
#   (若 ble 目录不存在会自动创建并写入 ble_manifest.json)
#
# 可选参数:
#   --checkpoint best_model.pt (默认，不存在时回退 model.pt)
#   --skip-blte / --skip-ble   仅导出其中一种
#   --dry-run                  只打印将写入的路径与 manifest，不实际 trace
#
# 导出示例 (与上方训练 run 对应):
#   /mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
#     experiment/scripts/train/export_encoder_predictor_torchscript.py \
#     --blte-dir experiment/models/predictors/encoder_expert_prefetch/mmlu-professional_law/switch-base-128/sparse-cache-b1-longest-v1/blte/src-simplenn-token-hardce-h384-l1-drop0p5-lr1e4-bs2-seed0-validtrim
#
# 评估脚本:
#   experiment/scripts/train/evaluate_encoder_predictor_prefetch.py
#
# 评估在 validation split 上、按 sample 去掉 padding 后逐层统计，与
# batch=1 无 padding 的运行时口径一致。默认 aggregator=noisy_or。
#
# ---- 3a. fixed-budget 预取曲线 (top-1 … top-num_expert) ----
# 对每个 sample-layer，把 token 级 expert 概率经 noisy_or 聚合成层内 expert
# score，再按 score 降序取前 m 个 expert 作为预取集合。
#   precision@m = 命中真实 expert 数 / m
#   recall@m    = 命中真实 expert 数 / 该层真实使用 expert 数
# m 从 1 扫到 num_experts，得到 top1-top-num_expert 预取准确率曲线。
#
# 输出:
#   curves.csv / precision_curve.png / recall_curve.png
#   layer_curves.csv  (逐层曲线)
#
# ---- 3b. noisy_or 动态 budget vs 真实 expert 数量 ----
# 对每层 expert score 做 noisy_or 后在 expert 维求和，得到 predictor 认为的
# 预取 budget（浮点）；与 trace 中该层真实使用的 expert 个数 true_count 对比。
#   noisy_or_budget = sum_e noisy_or_score[e]
#   gap = noisy_or_budget - true_count
#
# 输出:
#   noisy_or_count_gap.csv / noisy_or_count_gap.png
#   noisy_or_count_gap_by_layer.csv / noisy_or_count_gap_by_layer.png
#
# ---- 3c. 动态 budget 下的预取准确率 ----
# 用 predictor 自己估计的 budget：budget = ceil(noisy_or_budget)，取 top-budget
# 个 expert 作为预取集合，看真实使用的 expert 有多少落在这个集合里。
#   dynamic_budget_precision = overlap / budget   (预取命中率)
#   dynamic_budget_recall    = overlap / true_count (真实 expert 被覆盖比例)
#
# 输出:
#   dynamic_budget_prefetch_accuracy.csv / .png
#   dynamic_budget_prefetch_accuracy_by_layer.csv / .png
#
# ---- 3d. oracle-count 预取率 ----
# 把 budget 设为 oracle：直接用该层真实需要的 expert 个数 true_count 作为
# 预取数量，在 predictor 排序下取 top-true_count 个 expert，衡量「若已知
# 正确数量，predictor 排序能覆盖多少真实 expert」。
#   oracle_count_accuracy = overlap / true_count
#   (等价于 micro_precision = micro_recall)
#
# 输出:
#   oracle_count_metrics.csv / fixed_vs_oracle_accuracy.png
#
# 评估示例 (单模型):
#   /mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
#     experiment/scripts/train/evaluate_encoder_predictor_prefetch.py \
#     --model-dir experiment/models/predictors/encoder_expert_prefetch/mmlu-professional_law/switch-base-128/sparse-cache-b1-longest-v1/blte/src-simplenn-token-hardce-h384-l1-drop0p5-lr1e4-bs2-seed0-validtrim \
#     --trace-dir experiment/traces/switch-base-128-mmlu-professional_law/encoder_predictor_sparse_cache_trace \
#     --output-dir experiment/models/predictors/encoder_expert_prefetch/mmlu-professional_law/switch-base-128/sparse-cache-b1-longest-v1/reports/ble-noisyor-report \
#     --device auto \
#     --aggregator noisy_or
#
# 评估示例 (原模型 vs 新训练模型对比):
#   /mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
#     experiment/scripts/train/evaluate_encoder_predictor_prefetch.py \
#     --model-dirs \
#       performance_predictor/encoder/ERPP/implement/model/src-simplenn-token-hard-ce-h384-l1 \
#       experiment/models/predictors/encoder_expert_prefetch/mmlu-professional_law/switch-base-128/sparse-cache-b1-longest-v1/blte/src-simplenn-token-hardce-h384-l1-drop0p5-lr1e4-bs2-seed0-validtrim \
#     --labels \
#       original-src-simplenn-token-hard-ce-h384-l1 \
#       experiment-src-simplenn-token-hardce-h384-l1 \
#     --trace-dir experiment/traces/switch-base-128-mmlu-professional_law/encoder_predictor_sparse_cache_trace \
#     --output-dir experiment/models/predictors/encoder_expert_prefetch/mmlu-professional_law/switch-base-128/sparse-cache-b1-longest-v1/reports/compare-ble-noisyor-src-original-vs-experiment \
#     --device auto \
#     --aggregator noisy_or
#
# =============================================================================
# 以下为历史运行记录（注释掉的实际命令，可按需取消注释执行）
# =============================================================================

# # Step 1 — switch-base-128 hot experts
# python3 experiment/scripts/trace/export_hot_experts.py \
#     --model-path experiment/models/google/switch-base-128 \
#     --dataset mmlu \
#     --task-name professional_law \
#     --split test

# # Step 1 — switch-base-256 hot experts (b15a)
# python3 experiment/scripts/trace/export_hot_experts.py \
#   --model-path experiment/models/google/switch-base-256 \
#   --dataset mmlu \
#   --task-name professional_law \
#   --split test \
#   --router-topk auto \
#   --device-map auto \
#   --device cuda:1 \
#   --max-cpu-memory 256GiB \
#   --offload-folder /tmp/promoe-hf-offload/switch-base-256

# # Step 1 — switch-large-128 hot experts
# python3 experiment/scripts/trace/export_hot_experts.py \
#     --model-path experiment/models/google/switch-large-128 \
#     --dataset mmlu \
#     --task-name professional_law \
#     --split test \
#     --router-topk 1 \
#     --device-map auto \
#     --device cuda:0 \
#     --offload-folder /tmp/promoe-hf-offload/switch-large-128

# # Step 2 — encoder_predictor sparse cache trace (switch-base-128)
# /mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
#   experiment/scripts/trace/encoder_predictor/export_sparse_cache_encoder_trace.py \
#   --model-path experiment/models/google/switch-base-128 \
#   --dataset mmlu \
#   --task-name professional_law \
#   --device cuda:0 \
#   --print-status

# # Step 2 — 旧版 ERPP trace (e90d, 已弃用)
# /mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
#   experiment/scripts/trace/run_encoder_erpp_trace.py \
#   --model-path experiment/models/google/switch-base-128 \
#   --dataset mmlu \
#   --task-name professional_law \
#   --device cuda:0

# # Step 2 — 旧版 ERPP trace (2c6a, switch-base-256, 已弃用)
# /mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
#   experiment/scripts/trace/run_encoder_erpp_trace.py \
#   --model-path experiment/models/google/switch-base-256 \
#   --dataset mmlu \
#   --task-name professional_law \
#   --device cuda:0

# =============================================================================
# switch-base-256 训练与评估闭环正式运行命令
# =============================================================================
# 说明:
#   这些命令保留为注释，避免直接执行 workflow.sh 时误启动大模型任务。
#   需要运行时复制对应命令，或去掉每行前面的 "# "。
#
# 自动默认路径:
#   hot experts:
#     experiment/traces/switch-base-256-mmlu-professional_law-test/hot_experts/switch-base-256.test.json
#   encoder predictor trace:
#     experiment/traces/switch-base-256-mmlu-professional_law/encoder_predictor_sparse_cache_trace/
#   trained BLTE predictor:
#     experiment/models/predictors/encoder_expert_prefetch/mmlu-professional_law/switch-base-256/sparse-cache-b1-longest-v1/blte/src-simplenn-token-hardce-h384-l1-drop0p5-lr1e4-bs2-seed0-validtrim/
#   BLE/noisy-or report:
#     experiment/models/predictors/encoder_expert_prefetch/mmlu-professional_law/switch-base-256/sparse-cache-b1-longest-v1/reports/ble-noisyor-report/

# # 1. get_hot_expert: switch-base-256
# /mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
#   experiment/scripts/trace/export_hot_experts.py \
#   --model-path experiment/models/google/switch-base-256 \
#   --dataset mmlu \
#   --task-name professional_law \
#   --split test \
#   --router-topk auto \
#   --torch-dtype auto \
#   --device-map auto \
#   --device cuda:1 \
#   --max-cpu-memory 256GiB \
#   --offload-folder /tmp/promoe-hf-offload/switch-base-256

# # 2. get encoder_predictor hook data: switch-base-256
# /mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
#   experiment/scripts/trace/encoder_predictor/export_sparse_cache_encoder_trace.py \
#   --model-path experiment/models/google/switch-base-256 \
#   --dataset mmlu \
#   --task-name professional_law \
#   --device cuda:0 \
#   --print-status

  /mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
    experiment/scripts/trace/encoder_predictor/export_sparse_cache_encoder_trace.py \
    --model-path /mnt/huwf5/promoe/experiment/models/facebook/nllb-moe-54b \
    --dataset mmlu \
    --task-name professional_law \
    --device cuda:1 \
    --cache-rate 0.01 \
    --print-status

# # 3. train encoder_predictor: switch-base-256, src-simplenn-token-hard-ce-h384-l1
# CUDA_VISIBLE_DEVICES=0 \
# /mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
#   experiment/scripts/train/encoder_predictor_src_simplenn_token_hard_ce.py \
#   --model-path experiment/models/google/switch-large-128 \
#   --dataset mmlu \
#   --task-name professional_law \
#   --epochs 80 \
#   --batch-size 512 \
#   --lr 1e-4 \
#   --hidden-dim 384 \
#   --src-layers 1 \
#   --dropout 0.5 \
#   --device auto \
#   --early-stop \
#   --early-stop-window 8 \
#   --early-stop-threshold 0.003

cd /mnt/huwf5/promoe

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/mnt/huwf5/promoe/src:/mnt/huwf5/promoe/deps/transformers/src \
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
  experiment/scripts/train/encoder_predictor_src_simplenn_token_hard_ce.py \
  --trace-dir experiment/traces/nllb-moe-54b-mmlu-professional_law/encoder_predictor_sparse_cache_trace \
  --loss-type multi_label_bce \
  --device cuda:0 \
  --epochs 80 \
  --batch-size 2 \
  --lr 1e-4 \
  --hidden-dim 384 \
  --src-layers 1 \
  --dropout 0.5 \
  --early-stop \
  --early-stop-window 8 \
  --early-stop-threshold 0.003

  # PYTHONPATH=/mnt/huwf5/promoe/src:/mnt/huwf5/promoe/deps/transformers/src \
  # /mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
  #   experiment/scripts/train/encoder_predictor_src_simplenn_token_hard_ce.py \
  #   --trace-dir /mnt/huwf5/promoe/experiment/traces/nllb-moe-54b-mmlu-
  #   professional_law/encoder_predictor_sparse_cache_trace \
  #   --loss-type multi_label_bce \
  #   --no-use-expert-weights-in-loss \
  #   --device cuda:0

/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
  experiment/scripts/train/evaluate_encoder_predictor_prefetch.py \
  --model-dir /mnt/huwf5/promoe/experiment/models/predictors/encoder_expert_prefetch/mmlu-professional_law/nllb-moe-54b/sparse-cache-b1-longest-v1/blte/src-simplenn-token-bce-equal-top2-tokcnt0p1-layercnt0p05-h384-l1-drop0p5-lr1e4-bs512-seed0-validtrim \
  --device cuda:0

# # 3b. export encoder_predictor TorchScript: hidden-only BLTE + BLE
# # 默认输出:
# #   blte/<run-name>/encoder_predictor_blte.ts
# #   ble/noisyor-from-<run-name>/encoder_predictor_ble.ts

# /mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
#   experiment/scripts/train/export_encoder_predictor_torchscript.py \
#   --blte-dir experiment/models/predictors/encoder_expert_prefetch/mmlu-professional_law/switch-large-128/sparse-cache-b1-longest-v1/blte/src-simplenn-token-hardce-h384-l1-drop0p5-lr1e4-bs512-seed0-validtrim

# # 3b-alt. export encoder_predictor TorchScript: switch-base-256 已训练 bs512 版本
# /mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
#   experiment/scripts/train/export_encoder_predictor_torchscript.py \
#   --blte-dir experiment/models/predictors/encoder_expert_prefetch/mmlu-professional_law/switch-base-256/sparse-cache-b1-longest-v1/blte/src-simplenn-token-hardce-h384-l1-drop0p5-lr1e4-bs512-seed0-validtrim

# # 4. evaluate encoder_predictor: switch-base-256, BLE/noisy_or report
# # a. fixed-budget top1-top-num_expert 曲线:
# #      curves.csv, layer_curves.csv, precision_curve.png, recall_curve.png
# # b. noisy_or budget 与真实 expert 数量差距:
# #      noisy_or_count_gap.csv/.png, noisy_or_count_gap_by_layer.csv/.png
# # c. 模型 dynamic budget 下 top-budget 预取准确率:
# #      dynamic_budget_prefetch_accuracy.csv/.png,
# #      dynamic_budget_prefetch_accuracy_by_layer.csv/.png
# # d. oracle-count budget 下的预取准确率对比:
# #      oracle_count_metrics.csv, fixed_vs_oracle_accuracy.png

# /mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
#   experiment/scripts/train/evaluate_encoder_predictor_prefetch.py \
#   --model-dir experiment/models/predictors/encoder_expert_prefetch/mmlu-professional_law/switch-base-256/sparse-cache-b1-longest-v1/blte/src-simplenn-token-hardce-h384-l1-drop0p5-lr1e4-bs2-seed0-validtrim \
#   --trace-dir experiment/traces/switch-base-256-mmlu-professional_law/encoder_predictor_sparse_cache_trace \
#   --output-dir experiment/models/predictors/encoder_expert_prefetch/mmlu-professional_law/switch-base-256/sparse-cache-b1-longest-v1/reports/ble-noisyor-report \
#   --device auto \
#   --aggregator noisy_or

