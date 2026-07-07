from __future__ import annotations

import pytest
import torch

from experiment.scripts.evaluate.evaluate_dynamic_prefetch_policy_ablation import (
    best_fixed_policy_summary,
    dynamic_policy_summary_from_scores,
    hard_union_summary_from_logits,
    oracle_count_summary_from_scores,
)


def test_policy_summaries_report_micro_precision_recall_and_f1() -> None:
    scores = torch.tensor(
        [
            [[0.9, 0.8, 0.1, 0.0]],
            [[0.9, 0.2, 0.1, 0.0]],
        ],
        dtype=torch.float32,
    )
    true_set = torch.tensor(
        [
            [[True, True, False, False]],
            [[True, False, True, False]],
        ]
    )
    true_count = torch.tensor([[2], [2]], dtype=torch.long)

    fixed = best_fixed_policy_summary(scores, true_set, true_count)
    dynamic = dynamic_policy_summary_from_scores(scores, true_set, true_count)
    oracle = oracle_count_summary_from_scores(scores, true_set, true_count)

    assert fixed["policy"] == "best_fixed"
    assert fixed["budget"] == 3
    assert fixed["micro_precision"] == pytest.approx(4 / 6)
    assert fixed["micro_recall"] == pytest.approx(1.0)
    assert fixed["micro_f1"] == pytest.approx(0.8)

    assert dynamic["policy"] == "dynamic_noisyor"
    assert dynamic["micro_precision"] == pytest.approx(0.75)
    assert dynamic["micro_recall"] == pytest.approx(0.75)
    assert dynamic["micro_f1"] == pytest.approx(0.75)

    assert oracle["policy"] == "oracle_count"
    assert oracle["micro_precision"] == pytest.approx(0.75)
    assert oracle["micro_recall"] == pytest.approx(0.75)
    assert oracle["micro_f1"] == pytest.approx(0.75)


def test_hard_union_summary_uses_variable_union_size() -> None:
    logits = torch.tensor(
        [
            [[
                [9.0, 8.0, 0.0, 0.0],
                [8.0, 9.0, 0.0, 0.0],
            ]]
        ],
        dtype=torch.float32,
    )
    attention_mask = torch.tensor([[1, 1]], dtype=torch.long)
    true_set = torch.tensor([[[True, False, True, False]]])
    true_count = torch.tensor([[2]], dtype=torch.long)

    top1 = hard_union_summary_from_logits(logits, attention_mask, true_set, true_count, token_topk=1)
    top2 = hard_union_summary_from_logits(logits, attention_mask, true_set, true_count, token_topk=2)

    assert top1["policy"] == "hard_union_top1"
    assert top1["pred_budget_int_mean"] == pytest.approx(2.0)
    assert top1["micro_precision"] == pytest.approx(0.5)
    assert top1["micro_recall"] == pytest.approx(0.5)

    assert top2["policy"] == "hard_union_top2"
    assert top2["pred_budget_int_mean"] == pytest.approx(2.0)
    assert top2["micro_precision"] == pytest.approx(0.5)
    assert top2["micro_recall"] == pytest.approx(0.5)
