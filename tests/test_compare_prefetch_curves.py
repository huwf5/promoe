from __future__ import annotations

import torch

from performance_predictor.encoder.ERPP.implement.train.compare_prefetch_curves import (
    oracle_count_metrics_from_scores,
)


def test_oracle_count_metrics_prefetches_true_count_per_sample_layer() -> None:
    scores = torch.tensor(
        [
            [
                [5.0, 4.0, 3.0, 2.0],
                [1.0, 4.0, 3.0, 2.0],
            ]
        ]
    )
    true_set = torch.tensor(
        [
            [
                [1, 0, 1, 0],
                [0, 1, 1, 0],
            ]
        ],
        dtype=torch.float32,
    )
    true_count = torch.tensor([[2, 2]], dtype=torch.long)

    metrics = oracle_count_metrics_from_scores(scores, true_set, true_count)

    assert metrics["oracle_count_overlap_count"] == 3
    assert metrics["oracle_count_total_budget"] == 4
    assert metrics["oracle_count_accuracy"] == 0.75
    assert metrics["oracle_count_micro_precision"] == 0.75
    assert metrics["oracle_count_micro_recall"] == 0.75
    assert metrics["oracle_count_macro_recall"] == 0.75
    assert metrics["oracle_count_budget_mean"] == 2.0
