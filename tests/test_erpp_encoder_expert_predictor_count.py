from __future__ import annotations

import math

import torch

from performance_predictor.encoder.ERPP.implement.train.encoder_expert_predictor_count.metrics import (
    compute_count_metrics,
)
from performance_predictor.encoder.ERPP.implement.train.encoder_expert_predictor_count.noisy_or import (
    expected_count_from_noisy_or,
    noisy_or_scores_from_probs,
)
from performance_predictor.encoder.ERPP.implement.train.encoder_expert_predictor_count.features import (
    sida_count_features_from_logits,
)
from performance_predictor.encoder.ERPP.implement.train.encoder_expert_predictor_count.models import (
    AttentionPoolingCountPredictor,
    ResidualCountPredictor,
)


def test_attention_pooling_count_predictor_outputs_bounded_layer_counts_and_ignores_padding() -> None:
    torch.manual_seed(0)
    model = AttentionPoolingCountPredictor(
        input_dim=4,
        hidden_dim=8,
        num_router_layers=3,
        num_experts=10,
        dropout=0.0,
    )
    model.eval()
    x = torch.tensor(
        [
            [[1.0, 0.0, 0.5, -0.5], [0.5, 1.0, -0.5, 0.25], [9.0, 9.0, 9.0, 9.0]],
            [[-1.0, 0.5, 0.0, 0.25], [0.0, -0.5, 1.0, 0.5], [0.25, 0.25, -0.25, 1.0]],
        ]
    )
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long)

    output = model(x, attention_mask)

    assert set(output) == {"count_pred", "count_logits"}
    assert output["count_pred"].shape == (2, 3)
    assert output["count_logits"].shape == (2, 3)
    assert torch.isfinite(output["count_pred"]).all()
    assert output["count_pred"].min().item() >= 0.0
    assert output["count_pred"].max().item() <= 10.0

    x_with_changed_padding = x.clone()
    x_with_changed_padding[0, 2] = torch.tensor([999.0, -999.0, 777.0, -777.0])
    changed_output = model(x_with_changed_padding, attention_mask)

    torch.testing.assert_close(
        output["count_pred"][0],
        changed_output["count_pred"][0],
        rtol=0.0,
        atol=1e-6,
    )


def test_noisy_or_expected_count_ignores_padding_tokens() -> None:
    probs = torch.tensor(
        [
            [
                [
                    [0.5, 0.3, 0.2],
                    [0.2, 0.5, 0.3],
                    [0.9, 0.05, 0.05],
                ]
            ]
        ],
        dtype=torch.float32,
    )
    attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.long)

    scores = noisy_or_scores_from_probs(probs, attention_mask)
    expected = torch.tensor([[[0.6, 0.65, 0.44]]], dtype=torch.float32)

    torch.testing.assert_close(scores, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        expected_count_from_noisy_or(scores),
        torch.tensor([[1.69]], dtype=torch.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_sida_count_features_return_base_count_and_ignore_padding() -> None:
    logits = torch.log(torch.tensor(
        [
            [
                [
                    [0.5, 0.3, 0.2],
                    [0.2, 0.5, 0.3],
                    [0.9, 0.05, 0.05],
                ]
            ]
        ],
        dtype=torch.float32,
    ))
    attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.long)

    features, base_count, noisy_or_scores = sida_count_features_from_logits(logits, attention_mask)

    torch.testing.assert_close(noisy_or_scores, torch.tensor([[[0.6, 0.65, 0.44]]]), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(base_count, torch.tensor([[1.69]]), rtol=1e-6, atol=1e-6)
    assert features.shape[:2] == (1, 1)
    assert features.shape[-1] >= 10
    assert torch.isfinite(features).all()

    changed_logits = logits.clone()
    changed_logits[0, 0, 2] = torch.log(torch.tensor([0.01, 0.01, 0.98]))
    changed_features, changed_base_count, changed_scores = sida_count_features_from_logits(changed_logits, attention_mask)

    torch.testing.assert_close(features, changed_features, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(base_count, changed_base_count, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(noisy_or_scores, changed_scores, rtol=1e-6, atol=1e-6)


def test_residual_count_predictor_adds_delta_to_base_count_with_bounds() -> None:
    torch.manual_seed(0)
    model = ResidualCountPredictor(
        feature_dim=14,
        hidden_dim=8,
        num_router_layers=2,
        num_experts=10,
        dropout=0.0,
    )
    features = torch.randn(3, 2, 14)
    base_count = torch.tensor([[1.0, 5.0], [2.0, 7.0], [3.0, 9.0]])

    output = model(features, base_count)

    assert set(output) == {"delta_count", "count_pred"}
    assert output["delta_count"].shape == (3, 2)
    assert output["count_pred"].shape == (3, 2)
    assert torch.isfinite(output["count_pred"]).all()
    assert output["count_pred"].min().item() >= 0.0
    assert output["count_pred"].max().item() <= 10.0


def test_compute_count_metrics_reports_error_bias_and_rates() -> None:
    pred = torch.tensor([[2.0, 4.0, 6.0], [3.0, 3.0, 7.0]])
    true = torch.tensor([[1, 5, 6], [5, 2, 7]])

    metrics = compute_count_metrics(pred, true)

    assert metrics["num_sample_layer_items"] == 6
    assert math.isclose(metrics["count_mae"], 5.0 / 6.0, rel_tol=1e-6)
    assert math.isclose(metrics["count_bias"], -1.0 / 6.0, rel_tol=1e-6)
    assert math.isclose(metrics["under_count_rate"], 2.0 / 6.0, rel_tol=1e-6)
    assert math.isclose(metrics["over_count_rate"], 2.0 / 6.0, rel_tol=1e-6)
    assert metrics["per_layer_count_mae"] == [1.5, 1.0, 0.0]


if __name__ == "__main__":
    test_attention_pooling_count_predictor_outputs_bounded_layer_counts_and_ignores_padding()
    test_noisy_or_expected_count_ignores_padding_tokens()
    test_sida_count_features_return_base_count_and_ignore_padding()
    test_residual_count_predictor_adds_delta_to_base_count_with_bounds()
    test_compute_count_metrics_reports_error_bias_and_rates()
