from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from experiment.scripts.train.evaluate_encoder_predictor_prefetch import (
    build_model_from_config,
    budget_gap_curve,
    curve_from_scores,
    dynamic_budget_prefetch_by_layer,
    dynamic_budget_prefetch_metrics,
    resolve_report_output_dir,
    resolve_checkpoint_path,
    validate_report_output_dir,
    write_multi_csv,
    write_ble_artifact_if_taxonomy_blte,
    layer_curve_from_scores,
    noisy_or_budget_gap_by_layer,
    noisy_or_budget_gap_metrics,
    oracle_count_metrics_from_scores,
    sample_targets_from_expert_selection,
    token_scores_from_logits,
)


def test_sample_targets_ignore_padding_and_count_unique_experts() -> None:
    expert_selection = torch.tensor([[[[0], [2], [2], [1]]]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)

    expert_set, true_count = sample_targets_from_expert_selection(
        expert_selection,
        attention_mask,
        num_experts=4,
    )

    assert expert_set.shape == (1, 1, 4)
    assert true_count.shape == (1, 1)
    assert torch.equal(expert_set[0, 0], torch.tensor([True, False, True, False]))
    assert true_count[0, 0].item() == 2


def test_sample_targets_include_both_top2_experts_and_ignore_masked_slots() -> None:
    expert_selection = torch.tensor([[[[1, 3], [2, 0]]]], dtype=torch.long)
    expert_mask = torch.tensor([[[[True, True], [True, False]]]], dtype=torch.bool)
    attention_mask = torch.ones((1, 2), dtype=torch.long)

    expert_set, true_count = sample_targets_from_expert_selection(
        expert_selection,
        attention_mask,
        num_experts=4,
        expert_selection_mask=expert_mask,
    )

    assert expert_set[0, 0, 1]
    assert expert_set[0, 0, 3]
    assert expert_set[0, 0, 2]
    assert not expert_set[0, 0, 0]
    assert int(true_count[0, 0]) == 3


def test_token_scores_noisy_or_uses_only_valid_tokens() -> None:
    logits = torch.tensor([[[[4.0, 0.0], [0.0, 4.0], [100.0, -100.0]]]], dtype=torch.float32)
    attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.long)

    scores = token_scores_from_logits(logits, attention_mask, aggregator="noisy_or")

    probs = torch.softmax(logits[:, :, :2], dim=-1)
    expected = 1.0 - torch.prod(1.0 - probs, dim=2)
    assert torch.allclose(scores, expected)


def test_token_scores_from_logits_uses_sigmoid_for_multi_label_bce() -> None:
    logits = torch.tensor([[[[0.0, 2.0], [2.0, 0.0]]]], dtype=torch.float32)
    attention_mask = torch.tensor([[1, 1]], dtype=torch.long)

    scores = token_scores_from_logits(logits, attention_mask, aggregator="sum_prob", score_activation="sigmoid")

    expected = torch.sigmoid(logits).sum(dim=2)
    assert torch.allclose(scores, expected)


def test_curve_oracle_and_budget_gap_metrics() -> None:
    scores = torch.tensor([[[0.9, 0.8, 0.1]]], dtype=torch.float32)
    true_set = torch.tensor([[[True, False, True]]])
    true_count = torch.tensor([[2]], dtype=torch.long)

    curve = curve_from_scores(scores, true_set)
    oracle = oracle_count_metrics_from_scores(scores, true_set, true_count)
    gap = budget_gap_curve(true_count, max_budget=3)

    assert curve[0]["budget"] == 1
    assert curve[0]["precision"] == 1.0
    assert curve[0]["recall"] == 0.5
    assert curve[2]["budget"] == 3
    assert curve[2]["precision"] == pytest.approx(2 / 3)
    assert curve[2]["recall"] == 1.0
    assert oracle["oracle_count_total_budget"] == 2
    assert oracle["oracle_count_micro_precision"] == 0.5
    assert oracle["oracle_count_micro_recall"] == 0.5
    assert gap[0]["budget_minus_true_mean"] == -1.0
    assert gap[1]["budget_minus_true_mean"] == 0.0
    assert gap[2]["budget_minus_true_mean"] == 1.0


def test_build_src_simplenn_model_from_config() -> None:
    config = {
        "model": "src-simplenn-token",
        "hidden_dim": 384,
        "src_layers": 1,
        "dropout": 0.5,
        "metadata": {
            "hidden_size": 8,
            "num_encoder_moe_layers": 2,
            "num_experts": 5,
            "max_input_tokens": 4,
        },
    }

    model = build_model_from_config(config, torch.device("cpu"))
    state_dict = {name: value.detach().clone() for name, value in model.state_dict().items()}
    reloaded = build_model_from_config(config, torch.device("cpu"))
    reloaded.load_state_dict(state_dict, strict=True)
    output = reloaded(torch.randn(1, 4, 8))

    assert {"model.net.0.weight", "model.net.0.bias", "model.net.3.weight", "model.net.3.bias"}.issubset(state_dict)
    assert output.shape == (1, 2, 4, 5)
    layers = list(model.model.net)
    assert isinstance(layers[0], torch.nn.Linear)
    assert layers[0].out_features == 384
    assert isinstance(layers[2], torch.nn.Dropout)
    assert layers[2].p == 0.5


def test_layer_curve_from_scores_reports_each_layer_and_budget() -> None:
    scores = torch.tensor(
        [
            [[0.9, 0.2, 0.1, 0.0], [0.1, 0.7, 0.2, 0.0]],
            [[0.1, 0.2, 0.8, 0.0], [0.6, 0.1, 0.2, 0.0]],
        ],
        dtype=torch.float32,
    )
    true_set = torch.tensor(
        [
            [[True, False, False, False], [False, True, True, False]],
            [[False, False, True, False], [True, False, False, False]],
        ]
    )

    rows = layer_curve_from_scores(scores, true_set)

    assert {row["layer"] for row in rows} == {0, 1}
    assert {row["budget"] for row in rows} == {1, 2, 3, 4}
    assert all("precision" in row and "recall" in row for row in rows)
    assert all(row["num_sample_layer_items"] == 2 for row in rows)


def test_noisy_or_budget_gap_metrics_use_float_sum_budget() -> None:
    scores = torch.tensor([[[0.8, 0.1, 0.6, 0.0]]], dtype=torch.float32)
    true_count = torch.tensor([[2]], dtype=torch.long)

    rows = noisy_or_budget_gap_metrics(scores, true_count)
    layer_rows = noisy_or_budget_gap_by_layer(scores, true_count)

    assert rows[0]["noisy_or_budget_mean"] == pytest.approx(1.5)
    assert rows[0]["true_count_mean"] == pytest.approx(2.0)
    assert rows[0]["noisy_or_budget_minus_true_mean"] == pytest.approx(-0.5)
    assert rows[0]["noisy_or_budget_abs_gap_mean"] == pytest.approx(0.5)
    assert rows[0]["under_budget_rate"] == pytest.approx(1.0)
    assert layer_rows[0]["layer"] == 0
    assert layer_rows[0]["noisy_or_budget_mean"] == pytest.approx(1.5)


def test_dynamic_budget_prefetch_metrics_use_ceil_top_budget() -> None:
    scores = torch.tensor([[[0.6, 0.5, 0.1, 0.0]]], dtype=torch.float32)
    true_set = torch.tensor([[[True, False, True, False]]])
    true_count = torch.tensor([[2]], dtype=torch.long)

    rows = dynamic_budget_prefetch_metrics(scores, true_set, true_count)
    layer_rows = dynamic_budget_prefetch_by_layer(scores, true_set, true_count)

    assert rows[0]["pred_budget_mean"] == pytest.approx(1.2)
    assert rows[0]["pred_budget_int_mean"] == pytest.approx(2.0)
    assert rows[0]["true_count_mean"] == pytest.approx(2.0)
    assert rows[0]["micro_precision"] == pytest.approx(0.5)
    assert rows[0]["micro_recall"] == pytest.approx(0.5)
    assert rows[0]["dynamic_budget_precision"] == pytest.approx(0.5)
    assert rows[0]["dynamic_budget_recall"] == pytest.approx(0.5)
    assert layer_rows[0]["layer"] == 0
    assert layer_rows[0]["dynamic_budget_precision"] == pytest.approx(0.5)


def test_dynamic_budget_precision_and_recall_use_distinct_denominators() -> None:
    scores = torch.tensor([[[0.9, 0.7, 0.6, 0.4, 0.4]]], dtype=torch.float32)
    true_set = torch.tensor([[[True, False, True, False, False]]])
    true_count = torch.tensor([[2]], dtype=torch.long)

    rows = dynamic_budget_prefetch_metrics(scores, true_set, true_count)

    assert rows[0]["pred_budget_mean"] == pytest.approx(3.0)
    assert rows[0]["pred_budget_int_mean"] == pytest.approx(3.0)
    assert rows[0]["total_overlap"] == 2
    assert rows[0]["total_budget"] == 3
    assert rows[0]["total_true"] == 2
    assert rows[0]["micro_precision"] == pytest.approx(2 / 3)
    assert rows[0]["micro_recall"] == pytest.approx(1.0)


def test_dynamic_budget_int_clamps_to_valid_range() -> None:
    low_scores = torch.zeros(1, 1, 4, dtype=torch.float32)
    high_scores = torch.full((1, 1, 4), 1.5, dtype=torch.float32)
    true_set = torch.tensor([[[True, False, False, False]]])
    true_count = torch.tensor([[1]], dtype=torch.long)

    low_rows = dynamic_budget_prefetch_metrics(low_scores, true_set, true_count)
    high_rows = dynamic_budget_prefetch_metrics(high_scores, true_set, true_count)

    assert low_rows[0]["pred_budget_mean"] == pytest.approx(0.0)
    assert low_rows[0]["total_budget"] == 1
    assert high_rows[0]["pred_budget_mean"] == pytest.approx(6.0)
    assert high_rows[0]["total_budget"] == 4


def test_write_multi_csv_supports_single_and_multirow_payload(tmp_path) -> None:
    one_row = {"model-a": [{"metric": 1.0, "count": 2}]}
    multi_row = {"model-a": [{"budget": 1, "precision": 0.5}, {"budget": 2, "precision": 0.75}]}

    write_multi_csv(tmp_path / "one.csv", one_row)
    write_multi_csv(tmp_path / "multi.csv", multi_row)

    assert (tmp_path / "one.csv").read_text(encoding="utf-8").splitlines() == ["model,metric,count", "model-a,1.0,2"]
    assert (tmp_path / "multi.csv").read_text(encoding="utf-8").splitlines() == [
        "model,budget,precision",
        "model-a,1,0.5",
        "model-a,2,0.75",
    ]


def test_resolve_checkpoint_path_defaults_and_accepts_explicit_name(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.pt").write_text("model", encoding="utf-8")

    assert resolve_checkpoint_path(model_dir, None) == model_dir / "model.pt"

    (model_dir / "best_model.pt").write_text("best", encoding="utf-8")
    assert resolve_checkpoint_path(model_dir, None) == model_dir / "best_model.pt"
    assert resolve_checkpoint_path(model_dir, "model.pt") == model_dir / "model.pt"


def test_resolve_checkpoint_path_rejects_path_components(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    with pytest.raises(ValueError, match="checkpoint_name must be a file name"):
        resolve_checkpoint_path(model_dir, "../model.pt")


def test_report_output_dir_defaults_to_model_taxonomy_reports_dir() -> None:
    model_dir = Path(
        "/mnt/huwf5/promoe/experiment/models/predictors/encoder_expert_prefetch/"
        "mmlu-professional_law/nllb-moe-54b/sparse-cache-b1-longest-v1/blte/"
        "src-simplenn-token-bce-weighted-h384-l1-drop0p5-lr1e4-bs512-seed0-validtrim"
    )

    output_dir = resolve_report_output_dir(None, [model_dir])

    assert output_dir == Path(
        "/mnt/huwf5/promoe/experiment/models/predictors/encoder_expert_prefetch/"
        "mmlu-professional_law/nllb-moe-54b/sparse-cache-b1-longest-v1/reports/ble-noisyor-report"
    )


def test_report_output_dir_rejects_taxonomy_mismatch() -> None:
    model_dir = Path(
        "/mnt/huwf5/promoe/experiment/models/predictors/encoder_expert_prefetch/"
        "mmlu-professional_law/nllb-moe-54b/sparse-cache-b1-longest-v1/blte/"
        "src-simplenn-token-bce-weighted-h384-l1-drop0p5-lr1e4-bs512-seed0-validtrim"
    )
    output_dir = Path(
        "/mnt/huwf5/promoe/experiment/models/predictors/encoder_expert_prefetch/"
        "mmlu-professional_law/switch-base-128/sparse-cache-b1-longest-v1/reports/ble-noisyor-report"
    )

    with pytest.raises(ValueError, match="does not match model_dir taxonomy base"):
        validate_report_output_dir(output_dir, [model_dir])


def test_report_output_dir_allows_non_taxonomy_explicit_dir(tmp_path: Path) -> None:
    model_dir = Path(
        "/mnt/huwf5/promoe/experiment/models/predictors/encoder_expert_prefetch/"
        "mmlu-professional_law/nllb-moe-54b/sparse-cache-b1-longest-v1/blte/model-a"
    )
    output_dir = tmp_path / "custom-report"

    assert resolve_report_output_dir(output_dir, [model_dir]) == output_dir


def test_write_ble_artifact_only_for_new_taxonomy_blte(tmp_path: Path) -> None:
    base = tmp_path / "experiment/models/predictors/encoder_expert_prefetch/mmlu-professional_law/switch-base-128/sparse-cache-b1-longest-v1"
    run_name = "src-simplenn-token-hardce-h384-l1-drop0p5-lr1e4-bs2-seed0-validtrim"
    model_dir = base / "blte" / run_name
    model_dir.mkdir(parents=True)
    (model_dir / "run_manifest.json").write_text(json.dumps({"run_name": run_name}), encoding="utf-8")
    report_dir = base / "reports" / "ble-noisyor-report"

    output = write_ble_artifact_if_taxonomy_blte(
        model_dir=model_dir,
        config={"output_name": "ignored", "objective": "multi-label-bce", "loss_type": "multi_label_bce"},
        label="model-a",
        checkpoint_path=model_dir / "best_model.pt",
        trace_dir=tmp_path / "trace",
        report_dir_path=report_dir,
    )

    assert output == base / "ble" / f"noisyor-from-{run_name}"
    manifest = json.loads((output / "ble_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_level"] == "ble"
    assert manifest["output_layout"] == "BLE"
    assert manifest["source_blte_run_name"] == run_name
    assert manifest["budget_source"] == "sum_ble_score"
    assert manifest["budget_rounding"] == "ceil"
    assert manifest["source_objective"] == "multi-label-bce"
    assert manifest["source_loss_type"] == "multi_label_bce"
    assert manifest["probability_transform"] == "sigmoid"
    readme = (output / "README.md").read_text(encoding="utf-8")
    assert "source_objective: multi-label-bce" in readme
    assert "source_loss_type: multi_label_bce" in readme
    assert "probability_transform: sigmoid" in readme

    legacy_dir = tmp_path / "experiment/models/encoder_predictor/src-simplenn-token-hard-ce-h384-l1"
    legacy_dir.mkdir(parents=True)
    assert write_ble_artifact_if_taxonomy_blte(
        model_dir=legacy_dir,
        config={"output_name": "legacy"},
        label="legacy",
        checkpoint_path=legacy_dir / "best_model.pt",
        trace_dir=tmp_path / "trace",
        report_dir_path=report_dir,
    ) is None

