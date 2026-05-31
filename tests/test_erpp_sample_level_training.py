from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import torch

from performance_predictor.encoder.ERPP.implement.train.sample_level.labels import (
    build_sample_targets,
)
from performance_predictor.encoder.ERPP.implement.train.sample_level.losses import (
    hard_negative_ranking_loss,
)
from performance_predictor.encoder.ERPP.implement.train.sample_level.metrics import (
    compute_adaptive_prefetch_metrics,
    compute_layer_budget_prefetch_metrics,
    compute_sample_prefetch_metrics,
)
from performance_predictor.encoder.ERPP.implement.train.sample_level.models import (
    BUDGET_ADAPTIVE_MODEL,
    SAMPLE_MODEL_CHOICES,
    build_sample_model,
)
from performance_predictor.encoder.ERPP.implement.train.sample_level.train_sample_level import (
    build_train_budget_profiles,
    checkpoint_metric_mode,
    checkpoint_metric_value,
    estimate_pos_weight,
    is_improved_checkpoint_metric,
    main,
    parse_args,
    primary_score,
    resolve_rank_budget,
)


def write_tiny_sample_trace(root: Path) -> Path:
    trace_dir = root / "trace"
    train_dir = trace_dir / "train"
    validation_dir = trace_dir / "validation"
    train_dir.mkdir(parents=True)
    validation_dir.mkdir(parents=True)

    metadata = {
        "hidden_size": 4,
        "num_encoder_moe_layers": 2,
        "num_experts": 5,
        "routing_top_k": 1,
        "max_input_tokens": 3,
        "splits": {
            "train": {"num_samples": 2},
            "validation": {"num_samples": 1},
        },
    }
    (trace_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    def write_split(split_dir: Path, samples: int) -> None:
        layer0 = torch.arange(samples * 3 * 4, dtype=torch.float32).reshape(samples, 3, 4) / 10.0
        mask = torch.tensor([[1, 1, 0]] * samples, dtype=torch.long)
        logits = torch.zeros(samples, 2, 3, 5, dtype=torch.float32)
        probs = torch.softmax(logits, dim=-1)
        selection = torch.tensor(
            [[[[0], [2], [4]], [[1], [1], [3]]]] * samples,
            dtype=torch.long,
        )
        torch.save(layer0, split_dir / "layer0_attn_out.pt")
        torch.save(mask, split_dir / "attention_mask.pt")
        torch.save(logits, split_dir / "router_logits.pt")
        torch.save(probs, split_dir / "router_probs.pt")
        torch.save(selection, split_dir / "expert_selection.pt")

    write_split(train_dir, 2)
    write_split(validation_dir, 1)
    return trace_dir
def assert_primary_checkpoint_metrics_consistent(
    first: dict[str, object],
    second: dict[str, object],
    third: dict[str, object],
) -> None:
    for key in ("primary_score", "checkpoint_metric", "best_checkpoint_metric"):
        assert second[key] == first[key]
        assert third[key] == first[key]


def test_primary_score_weights_p90_recall_and_precision() -> None:
    score = primary_score(
        {
            "fixed_p90_set_recall": 0.8,
            "fixed_p90_prefetch_precision": 0.5,
        }
    )
    assert score == 0.6 * 0.8 + 0.4 * 0.5


def test_primary_score_requires_p90_recall_and_precision() -> None:
    with pytest.raises(KeyError, match="fixed_p90_prefetch_precision"):
        primary_score({"fixed_p90_set_recall": 0.8})


def test_checkpoint_metric_improvement_supports_maximize_and_minimize() -> None:
    assert checkpoint_metric_mode("validation_loss") == "min"
    assert checkpoint_metric_mode("primary_score") == "max"
    assert checkpoint_metric_value(
        "fixed_p90_set_recall", {"fixed_p90_set_recall": 0.9}
    ) == 0.9
    assert checkpoint_metric_value(
        "primary_score",
        {
            "fixed_p90_set_recall": 0.8,
            "fixed_p90_prefetch_precision": 0.5,
        },
    ) == pytest.approx(0.68)

    assert is_improved_checkpoint_metric(
        "primary_score", current=0.8, best=0.7, min_delta=0.01
    )
    assert not is_improved_checkpoint_metric(
        "primary_score", current=0.705, best=0.7, min_delta=0.01
    )
    assert is_improved_checkpoint_metric(
        "validation_loss", current=0.5, best=0.6, min_delta=0.01
    )
    assert not is_improved_checkpoint_metric(
        "validation_loss", current=0.595, best=0.6, min_delta=0.01
    )


def test_build_sample_targets_unions_valid_token_experts_and_ignores_padding() -> None:
    expert_selection = torch.tensor([[[[0], [2], [4]], [[1], [1], [3]]]], dtype=torch.long)
    router_probs = torch.zeros(1, 2, 3, 5, dtype=torch.float32)
    router_probs[0, 0, 0, 0] = 1.0
    router_probs[0, 0, 1, 2] = 1.0
    router_probs[0, 0, 2, 4] = 1.0
    router_probs[0, 1, 0, 1] = 0.25
    router_probs[0, 1, 1, 1] = 0.75
    router_probs[0, 1, 2, 3] = 1.0
    attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.long)

    targets = build_sample_targets(expert_selection, router_probs, attention_mask, num_experts=5)

    assert torch.equal(targets.expert_set[0, 0], torch.tensor([1, 0, 1, 0, 0], dtype=torch.float32))
    assert torch.equal(targets.expert_set[0, 1], torch.tensor([0, 1, 0, 0, 0], dtype=torch.float32))
    assert torch.equal(targets.true_count, torch.tensor([[2, 1]]))
    assert torch.equal(targets.expert_freq[0, 0], torch.tensor([0.5, 0.0, 0.5, 0.0, 0.0]))
    assert torch.equal(targets.router_prob_mean[0, 1], torch.tensor([0.0, 0.5, 0.0, 0.0, 0.0]))
    assert torch.equal(targets.router_prob_max[0, 1], torch.tensor([0.0, 0.75, 0.0, 0.0, 0.0]))


def test_sample_prefetch_metrics_report_set_overlap_precision_and_oracle_gap() -> None:
    true_set = torch.tensor([[[1, 0, 1, 0, 0], [0, 1, 0, 1, 0]]], dtype=torch.float32)
    true_freq = torch.tensor([[[0.6, 0.0, 0.4, 0.0, 0.0], [0.0, 0.7, 0.0, 0.3, 0.0]]])
    pred_logits = torch.tensor([[[5.0, 4.0, 3.0, 2.0, 1.0], [5.0, 4.0, 3.0, 2.0, 1.0]]])

    metrics = compute_sample_prefetch_metrics(pred_logits, true_set, true_freq, budgets=(1, 2))

    assert metrics["num_valid_sample_layer_items"] == 2
    assert metrics["overlap_count@1"] == 1
    assert metrics["set_recall@1"] == 0.25
    assert metrics["prefetch_precision@1"] == 0.5
    assert metrics["prefetch_waste@1"] == 0.5
    assert metrics["overlap_count@2"] == 2
    assert metrics["set_recall@2"] == 0.5
    assert metrics["prefetch_precision@2"] == 0.5
    assert metrics["oracle_recall@2"] == 1.0
    assert metrics["oracle_gap@2"] == 0.5
    assert metrics["true_set_size_mean"] == 2.0



def test_layer_budget_prefetch_metrics_use_per_layer_budgets() -> None:
    true_set = torch.tensor([[[1, 0, 1, 0, 0], [0, 1, 0, 1, 0]]], dtype=torch.float32)
    pred_logits = torch.tensor([[[5.0, 4.0, 3.0, 2.0, 1.0], [5.0, 4.0, 3.0, 2.0, 1.0]]])

    metrics = compute_layer_budget_prefetch_metrics(pred_logits, true_set, [2, 1], prefix="fixed_mean")

    assert metrics["fixed_mean_budgets"] == [2, 1]
    assert metrics["fixed_mean_budget_mean"] == 1.5
    assert metrics["fixed_mean_overlap_count"] == 1
    assert metrics["fixed_mean_set_recall"] == 0.25
    assert torch.isclose(torch.tensor(metrics["fixed_mean_prefetch_precision"]), torch.tensor(1 / 3))


def test_adaptive_prefetch_metrics_report_budget_and_count_quality() -> None:
    pred_logits = torch.tensor([[[4.0, 3.0, 2.0, 1.0], [1.0, 4.0, 3.0, 2.0]]])
    true_set = torch.tensor([[[1, 0, 1, 0], [0, 1, 1, 0]]], dtype=torch.float32)
    pred_budget = torch.tensor([[1, 3]], dtype=torch.long)
    true_count = torch.tensor([[2, 2]], dtype=torch.long)

    metrics = compute_adaptive_prefetch_metrics(pred_logits, true_set, pred_budget, true_count)

    assert metrics["adaptive_overlap_count"] == 3
    assert metrics["adaptive_set_recall"] == 0.75
    assert metrics["adaptive_precision"] == 0.75
    assert metrics["adaptive_budget_mean"] == 2.0
    assert metrics["count_mae"] == 1.0
    assert metrics["under_budget_rate"] == 0.5
    assert metrics["over_budget_rate"] == 0.5


def test_hard_negative_ranking_loss_penalizes_high_false_positive() -> None:
    logits = torch.tensor([[[0.0, 4.0, 3.0, 1.0]]], dtype=torch.float32)
    labels = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]], dtype=torch.float32)

    loss = hard_negative_ranking_loss(logits, labels, budget=2, margin=0.5)

    assert loss.item() > 0.0


def test_hard_negative_ranking_loss_zero_when_top_budget_has_no_false_positive() -> None:
    logits = torch.tensor([[[4.0, 0.0, 3.0, 1.0]]], dtype=torch.float32)
    labels = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]], dtype=torch.float32)

    loss = hard_negative_ranking_loss(logits, labels, budget=2, margin=0.5)

    assert torch.equal(loss, torch.tensor(0.0))


def test_hard_negative_ranking_loss_ignores_items_without_positive_labels() -> None:
    logits = torch.tensor([[[0.0, 4.0, 3.0, 1.0]]], dtype=torch.float32)
    labels = torch.zeros_like(logits)

    loss = hard_negative_ranking_loss(logits, labels, budget=2, margin=0.5)

    assert torch.equal(loss, torch.tensor(0.0))


def test_hard_negative_ranking_loss_averages_over_all_items() -> None:
    logits = torch.tensor(
        [[[0.0, 4.0, 3.0, 1.0], [0.0, 4.0, 3.0, 1.0]]],
        dtype=torch.float32,
    )
    labels = torch.tensor(
        [[[1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    loss = hard_negative_ranking_loss(logits, labels, budget=2, margin=0.5)

    assert torch.isclose(loss, torch.tensor(1.5))


def test_resolve_rank_budget_uses_explicit_positive_budget() -> None:
    args = argparse.Namespace(rank_budget=3, rank_budget_profile="fixed_p90")

    assert resolve_rank_budget(args, {"fixed_p90": [1, 2]}) == 3


def test_resolve_rank_budget_rejects_non_positive_explicit_budget() -> None:
    args = argparse.Namespace(rank_budget=0, rank_budget_profile="fixed_p90")

    with pytest.raises(ValueError, match="rank_budget"):
        resolve_rank_budget(args, {"fixed_p90": [1, 2]})


def test_resolve_rank_budget_rejects_missing_profile() -> None:
    args = argparse.Namespace(rank_budget=None, rank_budget_profile="missing")

    with pytest.raises(ValueError, match="missing"):
        resolve_rank_budget(args, {"fixed_p90": [1, 2]})


def test_resolve_rank_budget_rejects_empty_profile() -> None:
    args = argparse.Namespace(rank_budget=None, rank_budget_profile="fixed_p90")

    with pytest.raises(ValueError, match="fixed_p90"):
        resolve_rank_budget(args, {"fixed_p90": []})


def test_resolve_rank_budget_rejects_non_positive_resolved_budget() -> None:
    args = argparse.Namespace(rank_budget=None, rank_budget_profile="fixed_p90")

    with pytest.raises(ValueError, match="fixed_p90"):
        resolve_rank_budget(args, {"fixed_p90": [0, 0]})


def test_all_sample_models_return_sample_layer_logits_or_budget_outputs() -> None:
    x = torch.randn(2, 3, 4)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long)

    for model_name in SAMPLE_MODEL_CHOICES:
        model = build_sample_model(
            model_name,
            input_dim=4,
            hidden_dim=8,
            num_router_layers=2,
            num_experts=5,
            dropout=0.0,
            max_budget=4,
            set_transformer_seeds=2,
        )
        output = model(x, mask)
        if model_name == BUDGET_ADAPTIVE_MODEL:
            assert output["sample_logits"].shape == (2, 2, 5)
            assert output["count_logits"].shape == (2, 2, 5)
        else:
            assert output.shape == (2, 2, 5)


def test_hybrid_layer_fusion_has_per_layer_alpha_parameter() -> None:
    model = build_sample_model(
        "erpp-hybrid-tokenset",
        input_dim=4,
        hidden_dim=8,
        num_router_layers=2,
        num_experts=5,
        dropout=0.0,
        hybrid_fusion="layer",
    )

    assert model.fusion_logit.shape == (2,)
    assert model.fusion_logit.requires_grad
    model = model.double()
    assert model.fusion_logit.dtype == torch.float64
    output = model(
        torch.randn(1, 3, 4, dtype=torch.float64),
        torch.tensor([[1, 1, 1]], dtype=torch.long),
    )
    assert output.shape == (1, 2, 5)
    assert output.dtype == torch.float64


def test_hybrid_global_fusion_has_scalar_alpha_parameter() -> None:
    model = build_sample_model(
        "erpp-hybrid-tokenset",
        input_dim=4,
        hidden_dim=8,
        num_router_layers=2,
        num_experts=5,
        dropout=0.0,
        hybrid_fusion="global",
    )

    assert model.fusion_logit.shape == torch.Size([])
    assert model.fusion_logit.requires_grad
    output = model(
        torch.randn(1, 3, 4),
        torch.tensor([[1, 1, 1]], dtype=torch.long),
    )
    assert output.shape == (1, 2, 5)


def test_hybrid_fixed_fusion_keeps_no_fusion_parameter() -> None:
    model = build_sample_model(
        "erpp-hybrid-tokenset",
        input_dim=4,
        hidden_dim=8,
        num_router_layers=2,
        num_experts=5,
        dropout=0.0,
        hybrid_fusion="fixed",
    )

    assert not hasattr(model, "fusion_logit")


def test_hybrid_fusion_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="fusion"):
        build_sample_model(
            "erpp-hybrid-tokenset",
            input_dim=4,
            hidden_dim=8,
            num_router_layers=2,
            num_experts=5,
            dropout=0.0,
            hybrid_fusion="bad",
        )


def test_non_hybrid_models_ignore_hybrid_fusion_argument() -> None:
    model = build_sample_model(
        "erpp-setpool-mlp",
        input_dim=4,
        hidden_dim=8,
        num_router_layers=2,
        num_experts=5,
        dropout=0.0,
        hybrid_fusion="layer",
    )

    output = model(
        torch.randn(1, 3, 4),
        torch.tensor([[1, 1, 1]], dtype=torch.long),
    )
    assert output.shape == (1, 2, 5)
    assert not hasattr(model, "fusion_logit")


def test_sample_level_training_smoke_run_writes_artifacts(tmp_path: Path) -> None:
    trace_dir = write_tiny_sample_trace(tmp_path)
    output_root = tmp_path / "outputs"

    returned_metrics = main(
        [
            "--trace-dir",
            str(trace_dir),
            "--model",
            "erpp-setpool-mlp",
            "--output-root",
            str(output_root),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--hidden-dim",
            "8",
            "--dropout",
            "0",
            "--device",
            "cpu",
            "--max-train-batches",
            "1",
            "--max-eval-batches",
            "1",
        ]
    )

    model_dir = output_root / "erpp-setpool-mlp"
    for artifact_name in (
        "config.json",
        "metrics.json",
        "model.pt",
        "best_model.pt",
        "README.md",
        "train_log.jsonl",
    ):
        assert (model_dir / artifact_name).exists()

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    assert config["model"] == "erpp-setpool-mlp"
    assert "validation_loss" in metrics
    assert "set_recall@1" in metrics
    assert "prefetch_precision@1" in metrics
    expected_primary_score = (
        0.6 * metrics["fixed_p90_set_recall"]
        + 0.4 * metrics["fixed_p90_prefetch_precision"]
    )
    assert metrics["checkpoint_metric"] == "primary_score"
    assert metrics["primary_score"] == expected_primary_score
    assert metrics["best_checkpoint_metric"] == metrics["primary_score"]

    checkpoint = torch.load(
        model_dir / "model.pt", map_location="cpu", weights_only=True
    )
    checkpoint_metrics = checkpoint["metrics"]
    assert_primary_checkpoint_metrics_consistent(
        returned_metrics, metrics, checkpoint_metrics
    )



def test_metrics_include_parameter_and_checkpoint_size(tmp_path: Path) -> None:
    trace_dir = write_tiny_sample_trace(tmp_path)
    output_root = tmp_path / "metadata-output"

    returned_metrics = main(
        [
            "--trace-dir",
            str(trace_dir),
            "--model",
            "erpp-setpool-mlp",
            "--output-root",
            str(output_root),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--hidden-dim",
            "8",
            "--dropout",
            "0",
            "--device",
            "cpu",
            "--max-train-batches",
            "1",
            "--max-eval-batches",
            "1",
        ]
    )

    model_dir = output_root / "erpp-setpool-mlp"
    metrics = json.loads((model_dir / "metrics.json").read_text())
    model_checkpoint = torch.load(
        model_dir / "model.pt", map_location="cpu", weights_only=True
    )
    best_checkpoint = torch.load(
        model_dir / "best_model.pt", map_location="cpu", weights_only=True
    )

    for key in ("num_parameters", "trainable_parameters", "checkpoint_size_mb"):
        assert returned_metrics[key] == metrics[key]
        assert model_checkpoint["metrics"][key] == metrics[key]
        assert best_checkpoint["metrics"][key] == metrics[key]

    assert metrics["num_parameters"] > 0
    assert metrics["trainable_parameters"] > 0
    assert metrics["checkpoint_size_mb"] > 0.0

def test_sample_level_training_with_rank_loss_writes_rank_metrics(tmp_path: Path) -> None:
    trace_dir = write_tiny_sample_trace(tmp_path)
    output_root = tmp_path / "rank-outputs"

    returned_metrics = main(
        [
            "--trace-dir",
            str(trace_dir),
            "--model",
            "erpp-setpool-mlp",
            "--output-root",
            str(output_root),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--hidden-dim",
            "8",
            "--dropout",
            "0",
            "--device",
            "cpu",
            "--max-train-batches",
            "1",
            "--max-eval-batches",
            "1",
            "--lambda-rank",
            "0.3",
            "--rank-margin",
            "0.5",
            "--rank-budget",
            "2",
        ]
    )

    model_dir = output_root / "erpp-setpool-mlp"
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    log_rows = [
        json.loads(line)
        for line in (model_dir / "train_log.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert config["lambda_rank"] == 0.3
    assert config["rank_margin"] == 0.5
    assert config["rank_budget"] == 2
    assert config["rank_budget_arg"] == 2
    assert "validation_rank_loss" in metrics
    assert "validation_rank_loss" in returned_metrics
    assert "train_rank_loss" in log_rows[-1]
    assert "validation_rank_loss" in log_rows[-1]


def test_training_config_records_scheduler_and_grad_clip(tmp_path: Path) -> None:
    trace_dir = write_tiny_sample_trace(tmp_path)
    output_root = tmp_path / "scheduler-output"

    main(
        [
            "--trace-dir",
            str(trace_dir),
            "--model",
            "erpp-setpool-mlp",
            "--output-root",
            str(output_root),
            "--epochs",
            "2",
            "--batch-size",
            "1",
            "--hidden-dim",
            "8",
            "--dropout",
            "0",
            "--device",
            "cpu",
            "--max-train-batches",
            "1",
            "--max-eval-batches",
            "1",
            "--grad-clip-norm",
            "1.0",
            "--lr-scheduler",
            "cosine",
        ]
    )

    model_dir = output_root / "erpp-setpool-mlp"
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    log_rows = [
        json.loads(line)
        for line in (model_dir / "train_log.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert config["grad_clip_norm"] == 1.0
    assert config["lr_scheduler"] == "cosine"
    assert len(log_rows) >= 2
    assert all("lr" in row for row in log_rows)
    assert log_rows[1]["lr"] < log_rows[0]["lr"]


def test_parse_args_rejects_non_positive_grad_clip_norm(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--trace-dir",
                str(tmp_path),
                "--model",
                "erpp-setpool-mlp",
                "--grad-clip-norm",
                "0",
            ]
        )

    assert "positive" in capsys.readouterr().err


def test_budgetadaptive_default_max_budget_uses_num_experts() -> None:
    model = build_sample_model(
        "erpp-budgetadaptive-set",
        input_dim=4,
        hidden_dim=8,
        num_router_layers=2,
        num_experts=7,
        dropout=0.0,
    )

    output = model(
        torch.randn(1, 3, 4),
        torch.tensor([[1, 1, 1]], dtype=torch.long),
    )

    assert output["sample_logits"].shape == (1, 2, 7)
    assert output["count_logits"].shape == (1, 2, 8)


def test_sample_level_training_smoke_run_writes_budgetadaptive_metrics(tmp_path: Path) -> None:
    trace_dir = write_tiny_sample_trace(tmp_path)
    output_root = tmp_path / "budgetadaptive-outputs"

    main(
        [
            "--trace-dir",
            str(trace_dir),
            "--model",
            "erpp-budgetadaptive-set",
            "--output-root",
            str(output_root),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--hidden-dim",
            "8",
            "--dropout",
            "0",
            "--device",
            "cpu",
            "--max-train-batches",
            "1",
            "--max-eval-batches",
            "1",
        ]
    )

    metrics = json.loads((output_root / "erpp-budgetadaptive-set" / "metrics.json").read_text(encoding="utf-8"))
    assert "adaptive_set_recall" in metrics
    assert "adaptive_precision" in metrics
    assert metrics["adaptive_budget_mean"] >= 1.0
    assert "count_mae" in metrics


def test_thresholdset_training_smoke_writes_threshold_metrics(tmp_path: Path) -> None:
    trace_dir = write_tiny_sample_trace(tmp_path)
    output_root = tmp_path / "threshold-outputs"

    main(
        [
            "--trace-dir",
            str(trace_dir),
            "--model",
            "erpp-thresholdset",
            "--output-root",
            str(output_root),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--hidden-dim",
            "8",
            "--dropout",
            "0",
            "--device",
            "cpu",
            "--max-train-batches",
            "1",
            "--max-eval-batches",
            "1",
        ]
    )

    config = json.loads((output_root / "erpp-thresholdset" / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((output_root / "erpp-thresholdset" / "metrics.json").read_text(encoding="utf-8"))
    assert len(config["thresholds"]) == 2
    assert "threshold_set_recall" in metrics
    assert "threshold_budget_mean" in metrics


def test_estimate_pos_weight_uses_raw_negative_positive_ratio(tmp_path: Path) -> None:
    trace_dir = write_tiny_sample_trace(tmp_path)
    from performance_predictor.encoder.ERPP.implement.train.baseline.data import ErppTraceDataset

    dataset = ErppTraceDataset(trace_dir, "train")
    pos_weight = estimate_pos_weight(dataset, num_experts=5)

    # Each sample has 3 positive sample-layer experts out of 10 positions.
    assert torch.isclose(torch.tensor(pos_weight), torch.tensor(7 / 3))


def test_training_smoke_writes_fixed_budget_profiles(tmp_path: Path) -> None:
    trace_dir = write_tiny_sample_trace(tmp_path)
    output_root = tmp_path / "profile-outputs"

    main(
        [
            "--trace-dir",
            str(trace_dir),
            "--model",
            "erpp-setpool-mlp",
            "--output-root",
            str(output_root),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--hidden-dim",
            "8",
            "--dropout",
            "0",
            "--device",
            "cpu",
            "--max-train-batches",
            "1",
            "--max-eval-batches",
            "1",
        ]
    )

    config = json.loads((output_root / "erpp-setpool-mlp" / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((output_root / "erpp-setpool-mlp" / "metrics.json").read_text(encoding="utf-8"))
    assert config["budget_profiles"]["fixed_mean"] == [2, 1]
    assert config["budget_profiles"]["fixed_p50"] == [2, 1]
    assert "fixed_mean_set_recall" in metrics
    assert "fixed_p90_prefetch_precision" in metrics
    assert "fixed_p95_set_recall" in metrics
    assert "fixed_p99_budget_mean" in metrics


def test_build_train_budget_profiles_reports_mean_and_quantile_budgets(tmp_path: Path) -> None:
    trace_dir = write_tiny_sample_trace(tmp_path)
    from performance_predictor.encoder.ERPP.implement.train.baseline.data import ErppTraceDataset

    dataset = ErppTraceDataset(trace_dir, "train")
    profiles = build_train_budget_profiles(dataset, num_experts=5, quantiles=(0.50, 0.90, 0.95, 0.99))

    assert profiles == {
        "fixed_mean": [2, 1],
        "fixed_p50": [2, 1],
        "fixed_p90": [2, 1],
        "fixed_p95": [2, 1],
        "fixed_p99": [2, 1],
    }

def test_sample_models_handle_all_padding_without_nan() -> None:
    x = torch.randn(2, 3, 4)
    mask = torch.zeros(2, 3, dtype=torch.long)

    for model_name in SAMPLE_MODEL_CHOICES:
        model = build_sample_model(
            model_name,
            input_dim=4,
            hidden_dim=8,
            num_router_layers=2,
            num_experts=5,
            dropout=0.0,
            max_budget=4,
            set_transformer_seeds=2,
        )
        output = model(x, mask)
        logits = output["sample_logits"] if isinstance(output, dict) else output
        assert torch.isfinite(logits).all()



def test_eval_only_recomputes_metrics_from_checkpoint(tmp_path: Path) -> None:
    trace_dir = write_tiny_sample_trace(tmp_path)
    train_root = tmp_path / "train-output"
    eval_root = tmp_path / "eval-output"

    main(
        [
            "--trace-dir",
            str(trace_dir),
            "--model",
            "erpp-setpool-mlp",
            "--output-root",
            str(train_root),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--hidden-dim",
            "8",
            "--dropout",
            "0",
            "--device",
            "cpu",
            "--max-train-batches",
            "1",
            "--max-eval-batches",
            "1",
        ]
    )

    metrics = main(
        [
            "--trace-dir",
            str(trace_dir),
            "--model",
            "erpp-setpool-mlp",
            "--output-root",
            str(eval_root),
            "--output-name",
            "eval-only",
            "--hidden-dim",
            "8",
            "--dropout",
            "0",
            "--device",
            "cpu",
            "--max-eval-batches",
            "1",
            "--eval-only",
            "--checkpoint",
            str(train_root / "erpp-setpool-mlp" / "model.pt"),
        ]
    )

    written = json.loads((eval_root / "eval-only" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["stop_reason"] == "eval_only"
    assert written["stop_reason"] == "eval_only"
    assert "fixed_p99_set_recall" in written


def test_early_stop_writes_best_epoch_and_best_checkpoint(tmp_path: Path) -> None:
    trace_dir = write_tiny_sample_trace(tmp_path)
    output_root = tmp_path / "early-stop-output"

    metrics = main(
        [
            "--trace-dir",
            str(trace_dir),
            "--model",
            "erpp-setpool-mlp",
            "--output-root",
            str(output_root),
            "--epochs",
            "2",
            "--batch-size",
            "1",
            "--hidden-dim",
            "8",
            "--dropout",
            "0",
            "--device",
            "cpu",
            "--max-train-batches",
            "1",
            "--max-eval-batches",
            "1",
            "--early-stop",
            "--early-stop-patience",
            "1",
        ]
    )

    model_dir = output_root / "erpp-setpool-mlp"
    assert (model_dir / "best_model.pt").exists()
    assert "best_epoch" in metrics
    written_metrics = json.loads(
        (model_dir / "metrics.json").read_text(encoding="utf-8")
    )
    checkpoint = torch.load(
        model_dir / "best_model.pt", map_location="cpu", weights_only=True
    )
    checkpoint_metrics = checkpoint["metrics"]
    assert_primary_checkpoint_metrics_consistent(
        metrics, written_metrics, checkpoint_metrics
    )



def test_training_prints_progress_to_stdout(tmp_path: Path, capsys) -> None:
    trace_dir = write_tiny_sample_trace(tmp_path)
    output_root = tmp_path / "progress-output"

    main(
        [
            "--trace-dir",
            str(trace_dir),
            "--model",
            "erpp-setpool-mlp",
            "--output-root",
            str(output_root),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--hidden-dim",
            "8",
            "--dropout",
            "0",
            "--device",
            "cpu",
            "--max-train-batches",
            "1",
            "--max-eval-batches",
            "1",
            "--progress-interval",
            "1",
        ]
    )

    captured = capsys.readouterr().out
    assert "[train] epoch=1 batch=1/" in captured
    assert "[epoch] epoch=1" in captured
    assert "train_loss=" in captured
    assert "validation_loss=" in captured
