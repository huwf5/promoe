from __future__ import annotations

import json

import torch

from performance_predictor.encoder.ERPP.implement.train.high_precision_prefetch.losses import (
    asymmetric_bce_loss,
    hard_negative_precision_ranking_loss,
)
from performance_predictor.encoder.ERPP.implement.train.high_precision_prefetch.data import (
    build_high_precision_targets,
    split_validation_indices,
)
from performance_predictor.encoder.ERPP.implement.train.high_precision_prefetch.metrics import (
    compute_high_precision_metrics,
)
from performance_predictor.encoder.ERPP.implement.train.high_precision_prefetch.selection import (
    select_prefetch_sets,
)
from performance_predictor.encoder.ERPP.implement.train.high_precision_prefetch.models import (
    HP_MODEL_CHOICES,
    build_high_precision_model,
)
from performance_predictor.encoder.ERPP.implement.train.high_precision_prefetch.calibration import (
    calibrate_layer_thresholds_from_logits,
)
from performance_predictor.encoder.ERPP.implement.train.high_precision_prefetch.train import (
    calibrate_and_evaluate,
    make_config,
    parse_args,
    resolve_calibration_device,
)


def test_high_precision_prefetch_package_imports() -> None:
    import performance_predictor.encoder.ERPP.implement.train.high_precision_prefetch as hp

    assert hp.__all__ == []


def test_build_high_precision_targets_adds_priority_set() -> None:
    expert_selection = torch.tensor([[[[0], [1], [1], [2]]]])
    router_probs = torch.zeros(1, 1, 4, 4)
    attention_mask = torch.tensor([[1, 1, 1, 1]])

    targets = build_high_precision_targets(
        expert_selection,
        router_probs,
        attention_mask,
        num_experts=4,
        priority_ratio=0.5,
    )

    assert targets.true_count.tolist() == [[3]]
    assert targets.priority_set.tolist() == [[[False, True, False, False]]]


def test_split_validation_indices_is_deterministic_and_disjoint() -> None:
    a = split_validation_indices(10, calibration_fraction=0.4, seed=7)
    b = split_validation_indices(10, calibration_fraction=0.4, seed=7)

    assert torch.equal(a.calibration, b.calibration)
    assert torch.equal(a.heldout, b.heldout)
    assert set(a.calibration.tolist()).isdisjoint(set(a.heldout.tolist()))


def test_high_precision_models_return_expected_shapes() -> None:
    x = torch.randn(2, 5, 8)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])

    for name in HP_MODEL_CHOICES:
        model = build_high_precision_model(name, 8, 16, 3, 7, dropout=0.0, pma_seeds=2)

        out = model(x, mask)
        logits = out["sample_logits"] if isinstance(out, dict) else out

        assert logits.shape == (2, 3, 7)
        assert torch.isfinite(logits).all()


def test_select_prefetch_sets_applies_topk_and_logit_tau() -> None:
    logits = torch.tensor([[[5.0, 4.0, 0.0, -1.0]]])
    pred = select_prefetch_sets(logits, k_max=3, tau=1.0)
    assert pred.tolist() == [[[True, True, False, False]]]


def test_high_precision_metrics_tracks_empty_prefetch_without_fake_precision() -> None:
    pred = torch.zeros(1, 1, 4, dtype=torch.bool)
    true = torch.tensor([[[True, False, True, False]]])
    m = compute_high_precision_metrics(pred, true, prefix="hp")
    assert m["hp_micro_precision"] == 0.0
    assert m["hp_zero_prefetch_rate"] == 1.0


def test_asymmetric_bce_penalizes_false_positive_more_when_beta_is_large() -> None:
    logits = torch.tensor([[[2.0]]])
    labels = torch.tensor([[[0.0]]])

    assert asymmetric_bce_loss(
        logits, labels, beta=5.0, gamma_neg=0.0
    ) > asymmetric_bce_loss(logits, labels, beta=1.0, gamma_neg=0.0)


def test_hard_negative_precision_ranking_loss_penalizes_high_negative() -> None:
    logits = torch.tensor([[[0.0, 4.0, 3.0, 1.0]]])
    labels = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]])

    assert (
        hard_negative_precision_ranking_loss(
            logits, labels, hard_negatives=1, margin=1.0
        ).item()
        > 0
    )

def test_calibration_prefers_threshold_with_target_precision_and_more_hits() -> None:
    logits = torch.tensor([[[5.0, 4.0, 1.0, 0.0]], [[5.0, 3.0, 2.0, 0.0]]])
    true = torch.tensor(
        [[[True, True, False, False]], [[True, False, True, False]]]
    )

    result = calibrate_layer_thresholds_from_logits(
        logits,
        true,
        true.float(),
        k_max=3,
        target_precision=0.75,
        candidate_count=16,
    )

    assert len(result.tau) == 1
    assert result.micro_precision >= 0.75

def test_training_cli_parses_precision_and_kmax_lists_into_config() -> None:
    args = parse_args(
        [
            "--trace-dir",
            "trace-root",
            "--model",
            "hp-setpool-mlp",
            "--target-precision",
            "0.95,0.98",
            "--k-max",
            "32,64",
        ]
    )

    config = make_config(
        args,
        metadata={"hidden_size": 8, "num_encoder_moe_layers": 3, "num_experts": 7},
        device=torch.device("cpu"),
        output_name="hp-test",
    )

    assert args.target_precision == (0.95, 0.98)
    assert args.k_max == (32, 64)
    assert config["target_precision"] == [0.95, 0.98]
    assert config["k_max"] == [32, 64]

def test_calibration_uses_global_micro_precision_across_layers() -> None:
    logits = torch.tensor([[[5.0, 4.0, 0.0, -1.0], [5.0, 4.0, 0.0, -1.0]]])
    true = torch.tensor([[[True, True, False, False], [True, False, False, False]]])

    result = calibrate_layer_thresholds_from_logits(
        logits,
        true,
        true.float(),
        k_max=2,
        target_precision=0.75,
        candidate_count=8,
    )

    assert result.micro_precision >= 0.75
    assert result.useful_prefetch_count_mean == 1.5
    assert result.avg_prefetch_count == 2.0

def test_reporting_cli_writes_csv_and_markdown_for_hp_runs(tmp_path) -> None:
    import csv
    import json

    from performance_predictor.encoder.ERPP.implement.train.high_precision_prefetch.reporting import (
        main as reporting_main,
    )

    runs_root = tmp_path / "model"
    run_dir = runs_root / "hp-alpha"
    run_dir.mkdir(parents=True)
    (runs_root / "other-run").mkdir()
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "model": "hp-setpool-mlp",
                "loss_preset": "asymmetric-bce-rank",
                "target_precision": [0.95],
                "k_max": [32],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "heldout_micro_precision": 0.971,
                "heldout_avg_prefetch_count": 3.25,
                "heldout_zero_prefetch_rate": 0.125,
                "heldout_useful_prefetch_count_mean": 2.5,
                "heldout_weighted_useful_prefetch_count_mean": 2.75,
                "heldout_set_recall": 0.8125,
                "best_epoch": 4,
                "checkpoint_metric": "validation_loss",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "model.pt").write_bytes(b"checkpoint")
    output_csv = tmp_path / "summary.csv"
    output_md = tmp_path / "summary.md"

    rows = reporting_main(
        [
            "--runs-root",
            str(runs_root),
            "--prefix",
            "hp-",
            "--output-csv",
            str(output_csv),
            "--output-md",
            str(output_md),
        ]
    )

    assert rows[0]["run_name"] == "hp-alpha"
    with output_csv.open(newline="", encoding="utf-8") as fh:
        csv_rows = list(csv.DictReader(fh))
    assert [row["run_name"] for row in csv_rows] == ["hp-alpha"]
    assert csv_rows[0]["target_precision"] == "0.95"
    assert csv_rows[0]["heldout_micro_precision"] == "0.971"

    md = output_md.read_text(encoding="utf-8")
    assert "hp-alpha" in md
    assert "target_precision" in md
    assert "heldout_micro_precision" in md

def test_runtime_export_writes_schema_json_from_training_artifacts(tmp_path) -> None:
    from performance_predictor.encoder.ERPP.implement.train.high_precision_prefetch.runtime_export import (
        main as export_runtime,
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "metadata": {"num_encoder_moe_layers": 6},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "calibration.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "target_precision": 0.95,
                        "k_max": 64,
                        "tau": [0, 0, 0, 0, 0, 0],
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "runtime.json"

    export_runtime(
        [
            "--run-dir",
            str(run_dir),
            "--target-precision",
            "0.95",
            "--k-max",
            "64",
            "--fallback-mode",
            "strict",
            "--output",
            str(output),
        ]
    )

    output_text = output.read_text(encoding="utf-8")
    assert output_text.endswith("\n")
    assert output_text.startswith('{\n  "fallback_mode"')
    payload = json.loads(output_text)
    assert payload["schema_version"] == 1
    assert payload["model_type"] == "encoder_high_precision_prefetch"
    assert payload["score_space"] == "logit"
    assert payload["source_hook_name"] == "encoder_layer0_attention_output"
    assert payload["target_layers"] == [0, 1, 2, 3, 4, 5]
    assert payload["k_max_per_layer"] == [64, 64, 64, 64, 64, 64]
    assert payload["tau_per_layer"] == [0, 0, 0, 0, 0, 0]
    assert payload["target_precision"] == 0.95
    assert payload["fallback_mode"] == "strict"

def test_calibrate_and_evaluate_records_metrics_when_target_precision_unmet() -> None:
    logits = torch.tensor([[[0.0, -1.0, -2.0]]])
    true = torch.tensor([[[False, True, False]]])

    rows, metrics = calibrate_and_evaluate(
        logits,
        true,
        true.float(),
        true,
        target_precisions=(0.99,),
        k_max_values=(1,),
        calibration_fraction=0.5,
        candidate_count=4,
        seed=0,
    )

    assert len(rows) == 1
    assert "heldout_micro_precision" in metrics
    assert "calibration_micro_precision" in metrics
    assert metrics["target_precision"] == 0.99

def test_training_cli_records_calibration_device() -> None:
    args = parse_args(
        [
            "--trace-dir",
            "trace-root",
            "--model",
            "hp-setpool-mlp",
            "--calibration-device",
            "cuda",
        ]
    )

    config = make_config(
        args,
        metadata={"hidden_size": 8, "num_encoder_moe_layers": 3, "num_experts": 7},
        device=torch.device("cpu"),
        output_name="hp-test",
    )

    assert args.calibration_device == "cuda"
    assert config["calibration_device"] == "cuda"
    assert resolve_calibration_device("auto", torch.device("cuda:0")).type == "cuda"
