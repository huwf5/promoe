from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from experiment.scripts.train import encoder_predictor_src_simplenn_token_hard_ce as train_src


def write_tiny_sparse_cache_trace(root: Path) -> Path:
    trace_dir = root / "encoder_predictor_sparse_cache_trace"
    train_dir = trace_dir / "train"
    validation_dir = trace_dir / "validation"
    train_dir.mkdir(parents=True)
    validation_dir.mkdir(parents=True)

    metadata = {
        "model_config": {
            "hidden_size": 4,
            "num_experts": 3,
            "num_selected_experts": 1,
            "num_sparse_encoder_layers": 2,
        },
        "router_layer_to_model_block": [1, 3],
        "splits": {
            "train": {"num_samples": 2},
            "validation": {"num_samples": 1},
        },
    }
    (trace_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    def write_split(split_dir: Path, samples: int) -> None:
        layer0 = torch.arange(samples * 3 * 4, dtype=torch.float32).reshape(samples, 3, 4)
        attention_mask = torch.tensor([[1, 1, 0]] * samples, dtype=torch.long)
        expert_selection = torch.zeros(samples, 2, 3, 1, dtype=torch.long)
        torch.save(layer0, split_dir / "layer0_attn_out.pt")
        torch.save(attention_mask, split_dir / "attention_mask.pt")
        torch.save(expert_selection, split_dir / "expert_selection.pt")

    write_split(train_dir, 2)
    write_split(validation_dir, 1)
    return trace_dir


def test_src_simplenn_token_forward_shape_and_layers() -> None:
    model = train_src.SrcSimpleNNTokenPredictor(
        input_dim=4,
        hidden_dim=384,
        num_router_layers=3,
        num_experts=5,
        src_layers=1,
        dropout=0.5,
    )
    layers = list(model.model.net)

    assert isinstance(layers[0], torch.nn.Linear)
    assert layers[0].in_features == 4
    assert layers[0].out_features == 384
    assert isinstance(layers[1], torch.nn.ReLU)
    assert isinstance(layers[2], torch.nn.Dropout)
    assert layers[2].p == 0.5
    assert isinstance(layers[3], torch.nn.Linear)
    assert layers[3].out_features == 15
    output = model(torch.randn(2, 7, 4))
    assert output.shape == (2, 3, 7, 5)


def test_hard_ce_loss_ignores_padding() -> None:
    pred = torch.tensor(
        [[[
            [8.0, 0.0, 0.0],
            [0.0, 8.0, 0.0],
            [0.0, 0.0, 8.0],
        ]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    expert_selection = torch.tensor([[[[0], [1], [0]]]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.long)

    loss, parts = train_src.hard_ce_loss(pred, expert_selection, attention_mask)

    expected = torch.nn.functional.cross_entropy(
        pred[:, :, :2, :].reshape(2, 3),
        torch.tensor([0, 1]),
    )
    assert torch.allclose(loss, expected)
    assert parts["valid_items"] == 2


def test_multi_label_bce_uses_both_top2_experts() -> None:
    pred = torch.zeros((1, 1, 1, 4), dtype=torch.float32, requires_grad=True)
    expert_selection = torch.tensor([[[[1, 3]]]], dtype=torch.long)
    expert_weights = torch.tensor([[[[0.75, 0.25]]]], dtype=torch.float32)
    expert_mask = torch.tensor([[[[True, True]]]], dtype=torch.bool)
    attention_mask = torch.ones((1, 1), dtype=torch.long)

    loss, parts = train_src.multi_label_bce_loss(
        pred,
        expert_selection,
        attention_mask,
        expert_weights,
        expert_mask,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert parts["valid_items"] == 1
    assert parts["bce"].shape == ()
    assert pred.grad is not None
    assert pred.grad[0, 0, 0, 1] < 0
    assert pred.grad[0, 0, 0, 3] < pred.grad[0, 0, 0, 0]


def test_multi_label_bce_can_ignore_expert_weights_for_equal_top2_targets() -> None:
    pred = torch.zeros((1, 1, 1, 4), dtype=torch.float32, requires_grad=True)
    expert_selection = torch.tensor([[[[1, 3]]]], dtype=torch.long)
    expert_weights = torch.tensor([[[[0.75, 0.25]]]], dtype=torch.float32)
    expert_mask = torch.tensor([[[[True, True]]]], dtype=torch.bool)
    attention_mask = torch.ones((1, 1), dtype=torch.long)

    loss, parts = train_src.multi_label_bce_loss(
        pred,
        expert_selection,
        attention_mask,
        expert_weights,
        expert_mask,
        use_expert_weights=False,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert parts["valid_items"] == 1
    assert pred.grad is not None
    assert pred.grad[0, 0, 0, 1] == pred.grad[0, 0, 0, 3]


def test_multi_label_bce_soft_gate_auxiliary_is_optional() -> None:
    pred = torch.tensor([[[[0.0, 1.5, -0.5, 0.2]]]], dtype=torch.float32, requires_grad=True)
    expert_selection = torch.tensor([[[[1, 3]]]], dtype=torch.long)
    expert_weights = torch.tensor([[[[0.75, 0.25]]]], dtype=torch.float32)
    expert_mask = torch.tensor([[[[True, True]]]], dtype=torch.bool)
    attention_mask = torch.ones((1, 1), dtype=torch.long)

    base_loss, base_parts = train_src.loss_for_type(
        "multi_label_bce",
        pred,
        expert_selection,
        attention_mask,
        expert_weights=expert_weights,
        expert_selection_mask=expert_mask,
        use_expert_weights_in_loss=False,
    )
    aux_loss, aux_parts = train_src.loss_for_type(
        "multi_label_bce",
        pred,
        expert_selection,
        attention_mask,
        expert_weights=expert_weights,
        expert_selection_mask=expert_mask,
        use_expert_weights_in_loss=False,
        soft_gate_aux_weight=0.1,
        soft_gate_aux_type="kl",
    )

    assert "soft_gate_aux" not in base_parts
    assert aux_parts["soft_gate_aux"] > 0
    assert aux_loss > base_loss


def test_multi_label_bce_count_regularizers_are_optional() -> None:
    pred = torch.full((1, 1, 2, 4), -2.0, dtype=torch.float32, requires_grad=True)
    expert_selection = torch.tensor([[[[0, 1], [0, 2]]]], dtype=torch.long)
    expert_mask = torch.tensor([[[[True, True], [True, True]]]], dtype=torch.bool)
    attention_mask = torch.ones((1, 2), dtype=torch.long)

    base_loss, base_parts = train_src.loss_for_type(
        "multi_label_bce",
        pred,
        expert_selection,
        attention_mask,
        expert_selection_mask=expert_mask,
        use_expert_weights_in_loss=False,
    )
    count_loss, count_parts = train_src.loss_for_type(
        "multi_label_bce",
        pred,
        expert_selection,
        attention_mask,
        expert_selection_mask=expert_mask,
        use_expert_weights_in_loss=False,
        token_cardinality_loss_weight=0.1,
        layer_count_loss_weight=0.2,
    )

    assert "token_cardinality" not in base_parts
    assert "layer_count" not in base_parts
    assert count_parts["token_cardinality"] > 0
    assert count_parts["layer_count"] > 0
    assert count_loss > base_loss


def test_multi_label_bce_infers_mask_from_positive_weights_for_top2() -> None:
    pred = torch.zeros((1, 1, 2, 4), dtype=torch.float32, requires_grad=True)
    expert_selection = torch.tensor([[[[1, 3], [0, 2]]]], dtype=torch.long)
    expert_weights = torch.tensor([[[[0.75, 0.25], [0.0, 0.0]]]], dtype=torch.float32)
    attention_mask = torch.ones((1, 2), dtype=torch.long)

    loss, parts = train_src.multi_label_bce_loss(pred, expert_selection, attention_mask, expert_weights, None)
    loss.backward()

    assert parts["valid_items"] == 1
    assert pred.grad[0, 0, 0, 1] < 0
    assert pred.grad[0, 0, 0, 3] < 0
    assert torch.equal(pred.grad[0, 0, 1], torch.zeros(4))


def test_multi_label_bce_rejects_top2_without_mask_or_weights() -> None:
    pred = torch.zeros((1, 1, 1, 4), dtype=torch.float32)
    expert_selection = torch.tensor([[[[1, 3]]]], dtype=torch.long)
    attention_mask = torch.ones((1, 1), dtype=torch.long)

    try:
        train_src.multi_label_bce_loss(pred, expert_selection, attention_mask)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected missing top2 mask/weights to be rejected")

    assert "requires expert_selection_mask or expert_weights" in message


def test_multi_label_bce_rejects_nonfinite_weights() -> None:
    pred = torch.zeros((1, 1, 1, 4), dtype=torch.float32)
    expert_selection = torch.tensor([[[[1, 3]]]], dtype=torch.long)
    expert_weights = torch.tensor([[[[float("nan"), 0.25]]]], dtype=torch.float32)
    expert_mask = torch.tensor([[[[True, True]]]], dtype=torch.bool)
    attention_mask = torch.ones((1, 1), dtype=torch.long)

    try:
        train_src.multi_label_bce_loss(pred, expert_selection, attention_mask, expert_weights, expert_mask)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected NaN expert_weights to be rejected")

    assert "finite" in message


def test_multi_label_bce_rejects_out_of_range_weights() -> None:
    pred = torch.zeros((1, 1, 1, 4), dtype=torch.float32)
    expert_selection = torch.tensor([[[[1, 3]]]], dtype=torch.long)
    expert_weights = torch.tensor([[[[1.2, 0.25]]]], dtype=torch.float32)
    expert_mask = torch.tensor([[[[True, True]]]], dtype=torch.bool)
    attention_mask = torch.ones((1, 1), dtype=torch.long)

    try:
        train_src.multi_label_bce_loss(pred, expert_selection, attention_mask, expert_weights, expert_mask)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected out-of-range expert_weights to be rejected")

    assert "[0, 1]" in message


def test_valid_sample_tensors_crops_padding() -> None:
    batch = {
        "layer0_attn_out": torch.arange(1 * 5 * 2, dtype=torch.float32).reshape(1, 5, 2),
        "attention_mask": torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long),
        "expert_selection": torch.zeros(1, 2, 5, 1, dtype=torch.long),
    }

    sample = train_src.valid_sample_tensors(batch, 0)

    assert sample is not None
    x, expert_selection, mask = sample
    assert x.shape == (1, 3, 2)
    assert expert_selection.shape == (1, 2, 3, 1)
    assert mask.tolist() == [[1, 1, 1]]


def test_valid_sample_tensors_crops_optional_top2_weights_and_mask() -> None:
    batch = {
        "layer0_attn_out": torch.arange(1 * 5 * 2, dtype=torch.float32).reshape(1, 5, 2),
        "attention_mask": torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long),
        "expert_selection": torch.zeros(1, 2, 5, 2, dtype=torch.long),
        "expert_weights": torch.ones(1, 2, 5, 2, dtype=torch.float32),
        "expert_selection_mask": torch.ones(1, 2, 5, 2, dtype=torch.bool),
    }

    sample = train_src.valid_sample_tensors(batch, 0)

    assert sample is not None
    x, expert_selection, mask, expert_weights, expert_selection_mask = sample
    assert x.shape == (1, 3, 2)
    assert expert_selection.shape == (1, 2, 3, 2)
    assert expert_weights.shape == (1, 2, 3, 2)
    assert expert_selection_mask.shape == (1, 2, 3, 2)
    assert mask.tolist() == [[1, 1, 1]]


def test_trace_dataset_loads_only_hard_ce_required_fields(tmp_path: Path) -> None:
    trace_dir = write_tiny_sparse_cache_trace(tmp_path)

    dataset = train_src.EncoderPredictorTraceDataset(trace_dir, "train")
    item = dataset[0]

    assert set(item) == {"layer0_attn_out", "attention_mask", "expert_selection"}
    assert dataset.info.hidden_size == 4
    assert dataset.info.num_router_layers == 2
    assert dataset.info.num_experts == 3
    assert dataset.info.num_selected_experts == 1


def test_trace_dataset_loads_optional_top2_weights_and_mask(tmp_path: Path) -> None:
    trace_dir = write_tiny_sparse_cache_trace(tmp_path)
    metadata = json.loads((trace_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata["model_config"]["model_type"] = "nllb-moe"
    metadata["model_config"]["num_selected_experts"] = 2
    (trace_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    for split, samples in (("train", 2), ("validation", 1)):
        split_dir = trace_dir / split
        expert_selection = torch.zeros(samples, 2, 3, 2, dtype=torch.long)
        expert_selection[..., 1] = 1
        expert_weights = torch.full((samples, 2, 3, 2), 0.5, dtype=torch.float32)
        expert_selection_mask = torch.ones(samples, 2, 3, 2, dtype=torch.bool)
        torch.save(expert_selection, split_dir / "expert_selection.pt")
        torch.save(expert_weights, split_dir / "expert_weights.pt")
        torch.save(expert_selection_mask, split_dir / "expert_selection_mask.pt")

    dataset = train_src.EncoderPredictorTraceDataset(trace_dir, "train")
    item = dataset[0]

    assert set(item) == {
        "layer0_attn_out",
        "attention_mask",
        "expert_selection",
        "expert_weights",
        "expert_selection_mask",
    }
    assert item["expert_selection"].shape == (2, 3, 2)
    assert item["expert_weights"].shape == (2, 3, 2)
    assert item["expert_selection_mask"].dtype == torch.bool
    assert dataset.info.num_selected_experts == 2


def test_prefetch_metrics_count_both_top2_true_experts() -> None:
    pred_logits = torch.tensor([[[[0.0, 4.0, 1.0, 3.0]]]], dtype=torch.float32)
    expert_selection = torch.tensor([[[[1, 3]]]], dtype=torch.long)
    expert_selection_mask = torch.tensor([[[[True, True]]]], dtype=torch.bool)
    attention_mask = torch.ones((1, 1), dtype=torch.long)

    metrics = train_src.compute_prefetch_metrics(
        pred_logits,
        expert_selection,
        attention_mask,
        budgets=(1, 2),
        expert_selection_mask=expert_selection_mask,
    )

    assert metrics["num_valid_token_layer_items"] == 1
    assert metrics["overlap_count@1"] == 1
    assert metrics["overlap_count@2"] == 2
    assert metrics["topK_recall@1"] == 0.5
    assert metrics["topK_recall@2"] == 1.0


def test_train_epoch_forwards_only_valid_tokens(tmp_path: Path) -> None:
    trace_dir = write_tiny_sparse_cache_trace(tmp_path)
    dataset = train_src.EncoderPredictorTraceDataset(trace_dir, "train")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=train_src.collate_trace_batch)

    class RecordingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(4, 6)
            self.seen_lengths: list[int] = []

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.seen_lengths.append(int(x.shape[1]))
            return self.proj(x).reshape(x.shape[0], x.shape[1], 2, 3).permute(0, 2, 1, 3).contiguous()

    model = RecordingModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    metrics = train_src.train_one_epoch(model, loader, optimizer, torch.device("cpu"), max_batches=1)

    assert model.seen_lengths == [2]
    assert metrics["train_valid_items"] == 4


def test_make_config_uses_src_h384_l1_defaults(tmp_path: Path) -> None:
    info = train_src.TraceInfo(
        hidden_size=768,
        num_experts=128,
        num_selected_experts=1,
        num_router_layers=6,
        max_input_tokens=512,
    )
    args = train_src.parse_args([
        "--trace-dir",
        str(tmp_path / "trace"),
        "--output-dir",
        str(tmp_path / "out"),
    ])

    config = train_src.make_config(
        args,
        info,
        {"model_config": {"hidden_size": 768, "num_experts": 128}},
        torch.device("cpu"),
    )

    assert config["model"] == "src-simplenn-token"
    assert config["objective"] == "hard-ce"
    assert config["output_name"] == "src-simplenn-token-hardce-h384-l1-drop0p5-lr1e4-bs512-seed0-validtrim"
    assert config["hidden_dim"] == 384
    assert config["src_layers"] == 1
    assert config["dropout"] == 0.5
    assert config["metadata"]["num_encoder_moe_layers"] == 6
    assert config["use_expert_weights_in_loss"] is True


def test_make_config_records_soft_gate_auxiliary_settings(tmp_path: Path) -> None:
    info = train_src.TraceInfo(
        hidden_size=768,
        num_experts=128,
        num_selected_experts=2,
        num_router_layers=6,
        max_input_tokens=512,
    )
    args = train_src.parse_args([
        "--trace-dir",
        str(tmp_path / "trace"),
        "--output-dir",
        str(tmp_path / "out"),
        "--loss-type",
        "multi_label_bce",
        "--no-use-expert-weights-in-loss",
        "--soft-gate-aux-weight",
        "0.03",
        "--soft-gate-aux-type",
        "ce",
    ])

    config = train_src.make_config(
        args,
        info,
        {"model_config": {"model_type": "nllb-moe", "hidden_size": 768, "num_experts": 128}},
        torch.device("cpu"),
    )
    manifest = train_src.make_run_manifest(args, config)

    assert config["soft_gate_aux_weight"] == 0.03
    assert config["soft_gate_aux_type"] == "ce"
    assert manifest["soft_gate_aux_weight"] == 0.03
    assert manifest["soft_gate_aux_type"] == "ce"


def test_make_config_records_disabled_expert_weight_loss(tmp_path: Path) -> None:
    info = train_src.TraceInfo(
        hidden_size=768,
        num_experts=128,
        num_selected_experts=2,
        num_router_layers=6,
        max_input_tokens=512,
    )
    args = train_src.parse_args([
        "--trace-dir",
        str(tmp_path / "trace"),
        "--output-dir",
        str(tmp_path / "out"),
        "--no-use-expert-weights-in-loss",
    ])

    config = train_src.make_config(
        args,
        info,
        {"model_config": {"model_type": "nllb-moe", "hidden_size": 768, "num_experts": 128}},
        torch.device("cpu"),
    )

    assert config["loss_type"] == "multi_label_bce"
    assert config["use_expert_weights_in_loss"] is False


def test_output_name_distinguishes_weighted_and_equal_top2_loss(tmp_path: Path) -> None:
    weighted = train_src.parse_args([
        "--trace-dir",
        str(tmp_path / "trace"),
        "--loss-type",
        "multi_label_bce",
        "--use-expert-weights-in-loss",
    ])
    equal = train_src.parse_args([
        "--trace-dir",
        str(tmp_path / "trace"),
        "--loss-type",
        "multi_label_bce",
        "--no-use-expert-weights-in-loss",
    ])

    assert train_src.output_name_from_args(weighted) == (
        "src-simplenn-token-bce-weighted-h384-l1-drop0p5-lr1e4-bs512-seed0-validtrim"
    )
    assert train_src.output_name_from_args(equal) == (
        "src-simplenn-token-bce-equal-top2-h384-l1-drop0p5-lr1e4-bs512-seed0-validtrim"
    )
    assert train_src.resolve_output_dir(weighted) != train_src.resolve_output_dir(equal)


def test_output_name_distinguishes_soft_gate_auxiliary_loss(tmp_path: Path) -> None:
    args = train_src.parse_args([
        "--trace-dir",
        str(tmp_path / "trace"),
        "--loss-type",
        "multi_label_bce",
        "--no-use-expert-weights-in-loss",
        "--soft-gate-aux-weight",
        "0.01",
    ])

    assert train_src.output_name_from_args(args) == (
        "src-simplenn-token-bce-equal-top2-softgate0p01-h384-l1-drop0p5-lr1e4-bs512-seed0-validtrim"
    )


def test_output_name_distinguishes_count_regularized_equal_top2_loss(tmp_path: Path) -> None:
    plain = train_src.parse_args([
        "--trace-dir",
        str(tmp_path / "trace"),
        "--loss-type",
        "multi_label_bce",
        "--no-use-expert-weights-in-loss",
    ])
    count_regularized = train_src.parse_args([
        "--trace-dir",
        str(tmp_path / "trace"),
        "--loss-type",
        "multi_label_bce",
        "--no-use-expert-weights-in-loss",
        "--token-cardinality-loss-weight",
        "0.1",
        "--layer-count-loss-weight",
        "0.05",
    ])

    assert train_src.output_name_from_args(plain) == (
        "src-simplenn-token-bce-equal-top2-h384-l1-drop0p5-lr1e4-bs512-seed0-validtrim"
    )
    assert train_src.output_name_from_args(count_regularized) == (
        "src-simplenn-token-bce-equal-top2-tokcnt0p1-layercnt0p05-h384-l1-drop0p5-lr1e4-bs512-seed0-validtrim"
    )
    assert train_src.resolve_output_dir(plain) != train_src.resolve_output_dir(count_regularized)


def test_auto_nllb_training_resolves_bce_weight_mode_into_default_output_dir(tmp_path: Path) -> None:
    trace_dir = write_tiny_sparse_cache_trace(tmp_path)
    metadata = json.loads((trace_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata["model_config"]["model_type"] = "nllb-moe"
    metadata["model_config"]["num_selected_experts"] = 2
    (trace_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    for split, samples in (("train", 2), ("validation", 1)):
        split_dir = trace_dir / split
        expert_selection = torch.zeros(samples, 2, 3, 2, dtype=torch.long)
        expert_selection[..., 1] = 1
        torch.save(expert_selection, split_dir / "expert_selection.pt")
        torch.save(torch.full((samples, 2, 3, 2), 0.5, dtype=torch.float32), split_dir / "expert_weights.pt")
        torch.save(torch.ones(samples, 2, 3, 2, dtype=torch.bool), split_dir / "expert_selection_mask.pt")

    args = train_src.parse_args([
        "--trace-dir",
        str(trace_dir),
        "--hidden-dim",
        "8",
        "--batch-size",
        "1",
        "--no-use-expert-weights-in-loss",
    ])
    dataset = train_src.EncoderPredictorTraceDataset(trace_dir, "train")
    args.loss_type = train_src.resolve_loss_type(args.loss_type, dataset.metadata)

    expected_name = "src-simplenn-token-bce-equal-top2-h8-l1-drop0p5-lr1e4-bs1-seed0-validtrim"
    assert train_src.output_name_from_args(args) == expected_name
    assert train_src.resolve_output_dir(args).name == expected_name


def test_training_smoke_writes_artifacts(tmp_path: Path) -> None:
    trace_dir = write_tiny_sparse_cache_trace(tmp_path)
    output_dir = tmp_path / "model"

    metrics = train_src.main(
        [
            "--trace-dir",
            str(trace_dir),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--hidden-dim",
            "8",
            "--device",
            "cpu",
            "--max-train-batches",
            "1",
            "--max-eval-batches",
            "1",
        ]
    )

    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(output_dir / "best_model.pt", map_location="cpu", weights_only=True)

    assert (output_dir / "model.pt").exists()
    assert (output_dir / "best_model.pt").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "run_manifest.json").exists()
    assert (output_dir / "train_log.jsonl").exists()
    assert "model_state_dict" in checkpoint
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))

    assert config["model"] == "src-simplenn-token"
    assert config["objective"] == "hard-ce"
    assert config["hidden_dim"] == 8
    assert config["src_layers"] == 1
    assert config["use_expert_weights_in_loss"] is True
    assert config["metadata"]["hidden_size"] == 4
    assert config["metadata"]["num_encoder_moe_layers"] == 2
    assert config["metadata"]["max_input_tokens"] == 3
    assert manifest["artifact_level"] == "blte"
    assert manifest["output_layout"] == "BLTE"
    assert manifest["predictor_task"] == "encoder_expert_prefetch"
    assert manifest["workload_task"] == "mmlu-professional_law"
    assert manifest["base_model"] == "switch-base-128"
    assert manifest["trace_id"] == "sparse-cache-b1-longest-v1"
    assert manifest["data_mode"] == "validtrim"
    assert manifest["use_expert_weights_in_loss"] is True
    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "use_expert_weights_in_loss: True" in readme
    assert metrics["validation_valid_items"] == 4

def test_count_regularized_training_smoke_writes_artifact_settings(tmp_path: Path) -> None:
    trace_dir = write_tiny_sparse_cache_trace(tmp_path)
    metadata = json.loads((trace_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata["model_config"]["model_type"] = "nllb-moe"
    metadata["model_config"]["num_selected_experts"] = 2
    (trace_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    for split, samples in (("train", 2), ("validation", 1)):
        split_dir = trace_dir / split
        expert_selection = torch.zeros(samples, 2, 3, 2, dtype=torch.long)
        expert_selection[..., 1] = 1
        torch.save(expert_selection, split_dir / "expert_selection.pt")
        torch.save(torch.full((samples, 2, 3, 2), 0.5, dtype=torch.float32), split_dir / "expert_weights.pt")
        torch.save(torch.ones(samples, 2, 3, 2, dtype=torch.bool), split_dir / "expert_selection_mask.pt")
    output_dir = tmp_path / "count_model"

    train_src.main(
        [
            "--trace-dir",
            str(trace_dir),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--hidden-dim",
            "8",
            "--device",
            "cpu",
            "--max-train-batches",
            "1",
            "--max-eval-batches",
            "1",
            "--no-use-expert-weights-in-loss",
            "--token-cardinality-loss-weight",
            "0.1",
            "--layer-count-loss-weight",
            "0.05",
        ]
    )

    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    readme = (output_dir / "README.md").read_text(encoding="utf-8")

    assert config["loss_type"] == "multi_label_bce"
    assert config["use_expert_weights_in_loss"] is False
    assert config["token_cardinality_loss_weight"] == 0.1
    assert config["layer_count_loss_weight"] == 0.05
    assert manifest["token_cardinality_loss_weight"] == 0.1
    assert manifest["layer_count_loss_weight"] == 0.05
    assert "token_cardinality_loss_weight: 0.1" in readme
    assert "layer_count_loss_weight: 0.05" in readme

def test_count_regularized_entrypoint_runs_without_pythonpath(tmp_path: Path) -> None:
    trace_dir = write_tiny_sparse_cache_trace(tmp_path)
    metadata = json.loads((trace_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata["model_config"]["model_type"] = "nllb-moe"
    metadata["model_config"]["num_selected_experts"] = 2
    (trace_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    for split, samples in (("train", 2), ("validation", 1)):
        split_dir = trace_dir / split
        expert_selection = torch.zeros(samples, 2, 3, 2, dtype=torch.long)
        expert_selection[..., 1] = 1
        torch.save(expert_selection, split_dir / "expert_selection.pt")
        torch.save(torch.full((samples, 2, 3, 2), 0.5, dtype=torch.float32), split_dir / "expert_weights.pt")
        torch.save(torch.ones(samples, 2, 3, 2, dtype=torch.bool), split_dir / "expert_selection_mask.pt")
    output_dir = tmp_path / "entrypoint_model"

    result = subprocess.run(
        [
            sys.executable,
            "experiment/scripts/train/encoder_predictor_src_simplenn_token_bce_count_reg.py",
            "--trace-dir",
            str(trace_dir),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--hidden-dim",
            "8",
            "--device",
            "cpu",
            "--max-train-batches",
            "1",
            "--max-eval-batches",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert config["token_cardinality_loss_weight"] == 0.1
    assert config["layer_count_loss_weight"] == 0.05

