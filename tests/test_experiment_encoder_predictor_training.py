from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from experiment.scripts.train.encoder_predictor_sida_gru_sa_hard_ce import (
    EncoderPredictorTraceDataset,
    SidaGRUSparseAttentionPredictor,
    Sparsemax,
    collate_trace_batch,
    compute_prefetch_metrics,
    hard_ce_loss,
    multi_label_bce_loss,
    main,
    train_one_epoch,
    parse_trace_metadata,
)


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
        router_logits = torch.zeros(samples, 2, 3, 3, dtype=torch.float32)
        router_probs = torch.softmax(router_logits, dim=-1)
        expert_selection = torch.zeros(samples, 2, 3, 1, dtype=torch.long)
        torch.save(layer0, split_dir / "layer0_attn_out.pt")
        torch.save(attention_mask, split_dir / "attention_mask.pt")
        torch.save(router_logits, split_dir / "router_logits.pt")
        torch.save(router_probs, split_dir / "router_probs.pt")
        torch.save(expert_selection, split_dir / "expert_selection.pt")

    write_split(train_dir, 2)
    write_split(validation_dir, 1)
    return trace_dir


def test_trace_dataset_parses_sparse_cache_metadata_and_shapes(tmp_path: Path) -> None:
    trace_dir = write_tiny_sparse_cache_trace(tmp_path)
    dataset = EncoderPredictorTraceDataset(trace_dir, "train")

    info = parse_trace_metadata(dataset.metadata, dataset.tensors)
    item = dataset[0]

    assert len(dataset) == 2
    assert info.hidden_size == 4
    assert info.num_experts == 3
    assert info.num_router_layers == 2
    assert info.num_selected_experts == 1
    assert item["layer0_attn_out"].shape == (3, 4)
    assert item["attention_mask"].shape == (3,)
    assert item["expert_selection"].shape == (2, 3, 1)


def test_model_state_dict_uses_compare_compatible_sida_head_names() -> None:
    model = SidaGRUSparseAttentionPredictor(
        input_dim=4,
        hidden_dim=8,
        num_router_layers=2,
        num_experts=3,
        recurrent_layers=1,
    )

    state_keys = set(model.state_dict())

    assert "fc.router-0.weight" in state_keys
    assert "fc.router-1.bias" in state_keys


def test_sparsemax_matches_original_sida_formula() -> None:
    output = Sparsemax()(torch.tensor([[0.0, 2.0, -1.0]], dtype=torch.float32))

    assert torch.allclose(output, torch.tensor([[0.0, 2.0, 0.0]]), atol=1e-6)


def test_model_forward_keeps_original_unmasked_signature() -> None:
    model = SidaGRUSparseAttentionPredictor(
        input_dim=4,
        hidden_dim=8,
        num_router_layers=1,
        num_experts=3,
        recurrent_layers=1,
    )
    x = torch.randn(1, 3, 4, dtype=torch.float32)

    pred = model(x)

    assert pred.shape == (1, 1, 3, 3)
    with pytest.raises(TypeError):
        model(x, torch.ones(1, 3, dtype=torch.long))


def test_train_epoch_forwards_only_valid_tokens_without_model_mask(tmp_path: Path) -> None:
    trace_dir = write_tiny_sparse_cache_trace(tmp_path)
    dataset = EncoderPredictorTraceDataset(trace_dir, "train")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_trace_batch)

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

    metrics = train_one_epoch(model, loader, optimizer, torch.device("cpu"), max_batches=1)

    assert model.seen_lengths == [2]
    assert metrics["train_valid_items"] == 4


def test_all_padding_train_batch_does_not_update_parameters(tmp_path: Path) -> None:
    trace_dir = write_tiny_sparse_cache_trace(tmp_path)
    split_dir = trace_dir / "train"
    torch.save(torch.zeros(2, 3, dtype=torch.long), split_dir / "attention_mask.pt")
    dataset = EncoderPredictorTraceDataset(trace_dir, "train")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_trace_batch)
    model = SidaGRUSparseAttentionPredictor(
        input_dim=4,
        hidden_dim=8,
        num_router_layers=2,
        num_experts=3,
        recurrent_layers=1,
    )
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.5)

    metrics = train_one_epoch(model, loader, optimizer, torch.device("cpu"))

    assert metrics["train_valid_items"] == 0
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


def test_trace_dataset_rejects_invalid_mask_and_expert_ids(tmp_path: Path) -> None:
    trace_dir = write_tiny_sparse_cache_trace(tmp_path)
    train_dir = trace_dir / "train"
    mask = torch.load(train_dir / "attention_mask.pt", map_location="cpu", weights_only=True)
    mask[0, 0] = 2
    torch.save(mask, train_dir / "attention_mask.pt")

    with pytest.raises(ValueError, match="attention_mask"):
        EncoderPredictorTraceDataset(trace_dir, "train")

    trace_dir = write_tiny_sparse_cache_trace(tmp_path / "bad_expert")
    train_dir = trace_dir / "train"
    expert_selection = torch.load(train_dir / "expert_selection.pt", map_location="cpu", weights_only=True)
    expert_selection[0, 0, 0, 0] = 3
    torch.save(expert_selection, train_dir / "expert_selection.pt")

    with pytest.raises(ValueError, match="expert_selection"):
        EncoderPredictorTraceDataset(trace_dir, "train")


def test_hard_ce_loss_ignores_padding_tokens() -> None:
    pred_logits = torch.tensor(
        [[[[0.0, 20.0, -20.0], [20.0, -20.0, 0.0]]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    expert_selection = torch.tensor([[[[1], [1]]]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 0]], dtype=torch.long)

    loss, parts = hard_ce_loss(pred_logits, expert_selection, attention_mask)
    loss.backward()

    assert torch.isclose(loss.detach(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(parts["ce"].detach(), torch.tensor(0.0), atol=1e-6)
    assert parts["valid_items"] == 1
    assert pred_logits.grad is not None
    assert torch.equal(pred_logits.grad[0, 0, 1], torch.zeros(3))


def test_multi_label_bce_uses_both_top2_experts() -> None:
    pred = torch.zeros((1, 1, 1, 4), dtype=torch.float32, requires_grad=True)
    expert_selection = torch.tensor([[[[1, 3]]]], dtype=torch.long)
    expert_weights = torch.tensor([[[[0.75, 0.25]]]], dtype=torch.float32)
    expert_mask = torch.tensor([[[[True, True]]]], dtype=torch.bool)
    attention_mask = torch.ones((1, 1), dtype=torch.long)

    loss, parts = multi_label_bce_loss(
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


def test_multi_label_bce_infers_mask_from_positive_weights_for_top2() -> None:
    pred = torch.zeros((1, 1, 2, 4), dtype=torch.float32, requires_grad=True)
    expert_selection = torch.tensor([[[[1, 3], [0, 2]]]], dtype=torch.long)
    expert_weights = torch.tensor([[[[0.75, 0.25], [0.0, 0.0]]]], dtype=torch.float32)
    attention_mask = torch.ones((1, 2), dtype=torch.long)

    loss, parts = multi_label_bce_loss(pred, expert_selection, attention_mask, expert_weights, None)
    loss.backward()

    assert parts["valid_items"] == 1
    assert pred.grad[0, 0, 0, 1] < 0
    assert pred.grad[0, 0, 0, 3] < 0
    assert torch.equal(pred.grad[0, 0, 1], torch.zeros(4))


def test_multi_label_bce_rejects_top2_without_mask_or_weights() -> None:
    pred = torch.zeros((1, 1, 1, 4), dtype=torch.float32)
    expert_selection = torch.tensor([[[[1, 3]]]], dtype=torch.long)
    attention_mask = torch.ones((1, 1), dtype=torch.long)

    with pytest.raises(ValueError, match="requires expert_selection_mask or expert_weights"):
        multi_label_bce_loss(pred, expert_selection, attention_mask)


def test_multi_label_bce_rejects_nonfinite_weights() -> None:
    pred = torch.zeros((1, 1, 1, 4), dtype=torch.float32)
    expert_selection = torch.tensor([[[[1, 3]]]], dtype=torch.long)
    expert_weights = torch.tensor([[[[float("inf"), 0.25]]]], dtype=torch.float32)
    expert_mask = torch.tensor([[[[True, True]]]], dtype=torch.bool)
    attention_mask = torch.ones((1, 1), dtype=torch.long)

    with pytest.raises(ValueError, match="finite"):
        multi_label_bce_loss(pred, expert_selection, attention_mask, expert_weights, expert_mask)


def test_multi_label_bce_rejects_out_of_range_weights() -> None:
    pred = torch.zeros((1, 1, 1, 4), dtype=torch.float32)
    expert_selection = torch.tensor([[[[1, 3]]]], dtype=torch.long)
    expert_weights = torch.tensor([[[[1.2, 0.25]]]], dtype=torch.float32)
    expert_mask = torch.tensor([[[[True, True]]]], dtype=torch.bool)
    attention_mask = torch.ones((1, 1), dtype=torch.long)

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        multi_label_bce_loss(pred, expert_selection, attention_mask, expert_weights, expert_mask)


def test_prefetch_metrics_ignore_padding_tokens() -> None:
    pred_logits = torch.zeros(1, 1, 2, 3, dtype=torch.float32)
    expert_selection = torch.tensor([[[[1], [2]]]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 0]], dtype=torch.long)
    pred_logits[0, 0, 0] = torch.tensor([0.0, 2.0, 1.0])
    pred_logits[0, 0, 1] = torch.tensor([0.0, 0.0, 5.0])

    metrics = compute_prefetch_metrics(pred_logits, expert_selection, attention_mask, budgets=(1, 2))

    assert metrics["num_valid_token_layer_items"] == 1
    assert metrics["overlap_count@1"] == 1
    assert metrics["topK_recall@1"] == 1.0
    assert metrics["topK_precision@1"] == 1.0
    assert metrics["mean_true_rank"] == 1.0


def test_prefetch_metrics_count_both_top2_true_experts() -> None:
    pred_logits = torch.tensor([[[[0.0, 4.0, 1.0, 3.0]]]], dtype=torch.float32)
    expert_selection = torch.tensor([[[[1, 3]]]], dtype=torch.long)
    expert_selection_mask = torch.tensor([[[[True, True]]]], dtype=torch.bool)
    attention_mask = torch.ones((1, 1), dtype=torch.long)

    metrics = compute_prefetch_metrics(
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


def test_training_smoke_writes_artifacts_and_valid_item_counts(tmp_path: Path) -> None:
    trace_dir = write_tiny_sparse_cache_trace(tmp_path)
    output_dir = tmp_path / "model"

    metrics = main(
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

    assert (output_dir / "model.pt").exists()
    assert (output_dir / "best_model.pt").exists()
    checkpoint = torch.load(output_dir / "best_model.pt", map_location="cpu", weights_only=True)
    assert "model_state_dict" in checkpoint
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "train_log.jsonl").exists()
    assert config["model"] == "sida-gru-sa"
    assert config["objective"] == "hard-ce"
    assert config["hidden_dim"] == 8
    assert config["metadata"]["hidden_size"] == 4
    assert config["metadata"]["num_encoder_moe_layers"] == 2
    assert config["metadata"]["max_input_tokens"] == 3
    assert "sparse_cache_metadata" in config["metadata"]
    assert metrics["validation_valid_items"] == 4
