from __future__ import annotations

import json
from pathlib import Path

import torch

from performance_predictor.encoder.ERPP.implement.train.baseline.data import (
    ErppTraceDataset,
    collate_trace_batch,
)
from performance_predictor.encoder.ERPP.implement.train.baseline import (
    compute_prefetch_metrics,
    expert_frequency_targets,
    masked_router_loss,
    objective_router_loss,
    sample_gate_targets,
)
from performance_predictor.encoder.ERPP.implement.train.baseline.train_baseline import (
    build_model,
    main,
    parse_args,
)


def write_tiny_trace(root: Path) -> Path:
    trace_dir = root / "trace"
    train_dir = trace_dir / "train"
    validation_dir = trace_dir / "validation"
    train_dir.mkdir(parents=True)
    validation_dir.mkdir(parents=True)

    metadata = {
        "hidden_size": 4,
        "num_encoder_moe_layers": 2,
        "num_experts": 3,
        "routing_top_k": 1,
        "splits": {
            "train": {"num_samples": 2},
            "validation": {"num_samples": 1},
        },
    }
    (trace_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    def write_split(split_dir: Path, samples: int) -> None:
        layer0 = torch.arange(samples * 3 * 4, dtype=torch.float32).reshape(samples, 3, 4)
        mask = torch.tensor([[1, 1, 0]] * samples, dtype=torch.long)
        logits = torch.zeros(samples, 2, 3, 3, dtype=torch.float32)
        probs = torch.softmax(logits, dim=-1)
        selection = torch.zeros(samples, 2, 3, 1, dtype=torch.long)
        torch.save(layer0, split_dir / "layer0_attn_out.pt")
        torch.save(mask, split_dir / "attention_mask.pt")
        torch.save(logits, split_dir / "router_logits.pt")
        torch.save(probs, split_dir / "router_probs.pt")
        torch.save(selection, split_dir / "expert_selection.pt")

    write_split(train_dir, 2)
    write_split(validation_dir, 1)
    return trace_dir


def test_trace_dataset_loads_expected_item_shapes(tmp_path: Path) -> None:
    trace_dir = write_tiny_trace(tmp_path)
    dataset = ErppTraceDataset(trace_dir, "train")

    item = dataset[0]
    second_item = dataset[1]

    assert len(dataset) == 2
    assert item["layer0_attn_out"].shape == (3, 4)
    assert item["attention_mask"].shape == (3,)
    assert item["router_probs"].shape == (2, 3, 3)
    assert item["expert_selection"].shape == (2, 3, 1)
    assert not torch.equal(item["layer0_attn_out"], second_item["layer0_attn_out"])
    assert torch.equal(
        second_item["layer0_attn_out"],
        torch.arange(12, 24, dtype=torch.float32).reshape(3, 4),
    )


def test_collate_trace_batch_stacks_sequence_first_tensors(tmp_path: Path) -> None:
    trace_dir = write_tiny_trace(tmp_path)
    dataset = ErppTraceDataset(trace_dir, "train")

    first_item = dataset[0]
    second_item = dataset[1]
    batch = collate_trace_batch([first_item, second_item])

    assert batch["layer0_attn_out"].shape == (2, 3, 4)
    assert batch["attention_mask"].shape == (2, 3)
    assert batch["router_probs"].shape == (2, 2, 3, 3)
    assert batch["expert_selection"].shape == (2, 2, 3, 1)
    assert torch.equal(batch["layer0_attn_out"][0], first_item["layer0_attn_out"])
    assert torch.equal(batch["layer0_attn_out"][1], second_item["layer0_attn_out"])


def test_masked_router_loss_ignores_padding_tokens() -> None:
    pred_logits = torch.tensor(
        [[[[0.0, 50.0, 0.0], [50.0, 0.0, 0.0]]]],
        dtype=torch.float32,
    )
    router_probs = torch.tensor(
        [[[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]]],
        dtype=torch.float32,
    )
    expert_selection = torch.tensor([[[[1], [1]]]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 0]], dtype=torch.long)

    total, parts = masked_router_loss(
        pred_logits,
        router_probs,
        expert_selection,
        attention_mask,
    )

    assert torch.isclose(total, torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(parts["kl"], torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(parts["ce"], torch.tensor(0.0), atol=1e-6)


def test_masked_router_loss_expands_attention_mask_across_layers() -> None:
    batch, layers, tokens, experts = 2, 3, 4, 3
    attention_mask = torch.tensor(
        [[1, 0, 1, 0], [0, 1, 1, 0]],
        dtype=torch.long,
    )
    expert_selection = torch.ones(batch, layers, tokens, 1, dtype=torch.long)
    router_probs = torch.zeros(batch, layers, tokens, experts, dtype=torch.float32)
    router_probs[..., 1] = 1.0
    pred_logits = torch.full((batch, layers, tokens, experts), -50.0)

    valid = attention_mask.bool()[:, None, :].expand(batch, layers, tokens)
    pred_logits[..., 0] = 50.0
    pred_logits[valid] = torch.tensor([-50.0, 50.0, -50.0])

    total, parts = masked_router_loss(
        pred_logits,
        router_probs,
        expert_selection,
        attention_mask,
    )

    assert torch.isclose(total, torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(parts["kl"], torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(parts["ce"], torch.tensor(0.0), atol=1e-6)


def test_masked_router_loss_all_padding_keeps_zero_gradient() -> None:
    pred_logits = torch.randn(2, 3, 4, 5, requires_grad=True)
    router_probs = torch.softmax(torch.randn(2, 3, 4, 5), dim=-1)
    expert_selection = torch.randint(0, 5, (2, 3, 4, 1), dtype=torch.long)
    attention_mask = torch.zeros(2, 4, dtype=torch.long)

    total, parts = masked_router_loss(
        pred_logits,
        router_probs,
        expert_selection,
        attention_mask,
    )
    total.backward()

    assert torch.isclose(total, torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(parts["kl"], torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(parts["ce"], torch.tensor(0.0), atol=1e-6)
    assert pred_logits.grad is not None
    assert torch.equal(pred_logits.grad, torch.zeros_like(pred_logits))



def test_sample_gate_targets_build_layer_level_labels_from_router_logits() -> None:
    router_logits = torch.tensor(
        [[[[1.0, -1.0, 0.0], [3.0, 1.0, -2.0], [100.0, 100.0, 100.0]]]],
        dtype=torch.float32,
    )
    attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.long)

    valid_logits = router_logits[:, :, :2]
    max1_tokens = valid_logits / valid_logits.abs().amax(dim=-1, keepdim=True)
    max1_expected = max1_tokens.mean(dim=2)
    centered = valid_logits - valid_logits.mean(dim=-1, keepdim=True)
    std_expected = (centered / centered.std(dim=-1, keepdim=True, unbiased=False)).mean(dim=2)
    replace_expected = torch.softmax(valid_logits, dim=-1).mean(dim=2)

    assert torch.allclose(sample_gate_targets(router_logits, attention_mask, "max1"), max1_expected)
    assert torch.allclose(sample_gate_targets(router_logits, attention_mask, "std"), std_expected)
    assert torch.allclose(sample_gate_targets(router_logits, attention_mask, "replace"), replace_expected)


def test_sample_gate_objective_uses_layer_level_router_logits() -> None:
    pred = torch.zeros(1, 1, 3, dtype=torch.float32, requires_grad=True)
    router_logits = torch.tensor([[[[1.0, -1.0, 0.0], [3.0, 1.0, -2.0]]]], dtype=torch.float32)
    router_probs = torch.softmax(router_logits, dim=-1)
    expert_selection = torch.tensor([[[[0], [0]]]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1]], dtype=torch.long)

    total, parts = objective_router_loss(
        pred,
        router_probs,
        expert_selection,
        attention_mask,
        objective="sample-gate-replace-smoothl1",
        router_logits=router_logits,
    )
    total.backward()

    assert total.item() > 0
    assert torch.isclose(parts["aux"], total)
    assert pred.grad is not None


def test_prefetch_metrics_report_top1_and_extra_budget_hits() -> None:
    pred_logits = torch.zeros(1, 2, 2, 20, dtype=torch.float32)
    expert_selection = torch.tensor(
        [[[[0], [2]], [[4], [1]]]],
        dtype=torch.long,
    )
    router_probs = torch.softmax(torch.zeros_like(pred_logits), dim=-1)
    attention_mask = torch.tensor([[1, 1]], dtype=torch.long)

    # True expert ranks over valid token/layer items: 1, 2, 3, 5.
    pred_logits[0, 0, 0, [0, 1, 2, 3, 4]] = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    pred_logits[0, 0, 1, [0, 2, 1, 3, 4]] = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    pred_logits[0, 1, 0, [0, 1, 4, 2, 3]] = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    pred_logits[0, 1, 1, [0, 2, 3, 4, 1]] = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])

    metrics = compute_prefetch_metrics(
        pred_logits,
        expert_selection,
        router_probs,
        attention_mask,
        budgets=(1, 2, 3, 5),
    )

    assert metrics["num_valid_token_layer_items"] == 4
    assert metrics["overlap_count@1"] == 1
    assert metrics["topK_precision@1"] == 0.25
    assert metrics["topK_recall@1"] == 0.25
    assert metrics["MAP@1"] == 0.25
    assert metrics["NDCG@1"] == 0.25

    assert metrics["overlap_count@2"] == 2
    assert metrics["topK_precision@2"] == 0.25
    assert metrics["topK_recall@2"] == 0.5
    assert torch.isclose(torch.tensor(metrics["MAP@2"]), torch.tensor(0.375))
    assert torch.isclose(
        torch.tensor(metrics["NDCG@2"]),
        torch.tensor((1.0 + 1.0 / torch.log2(torch.tensor(3.0)).item()) / 4.0),
    )
    assert metrics["extra_overlap_gain@2"] == 1
    assert metrics["extra_recall_gain@2"] == 0.25
    assert metrics["marginal_precision@2"] == 0.25

    assert metrics["overlap_count@5"] == 4
    assert metrics["topK_recall@5"] == 1.0
    assert metrics["extra_overlap_gain@5"] == 3
    assert metrics["extra_recall_gain@5"] == 0.75
    assert metrics["marginal_precision@5"] == 0.1875
    assert metrics["mean_true_rank"] == 2.75
    assert metrics["median_true_rank"] == 2.5
    assert torch.isclose(torch.tensor(metrics["p90_true_rank"]), torch.tensor(4.4))
    assert torch.isclose(torch.tensor(metrics["p99_true_rank"]), torch.tensor(4.94))


def test_prefetch_metrics_ignore_padding_token_hits_and_misses() -> None:
    pred_logits = torch.tensor(
        [[[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [5.0, 0.0, 0.0]]]],
        dtype=torch.float32,
    )
    expert_selection = torch.tensor([[[[0], [2], [1]]]], dtype=torch.long)
    router_probs = torch.softmax(torch.zeros_like(pred_logits), dim=-1)
    attention_mask = torch.tensor([[1, 0, 0]], dtype=torch.long)

    metrics = compute_prefetch_metrics(
        pred_logits,
        expert_selection,
        router_probs,
        attention_mask,
        budgets=(1, 2),
    )

    assert metrics["num_valid_token_layer_items"] == 1
    assert metrics["overlap_count@1"] == 1
    assert metrics["topK_recall@1"] == 1.0
    assert metrics["overlap_count@2"] == 1
    assert metrics["topK_recall@2"] == 1.0
    assert metrics["extra_overlap_gain@2"] == 0



def test_prefetch_metrics_tied_logits_use_sorted_expert_membership() -> None:
    pred_logits = torch.zeros(1, 1, 1, 4, dtype=torch.float32)
    ranked_experts = torch.argsort(pred_logits[0, 0, 0], descending=True)
    true_expert = int(ranked_experts[1].item())
    expert_selection = torch.tensor([[[[true_expert]]]], dtype=torch.long)
    router_probs = torch.softmax(torch.zeros_like(pred_logits), dim=-1)
    attention_mask = torch.tensor([[1]], dtype=torch.long)

    metrics = compute_prefetch_metrics(
        pred_logits,
        expert_selection,
        router_probs,
        attention_mask,
        budgets=(1, 2),
    )

    assert metrics["overlap_count@1"] == 0
    assert metrics["topK_recall@1"] == 0.0
    assert metrics["overlap_count@2"] == 1
    assert metrics["topK_recall@2"] == 1.0
    assert metrics["mean_true_rank"] == 2.0


def test_prefetch_metrics_budget_larger_than_experts_keeps_original_precision_denominator() -> None:
    pred_logits = torch.tensor([[[[1.0, 0.0, 5.0]]]], dtype=torch.float32)
    expert_selection = torch.tensor([[[[2]]]], dtype=torch.long)
    router_probs = torch.softmax(torch.zeros_like(pred_logits), dim=-1)
    attention_mask = torch.tensor([[1]], dtype=torch.long)

    metrics = compute_prefetch_metrics(
        pred_logits,
        expert_selection,
        router_probs,
        attention_mask,
        budgets=(1, 5),
    )

    assert metrics["overlap_count@5"] == 1
    assert metrics["topK_precision@5"] == 1 / 5
    assert metrics["topK_recall@5"] == 1.0
    assert metrics["extra_overlap_gain@5"] == 0
    assert metrics["extra_recall_gain@5"] == 0.0
    assert metrics["marginal_precision@5"] == 0.0



def test_objective_router_loss_supports_sida_kd_and_hard_ce() -> None:
    pred_logits = torch.tensor([[[[3.0, 1.0, 0.0], [0.0, 3.0, 1.0]]]], dtype=torch.float32)
    router_logits = torch.tensor([[[[4.0, 1.0, 0.0], [0.0, 4.0, 1.0]]]], dtype=torch.float32)
    router_probs = torch.softmax(router_logits, dim=-1)
    expert_selection = torch.tensor([[[[0], [1]]]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 0]], dtype=torch.long)

    hard_loss, hard_parts = objective_router_loss(
        pred_logits,
        router_probs,
        expert_selection,
        attention_mask,
        objective="hard-ce",
    )
    sida_loss, sida_parts = objective_router_loss(
        pred_logits,
        router_probs,
        expert_selection,
        attention_mask,
        objective="sida-kd",
        router_logits=router_logits,
        sida_top_k=2,
    )

    assert hard_loss.item() > 0.0
    assert torch.equal(hard_loss, hard_parts["ce"])
    assert sida_loss.item() > 0.0
    assert sida_parts["aux"].item() >= 0.0
    assert torch.isclose(sida_loss, 0.005 * sida_parts["ce"] + sida_parts["aux"])


def test_objective_router_loss_supports_src_probability_regression() -> None:
    pred_logits = torch.tensor([[[[3.0, 1.0, 0.0]]]], dtype=torch.float32)
    router_probs = torch.softmax(torch.tensor([[[[4.0, 1.0, 0.0]]]], dtype=torch.float32), dim=-1)
    expert_selection = torch.tensor([[[[0]]]], dtype=torch.long)
    attention_mask = torch.tensor([[1]], dtype=torch.long)

    l1_loss, l1_parts = objective_router_loss(
        pred_logits,
        router_probs,
        expert_selection,
        attention_mask,
        objective="l1-prob",
    )
    smooth_loss, smooth_parts = objective_router_loss(
        pred_logits,
        router_probs,
        expert_selection,
        attention_mask,
        objective="smoothl1-prob",
    )

    assert torch.equal(l1_loss, l1_parts["aux"])
    assert torch.equal(smooth_loss, smooth_parts["aux"])
    assert l1_loss.item() >= smooth_loss.item()


def test_expert_frequency_targets_and_sample_objective() -> None:
    expert_selection = torch.tensor([[[[0], [1], [1]], [[2], [2], [1]]]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.long)

    target = expert_frequency_targets(expert_selection, attention_mask, num_experts=3)

    assert torch.equal(target[0, 0], torch.tensor([0.5, 0.5, 0.0]))
    assert torch.equal(target[0, 1], torch.tensor([0.0, 0.0, 1.0]))

    pred = target.clone() + 0.1
    loss, parts = objective_router_loss(
        pred,
        torch.zeros(1, 2, 3, 3),
        expert_selection,
        attention_mask,
        objective="sample-freq-l1",
    )
    assert torch.equal(loss, parts["aux"])
    assert loss.item() > 0.0


def test_build_src_simplenn_sample_returns_sample_layer_scores() -> None:
    model = build_model(
        "src-simplenn-sample",
        input_dim=4,
        hidden_dim=8,
        num_layers=1,
        num_experts=3,
        num_router_layers=2,
        src_layers=1,
        dropout=0.0,
        tokens=3,
    )

    y = model(torch.randn(2, 3, 4))

    assert y.shape == (2, 2, 3)

def test_build_src_simplenn_token_returns_token_level_logits() -> None:
    batch, tokens, hidden = 1, 3, 4
    layers, experts = 2, 5
    model = build_model(
        "src-simplenn-token",
        input_dim=hidden,
        hidden_dim=8,
        num_layers=2,
        num_experts=experts,
        num_router_layers=layers,
        src_layers=1,
        dropout=0.0,
    )

    class FakeTokenModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            rows = torch.arange(x.shape[0], dtype=torch.float32, device=x.device)
            layer_ids = torch.arange(layers, dtype=torch.float32, device=x.device)
            expert_ids = torch.arange(experts, dtype=torch.float32, device=x.device)
            return rows[:, None, None] * 100 + layer_ids[None, :, None] * 10 + expert_ids[None, None, :]

    model.model = FakeTokenModel()

    y = model(torch.randn(batch, tokens, hidden))

    assert y.shape == (batch, layers, tokens, experts)
    for token in range(tokens):
        for layer in range(layers):
            for expert in range(experts):
                assert y[0, layer, token, expert].item() == token * 100 + layer * 10 + expert


def test_build_sida_gru_returns_erpp_layout() -> None:
    model = build_model(
        "sida-gru-sa",
        input_dim=4,
        hidden_dim=8,
        num_layers=1,
        num_experts=3,
        num_router_layers=2,
        src_layers=1,
        dropout=0.0,
    )

    y = model(torch.randn(2, 3, 4))

    assert y.shape == (2, 2, 3, 3)


def test_build_sida_lstm_returns_erpp_layout() -> None:
    model = build_model(
        "sida-lstm-sa",
        input_dim=4,
        hidden_dim=8,
        num_layers=1,
        num_experts=3,
        num_router_layers=2,
        src_layers=1,
        dropout=0.0,
    )

    y = model(torch.randn(2, 3, 4))

    assert y.shape == (2, 2, 3, 3)



def test_parse_args_defaults_match_erpp_spec(tmp_path: Path) -> None:
    trace_dir = write_tiny_trace(tmp_path)

    args = parse_args(["--trace-dir", str(trace_dir), "--model", "src-simplenn-token"])

    assert args.output_root == Path("performance_predictor/encoder/ERPP/implement/model")
    assert args.epochs == 5
    assert args.batch_size == 2
    assert args.lr == 1e-4
    assert args.recurrent_layers == 2
    assert args.hidden_dim is None
    assert args.seed == 0


def test_src_simplenn_default_hidden_dim_resolves_to_1024(tmp_path: Path) -> None:
    trace_dir = write_tiny_trace(tmp_path)
    output_root = tmp_path / "default-hidden"

    main(
        [
            "--trace-dir",
            str(trace_dir),
            "--model",
            "src-simplenn-token",
            "--output-root",
            str(output_root),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--src-layers",
            "1",
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

    config = json.loads((output_root / "src-simplenn-token-erpp-kl-ce" / "config.json").read_text(encoding="utf-8"))
    assert config["hidden_dim"] == 1024
    assert config["hidden_dim_arg"] is None


def test_evaluate_reports_global_rank_quantiles_across_batches() -> None:
    from torch.utils.data import DataLoader

    from performance_predictor.encoder.ERPP.implement.train.baseline.train_baseline import evaluate

    class FixedRankModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            logits = torch.zeros(x.shape[0], 1, 1, 5, dtype=torch.float32, device=x.device)
            if x[0, 0, 0].item() == 0.0:
                logits[0, 0, 0] = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0], device=x.device)
            else:
                logits[0, 0, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=x.device)
            return logits

    items = [
        {
            "layer0_attn_out": torch.tensor([[0.0]]),
            "attention_mask": torch.tensor([1]),
            "router_probs": torch.softmax(torch.zeros(1, 1, 5), dim=-1),
            "expert_selection": torch.tensor([[[0]]]),
        },
        {
            "layer0_attn_out": torch.tensor([[1.0]]),
            "attention_mask": torch.tensor([1]),
            "router_probs": torch.softmax(torch.zeros(1, 1, 5), dim=-1),
            "expert_selection": torch.tensor([[[0]]]),
        },
    ]
    loader = DataLoader(items, batch_size=1, shuffle=False, collate_fn=collate_trace_batch)

    metrics = evaluate(
        FixedRankModel(),
        loader,
        torch.device("cpu"),
        objective="erpp-kl-ce",
        alpha_kl=1.0,
        alpha_ce=1.0,
        sida_top_k=30,
        budgets=(1, 5),
    )

    assert metrics["mean_true_rank"] == 3.0
    assert metrics["median_true_rank"] == 3.0
    assert torch.isclose(torch.tensor(metrics["p90_true_rank"]), torch.tensor(4.6))
    assert torch.isclose(torch.tensor(metrics["p99_true_rank"]), torch.tensor(4.96))

def test_training_smoke_run_writes_model_artifacts(tmp_path: Path) -> None:
    trace_dir = write_tiny_trace(tmp_path)
    output_root = tmp_path / "outputs"

    main(
        [
            "--trace-dir",
            str(trace_dir),
            "--model",
            "src-simplenn-token",
            "--output-root",
            str(output_root),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--hidden-dim",
            "8",
            "--src-layers",
            "1",
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

    model_dir = output_root / "src-simplenn-token-erpp-kl-ce"
    for artifact_name in (
        "config.json",
        "metrics.json",
        "model.pt",
        "README.md",
        "train_log.jsonl",
    ):
        assert (model_dir / artifact_name).exists()

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    assert config["model"] == "src-simplenn-token"
    assert config["epochs"] == 1
    assert config["batch_size"] == 1
    assert config["seed"] == 0

    metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "validation_loss" in metrics
    assert "topK_recall@1" in metrics
    assert "NDCG@1" in metrics

    log_lines = (model_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(log_lines) >= 1
    first_log = json.loads(log_lines[0])
    assert "train_loss" in first_log
    assert "validation_loss" in first_log

    checkpoint = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    assert "model_state_dict" in checkpoint
    assert "config" in checkpoint
    assert "metrics" in checkpoint



def test_training_smoke_run_writes_objective_specific_artifacts(tmp_path: Path) -> None:
    trace_dir = write_tiny_trace(tmp_path)
    output_root = tmp_path / "objective-outputs"

    main(
        [
            "--trace-dir",
            str(trace_dir),
            "--model",
            "src-simplenn-sample",
            "--objective",
            "sample-freq-l1",
            "--output-root",
            str(output_root),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--hidden-dim",
            "8",
            "--src-layers",
            "1",
            "--dropout",
            "0",
            "--device",
            "cpu",
            "--max-train-batches",
            "1",
            "--max-eval-batches",
            "1",
            "--early-stop",
            "--early-stop-window",
            "1",
        ]
    )

    model_dir = output_root / "src-simplenn-sample-sample-freq-l1"
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    assert config["model"] == "src-simplenn-sample"
    assert config["objective"] == "sample-freq-l1"
    assert config["early_stop"] is True
    assert "validation_aux" in metrics
    assert "topK_recall@1" in metrics
