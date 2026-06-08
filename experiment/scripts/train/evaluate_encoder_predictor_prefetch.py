#!/usr/bin/env python3
"""Evaluate encoder predictor prefetch curves and oracle-count accuracy."""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


def find_repo_root(start: Path | None = None) -> Path:
    current = Path.cwd() if start is None else Path(start).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "experiment").is_dir() and (candidate / "src").is_dir():
            return candidate
    return Path(__file__).resolve().parents[3]


REPO_ROOT = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.scripts.train.encoder_predictor_sida_gru_sa_hard_ce import (
    DEFAULT_TRACE_DIR,
    EncoderPredictorTraceDataset,
    SidaGRUSparseAttentionPredictor,
    collate_trace_batch,
    resolve_device,
)
from experiment.scripts.train.encoder_predictor_src_simplenn_token_hard_ce import SrcSimpleNNTokenPredictor
from experiment.scripts.train.encoder_predictor_taxonomy import (
    SRC_SIMPLENN_BLTE_RUN_NAME,
    ble_manifest,
    blte_artifact_dir,
    default_ble_report_dir,
)


DEFAULT_MODEL_DIR = blte_artifact_dir(SRC_SIMPLENN_BLTE_RUN_NAME)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def source_run_name(model_dir: Path, config: dict[str, Any], label: str) -> str:
    manifest_path = model_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        if manifest.get("run_name"):
            return str(manifest["run_name"])
    if config.get("output_name"):
        return str(config["output_name"])
    return label


def write_ble_artifact_if_taxonomy_blte(
    *,
    model_dir: Path,
    config: dict[str, Any],
    label: str,
    checkpoint_path: Path,
    trace_dir: Path,
    report_dir_path: Path,
) -> Path | None:
    if model_dir.parent.name != "blte":
        return None
    run_name = source_run_name(model_dir, config, label)
    output_dir = model_dir.parent.parent / "ble" / f"noisyor-from-{run_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = ble_manifest(
        source_blte_artifact=model_dir,
        source_checkpoint=checkpoint_path.name,
        source_run_name=run_name,
        output_dir=output_dir,
        trace_dir=trace_dir,
        report_dir_path=report_dir_path,
    )
    (output_dir / "ble_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    readme_lines = [
        "# BLE Noisy-Or View",
        "",
        f"- source_blte_artifact: {model_dir}",
        f"- source_checkpoint: {checkpoint_path.name}",
        "- output_layout: BLE",
        "- aggregation: noisy_or",
        "- budget_source: sum_ble_score",
        "- budget_rounding: ceil",
        "- selection_rule: topk_by_ble_score",
        f"- report_dir: {report_dir_path}",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    return output_dir


def sample_targets_from_expert_selection(
    expert_selection: torch.Tensor,
    attention_mask: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if expert_selection.ndim != 4 or expert_selection.shape[-1] < 1:
        raise ValueError("expert_selection must have shape [B,L,T,K>=1]")
    if attention_mask.shape != (expert_selection.shape[0], expert_selection.shape[2]):
        raise ValueError("attention_mask must have shape [B,T]")
    batch, layers, tokens, _ = expert_selection.shape
    selected = expert_selection[..., 0].long()
    valid = attention_mask.to(device=expert_selection.device).eq(1)
    expert_set = torch.zeros(batch, layers, num_experts, dtype=torch.bool, device=expert_selection.device)
    for expert_id in range(num_experts):
        expert_set[..., expert_id] = ((selected == expert_id) & valid[:, None, :]).any(dim=2)
    true_count = expert_set.sum(dim=-1).long()
    return expert_set, true_count


def token_scores_from_logits(
    logits: torch.Tensor,
    attention_mask: torch.Tensor,
    aggregator: str = "noisy_or",
) -> torch.Tensor:
    if logits.ndim != 4:
        raise ValueError("logits must have shape [B,L,T,E]")
    if attention_mask.shape != (logits.shape[0], logits.shape[2]):
        raise ValueError("attention_mask must have shape [B,T]")
    valid = attention_mask.to(device=logits.device).eq(1)[:, None, :, None]
    probs = torch.softmax(logits.float(), dim=-1).masked_fill(~valid, 0.0)
    if aggregator == "noisy_or":
        log_no_hit = torch.log1p(-probs.clamp(max=1.0 - 1e-6)).sum(dim=2)
        return 1.0 - torch.exp(log_no_hit)
    if aggregator == "sum_prob":
        return probs.sum(dim=2)
    if aggregator == "max_prob":
        return probs.amax(dim=2)
    raise ValueError(f"unknown aggregator: {aggregator}")


def curve_from_scores(scores: torch.Tensor, true_set: torch.Tensor) -> list[dict[str, float | int]]:
    if scores.shape != true_set.shape:
        raise ValueError("scores and true_set must both have shape [N,L,E]")
    true_bool = true_set.to(dtype=torch.bool, device=scores.device)
    ranked = torch.argsort(scores, dim=-1, descending=True)
    hits_ranked = torch.gather(true_bool, dim=-1, index=ranked).to(dtype=torch.float32)
    cumulative_hits = hits_ranked.cumsum(dim=-1)
    true_count = true_bool.sum(dim=-1).to(dtype=torch.float32).clamp(min=1.0)
    num_items = int(scores.shape[0] * scores.shape[1])
    rows: list[dict[str, float | int]] = []
    for budget_index in range(scores.shape[-1]):
        budget = budget_index + 1
        overlap = cumulative_hits[..., budget_index]
        rows.append(
            {
                "budget": budget,
                "precision": float((overlap / float(budget)).mean().item()),
                "recall": float((overlap / true_count).mean().item()),
                "mean_overlap": float(overlap.mean().item()),
                "total_overlap": int(overlap.sum().item()),
                "num_sample_layer_items": num_items,
            }
        )
    return rows


def layer_curve_from_scores(scores: torch.Tensor, true_set: torch.Tensor) -> list[dict[str, float | int]]:
    if scores.shape != true_set.shape:
        raise ValueError("scores and true_set must both have shape [N,L,E]")
    rows: list[dict[str, float | int]] = []
    for layer in range(scores.shape[1]):
        layer_rows = curve_from_scores(scores[:, layer : layer + 1, :], true_set[:, layer : layer + 1, :])
        for row in layer_rows:
            rows.append({"layer": layer, **row})
    return rows


def noisy_or_budget_gap_metrics(scores: torch.Tensor, true_count: torch.Tensor) -> list[dict[str, float | int]]:
    if true_count.shape != scores.shape[:2]:
        raise ValueError("true_count must have shape [N,L]")
    budget_float = scores.sum(dim=-1)
    true_float = true_count.to(device=scores.device, dtype=torch.float32)
    gap = budget_float - true_float
    return [
        {
            "noisy_or_budget_mean": float(budget_float.mean().item()),
            "true_count_mean": float(true_float.mean().item()),
            "noisy_or_budget_minus_true_mean": float(gap.mean().item()),
            "noisy_or_budget_abs_gap_mean": float(gap.abs().mean().item()),
            "noisy_or_budget_rmse": float(torch.sqrt((gap ** 2).mean()).item()),
            "under_budget_rate": float((gap < 0).to(dtype=torch.float32).mean().item()),
            "exact_budget_rate": float((gap == 0).to(dtype=torch.float32).mean().item()),
            "over_budget_rate": float((gap > 0).to(dtype=torch.float32).mean().item()),
            "num_sample_layer_items": int(true_count.numel()),
        }
    ]


def noisy_or_budget_gap_by_layer(scores: torch.Tensor, true_count: torch.Tensor) -> list[dict[str, float | int]]:
    if true_count.shape != scores.shape[:2]:
        raise ValueError("true_count must have shape [N,L]")
    rows: list[dict[str, float | int]] = []
    for layer in range(scores.shape[1]):
        row = noisy_or_budget_gap_metrics(scores[:, layer : layer + 1, :], true_count[:, layer : layer + 1])[0]
        rows.append({"layer": layer, **row})
    return rows


def _dynamic_budget_prediction(scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    budget_float = scores.sum(dim=-1)
    budget_int = torch.ceil(budget_float).long().clamp(min=1, max=scores.shape[-1])
    return budget_float, budget_int


def dynamic_budget_prefetch_metrics(
    scores: torch.Tensor,
    true_set: torch.Tensor,
    true_count: torch.Tensor,
) -> list[dict[str, float | int]]:
    if scores.shape != true_set.shape:
        raise ValueError("scores and true_set must both have shape [N,L,E]")
    if true_count.shape != scores.shape[:2]:
        raise ValueError("true_count must have shape [N,L]")
    budget_float, budget_int = _dynamic_budget_prediction(scores)
    true_bool = true_set.to(dtype=torch.bool, device=scores.device)
    pred_bool = torch.zeros_like(true_bool)
    for budget in torch.unique(budget_int):
        budget_value = int(budget.item())
        mask = budget_int.eq(budget_value)
        top_idx = torch.topk(scores[mask], k=budget_value, dim=-1).indices
        pred_bool[mask] = pred_bool[mask].scatter(-1, top_idx, True)
    overlap = (pred_bool & true_bool).sum(dim=-1).to(dtype=torch.float32)
    budget_int_float = budget_int.to(dtype=torch.float32)
    true_float = true_count.to(device=scores.device, dtype=torch.float32).clamp(min=1.0)
    overlap_total = overlap.sum()
    total_budget = budget_int_float.sum().clamp(min=1.0)
    total_true = true_float.sum().clamp(min=1.0)
    micro_precision = float((overlap_total / total_budget).item())
    micro_recall = float((overlap_total / total_true).item())
    return [
        {
            "pred_budget_mean": float(budget_float.mean().item()),
            "pred_budget_int_mean": float(budget_int_float.mean().item()),
            "true_count_mean": float(true_count.to(dtype=torch.float32).mean().item()),
            "mean_overlap": float(overlap.mean().item()),
            "total_overlap": int(overlap.sum().item()),
            "total_budget": int(budget_int.sum().item()),
            "total_true": int(true_count.sum().item()),
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "macro_precision": float((overlap / budget_int_float.clamp(min=1.0)).mean().item()),
            "macro_recall": float((overlap / true_float).mean().item()),
            "dynamic_budget_precision": micro_precision,
            "dynamic_budget_recall": micro_recall,
            "budget_abs_gap_mean": float((budget_float - true_count.to(device=scores.device, dtype=torch.float32)).abs().mean().item()),
            "num_sample_layer_items": int(true_count.numel()),
        }
    ]


def dynamic_budget_prefetch_by_layer(
    scores: torch.Tensor,
    true_set: torch.Tensor,
    true_count: torch.Tensor,
) -> list[dict[str, float | int]]:
    if scores.shape != true_set.shape:
        raise ValueError("scores and true_set must both have shape [N,L,E]")
    if true_count.shape != scores.shape[:2]:
        raise ValueError("true_count must have shape [N,L]")
    rows: list[dict[str, float | int]] = []
    for layer in range(scores.shape[1]):
        row = dynamic_budget_prefetch_metrics(
            scores[:, layer : layer + 1, :],
            true_set[:, layer : layer + 1, :],
            true_count[:, layer : layer + 1],
        )[0]
        rows.append({"layer": layer, **row})
    return rows


def oracle_count_metrics_from_scores(
    scores: torch.Tensor,
    true_set: torch.Tensor,
    true_count: torch.Tensor,
) -> dict[str, float | int]:
    if scores.shape != true_set.shape:
        raise ValueError("scores and true_set must both have shape [N,L,E]")
    if true_count.shape != scores.shape[:2]:
        raise ValueError("true_count must have shape [N,L]")
    true_bool = true_set.to(dtype=torch.bool, device=scores.device)
    budgets = true_count.to(device=scores.device, dtype=torch.long).clamp(min=0, max=scores.shape[-1])
    pred_bool = torch.zeros_like(true_bool)
    for budget in torch.unique(budgets):
        budget_int = int(budget.item())
        if budget_int <= 0:
            continue
        mask = budgets.eq(budget_int)
        top_idx = torch.topk(scores[mask], k=budget_int, dim=-1).indices
        pred_bool[mask] = pred_bool[mask].scatter(-1, top_idx, True)
    overlap = (pred_bool & true_bool).sum(dim=-1).to(dtype=torch.float32)
    waste = (pred_bool & ~true_bool).sum(dim=-1).to(dtype=torch.float32)
    budget_float = budgets.to(dtype=torch.float32)
    true_float = true_count.to(device=scores.device, dtype=torch.float32).clamp(min=0.0)
    overlap_total = overlap.sum()
    total_budget = budget_float.sum()
    total_true = true_float.sum()
    return {
        "oracle_count_overlap_count": int(overlap_total.item()),
        "oracle_count_total_budget": int(total_budget.item()),
        "oracle_count_total_true": int(total_true.item()),
        "oracle_count_accuracy": float((overlap_total / total_budget.clamp(min=1.0)).item()),
        "oracle_count_micro_precision": float((overlap_total / total_budget.clamp(min=1.0)).item()),
        "oracle_count_micro_recall": float((overlap_total / total_true.clamp(min=1.0)).item()),
        "oracle_count_macro_precision": float((overlap / budget_float.clamp(min=1.0)).mean().item()),
        "oracle_count_macro_recall": float((overlap / true_float.clamp(min=1.0)).mean().item()),
        "oracle_count_waste": float((waste.sum() / total_budget.clamp(min=1.0)).item()),
        "oracle_count_budget_mean": float(budget_float.mean().item()),
        "oracle_count_num_sample_layer_items": int(scores.shape[0] * scores.shape[1]),
    }


def budget_gap_curve(true_count: torch.Tensor, max_budget: int) -> list[dict[str, float | int]]:
    true_float = true_count.to(dtype=torch.float32)
    rows: list[dict[str, float | int]] = []
    for budget in range(1, max_budget + 1):
        gap = torch.full_like(true_float, float(budget)) - true_float
        rows.append(
            {
                "budget": budget,
                "true_count_mean": float(true_float.mean().item()),
                "budget_minus_true_mean": float(gap.mean().item()),
                "budget_abs_gap_mean": float(gap.abs().mean().item()),
                "under_budget_rate": float((gap < 0).to(dtype=torch.float32).mean().item()),
                "exact_budget_rate": float((gap == 0).to(dtype=torch.float32).mean().item()),
                "over_budget_rate": float((gap > 0).to(dtype=torch.float32).mean().item()),
                "num_sample_layer_items": int(true_count.numel()),
            }
        )
    return rows


def build_model_from_config(config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    metadata = config["metadata"]
    model_name = config.get("model")
    if model_name == "sida-gru-sa":
        model = SidaGRUSparseAttentionPredictor(
            input_dim=int(metadata["hidden_size"]),
            hidden_dim=int(config["hidden_dim"]),
            num_router_layers=int(metadata["num_encoder_moe_layers"]),
            num_experts=int(metadata["num_experts"]),
            recurrent_layers=int(config.get("recurrent_layers", 2)),
        )
        return model.to(device)
    if model_name == "src-simplenn-token":
        model = SrcSimpleNNTokenPredictor(
            input_dim=int(metadata["hidden_size"]),
            hidden_dim=int(config["hidden_dim"]),
            num_router_layers=int(metadata["num_encoder_moe_layers"]),
            num_experts=int(metadata["num_experts"]),
            src_layers=int(config.get("src_layers", 1)),
            dropout=float(config.get("dropout", 0.5)),
        )
        return model.to(device)
    raise ValueError(f"unknown encoder predictor model: {model_name}")


@torch.no_grad()
def evaluate_model(
    model_dir: Path,
    trace_dir_override: Path | None,
    split: str,
    device: torch.device,
    aggregator: str,
    max_batches: int | None,
    label: str,
    report_dir_path: Path,
) -> tuple[
    list[dict[str, float | int]],
    list[dict[str, float | int]],
    dict[str, float | int],
    list[dict[str, float | int]],
    list[dict[str, float | int]],
    list[dict[str, float | int]],
    list[dict[str, float | int]],
    list[dict[str, float | int]],
]:
    config = load_json(model_dir / "config.json")
    trace_dir = trace_dir_override or Path(config["trace_dir"])
    dataset = EncoderPredictorTraceDataset(trace_dir, split)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_trace_batch)
    model = build_model_from_config(config, device)
    checkpoint_path = model_dir / ("best_model.pt" if (model_dir / "best_model.pt").exists() else "model.pt")
    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    write_ble_artifact_if_taxonomy_blte(
        model_dir=model_dir,
        config=config,
        label=label,
        checkpoint_path=checkpoint_path,
        trace_dir=trace_dir,
        report_dir_path=report_dir_path,
    )

    score_parts: list[torch.Tensor] = []
    noisy_or_score_parts: list[torch.Tensor] = []
    true_parts: list[torch.Tensor] = []
    true_count_parts: list[torch.Tensor] = []
    num_experts = int(config["metadata"]["num_experts"])
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        valid_len = int(batch["attention_mask"][0].eq(1).sum().item())
        if valid_len == 0:
            continue
        x = batch["layer0_attn_out"][:, :valid_len].to(device).float()
        expert_selection = batch["expert_selection"][:, :, :valid_len].to(device)
        attention_mask = torch.ones(1, valid_len, dtype=torch.long, device=device)
        logits = model(x)
        scores = token_scores_from_logits(logits, attention_mask, aggregator=aggregator)
        noisy_or_scores = token_scores_from_logits(logits, attention_mask, aggregator="noisy_or")
        true_set, true_count = sample_targets_from_expert_selection(expert_selection, attention_mask, num_experts)
        score_parts.append(scores.cpu())
        noisy_or_score_parts.append(noisy_or_scores.cpu())
        true_parts.append(true_set.cpu())
        true_count_parts.append(true_count.cpu())
    if not score_parts:
        raise RuntimeError("no non-empty samples evaluated")
    scores_all = torch.cat(score_parts, dim=0)
    noisy_or_scores_all = torch.cat(noisy_or_score_parts, dim=0)
    true_all = torch.cat(true_parts, dim=0)
    true_count_all = torch.cat(true_count_parts, dim=0)
    return (
        curve_from_scores(scores_all, true_all),
        layer_curve_from_scores(scores_all, true_all),
        oracle_count_metrics_from_scores(noisy_or_scores_all, true_all, true_count_all),
        budget_gap_curve(true_count_all, max_budget=scores_all.shape[-1]),
        noisy_or_budget_gap_metrics(noisy_or_scores_all, true_count_all),
        noisy_or_budget_gap_by_layer(noisy_or_scores_all, true_count_all),
        dynamic_budget_prefetch_metrics(noisy_or_scores_all, true_all, true_count_all),
        dynamic_budget_prefetch_by_layer(noisy_or_scores_all, true_all, true_count_all),
    )


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_oracle_csv(path: Path, metrics: dict[str, float | int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def write_multi_csv(path: Path, payload: dict[str, list[dict[str, float | int]]]) -> None:
    first_rows = next(iter(payload.values()))
    fieldnames = ["model", *list(first_rows[0].keys())]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for label, rows in payload.items():
            for row in rows:
                writer.writerow({"model": label, **row})


def write_multi_oracle_csv(path: Path, payload: dict[str, dict[str, float | int]]) -> None:
    first_metrics = next(iter(payload.values()))
    fieldnames = ["model", *list(first_metrics.keys())]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for label, metrics in payload.items():
            writer.writerow({"model": label, **metrics})


def write_plots(output_dir: Path, curves: list[dict[str, float | int]], gaps: list[dict[str, float | int]], oracle: dict[str, float | int]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        write_svg_plot(output_dir / "precision_curve.svg", curves, "precision", "Validation precision@m")
        write_svg_plot(output_dir / "recall_curve.svg", curves, "recall", "Validation recall@m")
        write_svg_plot(output_dir / "budget_gap_curve.svg", gaps, "budget_minus_true_mean", "Budget gap versus true expert count")
        write_oracle_svg(output_dir / "fixed_vs_oracle_accuracy.svg", oracle)
        return
    budgets = [int(row["budget"]) for row in curves]
    for metric, title in (("precision", "Validation precision@m"), ("recall", "Validation recall@m")):
        plt.figure(figsize=(9, 5.5))
        plt.plot(budgets, [float(row[metric]) for row in curves], linewidth=2)
        plt.xlabel("Prefetch expert budget m")
        plt.ylabel(metric)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}_curve.png", dpi=180)
        plt.close()
    plt.figure(figsize=(9, 5.5))
    plt.plot([int(row["budget"]) for row in gaps], [float(row["budget_minus_true_mean"]) for row in gaps], linewidth=2)
    plt.xlabel("Prefetch expert budget m")
    plt.ylabel("mean(budget - true_count)")
    plt.title("Budget gap versus true expert count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "budget_gap_curve.png", dpi=180)
    plt.close()
    plt.figure(figsize=(6, 4.5))
    plt.bar(["oracle precision", "oracle recall"], [oracle["oracle_count_micro_precision"], oracle["oracle_count_micro_recall"]])
    plt.ylim(0.0, 1.0)
    plt.title("Oracle-count prefetch accuracy")
    plt.tight_layout()
    plt.savefig(output_dir / "fixed_vs_oracle_accuracy.png", dpi=180)
    plt.close()


def write_svg_plot(path: Path, rows: list[dict[str, float | int]], metric: str, title: str) -> None:
    budgets = [int(row["budget"]) for row in rows]
    values = [float(row[metric]) for row in rows]
    if metric in {"precision", "recall"}:
        min_value, max_value = 0.0, 1.0
    else:
        min_value, max_value = min(values), max(values)
        if min_value == max_value:
            min_value -= 1.0
            max_value += 1.0

    def x_pos(budget: int) -> float:
        return 60 + (budget - min(budgets)) / max(max(budgets) - min(budgets), 1) * 700

    def y_pos(value: float) -> float:
        return 430 - (value - min_value) / (max_value - min_value) * 360

    points = " ".join(f"{x_pos(budget):.2f},{y_pos(value):.2f}" for budget, value in zip(budgets, values))
    path.write_text(
        "\n".join(
            [
                '<svg xmlns="http://www.w3.org/2000/svg" width="860" height="500" viewBox="0 0 860 500">',
                '<rect width="100%" height="100%" fill="white"/>',
                f'<text x="430" y="28" text-anchor="middle" font-family="Arial" font-size="18">{html.escape(title)}</text>',
                '<line x1="60" y1="70" x2="60" y2="430" stroke="#333"/>',
                '<line x1="60" y1="430" x2="760" y2="430" stroke="#333"/>',
                f'<text x="54" y="74" text-anchor="end" font-family="Arial" font-size="11">{max_value:.2f}</text>',
                f'<text x="54" y="434" text-anchor="end" font-family="Arial" font-size="11">{min_value:.2f}</text>',
                f'<polyline fill="none" stroke="#1f77b4" stroke-width="2.4" points="{points}"/>',
                "</svg>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_oracle_svg(path: Path, oracle: dict[str, float | int]) -> None:
    precision = max(0.0, min(1.0, float(oracle["oracle_count_micro_precision"])))
    recall = max(0.0, min(1.0, float(oracle["oracle_count_micro_recall"])))
    bars = [("oracle precision", precision, 120), ("oracle recall", recall, 320)]
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="560" height="420" viewBox="0 0 560 420">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="280" y="30" text-anchor="middle" font-family="Arial" font-size="18">Oracle-count prefetch accuracy</text>',
        '<line x1="70" y1="350" x2="500" y2="350" stroke="#333"/>',
        '<line x1="70" y1="70" x2="70" y2="350" stroke="#333"/>',
    ]
    for label, value, x in bars:
        height = value * 260
        y = 350 - height
        lines.append(f'<rect x="{x}" y="{y:.2f}" width="120" height="{height:.2f}" fill="#1f77b4"/>')
        lines.append(f'<text x="{x + 60}" y="375" text-anchor="middle" font-family="Arial" font-size="12">{html.escape(label)}</text>')
        lines.append(f'<text x="{x + 60}" y="{y - 8:.2f}" text-anchor="middle" font-family="Arial" font-size="12">{value:.3f}</text>')
    lines.append('</svg>')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_multi_plots(
    output_dir: Path,
    curves_by_model: dict[str, list[dict[str, float | int]]],
    gaps_by_model: dict[str, list[dict[str, float | int]]],
    oracle_by_model: dict[str, dict[str, float | int]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        first_label = next(iter(curves_by_model))
        write_svg_plot(output_dir / "precision_curve.svg", curves_by_model[first_label], "precision", "Validation precision@m")
        write_svg_plot(output_dir / "recall_curve.svg", curves_by_model[first_label], "recall", "Validation recall@m")
        write_svg_plot(output_dir / "budget_gap_curve.svg", gaps_by_model[first_label], "budget_minus_true_mean", "Budget gap versus true expert count")
        write_oracle_svg(output_dir / "fixed_vs_oracle_accuracy.svg", oracle_by_model[first_label])
        return

    for metric, title in (("precision", "Validation precision@m"), ("recall", "Validation recall@m")):
        plt.figure(figsize=(10, 6))
        for label, rows in curves_by_model.items():
            budgets = [int(row["budget"]) for row in rows]
            values = [float(row[metric]) for row in rows]
            plt.plot(budgets, values, label=label, linewidth=2)
        plt.xlabel("Prefetch expert budget m")
        plt.ylabel(metric)
        plt.title(title)
        plt.xlim(1, max(int(row["budget"]) for row in next(iter(curves_by_model.values()))))
        plt.ylim(0.0, 1.02)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}_curve.png", dpi=180)
        plt.close()

    plt.figure(figsize=(10, 6))
    for label, rows in gaps_by_model.items():
        budgets = [int(row["budget"]) for row in rows]
        values = [float(row["budget_minus_true_mean"]) for row in rows]
        plt.plot(budgets, values, label=label, linewidth=2)
    plt.xlabel("Prefetch expert budget m")
    plt.ylabel("mean(budget - true_count)")
    plt.title("Budget gap versus true expert count")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "budget_gap_curve.png", dpi=180)
    plt.close()

    labels = list(oracle_by_model)
    precision_values = [float(oracle_by_model[label]["oracle_count_micro_precision"]) for label in labels]
    recall_values = [float(oracle_by_model[label]["oracle_count_micro_recall"]) for label in labels]
    positions = torch.arange(len(labels), dtype=torch.float32)
    width = 0.35
    plt.figure(figsize=(max(7, len(labels) * 2.0), 5))
    plt.bar((positions - width / 2).tolist(), precision_values, width=width, label="oracle precision")
    plt.bar((positions + width / 2).tolist(), recall_values, width=width, label="oracle recall")
    plt.xticks(positions.tolist(), labels, rotation=15, ha="right")
    plt.ylim(0.0, 1.0)
    plt.title("Oracle-count prefetch accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fixed_vs_oracle_accuracy.png", dpi=180)
    plt.close()


def write_extra_multi_plots(
    output_dir: Path,
    noisy_gap_by_model: dict[str, list[dict[str, float | int]]],
    noisy_gap_layer_by_model: dict[str, list[dict[str, float | int]]],
    dynamic_prefetch_by_model: dict[str, list[dict[str, float | int]]],
    dynamic_prefetch_layer_by_model: dict[str, list[dict[str, float | int]]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    labels = list(noisy_gap_by_model)
    positions = torch.arange(len(labels), dtype=torch.float32)
    width = 0.35
    mean_gap = [float(noisy_gap_by_model[label][0]["noisy_or_budget_minus_true_mean"]) for label in labels]
    abs_gap = [float(noisy_gap_by_model[label][0]["noisy_or_budget_abs_gap_mean"]) for label in labels]
    plt.figure(figsize=(max(7, len(labels) * 2.0), 5))
    plt.bar((positions - width / 2).tolist(), mean_gap, width=width, label="mean gap")
    plt.bar((positions + width / 2).tolist(), abs_gap, width=width, label="mean abs gap")
    plt.xticks(positions.tolist(), labels, rotation=15, ha="right")
    plt.ylabel("experts")
    plt.title("Noisy-or budget gap")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "noisy_or_count_gap.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    for label, rows in noisy_gap_layer_by_model.items():
        plt.plot(
            [int(row["layer"]) for row in rows],
            [float(row["noisy_or_budget_minus_true_mean"]) for row in rows],
            label=label,
            linewidth=2,
        )
    plt.xlabel("router layer")
    plt.ylabel("mean(noisy_or_budget - true_count)")
    plt.title("Noisy-or budget gap by layer")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "noisy_or_count_gap_by_layer.png", dpi=180)
    plt.close()

    precision = [float(dynamic_prefetch_by_model[label][0]["micro_precision"]) for label in labels]
    recall = [float(dynamic_prefetch_by_model[label][0]["micro_recall"]) for label in labels]
    plt.figure(figsize=(max(7, len(labels) * 2.0), 5))
    plt.bar((positions - width / 2).tolist(), precision, width=width, label="dynamic precision")
    plt.bar((positions + width / 2).tolist(), recall, width=width, label="dynamic recall")
    plt.xticks(positions.tolist(), labels, rotation=15, ha="right")
    plt.ylim(0.0, 1.0)
    plt.title("Dynamic-budget prefetch accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "dynamic_budget_prefetch_accuracy.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    for label, rows in dynamic_prefetch_layer_by_model.items():
        plt.plot(
            [int(row["layer"]) for row in rows],
            [float(row["dynamic_budget_precision"]) for row in rows],
            label=label,
            linewidth=2,
        )
    plt.xlabel("router layer")
    plt.ylabel("dynamic-budget precision")
    plt.ylim(0.0, 1.0)
    plt.title("Dynamic-budget precision by layer")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "dynamic_budget_prefetch_accuracy_by_layer.png", dpi=180)
    plt.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate encoder predictor prefetch curves.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-dirs", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--trace-dir", type=Path, help="Override config trace_dir and evaluate all models on the same trace.")
    parser.add_argument("--output-dir", type=Path, default=default_ble_report_dir())
    parser.add_argument("--split", default="validation")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--aggregator", default="noisy_or", choices=("noisy_or", "sum_prob", "max_prob"))
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model_dirs = args.model_dirs or [args.model_dir]
    labels = args.labels or [model_dir.name for model_dir in model_dirs]
    if len(labels) != len(model_dirs):
        raise ValueError("--labels must have the same length as --model-dirs")

    curves_by_model: dict[str, list[dict[str, float | int]]] = {}
    layer_curves_by_model: dict[str, list[dict[str, float | int]]] = {}
    oracle_by_model: dict[str, dict[str, float | int]] = {}
    gaps_by_model: dict[str, list[dict[str, float | int]]] = {}
    noisy_gap_by_model: dict[str, list[dict[str, float | int]]] = {}
    noisy_gap_layer_by_model: dict[str, list[dict[str, float | int]]] = {}
    dynamic_prefetch_by_model: dict[str, list[dict[str, float | int]]] = {}
    dynamic_prefetch_layer_by_model: dict[str, list[dict[str, float | int]]] = {}
    for label, model_dir in zip(labels, model_dirs):
        (
            curves,
            layer_curves,
            oracle,
            gaps,
            noisy_gap,
            noisy_gap_layer,
            dynamic_prefetch,
            dynamic_prefetch_layer,
        ) = evaluate_model(
            model_dir=model_dir,
            trace_dir_override=args.trace_dir,
            split=args.split,
            device=device,
            aggregator=args.aggregator,
            max_batches=args.max_batches,
            label=label,
            report_dir_path=args.output_dir,
        )
        curves_by_model[label] = curves
        layer_curves_by_model[label] = layer_curves
        oracle_by_model[label] = oracle
        gaps_by_model[label] = gaps
        noisy_gap_by_model[label] = noisy_gap
        noisy_gap_layer_by_model[label] = noisy_gap_layer
        dynamic_prefetch_by_model[label] = dynamic_prefetch
        dynamic_prefetch_layer_by_model[label] = dynamic_prefetch_layer

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "curves.json").write_text(
        json.dumps(
            {
                "curves": curves_by_model,
                "oracle_count_metrics": oracle_by_model,
                "budget_gap": gaps_by_model,
                "layer_curves": layer_curves_by_model,
                "noisy_or_count_gap": noisy_gap_by_model,
                "noisy_or_count_gap_by_layer": noisy_gap_layer_by_model,
                "dynamic_budget_prefetch_accuracy": dynamic_prefetch_by_model,
                "dynamic_budget_prefetch_accuracy_by_layer": dynamic_prefetch_layer_by_model,
                "metadata": {
                    "model_dirs": [str(model_dir) for model_dir in model_dirs],
                    "labels": labels,
                    "trace_dir_override": str(args.trace_dir) if args.trace_dir else None,
                    "split": args.split,
                    "aggregator": args.aggregator,
                    "max_batches": args.max_batches,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_multi_csv(args.output_dir / "curves.csv", curves_by_model)
    write_multi_csv(args.output_dir / "layer_curves.csv", layer_curves_by_model)
    write_multi_csv(args.output_dir / "budget_gap.csv", gaps_by_model)
    write_multi_csv(args.output_dir / "noisy_or_count_gap.csv", noisy_gap_by_model)
    write_multi_csv(args.output_dir / "noisy_or_count_gap_by_layer.csv", noisy_gap_layer_by_model)
    write_multi_csv(args.output_dir / "dynamic_budget_prefetch_accuracy.csv", dynamic_prefetch_by_model)
    write_multi_csv(args.output_dir / "dynamic_budget_prefetch_accuracy_by_layer.csv", dynamic_prefetch_layer_by_model)
    write_multi_oracle_csv(args.output_dir / "oracle_count_metrics.csv", oracle_by_model)
    write_multi_plots(args.output_dir, curves_by_model, gaps_by_model, oracle_by_model)
    write_extra_multi_plots(
        args.output_dir,
        noisy_gap_by_model,
        noisy_gap_layer_by_model,
        dynamic_prefetch_by_model,
        dynamic_prefetch_layer_by_model,
    )
    print(f"Wrote prefetch report to {args.output_dir}")


if __name__ == "__main__":
    main()
