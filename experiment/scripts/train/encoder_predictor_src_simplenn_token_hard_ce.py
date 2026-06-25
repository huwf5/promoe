#!/usr/bin/env python3
"""Train sparse-cache encoder predictor with SRC SimpleNN token hard CE."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[3]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from experiment.scripts.train.encoder_predictor_taxonomy import (
    DEFAULT_DATASET,
    DEFAULT_MODEL_PATH,
    DEFAULT_TASK_NAME,
    SRC_SIMPLENN_BLTE_RUN_NAME,
    TRACE_ID,
    blte_artifact_dir,
    blte_manifest,
    resolve_trace_context,
    src_simplenn_blte_run_name,
    trace_dir_for_model_task,
)


MODEL_NAME = "src-simplenn-token"
OBJECTIVE = "hard-ce"
LOSS_TYPES = ("auto", "hard_ce", "multi_label_bce")
OUTPUT_NAME = SRC_SIMPLENN_BLTE_RUN_NAME
DEFAULT_TRACE_DIR = trace_dir_for_model_task(model_path=DEFAULT_MODEL_PATH, dataset=DEFAULT_DATASET, task_name=DEFAULT_TASK_NAME)
DEFAULT_OUTPUT_DIR = blte_artifact_dir(OUTPUT_NAME)
DEFAULT_BUDGETS = (1, 2, 3, 5, 9, 17)


def find_repo_root(start: Path | None = None) -> Path:
    current = Path.cwd() if start is None else Path(start).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "experiment").is_dir() and (candidate / "src").is_dir():
            return candidate
    return Path(__file__).resolve().parents[3]


REPO_ROOT = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class TraceInfo:
    hidden_size: int
    num_experts: int
    num_selected_experts: int
    num_router_layers: int
    max_input_tokens: int


def _load_tensor(path: Path) -> torch.Tensor:
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _metadata_model_config(metadata: dict[str, Any]) -> dict[str, Any]:
    model_config = metadata.get("model_config")
    if isinstance(model_config, dict):
        return model_config
    return metadata


def parse_trace_metadata(metadata: dict[str, Any], tensors: dict[str, torch.Tensor]) -> TraceInfo:
    model_config = _metadata_model_config(metadata)
    layer0_attn_out = tensors["layer0_attn_out"]
    expert_selection = tensors["expert_selection"]

    if layer0_attn_out.ndim != 3:
        raise ValueError("layer0_attn_out must have shape [S,T,H]")
    if expert_selection.ndim != 4:
        raise ValueError("expert_selection must have shape [S,L,T,K]")

    hidden_size = int(model_config.get("hidden_size", layer0_attn_out.shape[-1]))
    num_experts = int(model_config.get("num_experts", 0))
    if num_experts <= 0:
        raise ValueError("metadata model_config.num_experts is required for hard-ce training")

    num_selected_experts = int(model_config.get("num_selected_experts", expert_selection.shape[-1]))
    router_layers = metadata.get("router_layers")
    layer_to_block = metadata.get("router_layer_to_model_block")
    if isinstance(router_layers, list) and router_layers:
        num_router_layers = len(router_layers)
    elif isinstance(layer_to_block, list) and layer_to_block:
        num_router_layers = len(layer_to_block)
    else:
        num_router_layers = int(model_config.get("num_sparse_encoder_layers", expert_selection.shape[1]))

    info = TraceInfo(
        hidden_size=hidden_size,
        num_experts=num_experts,
        num_selected_experts=num_selected_experts,
        num_router_layers=num_router_layers,
        max_input_tokens=int(layer0_attn_out.shape[1]),
    )
    if info.hidden_size != int(layer0_attn_out.shape[-1]):
        raise ValueError(f"metadata hidden_size={info.hidden_size} does not match layer0_attn_out H={layer0_attn_out.shape[-1]}")
    if info.num_selected_experts != int(expert_selection.shape[-1]):
        raise ValueError(
            f"metadata num_selected_experts={info.num_selected_experts} does not match expert_selection K={expert_selection.shape[-1]}"
        )
    if info.num_router_layers != int(expert_selection.shape[1]):
        raise ValueError(
            f"metadata router layers={info.num_router_layers} does not match expert_selection L={expert_selection.shape[1]}"
        )
    if info.num_selected_experts < 1:
        raise ValueError("expert_selection must include at least one selected expert")
    return info


class EncoderPredictorTraceDataset(Dataset[dict[str, torch.Tensor]]):
    """Tensor-backed sparse-cache encoder predictor trace split."""

    REQUIRED_FILES = (
        "layer0_attn_out.pt",
        "attention_mask.pt",
        "expert_selection.pt",
    )

    def __init__(self, trace_dir: str | Path, split: str) -> None:
        self.trace_dir = Path(trace_dir)
        self.split = split
        self.split_dir = self.trace_dir / split
        metadata_path = self.trace_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"missing trace metadata: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        missing = [name for name in self.REQUIRED_FILES if not (self.split_dir / name).exists()]
        if missing:
            raise FileNotFoundError(f"missing {split} trace files: {missing}")

        self.tensors = {
            "layer0_attn_out": _load_tensor(self.split_dir / "layer0_attn_out.pt"),
            "attention_mask": _load_tensor(self.split_dir / "attention_mask.pt"),
            "expert_selection": _load_tensor(self.split_dir / "expert_selection.pt"),
        }
        optional_files = {
            "expert_weights": self.split_dir / "expert_weights.pt",
            "expert_selection_mask": self.split_dir / "expert_selection_mask.pt",
        }
        for name, optional_path in optional_files.items():
            if optional_path.exists():
                self.tensors[name] = _load_tensor(optional_path)
        self.info = parse_trace_metadata(self.metadata, self.tensors)
        self._validate_shapes()

    def _validate_shapes(self) -> None:
        layer0 = self.tensors["layer0_attn_out"]
        mask = self.tensors["attention_mask"]
        expert_selection = self.tensors["expert_selection"]
        if mask.shape != layer0.shape[:2]:
            raise ValueError("attention_mask must have shape [S,T] matching layer0_attn_out")
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("attention_mask must contain only 0/1 values")
        if expert_selection.shape[:1] + expert_selection.shape[2:3] != mask.shape:
            raise ValueError("expert_selection must align with attention_mask on [S,T]")
        if expert_selection.shape[-1] < 1:
            raise ValueError("expert_selection must have K >= 1")
        if bool((expert_selection < 0).any()) or bool((expert_selection >= self.info.num_experts).any()):
            raise ValueError("expert_selection ids must be in [0, num_experts)")
        expert_weights = self.tensors.get("expert_weights")
        if expert_weights is not None:
            if expert_weights.shape != expert_selection.shape:
                raise ValueError("expert_weights must match expert_selection shape [S,L,T,K]")
            if not torch.is_floating_point(expert_weights):
                self.tensors["expert_weights"] = expert_weights.float()
        expert_selection_mask = self.tensors.get("expert_selection_mask")
        if expert_selection_mask is not None:
            if expert_selection_mask.shape != expert_selection.shape:
                raise ValueError("expert_selection_mask must match expert_selection shape [S,L,T,K]")
            if expert_selection_mask.dtype != torch.bool:
                self.tensors["expert_selection_mask"] = expert_selection_mask.bool()

    def __len__(self) -> int:
        return int(self.tensors["layer0_attn_out"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {name: tensor[index] for name, tensor in self.tensors.items()}


def collate_trace_batch(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in items], dim=0) for key in items[0]}


class SrcSimpleNNBaseline(nn.Module):
    """Feed-forward MLP equivalent to the original src SimpleNN baseline."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 1024,
        output_size: int = 128,
        n_layer: int = 2,
        dropout: float = 0.5,
        output_shape: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        for _ in range(n_layer - 1):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(hidden_size, output_size))
        self.net = nn.Sequential(*layers)
        self.output_shape = output_shape

    def forward_flat(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return self.net(x.reshape(batch_size, -1).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.forward_flat(x)
        if self.output_shape is None:
            return y
        return y.reshape((x.shape[0],) + self.output_shape)


class SrcSimpleNNTokenPredictor(nn.Module):
    """Apply the SRC SimpleNN baseline independently to each token."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_router_layers: int,
        num_experts: int,
        src_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_router_layers = int(num_router_layers)
        self.num_experts = int(num_experts)
        self.model = SrcSimpleNNBaseline(
            input_size=input_dim,
            hidden_size=hidden_dim,
            output_size=self.num_router_layers * self.num_experts,
            n_layer=src_layers,
            dropout=dropout,
            output_shape=(self.num_router_layers, self.num_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, tokens, hidden_dim = x.shape
        token_logits = self.model(x.reshape(batch_size * tokens, hidden_dim))
        return token_logits.reshape(
            batch_size,
            tokens,
            self.num_router_layers,
            self.num_experts,
        ).permute(0, 2, 1, 3).contiguous()


def valid_token_layer_mask(pred_logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if pred_logits.ndim != 4:
        raise ValueError("pred_logits must have shape [B,L,T,E]")
    if attention_mask.shape != (pred_logits.shape[0], pred_logits.shape[2]):
        raise ValueError("attention_mask must have shape [B,T]")
    return attention_mask.to(device=pred_logits.device).eq(1)[:, None, :].expand(pred_logits.shape[:3])


def hard_ce_loss(
    pred_logits: torch.Tensor,
    expert_selection: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    if expert_selection.shape[:3] != pred_logits.shape[:3] or expert_selection.shape[-1] < 1:
        raise ValueError("expert_selection must have shape [B,L,T,K>=1] aligned with pred_logits")
    valid = valid_token_layer_mask(pred_logits, attention_mask)
    valid_items = int(valid.sum().item())
    if valid_items == 0:
        zero = pred_logits.sum() * 0.0
        return zero, {"ce": zero, "valid_items": 0}
    labels = expert_selection.to(device=pred_logits.device)[..., 0].long()
    ce = F.cross_entropy(pred_logits[valid], labels[valid])
    return ce, {"ce": ce, "valid_items": valid_items}


def _validate_optional_expert_tensor(
    name: str,
    tensor: torch.Tensor | None,
    expert_selection: torch.Tensor,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.shape != expert_selection.shape:
        raise ValueError(f"{name} must match expert_selection shape [B,L,T,K]")
    tensor = tensor.to(device=expert_selection.device)
    if name == "expert_weights":
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("expert_weights must contain only finite values")
        if bool((tensor < 0).any()) or bool((tensor > 1).any()):
            raise ValueError("expert_weights must be in [0, 1]")
    return tensor


def _effective_expert_selection_mask(
    expert_selection: torch.Tensor,
    expert_selection_mask: torch.Tensor | None,
) -> torch.Tensor:
    if expert_selection_mask is None:
        mask = torch.zeros_like(expert_selection, dtype=torch.bool)
        mask[..., 0] = True
        return mask
    return expert_selection_mask.to(device=expert_selection.device, dtype=torch.bool)


def _float_weight_token(value: float) -> str:
    text = f"{value:g}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        mantissa = mantissa.replace(".", "p").replace("-", "m")
        exponent = exponent.replace("+", "").replace("-", "m")
        return f"{mantissa}e{exponent}"
    return text.replace(".", "p").replace("-", "m")


def _smooth_l1_or_zero(pred_logits: torch.Tensor, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.numel() == 0:
        return pred_logits.sum() * 0.0
    return F.smooth_l1_loss(pred, target.to(device=pred.device, dtype=pred.dtype))


def token_cardinality_loss_from_target(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    probs = torch.sigmoid(pred_logits.float())
    pred_cardinality = probs.sum(dim=-1)
    true_cardinality = target.sum(dim=-1)
    return _smooth_l1_or_zero(pred_logits, pred_cardinality[valid], true_cardinality[valid])


def layer_noisy_or_count_loss_from_target(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    probs = torch.sigmoid(pred_logits.float()).masked_fill(~valid[..., None], 0.0)
    log_no_hit = torch.log1p(-probs.clamp(max=1.0 - 1e-6)).sum(dim=2)
    pred_count = (1.0 - torch.exp(log_no_hit)).sum(dim=-1)
    true_count = (target.to(dtype=torch.bool) & valid[..., None]).any(dim=2).sum(dim=-1)
    layer_valid = valid.any(dim=-1)
    return _smooth_l1_or_zero(pred_logits, pred_count[layer_valid], true_count[layer_valid])


def soft_gate_auxiliary_loss(
    pred_logits: torch.Tensor,
    expert_selection: torch.Tensor,
    expert_weights: torch.Tensor | None,
    slot_mask: torch.Tensor,
    valid: torch.Tensor,
    aux_type: str = "kl",
) -> torch.Tensor:
    if aux_type not in {"kl", "ce"}:
        raise ValueError(f"unknown soft_gate_aux_type: {aux_type}")
    if expert_weights is None:
        raise ValueError("soft gate auxiliary loss requires expert_weights")
    if expert_selection.shape[-1] < 2:
        return pred_logits.sum() * 0.0
    selected_logits = torch.gather(pred_logits, dim=-1, index=expert_selection)
    selected_logits = selected_logits.masked_fill(~slot_mask, -1.0e9)
    weights = expert_weights.to(device=pred_logits.device, dtype=pred_logits.dtype) * slot_mask.to(dtype=pred_logits.dtype)
    weight_sum = weights.sum(dim=-1, keepdim=True)
    valid_gate = valid & (weight_sum.squeeze(-1) > 0.0)
    if not bool(valid_gate.any()):
        return pred_logits.sum() * 0.0
    true_gate = weights / weight_sum.clamp(min=1e-12)
    log_pred_gate = F.log_softmax(selected_logits.float(), dim=-1).to(dtype=pred_logits.dtype)
    token_ce = -(true_gate * log_pred_gate).sum(dim=-1)
    if aux_type == "ce":
        return token_ce[valid_gate].mean()
    token_kl = token_ce + (true_gate * true_gate.clamp(min=1e-12).log()).sum(dim=-1)
    return token_kl[valid_gate].mean()


def multi_label_bce_loss(
    pred_logits: torch.Tensor,
    expert_selection: torch.Tensor,
    attention_mask: torch.Tensor,
    expert_weights: torch.Tensor | None = None,
    expert_selection_mask: torch.Tensor | None = None,
    use_expert_weights: bool = True,
    token_cardinality_loss_weight: float = 0.0,
    layer_count_loss_weight: float = 0.0,
    soft_gate_aux_weight: float = 0.0,
    soft_gate_aux_type: str = "kl",
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    if expert_selection.shape[:3] != pred_logits.shape[:3] or expert_selection.shape[-1] < 1:
        raise ValueError("expert_selection must have shape [B,L,T,K>=1] aligned with pred_logits")
    expert_selection = expert_selection.to(device=pred_logits.device).long()
    expert_weights = _validate_optional_expert_tensor("expert_weights", expert_weights, expert_selection)
    expert_selection_mask = _validate_optional_expert_tensor("expert_selection_mask", expert_selection_mask, expert_selection)
    if expert_selection_mask is None and expert_selection.shape[-1] > 1:
        if expert_weights is None:
            raise ValueError("multi_label_bce_loss requires expert_selection_mask or expert_weights for K > 1")
        expert_selection_mask = expert_weights > 0
    slot_mask = _effective_expert_selection_mask(expert_selection, expert_selection_mask)
    valid = valid_token_layer_mask(pred_logits, attention_mask) & slot_mask.any(dim=-1)
    valid_items = int(valid.sum().item())
    if valid_items == 0:
        zero = pred_logits.sum() * 0.0
        return zero, {"bce": zero, "valid_items": 0}

    target = torch.zeros_like(pred_logits, dtype=pred_logits.dtype)
    positive_slots = slot_mask.to(dtype=pred_logits.dtype)
    target.scatter_add_(dim=-1, index=expert_selection, src=positive_slots)
    target = target.clamp(min=0.0, max=1.0)

    element_weights = torch.ones_like(pred_logits, dtype=pred_logits.dtype)
    if expert_weights is not None and use_expert_weights:
        selected_weights = torch.zeros_like(pred_logits, dtype=pred_logits.dtype)
        selected_weights.scatter_add_(
            dim=-1,
            index=expert_selection,
            src=expert_weights.to(device=pred_logits.device, dtype=pred_logits.dtype) * positive_slots,
        )
        element_weights = torch.where(target > 0, selected_weights.clamp(min=1e-6), element_weights)

    bce_per_expert = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")
    weighted = bce_per_expert[valid] * element_weights[valid]
    bce = weighted.sum() / element_weights[valid].sum().clamp(min=1e-12)
    loss = bce
    parts: dict[str, torch.Tensor | int] = {"bce": bce, "valid_items": valid_items}
    if token_cardinality_loss_weight > 0.0:
        token_cardinality = token_cardinality_loss_from_target(pred_logits, target, valid)
        loss = loss + float(token_cardinality_loss_weight) * token_cardinality
        parts["token_cardinality"] = token_cardinality
    if layer_count_loss_weight > 0.0:
        layer_count = layer_noisy_or_count_loss_from_target(pred_logits, target, valid)
        loss = loss + float(layer_count_loss_weight) * layer_count
        parts["layer_count"] = layer_count
    if soft_gate_aux_weight > 0.0:
        soft_gate_aux = soft_gate_auxiliary_loss(
            pred_logits,
            expert_selection,
            expert_weights,
            slot_mask,
            valid,
            aux_type=soft_gate_aux_type,
        )
        loss = loss + float(soft_gate_aux_weight) * soft_gate_aux
        parts["soft_gate_aux"] = soft_gate_aux
    return loss, parts


def loss_for_type(
    loss_type: str,
    pred_logits: torch.Tensor,
    expert_selection: torch.Tensor,
    attention_mask: torch.Tensor,
    expert_weights: torch.Tensor | None = None,
    expert_selection_mask: torch.Tensor | None = None,
    use_expert_weights_in_loss: bool = True,
    token_cardinality_loss_weight: float = 0.0,
    layer_count_loss_weight: float = 0.0,
    soft_gate_aux_weight: float = 0.0,
    soft_gate_aux_type: str = "kl",
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    if token_cardinality_loss_weight < 0.0 or layer_count_loss_weight < 0.0 or soft_gate_aux_weight < 0.0:
        raise ValueError("auxiliary loss weights must be non-negative")
    if loss_type == "hard_ce":
        if token_cardinality_loss_weight > 0.0 or layer_count_loss_weight > 0.0 or soft_gate_aux_weight > 0.0:
            raise ValueError("auxiliary losses require multi_label_bce")
        return hard_ce_loss(pred_logits, expert_selection, attention_mask)
    if loss_type == "multi_label_bce":
        return multi_label_bce_loss(
            pred_logits,
            expert_selection,
            attention_mask,
            expert_weights,
            expert_selection_mask,
            use_expert_weights=use_expert_weights_in_loss,
            token_cardinality_loss_weight=token_cardinality_loss_weight,
            layer_count_loss_weight=layer_count_loss_weight,
            soft_gate_aux_weight=soft_gate_aux_weight,
            soft_gate_aux_type=soft_gate_aux_type,
        )
    raise ValueError(f"unknown loss_type: {loss_type}")


def valid_sample_tensors(
    batch: dict[str, torch.Tensor], sample_index: int
) -> tuple[torch.Tensor, ...] | None:
    attention_mask = batch["attention_mask"][sample_index]
    valid_len = int(attention_mask.eq(1).sum().item())
    if valid_len == 0:
        return None
    x = batch["layer0_attn_out"][sample_index : sample_index + 1, :valid_len].float()
    expert_selection = batch["expert_selection"][sample_index : sample_index + 1, :, :valid_len]
    valid_mask = torch.ones(1, valid_len, dtype=batch["attention_mask"].dtype, device=batch["attention_mask"].device)
    if "expert_weights" not in batch and "expert_selection_mask" not in batch:
        return x, expert_selection, valid_mask
    expert_weights = batch.get("expert_weights")
    expert_selection_mask = batch.get("expert_selection_mask")
    if expert_weights is not None:
        expert_weights = expert_weights[sample_index : sample_index + 1, :, :valid_len]
    if expert_selection_mask is not None:
        expert_selection_mask = expert_selection_mask[sample_index : sample_index + 1, :, :valid_len]
    return x, expert_selection, valid_mask, expert_weights, expert_selection_mask


def _unpack_sample_tensors(sample: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if len(sample) == 3:
        x, expert_selection, valid_mask = sample
        return x, expert_selection, valid_mask, None, None
    x, expert_selection, valid_mask, expert_weights, expert_selection_mask = sample
    return x, expert_selection, valid_mask, expert_weights, expert_selection_mask


def _empty_prefetch_metrics(budgets: tuple[int, ...]) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {"num_valid_token_layer_items": 0}
    for budget in budgets:
        metrics[f"overlap_count@{budget}"] = 0
        metrics[f"topK_precision@{budget}"] = 0.0
        metrics[f"topK_recall@{budget}"] = 0.0
        metrics[f"MAP@{budget}"] = 0.0
        metrics[f"NDCG@{budget}"] = 0.0
        if budget > 1:
            metrics[f"extra_overlap_gain@{budget}"] = 0
            metrics[f"extra_recall_gain@{budget}"] = 0.0
            metrics[f"marginal_precision@{budget}"] = 0.0
    metrics.update({"mean_true_rank": 0.0, "median_true_rank": 0.0, "p90_true_rank": 0.0, "p99_true_rank": 0.0})
    return metrics


def _expert_targets_for_metrics(
    pred_logits: torch.Tensor,
    expert_selection: torch.Tensor,
    attention_mask: torch.Tensor,
    expert_selection_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if expert_selection.shape[:3] != pred_logits.shape[:3] or expert_selection.shape[-1] < 1:
        raise ValueError("expert_selection must have shape [B,L,T,K>=1] aligned with pred_logits")
    expert_selection = expert_selection.to(device=pred_logits.device).long()
    expert_selection_mask = _validate_optional_expert_tensor("expert_selection_mask", expert_selection_mask, expert_selection)
    slot_mask = _effective_expert_selection_mask(expert_selection, expert_selection_mask)
    valid = valid_token_layer_mask(pred_logits, attention_mask) & slot_mask.any(dim=-1)
    if not bool(valid.any()):
        empty_logits = pred_logits.new_zeros((0, pred_logits.shape[-1]))
        empty_set = torch.zeros((0, pred_logits.shape[-1]), dtype=torch.bool, device=pred_logits.device)
        empty_count = torch.zeros((0,), dtype=torch.long, device=pred_logits.device)
        return empty_logits, empty_set, empty_count

    valid_logits = pred_logits[valid]
    selected = expert_selection[valid]
    selected_mask = slot_mask[valid]
    true_set = torch.zeros(selected.shape[0], pred_logits.shape[-1], dtype=torch.bool, device=pred_logits.device)
    true_set.scatter_(dim=-1, index=selected, src=selected_mask)
    true_count = true_set.sum(dim=-1).long()
    keep = true_count > 0
    return valid_logits[keep], true_set[keep], true_count[keep]


def compute_prefetch_metrics(
    pred_logits: torch.Tensor,
    expert_selection: torch.Tensor,
    attention_mask: torch.Tensor,
    budgets: Iterable[int] = DEFAULT_BUDGETS,
    expert_selection_mask: torch.Tensor | None = None,
) -> dict[str, float | int]:
    budget_values = tuple(int(budget) for budget in budgets)
    if not budget_values or any(budget <= 0 for budget in budget_values):
        raise ValueError("budgets must be positive integers")

    valid_logits, true_set, true_count = _expert_targets_for_metrics(
        pred_logits,
        expert_selection,
        attention_mask,
        expert_selection_mask,
    )
    num_valid = int(valid_logits.shape[0])
    if num_valid == 0:
        return _empty_prefetch_metrics(budget_values)

    ranked = torch.argsort(valid_logits, dim=-1, descending=True)
    true_by_rank = torch.gather(true_set, dim=-1, index=ranked)
    rank_numbers = torch.arange(1, ranked.shape[-1] + 1, device=ranked.device, dtype=torch.float32)[None, :]
    true_rank_positions = true_by_rank.nonzero(as_tuple=False)[:, 1]
    true_ranks_float = (true_rank_positions + 1).to(dtype=torch.float32)
    top1_overlap = int(true_by_rank[:, :1].sum().item())
    total_true = int(true_count.sum().item())
    top1_recall = top1_overlap / total_true if total_true else 0.0

    metrics: dict[str, float | int] = {"num_valid_token_layer_items": num_valid}
    experts = int(pred_logits.shape[-1])
    for budget in budget_values:
        effective_budget = min(budget, experts)
        hits_ranked = true_by_rank[:, :effective_budget].to(dtype=torch.float32)
        overlap_per_item = hits_ranked.sum(dim=-1)
        overlap_count = int(overlap_per_item.sum().item())
        precision_at_hits = hits_ranked * (hits_ranked.cumsum(dim=-1) / rank_numbers[:, :effective_budget])
        ap = precision_at_hits.sum(dim=-1) / true_count.to(dtype=torch.float32).clamp(min=1.0)
        discounts = 1.0 / torch.log2(rank_numbers[:, :effective_budget] + 1.0)
        dcg = (hits_ranked * discounts).sum(dim=-1)
        ideal_hits = torch.arange(effective_budget, device=ranked.device)[None, :] < true_count[:, None].clamp(max=effective_budget)
        idcg = (ideal_hits.to(dtype=torch.float32) * discounts).sum(dim=-1).clamp(min=1e-12)
        recall = overlap_count / total_true if total_true else 0.0
        metrics[f"overlap_count@{budget}"] = overlap_count
        metrics[f"topK_precision@{budget}"] = overlap_count / (num_valid * effective_budget)
        metrics[f"topK_recall@{budget}"] = recall
        metrics[f"MAP@{budget}"] = float(ap.mean().item())
        metrics[f"NDCG@{budget}"] = float((dcg / idcg).mean().item())
        if budget > 1:
            extra_overlap = overlap_count - top1_overlap
            metrics[f"extra_overlap_gain@{budget}"] = extra_overlap
            metrics[f"extra_recall_gain@{budget}"] = recall - top1_recall
            extra_slots = max(effective_budget - 1, 1) * num_valid
            metrics[f"marginal_precision@{budget}"] = extra_overlap / extra_slots

    metrics["mean_true_rank"] = float(true_ranks_float.mean().item())
    metrics["median_true_rank"] = float(torch.quantile(true_ranks_float, 0.5).item())
    metrics["p90_true_rank"] = float(torch.quantile(true_ranks_float, 0.9).item())
    metrics["p99_true_rank"] = float(torch.quantile(true_ranks_float, 0.99).item())
    return metrics


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available")
    return device


def resolve_loss_type(requested_loss_type: str, metadata: dict[str, Any]) -> str:
    if requested_loss_type not in LOSS_TYPES:
        raise ValueError(f"unknown loss_type: {requested_loss_type}")
    if requested_loss_type != "auto":
        return requested_loss_type
    model_type = str(_metadata_model_config(metadata).get("model_type", ""))
    return "multi_label_bce" if model_type == "nllb-moe" else "hard_ce"


def objective_from_loss_type(loss_type: str) -> str:
    if loss_type == "hard_ce":
        return "hard-ce"
    if loss_type == "multi_label_bce":
        return "multi-label-bce"
    raise ValueError(f"unknown loss_type: {loss_type}")


def _loss_metric_name(loss_type: str) -> str:
    return "ce" if loss_type == "hard_ce" else "bce"


def _empty_loss_summary(prefix: str, loss_type: str = "hard_ce") -> dict[str, float | int]:
    return {f"{prefix}_loss": 0.0, f"{prefix}_{_loss_metric_name(loss_type)}": 0.0, f"{prefix}_valid_items": 0}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None = None,
    loss_type: str = "hard_ce",
    use_expert_weights_in_loss: bool = True,
    token_cardinality_loss_weight: float = 0.0,
    layer_count_loss_weight: float = 0.0,
    soft_gate_aux_weight: float = 0.0,
    soft_gate_aux_type: str = "kl",
) -> dict[str, float | int]:
    model.train()
    aux_key = _loss_metric_name(loss_type)
    total_loss = 0.0
    total_aux = 0.0
    total_valid = 0
    extra_part_totals: dict[str, float] = {}
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        weighted_batch_loss: torch.Tensor | None = None
        batch_valid = 0
        for sample_index in range(int(batch["attention_mask"].shape[0])):
            sample = valid_sample_tensors(batch, sample_index)
            if sample is None:
                continue
            x, expert_selection, valid_mask, expert_weights, expert_selection_mask = _unpack_sample_tensors(sample)
            pred = model(x)
            loss, parts = loss_for_type(
                loss_type,
                pred,
                expert_selection,
                valid_mask,
                expert_weights,
                expert_selection_mask,
                use_expert_weights_in_loss=use_expert_weights_in_loss,
                token_cardinality_loss_weight=token_cardinality_loss_weight,
                layer_count_loss_weight=layer_count_loss_weight,
                soft_gate_aux_weight=soft_gate_aux_weight,
                soft_gate_aux_type=soft_gate_aux_type,
            )
            valid_items = int(parts["valid_items"])
            weighted = loss * valid_items
            weighted_batch_loss = weighted if weighted_batch_loss is None else weighted_batch_loss + weighted
            batch_valid += valid_items
            total_loss += float(loss.detach().cpu().item()) * valid_items
            total_aux += float(parts[aux_key].detach().cpu().item()) * valid_items
            for part_name, part_value in parts.items():
                if part_name in {"valid_items", aux_key}:
                    continue
                if isinstance(part_value, torch.Tensor):
                    extra_part_totals[part_name] = extra_part_totals.get(part_name, 0.0) + float(part_value.detach().cpu().item()) * valid_items
            total_valid += valid_items
        if weighted_batch_loss is not None and batch_valid > 0:
            (weighted_batch_loss / batch_valid).backward()
            optimizer.step()
    if total_valid == 0:
        return _empty_loss_summary("train", loss_type)
    summary = {"train_loss": total_loss / total_valid, f"train_{aux_key}": total_aux / total_valid, "train_valid_items": total_valid}
    for part_name, value in extra_part_totals.items():
        summary[f"train_{part_name}"] = value / total_valid
    return summary


def _is_count_metric(name: str) -> bool:
    return name.startswith("overlap_count@") or name.startswith("extra_overlap_gain@")


def _is_rank_stat(name: str) -> bool:
    return name in {"mean_true_rank", "median_true_rank", "p90_true_rank", "p99_true_rank"}


def _rank_histogram(
    pred_logits: torch.Tensor,
    expert_selection: torch.Tensor,
    attention_mask: torch.Tensor,
    expert_selection_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    valid_logits, true_set, _ = _expert_targets_for_metrics(pred_logits, expert_selection, attention_mask, expert_selection_mask)
    if int(valid_logits.shape[0]) == 0:
        return torch.zeros(pred_logits.shape[-1] + 1, dtype=torch.long)
    ranked = torch.argsort(valid_logits, dim=-1, descending=True)
    true_by_rank = torch.gather(true_set, dim=-1, index=ranked)
    true_rank_positions = true_by_rank.nonzero(as_tuple=False)[:, 1]
    return torch.bincount((true_rank_positions + 1).cpu(), minlength=pred_logits.shape[-1] + 1)


def _rank_stats_from_histogram(histogram: torch.Tensor) -> dict[str, float]:
    ranks = torch.repeat_interleave(torch.arange(histogram.numel(), dtype=torch.float32), histogram.to(torch.long))
    ranks = ranks[ranks > 0]
    if ranks.numel() == 0:
        return {"mean_true_rank": 0.0, "median_true_rank": 0.0, "p90_true_rank": 0.0, "p99_true_rank": 0.0}
    return {
        "mean_true_rank": float(ranks.mean().item()),
        "median_true_rank": float(torch.quantile(ranks, 0.5).item()),
        "p90_true_rank": float(torch.quantile(ranks, 0.9).item()),
        "p99_true_rank": float(torch.quantile(ranks, 0.99).item()),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    budgets: tuple[int, ...],
    max_batches: int | None = None,
    loss_type: str = "hard_ce",
    use_expert_weights_in_loss: bool = True,
    token_cardinality_loss_weight: float = 0.0,
    layer_count_loss_weight: float = 0.0,
    soft_gate_aux_weight: float = 0.0,
    soft_gate_aux_type: str = "kl",
) -> dict[str, float | int]:
    model.eval()
    aux_key = _loss_metric_name(loss_type)
    total_loss = 0.0
    total_aux = 0.0
    total_valid = 0
    extra_part_totals: dict[str, float] = {}
    metric_totals: dict[str, float] = {}
    rank_hist: torch.Tensor | None = None

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch(batch, device)
        for sample_index in range(int(batch["attention_mask"].shape[0])):
            sample = valid_sample_tensors(batch, sample_index)
            if sample is None:
                continue
            x, expert_selection, valid_mask, expert_weights, expert_selection_mask = _unpack_sample_tensors(sample)
            pred = model(x)
            loss, parts = loss_for_type(
                loss_type,
                pred,
                expert_selection,
                valid_mask,
                expert_weights,
                expert_selection_mask,
                use_expert_weights_in_loss=use_expert_weights_in_loss,
                token_cardinality_loss_weight=token_cardinality_loss_weight,
                layer_count_loss_weight=layer_count_loss_weight,
                soft_gate_aux_weight=soft_gate_aux_weight,
                soft_gate_aux_type=soft_gate_aux_type,
            )
            valid_items = int(parts["valid_items"])
            total_loss += float(loss.detach().cpu().item()) * valid_items
            total_aux += float(parts[aux_key].detach().cpu().item()) * valid_items
            for part_name, part_value in parts.items():
                if part_name in {"valid_items", aux_key}:
                    continue
                if isinstance(part_value, torch.Tensor):
                    extra_part_totals[part_name] = extra_part_totals.get(part_name, 0.0) + float(part_value.detach().cpu().item()) * valid_items
            total_valid += valid_items

            sample_metrics = compute_prefetch_metrics(
                pred.detach().cpu(),
                expert_selection.detach().cpu(),
                valid_mask.detach().cpu(),
                budgets=budgets,
                expert_selection_mask=expert_selection_mask.detach().cpu() if expert_selection_mask is not None else None,
            )
            sample_valid = int(sample_metrics["num_valid_token_layer_items"])
            if sample_valid:
                for key, value in sample_metrics.items():
                    if key == "num_valid_token_layer_items" or _is_rank_stat(key):
                        continue
                    if _is_count_metric(key):
                        metric_totals[key] = metric_totals.get(key, 0.0) + float(value)
                    else:
                        metric_totals[key] = metric_totals.get(key, 0.0) + float(value) * sample_valid
            sample_rank_hist = _rank_histogram(pred.detach(), expert_selection, valid_mask, expert_selection_mask)
            if rank_hist is None:
                rank_hist = sample_rank_hist
            else:
                if rank_hist.numel() < sample_rank_hist.numel():
                    rank_hist = F.pad(rank_hist, (0, sample_rank_hist.numel() - rank_hist.numel()))
                rank_hist[: sample_rank_hist.numel()] += sample_rank_hist

    summary = _empty_loss_summary("validation", loss_type)
    if total_valid:
        summary = {
            "validation_loss": total_loss / total_valid,
            f"validation_{aux_key}": total_aux / total_valid,
            "validation_valid_items": total_valid,
        }
        for part_name, value in extra_part_totals.items():
            summary[f"validation_{part_name}"] = value / total_valid
    metrics: dict[str, float | int] = {"num_valid_token_layer_items": total_valid}
    for key, value in metric_totals.items():
        metrics[key] = int(value) if _is_count_metric(key) else value / total_valid if total_valid else 0.0
    metrics.update(_rank_stats_from_histogram(rank_hist if rank_hist is not None else torch.zeros(1, dtype=torch.long)))
    return {**summary, **metrics}


def _parse_budgets(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("budgets must be comma-separated positive integers")
    return values



def loss_token_from_args(args: argparse.Namespace) -> str:
    loss_type = getattr(args, "loss_type", "hard_ce")
    if loss_type == "multi_label_bce":
        token = "bce-weighted" if args.use_expert_weights_in_loss else "bce-equal-top2"
        token_cardinality_weight = float(getattr(args, "token_cardinality_loss_weight", 0.0))
        layer_count_weight = float(getattr(args, "layer_count_loss_weight", 0.0))
        soft_gate_weight = float(getattr(args, "soft_gate_aux_weight", 0.0))
        if soft_gate_weight > 0.0:
            token += f"-softgate{_float_weight_token(soft_gate_weight)}"
        if token_cardinality_weight > 0.0:
            token += f"-tokcnt{_float_weight_token(token_cardinality_weight)}"
        if layer_count_weight > 0.0:
            token += f"-layercnt{_float_weight_token(layer_count_weight)}"
        return token
    return "hardce"


def output_name_from_args(args: argparse.Namespace) -> str:
    return src_simplenn_blte_run_name(
        hidden_dim=args.hidden_dim,
        src_layers=args.src_layers,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        loss_token=loss_token_from_args(args),
    )


def resolve_trace_dir(args: argparse.Namespace) -> Path:
    if args.trace_dir is not None:
        return args.trace_dir
    return trace_dir_for_model_task(
        model_path=args.model_path if args.model_path is not None else DEFAULT_MODEL_PATH,
        model_name=args.model_name,
        dataset=args.dataset,
        task_name=args.task_name,
    )


def _trace_context_from_args(args: argparse.Namespace):
    return resolve_trace_context(
        trace_dir=args.trace_dir,
        model_path=args.model_path,
        model_name=args.model_name,
        dataset=args.dataset,
        task_name=args.task_name,
        trace_id=args.trace_id,
    )


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    context = _trace_context_from_args(args)
    return blte_artifact_dir(
        output_name_from_args(args),
        workload_task=context.workload_task,
        base_model=context.base_model,
        trace_id=context.trace_id,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sparse-cache encoder predictor SRC-SimpleNN-token hard CE.")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--trace-id", default=TRACE_ID)
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--src-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--budgets", type=_parse_budgets, default=DEFAULT_BUDGETS)
    parser.add_argument("--loss-type", choices=LOSS_TYPES, default="auto")
    parser.add_argument("--use-expert-weights-in-loss", dest="use_expert_weights_in_loss", action="store_true", default=True)
    parser.add_argument("--no-use-expert-weights-in-loss", dest="use_expert_weights_in_loss", action="store_false")
    parser.add_argument("--token-cardinality-loss-weight", type=float, default=0.0)
    parser.add_argument("--layer-count-loss-weight", type=float, default=0.0)
    parser.add_argument("--soft-gate-aux-weight", type=float, default=0.0)
    parser.add_argument("--soft-gate-aux-type", choices=("kl", "ce"), default="kl")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--early-stop", dest="early_stop", action="store_true", default=True)
    parser.add_argument("--no-early-stop", dest="early_stop", action="store_false")
    parser.add_argument("--early-stop-window", type=int, default=8)
    parser.add_argument("--early-stop-threshold", type=float, default=0.003)
    return parser.parse_args(argv)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _loss_delta(losses: list[float], window: int) -> float:
    if len(losses) <= window * 2:
        return math.nan
    prev = torch.tensor(losses[-2 * window : -window], dtype=torch.float64).mean()
    last = torch.tensor(losses[-window:], dtype=torch.float64).mean()
    if float(prev.abs().item()) == 0.0:
        return math.inf
    return float(((prev - last).abs() / prev.abs()).item())


def normalized_compare_metadata(info: TraceInfo, metadata: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "hidden_size": info.hidden_size,
        "num_encoder_moe_layers": info.num_router_layers,
        "num_experts": info.num_experts,
        "routing_top_k": info.num_selected_experts,
        "num_selected_experts": info.num_selected_experts,
        "max_input_tokens": info.max_input_tokens,
        "sparse_cache_metadata": metadata,
    }
    model_config = _metadata_model_config(metadata)
    for key in ("model_type", "num_sparse_encoder_layers", "encoder_sparse_step"):
        if key in model_config:
            normalized[key] = model_config[key]
    return normalized


def make_config(args: argparse.Namespace, info: TraceInfo, metadata: dict[str, Any], device: torch.device) -> dict[str, Any]:
    actual_loss_type = resolve_loss_type(args.loss_type, metadata)
    return {
        "model": MODEL_NAME,
        "objective": objective_from_loss_type(actual_loss_type),
        "loss_type": actual_loss_type,
        "requested_loss_type": args.loss_type,
        "use_expert_weights_in_loss": bool(args.use_expert_weights_in_loss),
        "token_cardinality_loss_weight": float(args.token_cardinality_loss_weight),
        "layer_count_loss_weight": float(args.layer_count_loss_weight),
        "soft_gate_aux_weight": float(args.soft_gate_aux_weight),
        "soft_gate_aux_type": args.soft_gate_aux_type,
        "output_name": output_name_from_args(args),
        "trace_dir": str(args.trace_dir),
        "output_dir": str(args.output_dir),
        "workload_task": _trace_context_from_args(args).workload_task,
        "base_model": _trace_context_from_args(args).base_model,
        "trace_id": _trace_context_from_args(args).trace_id,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "src_layers": args.src_layers,
        "dropout": args.dropout,
        "budgets": list(args.budgets),
        "num_workers": args.num_workers,
        "device": str(device),
        "seed": args.seed,
        "max_train_batches": args.max_train_batches,
        "max_eval_batches": args.max_eval_batches,
        "early_stop": args.early_stop,
        "early_stop_window": args.early_stop_window,
        "early_stop_threshold": args.early_stop_threshold,
        "trace_info": {
            "hidden_size": info.hidden_size,
            "num_experts": info.num_experts,
            "num_selected_experts": info.num_selected_experts,
            "num_router_layers": info.num_router_layers,
            "max_input_tokens": info.max_input_tokens,
        },
        "metadata": normalized_compare_metadata(info, metadata),
        "torch_version": str(torch.__version__),
        "cuda_available": torch.cuda.is_available(),
    }



def make_run_manifest(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    context = _trace_context_from_args(args)
    return blte_manifest(
        run_name=config["output_name"],
        model_arch=MODEL_NAME,
        objective=objective_from_loss_type(config.get("loss_type", "hard_ce")),
        output_dir=args.output_dir,
        trace_dir=args.trace_dir,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        src_layers=args.src_layers,
        dropout=args.dropout,
        workload_task=context.workload_task,
        base_model=context.base_model,
        trace_id=context.trace_id,
        extra={
            "trace_info": config["trace_info"],
            "loss_type": config.get("loss_type", "hard_ce"),
            "objective": config.get("objective", objective_from_loss_type(config.get("loss_type", "hard_ce"))),
            "use_expert_weights_in_loss": config.get("use_expert_weights_in_loss", True),
            "token_cardinality_loss_weight": config.get("token_cardinality_loss_weight", 0.0),
            "layer_count_loss_weight": config.get("layer_count_loss_weight", 0.0),
            "soft_gate_aux_weight": config.get("soft_gate_aux_weight", 0.0),
            "soft_gate_aux_type": config.get("soft_gate_aux_type", "kl"),
        },
    )


def write_readme(path: Path, config: dict[str, Any], metrics: dict[str, Any]) -> None:
    lines = [
        "# Encoder Predictor SRC-SimpleNN Token Hard CE",
        "",
        "Training artifact produced by experiment/scripts/train/encoder_predictor_src_simplenn_token_hard_ce.py.",
        "",
        f"- model: {config['model']}",
        f"- objective: {config['objective']}",
        f"- loss_type: {config.get('loss_type', 'hard_ce')}",
        f"- use_expert_weights_in_loss: {config.get('use_expert_weights_in_loss', True)}",
        f"- token_cardinality_loss_weight: {config.get('token_cardinality_loss_weight', 0.0)}",
        f"- layer_count_loss_weight: {config.get('layer_count_loss_weight', 0.0)}",
        f"- soft_gate_aux_weight: {config.get('soft_gate_aux_weight', 0.0)}",
        f"- soft_gate_aux_type: {config.get('soft_gate_aux_type', 'kl')}",
        f"- trace_dir: {config['trace_dir']}",
        f"- epochs_requested: {config['epochs']}",
        f"- epoch_final: {metrics.get('epoch', 0)}",
        f"- best_epoch: {metrics.get('best_epoch', 0)}",
        f"- validation_loss: {metrics.get('validation_loss', 0.0)}",
        f"- validation_valid_items: {metrics.get('validation_valid_items', 0)}",
        f"- topK_recall@1: {metrics.get('topK_recall@1', 0.0)}",
        f"- topK_recall@17: {metrics.get('topK_recall@17', 0.0)}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    args.trace_dir = resolve_trace_dir(args)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    train_dataset = EncoderPredictorTraceDataset(args.trace_dir, "train")
    validation_dataset = EncoderPredictorTraceDataset(args.trace_dir, "validation")
    info = train_dataset.info
    if validation_dataset.info != info:
        raise ValueError("train and validation trace metadata/shapes disagree")
    args.loss_type = resolve_loss_type(args.loss_type, train_dataset.metadata)
    args.output_dir = resolve_output_dir(args)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        collate_fn=collate_trace_batch,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_trace_batch,
    )

    model = SrcSimpleNNTokenPredictor(
        input_dim=info.hidden_size,
        hidden_dim=args.hidden_dim,
        num_router_layers=info.num_router_layers,
        num_experts=info.num_experts,
        src_layers=args.src_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = make_config(args, info, train_dataset.metadata, device)
    log_path = args.output_dir / "train_log.jsonl"
    validation_losses: list[float] = []
    best_loss = math.inf
    best_metrics: dict[str, Any] = {}
    final_metrics: dict[str, Any] = {}

    with log_path.open("w", encoding="utf-8") as log_fh:
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                max_batches=args.max_train_batches,
                loss_type=args.loss_type,
                use_expert_weights_in_loss=args.use_expert_weights_in_loss,
                token_cardinality_loss_weight=args.token_cardinality_loss_weight,
                layer_count_loss_weight=args.layer_count_loss_weight,
                soft_gate_aux_weight=args.soft_gate_aux_weight,
                soft_gate_aux_type=args.soft_gate_aux_type,
            )
            eval_metrics = evaluate(
                model,
                validation_loader,
                device,
                budgets=args.budgets,
                max_batches=args.max_eval_batches,
                loss_type=args.loss_type,
                use_expert_weights_in_loss=args.use_expert_weights_in_loss,
                token_cardinality_loss_weight=args.token_cardinality_loss_weight,
                layer_count_loss_weight=args.layer_count_loss_weight,
                soft_gate_aux_weight=args.soft_gate_aux_weight,
                soft_gate_aux_type=args.soft_gate_aux_type,
            )
            validation_loss = float(eval_metrics["validation_loss"])
            validation_losses.append(validation_loss)
            delta = _loss_delta(validation_losses, args.early_stop_window)
            stopped_early = bool(args.early_stop and not math.isnan(delta) and delta < args.early_stop_threshold)
            final_metrics = {
                "epoch": epoch,
                "loss_delta": delta,
                "stopped_early": stopped_early,
                "stop_reason": "early_stop_loss_delta" if stopped_early else "max_epochs",
                **eval_metrics,
            }
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_metrics = {"best_epoch": epoch, **final_metrics}
                torch.save(
                    {"model_state_dict": model.state_dict(), "config": config, "metrics": best_metrics},
                    args.output_dir / "best_model.pt",
                )
            log_fh.write(json.dumps({"epoch": epoch, **train_metrics, **eval_metrics}, sort_keys=True) + "\n")
            log_fh.flush()
            if stopped_early:
                break

    final_metrics = {"best_epoch": best_metrics.get("best_epoch", 0), **final_metrics}
    write_json(args.output_dir / "config.json", config)
    write_json(args.output_dir / "run_manifest.json", make_run_manifest(args, config))
    write_json(args.output_dir / "metrics.json", final_metrics)
    torch.save({"model_state_dict": model.state_dict(), "config": config, "metrics": final_metrics}, args.output_dir / "model.pt")
    if not best_metrics:
        torch.save({"model_state_dict": model.state_dict(), "config": config, "metrics": final_metrics}, args.output_dir / "best_model.pt")
    write_readme(args.output_dir / "README.md", config, final_metrics)
    return final_metrics


if __name__ == "__main__":
    main()
