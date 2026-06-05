#!/usr/bin/env python3
"""Verify an ERPP encoder trace directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _load(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=True)


def _assert_floating(name: str, x: torch.Tensor) -> None:
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise AssertionError(f"{name} dtype {x.dtype} is not a supported floating dtype")
    if not torch.isfinite(x.float()).all():
        raise AssertionError(f"{name} contains non-finite values")


def verify_split(root: Path, split: str, meta: dict) -> None:
    d = root / split
    required = ["input_ids.pt", "attention_mask.pt", "layer0_attn_out.pt", "router_logits.pt", "expert_selection.pt"]
    for name in required:
        if not (d / name).is_file():
            raise AssertionError(f"missing {split}/{name}")

    input_ids = _load(d / "input_ids.pt")
    attention_mask = _load(d / "attention_mask.pt")
    layer0_attn_out = _load(d / "layer0_attn_out.pt")
    router_logits = _load(d / "router_logits.pt")
    expert_selection = _load(d / "expert_selection.pt")

    if input_ids.dtype != torch.int64:
        raise AssertionError(f"{split}/input_ids dtype {input_ids.dtype} != int64")
    if attention_mask.dtype not in (torch.int64, torch.int32, torch.bool):
        raise AssertionError(f"{split}/attention_mask dtype {attention_mask.dtype} is not supported")
    if expert_selection.dtype != torch.int64:
        raise AssertionError(f"{split}/expert_selection dtype {expert_selection.dtype} != int64")
    _assert_floating(f"{split}/layer0_attn_out", layer0_attn_out)
    _assert_floating(f"{split}/router_logits", router_logits)

    if input_ids.dim() != 2:
        raise AssertionError(f"{split}/input_ids must be [S,T], got {tuple(input_ids.shape)}")
    s, t = input_ids.shape
    if s <= 0 or t <= 0:
        raise AssertionError(f"{split} must be non-empty, got {tuple(input_ids.shape)}")
    if attention_mask.shape != input_ids.shape:
        raise AssertionError(f"{split}/attention_mask shape {tuple(attention_mask.shape)} != input_ids {tuple(input_ids.shape)}")
    if layer0_attn_out.shape[:2] != input_ids.shape:
        raise AssertionError(f"{split}/layer0_attn_out shape {tuple(layer0_attn_out.shape)} not aligned with input_ids")
    if router_logits.dim() != 4:
        raise AssertionError(f"{split}/router_logits must be [S,L,T,E], got {tuple(router_logits.shape)}")
    if expert_selection.dim() != 4:
        raise AssertionError(f"{split}/expert_selection must be [S,L,T,K], got {tuple(expert_selection.shape)}")
    if router_logits.shape[0] != s or router_logits.shape[2] != t:
        raise AssertionError(f"{split}/router_logits shape {tuple(router_logits.shape)} not aligned with input_ids")
    if expert_selection.shape[:3] != router_logits.shape[:3]:
        raise AssertionError(f"{split}/expert_selection shape {tuple(expert_selection.shape)} not aligned with router_logits")

    hidden_size = int(meta["hidden_size"])
    num_layers = int(meta["num_encoder_moe_layers"])
    num_experts = int(meta["num_experts"])
    top_k = int(meta["routing_top_k"])
    if layer0_attn_out.shape[-1] != hidden_size:
        raise AssertionError(f"{split}/hidden size {layer0_attn_out.shape[-1]} != metadata {hidden_size}")
    if router_logits.shape[1] != num_layers or router_logits.shape[-1] != num_experts:
        raise AssertionError(f"{split}/router logits shape {tuple(router_logits.shape)} mismatches metadata")
    if expert_selection.shape[-1] != top_k:
        raise AssertionError(f"{split}/expert top_k {expert_selection.shape[-1]} != metadata {top_k}")
    if int(expert_selection.min()) < 0 or int(expert_selection.max()) >= num_experts:
        raise AssertionError(f"{split}/expert_selection values out of range [0,{num_experts})")

    mask_values = attention_mask.to(torch.int64).unique().tolist()
    if any(v not in (0, 1) for v in mask_values):
        raise AssertionError(f"{split}/attention_mask has values outside {{0,1}}: {mask_values}")

    probs_path = d / "router_probs.pt"
    if probs_path.exists():
        router_probs = _load(probs_path)
        if router_probs.shape != router_logits.shape:
            raise AssertionError(f"{split}/router_probs shape {tuple(router_probs.shape)} != router_logits {tuple(router_logits.shape)}")
        _assert_floating(f"{split}/router_probs", router_probs)
        valid = attention_mask.bool().unsqueeze(1)
        sums = router_probs.float().sum(dim=-1)
        if valid.any():
            max_err = (sums[valid.expand_as(sums)] - 1.0).abs().max().item()
            if max_err > 1e-2:
                raise AssertionError(f"{split}/router_probs sum error too large: {max_err}")

    weights_path = d / "expert_weights.pt"
    if weights_path.exists():
        expert_weights = _load(weights_path)
        if expert_weights.shape != expert_selection.shape:
            raise AssertionError(f"{split}/expert_weights shape {tuple(expert_weights.shape)} != expert_selection {tuple(expert_selection.shape)}")
        _assert_floating(f"{split}/expert_weights", expert_weights)


def verify_trace(trace_dir: Path) -> None:
    trace_dir = Path(trace_dir)
    meta_path = trace_dir / "metadata.json"
    if not meta_path.is_file():
        raise AssertionError("missing metadata.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("schema") != "erpp_encoder_trace":
        raise AssertionError(f"unexpected schema {meta.get('schema')!r}")
    if meta.get("stage") != "encoder":
        raise AssertionError(f"unexpected stage {meta.get('stage')!r}")
    for key in ("hidden_size", "num_experts", "routing_top_k", "num_encoder_moe_layers", "router_layer_to_model_block"):
        if key not in meta:
            raise AssertionError(f"metadata missing {key}")
    if len(meta["router_layer_to_model_block"]) != int(meta["num_encoder_moe_layers"]):
        raise AssertionError("router_layer_to_model_block length mismatches num_encoder_moe_layers")
    for split in ("train", "validation"):
        verify_split(trace_dir, split, meta)


def main() -> int:
    p = argparse.ArgumentParser(description="Verify ERPP encoder trace")
    p.add_argument("--trace-dir", required=True, type=Path)
    args = p.parse_args()
    verify_trace(args.trace_dir)
    print("ERPP verifier: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
