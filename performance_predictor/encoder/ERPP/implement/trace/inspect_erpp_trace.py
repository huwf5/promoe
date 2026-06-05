#!/usr/bin/env python3
"""Inspect one sample/token from an ERPP trace."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _load(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Inspect ERPP encoder trace sample")
    p.add_argument("--trace-dir", required=True, type=Path)
    p.add_argument("--split", default="validation", choices=["train", "validation"])
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--token", type=int, default=0)
    p.add_argument("--topk", type=int, default=5)
    args = p.parse_args()

    root = args.trace_dir
    meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    d = root / args.split
    input_ids = _load(d / "input_ids.pt")
    attention_mask = _load(d / "attention_mask.pt")
    layer0_attn_out = _load(d / "layer0_attn_out.pt")
    router_logits = _load(d / "router_logits.pt")
    expert_selection = _load(d / "expert_selection.pt")
    router_probs_path = d / "router_probs.pt"
    router_probs = _load(router_probs_path) if router_probs_path.exists() else torch.softmax(router_logits.float(), dim=-1)

    s = args.sample
    t = args.token
    if s < 0 or s >= input_ids.shape[0]:
        raise IndexError(f"sample {s} outside [0,{input_ids.shape[0]})")
    if t < 0 or t >= input_ids.shape[1]:
        raise IndexError(f"token {t} outside [0,{input_ids.shape[1]})")

    print(f"schema: {meta.get('schema')} v{meta.get('schema_version')}")
    print(f"split={args.split} sample={s} token={t}")
    print(f"input_id={int(input_ids[s, t])} attention_mask={int(attention_mask[s, t])}")
    print(f"layer0_attn_out_norm={float(layer0_attn_out[s, t].float().norm()) :.6f}")
    print("router layers:")
    for layer in range(router_logits.shape[1]):
        logits = router_logits[s, layer, t].float()
        probs = router_probs[s, layer, t].float()
        k = min(args.topk, logits.numel())
        values, indices = torch.topk(logits, k=k)
        actual = expert_selection[s, layer, t].tolist()
        prob_values = probs[indices]
        pairs = ", ".join(
            f"{int(idx)}:logit={float(val):.4f},prob={float(prob):.4f}"
            for idx, val, prob in zip(indices, values, prob_values)
        )
        block = meta.get("router_layer_to_model_block", [None] * router_logits.shape[1])[layer]
        print(f"  router_layer={layer} model_block={block} actual={actual} top{k}=[{pairs}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
