#!/usr/bin/env python3
"""Compare two encoder predictor trace directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def load_tensor(path: Path) -> torch.Tensor:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def read_prompt_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def expert_diff_summary(left: torch.Tensor, right: torch.Tensor, attention_mask: torch.Tensor) -> dict:
    if left.shape != right.shape:
        raise ValueError(f"expert_selection shape mismatch: {tuple(left.shape)} vs {tuple(right.shape)}")
    mask = attention_mask.to(torch.bool)[:, None, :, None].expand_as(left)
    diff = left != right
    per_layer = []
    for layer in range(left.shape[1]):
        layer_diff = diff[:, layer]
        layer_mask = mask[:, layer]
        per_layer.append(
            {
                "layer": int(layer),
                "valid_diff": int((layer_diff & layer_mask).sum()),
                "padding_diff": int((layer_diff & ~layer_mask).sum()),
            }
        )
    return {
        "total_diff": int(diff.sum()),
        "valid_diff": int((diff & mask).sum()),
        "padding_diff": int((diff & ~mask).sum()),
        "valid_positions": int(mask.sum()),
        "padding_positions": int((~mask).sum()),
        "per_layer": per_layer,
    }


def float_diff_summary(left: torch.Tensor, right: torch.Tensor, attention_mask: torch.Tensor, kind: str) -> dict:
    if left.shape != right.shape:
        raise ValueError(f"{kind} shape mismatch: {tuple(left.shape)} vs {tuple(right.shape)}")
    diff = (left.float() - right.float()).abs()
    if kind == "layer0_attn_out.pt":
        mask = attention_mask.to(torch.bool)[:, :, None].expand_as(diff)
    elif kind in {"router_logits.pt", "router_probs.pt"}:
        mask = attention_mask.to(torch.bool)[:, None, :, None].expand_as(diff)
    elif kind == "expert_weights.pt":
        mask = attention_mask.to(torch.bool)[:, None, :, None].expand_as(diff)
    else:
        mask = torch.ones_like(diff, dtype=torch.bool)
    valid = diff[mask]
    padding = diff[~mask]
    result = {
        "all_max": float(diff.max()) if diff.numel() else 0.0,
        "all_mean": float(diff.mean()) if diff.numel() else 0.0,
        "valid_max": float(valid.max()) if valid.numel() else 0.0,
        "valid_mean": float(valid.mean()) if valid.numel() else 0.0,
        "padding_max": float(padding.max()) if padding.numel() else 0.0,
        "padding_mean": float(padding.mean()) if padding.numel() else 0.0,
    }
    if kind == "router_logits.pt":
        per_pos = diff.max(dim=-1).values
        pos_mask = attention_mask.to(torch.bool)[:, None, :].expand_as(per_pos)
        result["valid_positions_maxdiff_gt_1e_2"] = int(((per_pos > 1e-2) & pos_mask).sum())
        result["per_layer"] = []
        for layer in range(left.shape[1]):
            layer_diff = diff[:, layer]
            layer_mask = attention_mask.to(torch.bool)[:, :, None].expand_as(layer_diff)
            values = layer_diff[layer_mask]
            result["per_layer"].append(
                {
                    "layer": int(layer),
                    "valid_max": float(values.max()) if values.numel() else 0.0,
                    "valid_mean": float(values.mean()) if values.numel() else 0.0,
                    "valid_elements_gt_1e_2": int((values > 1e-2).sum()) if values.numel() else 0,
                }
            )
    return result


def compare_split(left: Path, right: Path, split: str) -> dict:
    left_split = left / split
    right_split = right / split
    attention_mask = load_tensor(left_split / "attention_mask.pt")
    result = {"split": split, "integer_equal": {}, "float_diff": {}}

    for name in ("input_ids.pt", "attention_mask.pt", "seq_ids.pt"):
        a = load_tensor(left_split / name)
        b = load_tensor(right_split / name)
        result["integer_equal"][name] = {
            "left_shape": list(a.shape),
            "right_shape": list(b.shape),
            "equal": bool(torch.equal(a, b)),
        }

    left_prompts = read_prompt_lines(left_split / "prompt_texts.jsonl")
    right_prompts = read_prompt_lines(right_split / "prompt_texts.jsonl")
    result["prompt_texts_equal"] = left_prompts == right_prompts
    result["prompt_texts_lines"] = [len(left_prompts), len(right_prompts)]

    result["expert_selection"] = expert_diff_summary(
        load_tensor(left_split / "expert_selection.pt"),
        load_tensor(right_split / "expert_selection.pt"),
        attention_mask,
    )

    for name in ("layer0_attn_out.pt", "router_logits.pt", "router_probs.pt", "expert_weights.pt"):
        left_file = left_split / name
        right_file = right_split / name
        if left_file.exists() and right_file.exists():
            result["float_diff"][name] = float_diff_summary(
                load_tensor(left_file),
                load_tensor(right_file),
                attention_mask,
                name,
            )
        else:
            result["float_diff"][name] = {
                "left_exists": left_file.exists(),
                "right_exists": right_file.exists(),
            }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two encoder predictor trace directories.")
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = {
        "left": str(args.left),
        "right": str(args.right),
        "splits": [compare_split(args.left, args.right, split) for split in args.splits],
    }
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        for split_report in report["splits"]:
            print(f"=== {split_report['split']} ===")
            print(json.dumps(split_report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
