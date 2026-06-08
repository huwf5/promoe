#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export MMLU subset to prompt_list.txt/.pt (question-only)."
    )
    parser.add_argument("--dataset", default="cais/mmlu", help="HF dataset name")
    parser.add_argument("--task", default="professional_law", help="MMLU subset/config name")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test", "validation"],
        help="Dataset splits to export",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Root output dir (default: current mmlu dir)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed (matches legacy implementation)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.output_root / args.task
    base_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        out_dir = base_dir / split
        out_dir.mkdir(parents=True, exist_ok=True)

        ds = load_dataset(args.dataset, args.task, split=split).shuffle(seed=args.seed)
        prompts = [row["question"] for row in ds]

        torch.save(prompts, out_dir / "prompt_list.pt")
        with (out_dir / "prompt_list.txt").open("w", encoding="utf-8") as f:
            for item in prompts:
                f.write(item.replace("\n", "\\n") + "\n")

        print(f"{split}: {len(prompts)} prompts -> {out_dir}")


if __name__ == "__main__":
    main()
