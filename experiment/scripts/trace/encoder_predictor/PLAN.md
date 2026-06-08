# Sparse Cache Encoder Predictor Trace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This project explicitly forbids git worktrees and git commits.

**Goal:** Build a sparse-cache-backed encoder predictor trace exporter under `experiment/scripts/trace/encoder_predictor/` that reuses `sparse_llm_cache.utils.hack_transformers(...)` and emits ERPP-style encoder trace tensors.

**Architecture:** The exporter is a local experiment tool. It patches Transformers through `sparse_llm_cache`, loads the model with `device_map=0`, drives generation to match the small-demo execution path, and collects encoder hook outputs into split accumulators. A separate comparison utility diagnoses differences against existing trace directories using `attention_mask.pt`.

**Tech Stack:** Python 3.11, PyTorch, local Transformers under `deps/transformers/src`, local `src/sparse_llm_cache`, JSON, `torch.save`, pytest-compatible local tests.

---

## Global Constraints

- Do not modify files outside `experiment/scripts/trace/encoder_predictor/`.
- Do not modify `src/`.
- Do not modify `deps/`.
- Do not create a worktree.
- Do not commit git changes.
- Use `/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python` for runtime verification when available.
- Keep all tests and helper scripts inside `experiment/scripts/trace/encoder_predictor/`.

## Files

- Create: `experiment/scripts/trace/encoder_predictor/README.md`
  - User-facing usage guide, trace口径 explanation, training warnings.
- Create: `experiment/scripts/trace/encoder_predictor/__init__.py`
  - Empty package marker for local imports.
- Create: `experiment/scripts/trace/encoder_predictor/export_sparse_cache_encoder_trace.py`
  - Main exporter CLI and implementation.
- Create: `experiment/scripts/trace/encoder_predictor/compare_encoder_traces.py`
  - Trace comparison CLI.
- Create: `experiment/scripts/trace/encoder_predictor/test_encoder_predictor_trace.py`
  - Local unit tests for argument/path logic, tensor padding, output verification, and comparison masking.
- Existing: `experiment/scripts/trace/encoder_predictor/SPEC.md`
  - Requirements source for implementation.
- Existing: `experiment/scripts/trace/encoder_predictor/PLAN.md`
  - This plan.

## Task 1: Local Package, README, and Test Skeleton

**Files:**

- Create: `experiment/scripts/trace/encoder_predictor/__init__.py`
- Create: `experiment/scripts/trace/encoder_predictor/README.md`
- Create: `experiment/scripts/trace/encoder_predictor/test_encoder_predictor_trace.py`

- [ ] **Step 1: Create package marker**

Create `experiment/scripts/trace/encoder_predictor/__init__.py` with:

```python
"""Sparse-cache-backed encoder predictor trace tools."""
```

- [ ] **Step 2: Create README**

Create `experiment/scripts/trace/encoder_predictor/README.md` with:

```markdown
# Encoder Predictor Sparse Cache Trace

This directory contains trace tools for encoder predictor training data.

The exporter uses the same sparse cache entry point as `examples/small-demo/transformers-app.py`:

```python
sparse_llm_cache.utils.hack_transformers(...)
```

It is intentionally different from pure HuggingFace CPU/GPU offload exporters. Expert parameters may be stored on CPU while idle, but selected experts are staged to GPU before expert forward. The trace is meant to avoid CPU expert computation while still supporting memory-constrained runs.

## Default Trace口径

- `padding=longest`
- `batch_size=1`
- `max_input_tokens=512`
- `max_new_tokens=1`
- `model_torch_dtype=auto`
- `device=cuda:0`
- `num_predict_expert_per_layer=0`
- `reorder_experts=False`
- `early_preempt=False`

The exporter writes tensors padded at save time when needed. Training code must use `attention_mask.pt` to ignore positions where `attention_mask == 0`.

## Export

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
  experiment/scripts/trace/encoder_predictor/export_sparse_cache_encoder_trace.py \
  --model-path experiment/models/google/switch-base-128 \
  --dataset mmlu \
  --task-name professional_law \
  --device cuda:0 \
  --print-status
```

Default output:

```text
experiment/traces/switch-base-128-mmlu-professional_law/encoder_predictor_sparse_cache_trace/
```

## Compare

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
  experiment/scripts/trace/encoder_predictor/compare_encoder_traces.py \
  --left performance_predictor/encoder/ERPP/data/traces/switch-base-128-mmlu-professional_law-erpp \
  --right experiment/traces/switch-base-128-mmlu-professional_law/encoder_predictor_sparse_cache_trace
```

The comparison report separates real-token differences from padding-token differences by using `attention_mask.pt`.
```

- [ ] **Step 3: Create initial tests that define local helper expectations**

Create `experiment/scripts/trace/encoder_predictor/test_encoder_predictor_trace.py` with these tests:

```python
from pathlib import Path

import torch

from export_sparse_cache_encoder_trace import (
    SplitAccumulator,
    derive_model_name,
    resolve_default_output_dir,
    storage_dtype,
)
from compare_encoder_traces import expert_diff_summary


def test_derive_model_name_from_local_path():
    assert derive_model_name(Path("experiment/models/google/switch-base-128")) == "switch-base-128"


def test_resolve_default_output_dir():
    out = resolve_default_output_dir(
        repo_root=Path("/repo"),
        model_path=Path("experiment/models/google/switch-base-128"),
        dataset="mmlu",
        task_name="professional_law",
    )
    assert out == Path("/repo/experiment/traces/switch-base-128-mmlu-professional_law/encoder_predictor_sparse_cache_trace")


def test_storage_dtype_names():
    assert storage_dtype("float32") is torch.float32
    assert storage_dtype("float16") is torch.float16
    assert storage_dtype("bfloat16") is torch.bfloat16


def test_split_accumulator_pads_sequence_dimension():
    acc = SplitAccumulator()
    acc.append(
        input_ids=torch.tensor([[1, 2, 3]], dtype=torch.int64),
        attention_mask=torch.tensor([[1, 1, 1]], dtype=torch.int64),
        layer0_attn_out=torch.ones((1, 3, 2)),
        router_logits=torch.ones((1, 2, 3, 4)),
        expert_selection=torch.ones((1, 2, 3, 1), dtype=torch.int64),
        seq_ids=torch.tensor([0], dtype=torch.int64),
        prompts=["a"],
    )
    acc.append(
        input_ids=torch.tensor([[4, 5]], dtype=torch.int64),
        attention_mask=torch.tensor([[1, 1]], dtype=torch.int64),
        layer0_attn_out=2 * torch.ones((1, 2, 2)),
        router_logits=2 * torch.ones((1, 2, 2, 4)),
        expert_selection=2 * torch.ones((1, 2, 2, 1), dtype=torch.int64),
        seq_ids=torch.tensor([1], dtype=torch.int64),
        prompts=["b"],
    )

    input_ids = acc.cat("input_ids")
    router_logits = acc.cat("router_logits")
    expert_selection = acc.cat("expert_selection")

    assert input_ids.shape == (2, 3)
    assert input_ids[1, 2].item() == 0
    assert router_logits.shape == (2, 2, 3, 4)
    assert torch.all(router_logits[1, :, 2, :] == 0)
    assert expert_selection.shape == (2, 2, 3, 1)
    assert torch.all(expert_selection[1, :, 2, :] == 0)


def test_expert_diff_summary_splits_valid_and_padding():
    left = torch.tensor([[[[1], [2], [3]]]], dtype=torch.int64)
    right = torch.tensor([[[[1], [7], [9]]]], dtype=torch.int64)
    attention_mask = torch.tensor([[1, 0, 0]], dtype=torch.int64)

    summary = expert_diff_summary(left, right, attention_mask)

    assert summary["total_diff"] == 2
    assert summary["valid_diff"] == 0
    assert summary["padding_diff"] == 2
```

- [ ] **Step 4: Run tests and confirm they fail before implementation**

Run:

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python -m pytest \
  experiment/scripts/trace/encoder_predictor/test_encoder_predictor_trace.py -v
```

Expected result at this stage:

```text
ModuleNotFoundError: No module named 'export_sparse_cache_encoder_trace'
```

If pytest is unavailable in the conda environment, run:

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
  experiment/scripts/trace/encoder_predictor/test_encoder_predictor_trace.py
```

Expected result at this stage:

```text
ModuleNotFoundError
```

## Task 2: Exporter Core Utilities and Accumulator

**Files:**

- Create: `experiment/scripts/trace/encoder_predictor/export_sparse_cache_encoder_trace.py`
- Modify: `experiment/scripts/trace/encoder_predictor/test_encoder_predictor_trace.py`

- [ ] **Step 1: Implement import path setup and pure helpers**

Create `export_sparse_cache_encoder_trace.py` with:

```python
#!/usr/bin/env python3
"""Export encoder predictor traces through sparse_llm_cache patched Transformers."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def find_repo_root(start: Optional[Path] = None) -> Path:
    current = Path.cwd() if start is None else Path(start).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "experiment").is_dir() and (candidate / "src").is_dir():
            return candidate
    return Path(__file__).resolve().parents[4]


REPO_ROOT = find_repo_root()
SRC_DIR = REPO_ROOT / "src"
TRANSFORMERS_SRC = REPO_ROOT / "deps" / "transformers" / "src"
for path in (SRC_DIR, TRANSFORMERS_SRC):
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


def storage_dtype(name: str) -> torch.dtype:
    table = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return table[name]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"unsupported storage dtype {name!r}") from exc


def derive_model_name(model_path: Path) -> str:
    return Path(model_path).name


def resolve_default_output_dir(repo_root: Path, model_path: Path, dataset: str, task_name: str) -> Path:
    model_name = derive_model_name(model_path)
    return repo_root / "experiment" / "traces" / f"{model_name}-{dataset}-{task_name}" / "encoder_predictor_sparse_cache_trace"


def resolve_prompt_file(repo_root: Path, dataset: str, task_name: str, split: str) -> Path:
    return repo_root / "experiment" / "datasets" / dataset / task_name / split / "prompt_list.txt"


def load_prompts(path: Path) -> list[str]:
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"prompt file has no non-empty lines: {path}")
    return prompts


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
```

- [ ] **Step 2: Implement split accumulator**

Append this code to `export_sparse_cache_encoder_trace.py`:

```python
@dataclass
class SplitAccumulator:
    input_ids: list[torch.Tensor] = field(default_factory=list)
    attention_mask: list[torch.Tensor] = field(default_factory=list)
    layer0_attn_out: list[torch.Tensor] = field(default_factory=list)
    router_logits: list[torch.Tensor] = field(default_factory=list)
    expert_selection: list[torch.Tensor] = field(default_factory=list)
    seq_ids: list[torch.Tensor] = field(default_factory=list)
    prompt_records: list[dict] = field(default_factory=list)

    def append(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layer0_attn_out: torch.Tensor,
        router_logits: torch.Tensor,
        expert_selection: torch.Tensor,
        seq_ids: torch.Tensor,
        prompts: list[str],
    ) -> None:
        self.input_ids.append(input_ids.detach().cpu())
        self.attention_mask.append(attention_mask.detach().cpu())
        self.layer0_attn_out.append(layer0_attn_out.detach().cpu())
        self.router_logits.append(router_logits.detach().cpu())
        self.expert_selection.append(expert_selection.detach().to(torch.int64).cpu())
        self.seq_ids.append(seq_ids.detach().to(torch.int64).cpu())
        for seq_id, text in zip(seq_ids.tolist(), prompts):
            self.prompt_records.append({"seq_id": int(seq_id), "text": text})

    def total_samples(self) -> int:
        return sum(int(t.shape[0]) for t in self.input_ids)

    @staticmethod
    def _sequence_dim(name: str) -> Optional[int]:
        if name in {"input_ids", "attention_mask", "layer0_attn_out"}:
            return 1
        if name in {"router_logits", "expert_selection"}:
            return 2
        return None

    def cat(self, name: str) -> torch.Tensor:
        values = getattr(self, name)
        if not values:
            raise RuntimeError(f"split accumulator has no tensors for {name}")
        seq_dim = self._sequence_dim(name)
        if seq_dim is None:
            return torch.cat(values, dim=0)
        max_len = max(int(t.shape[seq_dim]) for t in values)
        padded: list[torch.Tensor] = []
        for tensor in values:
            current_len = int(tensor.shape[seq_dim])
            if current_len == max_len:
                padded.append(tensor)
                continue
            shape = list(tensor.shape)
            shape[seq_dim] = max_len
            out = tensor.new_zeros(shape)
            index = [slice(None)] * tensor.dim()
            index[seq_dim] = slice(0, current_len)
            out[tuple(index)] = tensor
            padded.append(out)
        return torch.cat(padded, dim=0)

    def true_token_lengths(self) -> list[int]:
        lengths: list[int] = []
        for attention_mask in self.attention_mask:
            lengths.extend(int(x) for x in attention_mask.to(torch.int64).sum(dim=1).tolist())
        return lengths
```

- [ ] **Step 3: Implement split writer**

Append this code:

```python
class TraceWriter:
    def __init__(self, output_dir: Path, dtype: torch.dtype) -> None:
        self.output_dir = Path(output_dir)
        self.dtype = dtype

    def write_split(self, split: str, acc: SplitAccumulator) -> dict:
        if acc.total_samples() == 0:
            raise ValueError(f"refusing to write empty split {split!r}")
        split_dir = self.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        input_ids = acc.cat("input_ids").to(torch.int64)
        attention_mask = acc.cat("attention_mask").to(torch.int64)
        layer0_attn_out = acc.cat("layer0_attn_out").to(self.dtype)
        router_logits = acc.cat("router_logits").to(self.dtype)
        expert_selection = acc.cat("expert_selection").to(torch.int64)
        seq_ids = acc.cat("seq_ids").to(torch.int64)

        router_probs = torch.softmax(router_logits.float(), dim=-1).to(self.dtype)
        expert_weights = torch.gather(
            router_probs.float(),
            dim=-1,
            index=expert_selection,
        ).to(self.dtype)

        torch.save(input_ids, split_dir / "input_ids.pt")
        torch.save(attention_mask, split_dir / "attention_mask.pt")
        torch.save(layer0_attn_out, split_dir / "layer0_attn_out.pt")
        torch.save(router_logits, split_dir / "router_logits.pt")
        torch.save(router_probs, split_dir / "router_probs.pt")
        torch.save(expert_selection, split_dir / "expert_selection.pt")
        torch.save(expert_weights, split_dir / "expert_weights.pt")
        torch.save(seq_ids, split_dir / "seq_ids.pt")

        with (split_dir / "prompt_texts.jsonl").open("w", encoding="utf-8") as f:
            for record in acc.prompt_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return {
            "num_samples": int(input_ids.shape[0]),
            "true_token_lengths": acc.true_token_lengths(),
            "input_ids_shape": list(input_ids.shape),
            "attention_mask_shape": list(attention_mask.shape),
            "layer0_attn_out_shape": list(layer0_attn_out.shape),
            "router_logits_shape": list(router_logits.shape),
            "router_probs_shape": list(router_probs.shape),
            "expert_selection_shape": list(expert_selection.shape),
            "expert_weights_shape": list(expert_weights.shape),
            "seq_ids_shape": list(seq_ids.shape),
        }
```

- [ ] **Step 4: Run local tests for utilities**

Run:

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python -m pytest \
  experiment/scripts/trace/encoder_predictor/test_encoder_predictor_trace.py -v
```

Expected result after Task 2:

```text
ModuleNotFoundError: No module named 'compare_encoder_traces'
```

The exporter-related tests should no longer fail.

## Task 3: Comparison Utility

**Files:**

- Create: `experiment/scripts/trace/encoder_predictor/compare_encoder_traces.py`
- Modify: `experiment/scripts/trace/encoder_predictor/test_encoder_predictor_trace.py`

- [ ] **Step 1: Implement comparison masks and summaries**

Create `compare_encoder_traces.py` with:

```python
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
        per_layer.append({
            "layer": int(layer),
            "valid_diff": int((layer_diff & layer_mask).sum()),
            "padding_diff": int((layer_diff & ~layer_mask).sum()),
        })
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
            result["per_layer"].append({
                "layer": int(layer),
                "valid_max": float(values.max()) if values.numel() else 0.0,
                "valid_mean": float(values.mean()) if values.numel() else 0.0,
                "valid_elements_gt_1e_2": int((values > 1e-2).sum()) if values.numel() else 0,
            })
    return result
```

- [ ] **Step 2: Implement split comparison and CLI**

Append this code:

```python
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
```

- [ ] **Step 3: Run local tests**

Run:

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python -m pytest \
  experiment/scripts/trace/encoder_predictor/test_encoder_predictor_trace.py -v
```

Expected result after Task 3:

```text
5 passed
```

## Task 4: Sparse Cache Runtime Loader and Hooks

**Files:**

- Modify: `experiment/scripts/trace/encoder_predictor/export_sparse_cache_encoder_trace.py`
- Modify: `experiment/scripts/trace/encoder_predictor/test_encoder_predictor_trace.py`

- [ ] **Step 1: Add model dtype resolver, sparse cache config builder, and parser**

Append this code to `export_sparse_cache_encoder_trace.py`:

```python
def bool_arg(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {value!r}")


def build_sparse_cache_config(args: argparse.Namespace) -> dict:
    return {
        "model_id": str(args.model_path),
        "cache_rate": args.cache_rate,
        "cache_policy": args.cache_policy,
        "per_layer_cache": args.per_layer_cache,
        "num_predict_expert_per_layer": args.num_predict_expert_per_layer,
        "reorder_experts": args.reorder_experts,
        "early_preempt": args.early_preempt,
        "chunk_prefetch": args.chunk_prefetch,
        "predictor_model_path": args.predictor_model_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export sparse-cache-backed encoder predictor traces.")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--train-split", default="test")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--train-prompt-file", type=Path)
    parser.add_argument("--validation-prompt-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--padding", choices=["longest", "max_length"], default="longest")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--storage-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--model-torch-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-rate", type=float, default=0.375)
    parser.add_argument("--cache-policy", default="lru")
    parser.add_argument("--per-layer-cache", type=bool_arg, default=True)
    parser.add_argument("--num-predict-expert-per-layer", type=int, default=0)
    parser.add_argument("--reorder-experts", type=bool_arg, default=False)
    parser.add_argument("--early-preempt", type=bool_arg, default=False)
    parser.add_argument("--chunk-prefetch", type=bool_arg, default=False)
    parser.add_argument("--predictor-model-path")
    parser.add_argument("--assert-gpu-expert-forward", dest="assert_gpu_expert_forward", action="store_true", default=True)
    parser.add_argument("--no-assert-gpu-expert-forward", dest="assert_gpu_expert_forward", action="store_false")
    parser.add_argument("--no-verify", dest="verify", action="store_false", default=True)
    parser.add_argument("--print-status", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Resolve paths/config and exit without loading model")
    return parser
```

- [ ] **Step 2: Add GPU expert forward assertion hook**

Append this code:

```python
class GpuExpertForwardAsserter:
    def __init__(self, expected_device: str) -> None:
        self.expected_device = torch.device(expected_device)
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.observed = 0

    def attach(self, model: torch.nn.Module) -> None:
        for module in model.modules():
            if hasattr(module, "_expert_id") and hasattr(module, "_layer_id"):
                self.handles.append(module.register_forward_pre_hook(self._pre_hook))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _pre_hook(self, module: torch.nn.Module, inputs: tuple) -> None:
        self.observed += 1
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError("expert forward did not receive tensor input")
        if inputs[0].device.type != "cuda":
            raise RuntimeError(f"expert input is not CUDA: {inputs[0].device}")
        for param in module.parameters(recurse=True):
            if param.device.type != "cuda":
                raise RuntimeError(
                    f"expert L{getattr(module, '_layer_id', '?')} E{getattr(module, '_expert_id', '?')} "
                    f"parameter is not CUDA during forward: {param.device}"
                )
            break
```

- [ ] **Step 3: Add runtime class with hooks**

Append this code:

```python
class SparseCacheTraceRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model = None
        self.tokenizer = None
        self.input_device = torch.device(args.device)
        self.layer0_attn_out: torch.Tensor | None = None
        self.router_logits_by_layer: dict[int, torch.Tensor] = {}
        self.expert_selection_by_layer: dict[int, torch.Tensor] = {}
        self.router_layer_to_model_block: list[int] = []
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.gpu_asserter: GpuExpertForwardAsserter | None = None

    def status(self, message: str) -> None:
        if self.args.print_status:
            print(f"[encoder-predictor-trace] {message}", flush=True)

    def load(self) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HUGGINGFACE_OFFLINE", "1")
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if self.input_device.type == "cuda":
            torch.cuda.set_device(self.input_device)

        import sparse_llm_cache
        from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration

        cache_config = build_sparse_cache_config(self.args)
        sparse_llm_cache.utils.hack_transformers(
            **cache_config,
            pin_memory=True,
            enable_model_timer=False,
        )

        self.status(f"loading model from {self.args.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = SwitchTransformersForConditionalGeneration.from_pretrained(
            self.args.model_path,
            torch_dtype=self.args.model_torch_dtype,
            local_files_only=True,
            device_map=0,
            trust_remote_code=True,
        )
        if getattr(self.model.config, "is_encoder_decoder", False) and self.model.config.decoder_start_token_id is None:
            self.model.config.decoder_start_token_id = self.tokenizer.pad_token_id
            self.model.generation_config.decoder_start_token_id = self.tokenizer.pad_token_id
        self.model.eval()
        self.attach_trace_hooks()
        if self.args.assert_gpu_expert_forward:
            self.gpu_asserter = GpuExpertForwardAsserter(self.args.device)
            self.gpu_asserter.attach(self.model)

    def attach_trace_hooks(self) -> None:
        from transformers.models.switch_transformers.modeling_switch_transformers import SwitchTransformersSparseMLP

        encoder_blocks = getattr(self.model.encoder, "block", None)
        if encoder_blocks is None:
            raise RuntimeError("model has no encoder.block")
        self.handles.append(encoder_blocks[0].layer[0].register_forward_hook(self._layer0_attention_hook))

        router_layer_id = 0
        self.router_layer_to_model_block.clear()
        for block_id, block in enumerate(self.model.encoder.block):
            for layer in block.layer:
                mlp = getattr(layer, "mlp", None)
                if isinstance(mlp, SwitchTransformersSparseMLP):
                    mlp._encoder_predictor_trace_layer_id = router_layer_id
                    self.router_layer_to_model_block.append(int(block_id))
                    self.handles.append(mlp.register_forward_hook(self._sparse_mlp_hook))
                    router_layer_id += 1
        if router_layer_id == 0:
            raise RuntimeError("found no encoder SwitchTransformersSparseMLP modules")

    def _layer0_attention_hook(self, module, inputs, output) -> None:
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(hidden, torch.Tensor) or hidden.dim() != 3:
            raise RuntimeError(f"layer0 attention hook expected [B,T,H], got {type(hidden).__name__}")
        self.layer0_attn_out = hidden.detach().cpu()

    def _sparse_mlp_hook(self, module, inputs, output) -> None:
        layer_id = int(module._encoder_predictor_trace_layer_id)
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError("SparseMLP hook expected hidden states in inputs[0]")
        hidden = inputs[0]
        bsz, seq_len = int(hidden.shape[0]), int(hidden.shape[1])
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            raise RuntimeError("SparseMLP output must contain router tuple")
        router_tuple = output[1]
        if not isinstance(router_tuple, (tuple, list)) or len(router_tuple) < 2:
            raise RuntimeError("SparseMLP router tuple must contain (router_logits, expert_index)")
        router_logits = router_tuple[0]
        expert_index = router_tuple[1]
        if router_logits.dim() == 2:
            router_logits = router_logits.view(bsz, seq_len, -1)
        if expert_index.dim() == 2:
            expert_index = expert_index.unsqueeze(-1)
        self.router_logits_by_layer[layer_id] = router_logits.detach().cpu()
        self.expert_selection_by_layer[layer_id] = expert_index.detach().to(torch.int64).cpu()
```

- [ ] **Step 4: Run syntax check**

Run:

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python -m py_compile \
  experiment/scripts/trace/encoder_predictor/export_sparse_cache_encoder_trace.py
```

Expected:

```text
no output, exit code 0
```

## Task 5: Batch Execution, Generate Path, Verification, and CLI Main

**Files:**

- Modify: `experiment/scripts/trace/encoder_predictor/export_sparse_cache_encoder_trace.py`

- [ ] **Step 1: Add batch execution methods**

Append this code inside `SparseCacheTraceRunner` after `_sparse_mlp_hook`:

```python
    def run_split(self, prompts: list[str], *, batch_size: int, seq_id_start: int = 0) -> SplitAccumulator:
        if self.model is None:
            self.load()
        acc = SplitAccumulator()
        total_batches = (len(prompts) + batch_size - 1) // batch_size
        for batch_idx, start in enumerate(range(0, len(prompts), batch_size), start=1):
            batch_prompts = prompts[start:start + batch_size]
            seq_ids = torch.arange(seq_id_start + start, seq_id_start + start + len(batch_prompts), dtype=torch.int64)
            self.status(f"batch {batch_idx}/{total_batches}: prompts={len(batch_prompts)} seq_id={int(seq_ids[0])}..{int(seq_ids[-1])}")
            batch = self.run_batch(batch_prompts)
            acc.append(seq_ids=seq_ids, prompts=batch_prompts, **batch)
        return acc

    def run_batch(self, prompts: list[str]) -> dict[str, torch.Tensor]:
        self.layer0_attn_out = None
        self.router_logits_by_layer.clear()
        self.expert_selection_by_layer.clear()

        enc_inputs = self.tokenizer(
            prompts,
            padding=self.args.padding,
            truncation=True,
            max_length=self.args.max_input_tokens,
            return_tensors="pt",
        )
        input_ids = enc_inputs["input_ids"].to(self.input_device)
        attention_mask = enc_inputs["attention_mask"].to(self.input_device)

        with torch.inference_mode():
            self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.args.max_new_tokens,
                do_sample=False,
            )

        if self.layer0_attn_out is None:
            raise RuntimeError("missing layer0 attention output; hook did not fire")
        num_layers = len(self.router_layer_to_model_block)
        missing = [i for i in range(num_layers) if i not in self.router_logits_by_layer]
        if missing:
            raise RuntimeError(f"missing encoder router logits for layers {missing}")

        router_logits = torch.stack([self.router_logits_by_layer[i] for i in range(num_layers)], dim=1)
        expert_selection = torch.stack([self.expert_selection_by_layer[i] for i in range(num_layers)], dim=1)

        expected_bt = tuple(input_ids.shape)
        if tuple(self.layer0_attn_out.shape[:2]) != expected_bt:
            raise RuntimeError(f"layer0 attention shape {tuple(self.layer0_attn_out.shape)} does not align with input_ids {expected_bt}")
        if tuple(router_logits.shape[:1] + router_logits.shape[2:3]) != expected_bt:
            raise RuntimeError(f"router logits shape {tuple(router_logits.shape)} does not align with input_ids {expected_bt}")
        if tuple(expert_selection.shape[:1] + expert_selection.shape[2:3]) != expected_bt:
            raise RuntimeError(f"expert selection shape {tuple(expert_selection.shape)} does not align with input_ids {expected_bt}")

        return {
            "input_ids": input_ids.detach().cpu(),
            "attention_mask": attention_mask.detach().cpu(),
            "layer0_attn_out": self.layer0_attn_out,
            "router_logits": router_logits,
            "expert_selection": expert_selection,
        }
```

- [ ] **Step 2: Add verification function**

Append this code outside the class:

```python
def load_saved_tensor(path: Path) -> torch.Tensor:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def verify_trace(output_dir: Path, metadata: dict) -> dict:
    required = [
        "attention_mask.pt",
        "expert_selection.pt",
        "expert_weights.pt",
        "input_ids.pt",
        "layer0_attn_out.pt",
        "prompt_texts.jsonl",
        "router_logits.pt",
        "router_probs.pt",
        "seq_ids.pt",
    ]
    result = {"ok": True, "splits": {}}
    num_experts = int(metadata["model_config"]["num_experts"])
    top_k = int(metadata["model_config"]["num_selected_experts"])
    for split in ("train", "validation"):
        split_dir = output_dir / split
        for name in required:
            path = split_dir / name
            if not path.exists():
                raise AssertionError(f"missing {split}/{name}")
        input_ids = load_saved_tensor(split_dir / "input_ids.pt")
        attention_mask = load_saved_tensor(split_dir / "attention_mask.pt")
        router_logits = load_saved_tensor(split_dir / "router_logits.pt")
        router_probs = load_saved_tensor(split_dir / "router_probs.pt")
        expert_selection = load_saved_tensor(split_dir / "expert_selection.pt")
        expert_weights = load_saved_tensor(split_dir / "expert_weights.pt")
        seq_ids = load_saved_tensor(split_dir / "seq_ids.pt")
        if input_ids.shape != attention_mask.shape:
            raise AssertionError(f"{split}: input_ids shape does not match attention_mask")
        if router_logits.shape != router_probs.shape:
            raise AssertionError(f"{split}: router_logits shape does not match router_probs")
        if router_logits.shape[:3] != expert_selection.shape[:3]:
            raise AssertionError(f"{split}: router logits do not align with expert_selection")
        if expert_selection.shape != expert_weights.shape:
            raise AssertionError(f"{split}: expert_selection does not align with expert_weights")
        if expert_selection.shape[-1] != top_k:
            raise AssertionError(f"{split}: expert top_k mismatch")
        if int(expert_selection.min()) < 0 or int(expert_selection.max()) >= num_experts:
            raise AssertionError(f"{split}: expert ids out of range")
        prompt_lines = (split_dir / "prompt_texts.jsonl").read_text(encoding="utf-8").splitlines()
        if len(prompt_lines) != int(input_ids.shape[0]):
            raise AssertionError(f"{split}: prompt line count mismatch")
        if seq_ids.shape[0] != input_ids.shape[0]:
            raise AssertionError(f"{split}: seq_ids length mismatch")
        result["splits"][split] = {
            "num_samples": int(input_ids.shape[0]),
            "seq_len": int(input_ids.shape[1]),
            "num_router_layers": int(router_logits.shape[1]),
            "num_experts": int(router_logits.shape[-1]),
        }
    return result
```

- [ ] **Step 3: Add main function**

Append this code:

```python
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = REPO_ROOT
    args.model_path = Path(args.model_path)
    train_prompt_file = args.train_prompt_file or resolve_prompt_file(repo_root, args.dataset, args.task_name, args.train_split)
    validation_prompt_file = args.validation_prompt_file or resolve_prompt_file(repo_root, args.dataset, args.task_name, args.validation_split)
    output_dir = args.output_dir or resolve_default_output_dir(repo_root, args.model_path, args.dataset, args.task_name)
    output_dir = Path(output_dir)

    resolved = {
        "repo_root": str(repo_root),
        "model_path": str(args.model_path),
        "train_prompt_file": str(train_prompt_file),
        "validation_prompt_file": str(validation_prompt_file),
        "output_dir": str(output_dir),
        "sparse_cache_config": build_sparse_cache_config(args),
    }
    if args.dry_run:
        print(json.dumps(resolved, indent=2, ensure_ascii=False))
        return 0

    start = time.time()
    train_prompts = load_prompts(train_prompt_file)
    validation_prompts = load_prompts(validation_prompt_file)

    runner = SparseCacheTraceRunner(args)
    runner.load()
    writer = TraceWriter(output_dir, storage_dtype(args.storage_dtype))
    train_acc = runner.run_split(train_prompts, batch_size=args.batch_size, seq_id_start=0)
    validation_acc = runner.run_split(validation_prompts, batch_size=args.batch_size, seq_id_start=0)

    model_config = {
        "model_type": getattr(runner.model.config, "model_type", None),
        "hidden_size": int(getattr(runner.model.config, "hidden_size")),
        "num_experts": int(getattr(runner.model.config, "num_experts")),
        "num_selected_experts": int(getattr(runner.model.config, "num_selected_experts")),
        "num_sparse_encoder_layers": int(getattr(runner.model.config, "num_sparse_encoder_layers")),
        "num_sparse_decoder_layers": int(getattr(runner.model.config, "num_sparse_decoder_layers")),
        "encoder_sparse_step": int(getattr(runner.model.config, "encoder_sparse_step")),
        "decoder_sparse_step": int(getattr(runner.model.config, "decoder_sparse_step")),
    }
    metadata = {
        "args": vars(args),
        "resolved": resolved,
        "device": args.device,
        "padding": args.padding,
        "batch_size": args.batch_size,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "storage_dtype": args.storage_dtype,
        "model_torch_dtype": args.model_torch_dtype,
        "model_config": model_config,
        "router_layer_to_model_block": runner.router_layer_to_model_block,
        "gpu_expert_forward_assertion": {
            "enabled": bool(args.assert_gpu_expert_forward),
            "observed_forwards": int(runner.gpu_asserter.observed) if runner.gpu_asserter else 0,
        },
        "splits": {
            "train": writer.write_split("train", train_acc),
            "validation": writer.write_split("validation", validation_acc),
        },
    }
    if args.verify:
        metadata["verification"] = verify_trace(output_dir, metadata)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metadata.json", metadata)
    write_json(output_dir / "run_log.json", {
        "start_time": start,
        "end_time": time.time(),
        "elapsed_seconds": time.time() - start,
        "python": sys.executable,
        "cuda_available": torch.cuda.is_available(),
        "device": args.device,
        "cuda_device_name": torch.cuda.get_device_name(torch.device(args.device)) if torch.cuda.is_available() and str(args.device).startswith("cuda") else None,
        "output_dir": str(output_dir),
        "num_train_prompts": len(train_prompts),
        "num_validation_prompts": len(validation_prompts),
    })
    print(f"wrote encoder predictor sparse cache trace to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run dry-run**

Run:

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
  experiment/scripts/trace/encoder_predictor/export_sparse_cache_encoder_trace.py \
  --model-path experiment/models/google/switch-base-128 \
  --dataset mmlu \
  --task-name professional_law \
  --dry-run
```

Expected output contains:

```text
"output_dir": "/mnt/huwf5/promoe/experiment/traces/switch-base-128-mmlu-professional_law/encoder_predictor_sparse_cache_trace"
```

- [ ] **Step 5: Run syntax and local tests**

Run:

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python -m py_compile \
  experiment/scripts/trace/encoder_predictor/export_sparse_cache_encoder_trace.py \
  experiment/scripts/trace/encoder_predictor/compare_encoder_traces.py
```

Expected:

```text
no output, exit code 0
```

Run:

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python -m pytest \
  experiment/scripts/trace/encoder_predictor/test_encoder_predictor_trace.py -v
```

Expected:

```text
5 passed
```

## Task 6: Real-Run Smoke Test and Trace Comparison

**Files:**

- Modify only if needed: `experiment/scripts/trace/encoder_predictor/export_sparse_cache_encoder_trace.py`
- Modify only if needed: `experiment/scripts/trace/encoder_predictor/README.md`

- [ ] **Step 1: Run a one-sample smoke export using explicit prompt files**

Create temporary prompt files under `/tmp` with:

```bash
printf 'What is the answer?\\n' > /tmp/encoder_predictor_trace_train_prompts.txt
printf 'What is the answer?\\n' > /tmp/encoder_predictor_trace_validation_prompts.txt
```

Run:

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
  experiment/scripts/trace/encoder_predictor/export_sparse_cache_encoder_trace.py \
  --model-path experiment/models/google/switch-base-128 \
  --dataset mmlu \
  --task-name professional_law \
  --train-prompt-file /tmp/encoder_predictor_trace_train_prompts.txt \
  --validation-prompt-file /tmp/encoder_predictor_trace_validation_prompts.txt \
  --output-dir /tmp/encoder_predictor_sparse_cache_trace_smoke \
  --device cuda:0 \
  --batch-size 1 \
  --max-input-tokens 512 \
  --max-new-tokens 1 \
  --print-status
```

Expected:

```text
wrote encoder predictor sparse cache trace to /tmp/encoder_predictor_sparse_cache_trace_smoke
```

If this fails with CUDA OOM, record the OOM in the final implementation report and do not weaken dtype or move expert compute to CPU.

- [ ] **Step 2: Inspect smoke metadata**

Run:

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python - <<'PY'
import json
from pathlib import Path
meta = json.loads(Path('/tmp/encoder_predictor_sparse_cache_trace_smoke/metadata.json').read_text())
print(meta['sparse_cache_config'] if 'sparse_cache_config' in meta else meta['resolved']['sparse_cache_config'])
print(meta['gpu_expert_forward_assertion'])
print(meta['splits']['train']['router_logits_shape'])
PY
```

Expected:

```text
num_predict_expert_per_layer is 0
observed_forwards is greater than 0
router_logits_shape has four dimensions
```

- [ ] **Step 3: Run comparison utility against old ERPP trace if a full export exists**

Run:

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
  experiment/scripts/trace/encoder_predictor/compare_encoder_traces.py \
  --left performance_predictor/encoder/ERPP/data/traces/switch-base-128-mmlu-professional_law-erpp \
  --right experiment/traces/switch-base-128-mmlu-professional_law/encoder_predictor_sparse_cache_trace
```

Expected:

```text
The report prints train and validation sections.
Input equality fields are visible.
Expert differences are split into valid_diff and padding_diff.
```

If the full export path does not exist yet, skip this command and state that comparison awaits the real export.

## Task 7: Final Self-Review

**Files:**

- Review only: `experiment/scripts/trace/encoder_predictor/*`

- [ ] **Step 1: Enforce file boundary**

Run:

```bash
git diff --name-only
```

Expected: every changed file path starts with:

```text
experiment/scripts/trace/encoder_predictor/
```

If any changed file is outside that directory, stop and report the violation instead of modifying it.

- [ ] **Step 2: Confirm no forbidden terms in implementation choices**

Run:

```bash
grep -R -n "device_map=.auto.\\|device_map=\\\"auto\\\"\\|dispatch_model\\|infer_auto_device_map" \
  experiment/scripts/trace/encoder_predictor || true
```

Expected: no exporter implementation uses auto device map or Accelerate CPU offload.

- [ ] **Step 3: Confirm sparse cache entry point is used**

Run:

```bash
grep -R -n "hack_transformers" experiment/scripts/trace/encoder_predictor
```

Expected: `export_sparse_cache_encoder_trace.py` calls `sparse_llm_cache.utils.hack_transformers(...)`.

- [ ] **Step 4: Final verification commands**

Run:

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python -m py_compile \
  experiment/scripts/trace/encoder_predictor/export_sparse_cache_encoder_trace.py \
  experiment/scripts/trace/encoder_predictor/compare_encoder_traces.py
```

Run:

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python -m pytest \
  experiment/scripts/trace/encoder_predictor/test_encoder_predictor_trace.py -v
```

Expected: syntax check exits 0 and local tests pass.

## Subagent Execution Protocol

When implementation starts:

1. The controller reads this plan once.
2. The controller creates a task list from Tasks 1 through 7.
3. For each task, dispatch one fresh implementer subagent with:
   - the full text of that task;
   - the global constraints;
   - the relevant spec sections from `SPEC.md`;
   - a hard instruction that it may only edit `experiment/scripts/trace/encoder_predictor/`.
4. After each implementer returns, dispatch a spec-review subagent.
5. Only after spec review passes, dispatch a code-quality review subagent.
6. If a reviewer finds issues, send the same task back for fixes before moving to the next task.
7. Do not commit.
8. Do not ask the user between tasks unless the plan is blocked by a real ambiguity or a forbidden edit would be required.

## Plan Self-Review

- Spec coverage: the plan covers exporter CLI, sparse cache runtime, GPU expert forward assertion, trace outputs, metadata, comparison utility, local tests, and documentation.
- Placeholder scan: no task uses unresolved `TBD`, `TODO`, or unspecified implementation steps.
- Boundary check: every planned create/modify path is under `experiment/scripts/trace/encoder_predictor/`.
- Git policy: all commit steps from the generic Superpowers plan template are intentionally omitted because the user explicitly forbids commits.
