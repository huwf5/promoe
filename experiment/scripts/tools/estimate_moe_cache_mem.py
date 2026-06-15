#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
  sys.path.insert(0, str(SRC_ROOT))


GIB = 1024 ** 3
MIB = 1024 ** 2


def round_like_cpp(value: float) -> int:
  return math.floor(float(value) + 0.5)


def parse_bytes(value: str) -> int:
  text = str(value).strip()
  if not text:
    raise ValueError("empty byte value")
  units = {
    "b": 1,
    "bytes": 1,
    "kib": 1024,
    "kb": 1000,
    "mib": MIB,
    "mb": 1000 ** 2,
    "gib": GIB,
    "gb": 1000 ** 3,
  }
  lowered = text.lower()
  for unit in sorted(units, key=len, reverse=True):
    if lowered.endswith(unit):
      number = lowered[: -len(unit)].strip()
      return int(float(number) * units[unit])
  return int(float(text))


def format_bytes(num_bytes: int) -> str:
  return f"{num_bytes / GIB:.4f} GiB ({num_bytes / MIB:.2f} MiB)"


@dataclass(frozen=True)
class CacheEstimate:
  cache_rate: float
  total_experts: int
  cache_slots: int
  max_single_expert_bytes: int
  cache_bytes: int

  @property
  def effective_cache_rate(self) -> float:
    if self.total_experts <= 0:
      return 0.0
    return self.cache_slots / self.total_experts


@dataclass(frozen=True)
class ExpertSize:
  name: str
  layer_id: int | None
  expert_id: int | None
  bytes: int


def estimate_cache_memory(
    *,
    max_single_expert_bytes: int,
    num_moe_layer: int,
    num_expert_per_layer: int,
    cache_rate: float,
) -> CacheEstimate:
  if max_single_expert_bytes <= 0:
    raise ValueError("max_single_expert_bytes must be > 0")
  if num_moe_layer <= 0:
    raise ValueError("num_moe_layer must be > 0")
  if num_expert_per_layer <= 0:
    raise ValueError("num_expert_per_layer must be > 0")
  if cache_rate < 0:
    raise ValueError("cache_rate must be >= 0")

  total_experts = int(num_moe_layer) * int(num_expert_per_layer)
  cache_slots = round_like_cpp(float(cache_rate) * total_experts)
  return CacheEstimate(
    cache_rate=float(cache_rate),
    total_experts=total_experts,
    cache_slots=cache_slots,
    max_single_expert_bytes=int(max_single_expert_bytes),
    cache_bytes=cache_slots * int(max_single_expert_bytes),
  )


def max_cache_rate_for_budget(
    *,
    budget_bytes: int,
    max_single_expert_bytes: int,
    num_moe_layer: int,
    num_expert_per_layer: int,
) -> tuple[int, float]:
  if budget_bytes < 0:
    raise ValueError("budget_bytes must be >= 0")
  estimate_cache_memory(
    max_single_expert_bytes=max_single_expert_bytes,
    num_moe_layer=num_moe_layer,
    num_expert_per_layer=num_expert_per_layer,
    cache_rate=0.0,
  )
  total_experts = int(num_moe_layer) * int(num_expert_per_layer)
  max_slots = min(total_experts, int(budget_bytes) // int(max_single_expert_bytes))
  return max_slots, max_slots / total_experts


def module_prefetch_nbytes(module) -> int:
  from sparse_llm_cache.utils import param_buffer_name_to_prefetch

  total = 0
  for _, tensor in module.named_parameters():
    total += int(tensor.numel()) * int(tensor.element_size())
  for name, tensor in module.named_buffers():
    if param_buffer_name_to_prefetch(name):
      total += int(tensor.numel()) * int(tensor.element_size())
  return total


def expert_sizes_from_model(model, adapter) -> list[ExpertSize]:
  sizes: list[ExpertSize] = []
  for name, module in model.named_modules():
    if not adapter.expert_name_filter(name):
      continue
    meta = adapter.expert_meta_parser(name)
    layer_id = int(meta[0]) if meta is not None else getattr(module, "_layer_id", None)
    expert_id = int(meta[1]) if meta is not None else getattr(module, "_expert_id", None)
    sizes.append(
      ExpertSize(
        name=name,
        layer_id=layer_id,
        expert_id=expert_id,
        bytes=module_prefetch_nbytes(module),
      )
    )
  if not sizes:
    raise ValueError("no expert modules matched the selected model adapter")
  return sizes


def load_model_for_sizing(model_id: str, model_revision: str | None, local_files_only: bool):
  from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, SwitchTransformersForConditionalGeneration

  from sparse_llm_cache.utils.model_loading import (
    is_nllb_moe_model_id,
    is_switch_model_id,
    resolve_transformers_module_model_path,
  )

  model_load_id = resolve_transformers_module_model_path(model_id, repo_root=REPO_ROOT)
  if is_switch_model_id(model_id):
    model_cls = SwitchTransformersForConditionalGeneration
  elif is_nllb_moe_model_id(model_id):
    model_cls = AutoModelForSeq2SeqLM
  else:
    model_cls = AutoModelForCausalLM

  kwargs = {
    "trust_remote_code": True,
    "local_files_only": local_files_only,
  }
  if model_revision:
    kwargs["revision"] = model_revision
  return model_cls.from_pretrained(model_load_id, torch_dtype="auto", device_map={"": "cpu"}, **kwargs)


def summarize_experts(sizes: Iterable[ExpertSize]) -> dict[str, object]:
  materialized = list(sizes)
  largest = max(materialized, key=lambda item: item.bytes)
  smallest = min(materialized, key=lambda item: item.bytes)
  return {
    "count": len(materialized),
    "max": asdict(largest),
    "min": asdict(smallest),
    "all_same_size": smallest.bytes == largest.bytes,
  }


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Estimate Promoe MoE expert cache GPU memory from expert tensor sizes.",
  )
  parser.add_argument("--model_id", help="Hugging Face/local model id to inspect.")
  parser.add_argument("--model_revision")
  parser.add_argument("--allow_download", action="store_true", help="Allow Transformers to fetch missing model files.")
  parser.add_argument("--expert_bytes", help="Single expert size, e.g. 128MiB. Skips model loading.")
  parser.add_argument("--num_moe_layer", type=int, help="Required with --expert_bytes.")
  parser.add_argument("--num_expert_per_layer", type=int, help="Required with --expert_bytes.")
  parser.add_argument("--cache_rate", type=float, action="append", required=True, help="May be passed multiple times.")
  parser.add_argument("--gpu_mem_gb", type=float, help="Optional GPU memory budget for reverse cache_rate calculation.")
  parser.add_argument("--reserved_gb", type=float, default=0.0, help="Memory to reserve for non-cache runtime.")
  parser.add_argument("--safety_fraction", type=float, default=0.90, help="Fraction of remaining budget usable by cache.")
  parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
  return parser


def main(argv: list[str] | None = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)

  if args.expert_bytes:
    if args.num_moe_layer is None or args.num_expert_per_layer is None:
      parser.error("--expert_bytes requires --num_moe_layer and --num_expert_per_layer")
    max_expert_bytes = parse_bytes(args.expert_bytes)
    num_moe_layer = args.num_moe_layer
    num_expert_per_layer = args.num_expert_per_layer
    expert_summary = {
      "source": "expert_bytes",
      "max": {"bytes": max_expert_bytes},
      "all_same_size": None,
    }
  else:
    if not args.model_id:
      parser.error("pass either --model_id or --expert_bytes")
    from sparse_llm_cache.model_adapters import get_model_adapter

    model = load_model_for_sizing(args.model_id, args.model_revision, not args.allow_download)
    adapter = get_model_adapter(model, args.model_id)
    sizes = expert_sizes_from_model(model, adapter)
    expert_summary = summarize_experts(sizes)
    max_expert_bytes = int(expert_summary["max"]["bytes"])
    num_moe_layer = int(adapter.num_moe_layer)
    num_expert_per_layer = int(adapter.num_expert_per_layer)

  estimates = [
    estimate_cache_memory(
      max_single_expert_bytes=max_expert_bytes,
      num_moe_layer=num_moe_layer,
      num_expert_per_layer=num_expert_per_layer,
      cache_rate=rate,
    )
    for rate in args.cache_rate
  ]

  reverse = None
  if args.gpu_mem_gb is not None:
    if args.gpu_mem_gb <= 0:
      parser.error("--gpu_mem_gb must be > 0")
    if args.reserved_gb < 0:
      parser.error("--reserved_gb must be >= 0")
    if args.safety_fraction <= 0 or args.safety_fraction > 1:
      parser.error("--safety_fraction must be in (0, 1]")
    budget_bytes = max(0, int((args.gpu_mem_gb - args.reserved_gb) * GIB * args.safety_fraction))
    max_slots, max_rate = max_cache_rate_for_budget(
      budget_bytes=budget_bytes,
      max_single_expert_bytes=max_expert_bytes,
      num_moe_layer=num_moe_layer,
      num_expert_per_layer=num_expert_per_layer,
    )
    reverse = {
      "gpu_mem_gb": args.gpu_mem_gb,
      "reserved_gb": args.reserved_gb,
      "safety_fraction": args.safety_fraction,
      "cache_budget_bytes": budget_bytes,
      "max_cache_slots": max_slots,
      "max_cache_rate": max_rate,
    }

  result = {
    "num_moe_layer": num_moe_layer,
    "num_expert_per_layer": num_expert_per_layer,
    "total_experts": num_moe_layer * num_expert_per_layer,
    "max_single_expert_bytes": max_expert_bytes,
    "max_single_expert_human": format_bytes(max_expert_bytes),
    "expert_summary": expert_summary,
    "estimates": [asdict(item) | {"cache_human": format_bytes(item.cache_bytes), "effective_cache_rate": item.effective_cache_rate} for item in estimates],
    "reverse_budget": reverse,
  }

  if args.json:
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0

  print(f"num_moe_layer: {num_moe_layer}")
  print(f"num_expert_per_layer: {num_expert_per_layer}")
  print(f"total_experts: {num_moe_layer * num_expert_per_layer}")
  print(f"max_single_expert: {format_bytes(max_expert_bytes)}")
  if expert_summary.get("source") != "expert_bytes":
    print(f"expert_count_scanned: {expert_summary['count']}")
    print(f"all_experts_same_size: {expert_summary['all_same_size']}")
    print(f"max_expert_name: {expert_summary['max']['name']}")
  for estimate in estimates:
    print("")
    print(f"cache_rate: {estimate.cache_rate}")
    print(f"cache_slots: {estimate.cache_slots}")
    print(f"effective_cache_rate: {estimate.effective_cache_rate:.8f}")
    print(f"cache_memory: {format_bytes(estimate.cache_bytes)}")
  if reverse is not None:
    print("")
    print("reverse_budget:")
    print(f"cache_budget: {format_bytes(reverse['cache_budget_bytes'])}")
    print(f"max_cache_slots: {reverse['max_cache_slots']}")
    print(f"max_cache_rate: {reverse['max_cache_rate']:.8f}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
