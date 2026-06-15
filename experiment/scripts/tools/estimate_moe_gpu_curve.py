#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
for path in (str(TOOLS_ROOT), str(SRC_ROOT)):
  if path not in sys.path:
    sys.path.insert(0, path)

from estimate_moe_cache_mem import (  # noqa: E402
  GIB,
  estimate_cache_memory,
  expert_sizes_from_model,
  format_bytes,
  load_model_for_sizing,
  parse_bytes,
  summarize_experts,
)


@dataclass(frozen=True)
class CurvePoint:
  cache_rate: float
  cache_slots: int
  effective_cache_rate: float
  cache_bytes: int
  cache_gib: float
  lower_bound_total_bytes: int
  lower_bound_total_gib: float
  estimated_total_bytes: int
  estimated_total_gib: float


def parse_float_list(value: str) -> list[float]:
  items = [item.strip() for item in str(value).split(",")]
  if not items or any(item == "" for item in items):
    raise ValueError(f"invalid comma-separated float list: {value!r}")
  return [float(item) for item in items]


def sanitize_filename(value: str) -> str:
  sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
  sanitized = sanitized.strip("-._")
  return sanitized or "model"


def default_model_output_name(model_id: str) -> str:
  parts = [part for part in re.split(r"[\/]+", str(model_id).strip()) if part and part != "."]
  if len(parts) >= 2:
    return sanitize_filename(f"{parts[-2]}-{parts[-1]}")
  if parts:
    return sanitize_filename(parts[-1])
  return "model"


def build_curve_points(
    *,
    cache_rates: list[float],
    max_single_expert_bytes: int,
    num_moe_layer: int,
    num_expert_per_layer: int,
    base_runtime_bytes: int,
    safety_margin_bytes: int,
) -> list[CurvePoint]:
  points: list[CurvePoint] = []
  for cache_rate in cache_rates:
    estimate = estimate_cache_memory(
      max_single_expert_bytes=max_single_expert_bytes,
      num_moe_layer=num_moe_layer,
      num_expert_per_layer=num_expert_per_layer,
      cache_rate=cache_rate,
    )
    lower_bound_total = int(base_runtime_bytes) + int(estimate.cache_bytes)
    total = lower_bound_total + int(safety_margin_bytes)
    points.append(
      CurvePoint(
        cache_rate=float(cache_rate),
        cache_slots=int(estimate.cache_slots),
        effective_cache_rate=float(estimate.effective_cache_rate),
        cache_bytes=int(estimate.cache_bytes),
        cache_gib=estimate.cache_bytes / GIB,
        lower_bound_total_bytes=lower_bound_total,
        lower_bound_total_gib=lower_bound_total / GIB,
        estimated_total_bytes=total,
        estimated_total_gib=total / GIB,
      )
    )
  return points


def fit_columns(point: CurvePoint, gpu_sizes_gb: list[float]) -> dict[str, bool]:
  return {f"fit_{size:g}g": point.estimated_total_gib <= float(size) for size in gpu_sizes_gb}


def write_csv(path: Path, points: list[CurvePoint], gpu_sizes_gb: list[float]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  fit_keys = [f"fit_{size:g}g" for size in gpu_sizes_gb]
  fieldnames = [
    "cache_rate",
    "cache_slots",
    "effective_cache_rate",
    "cache_gib",
    "lower_bound_total_gib",
    "safety_margin_gib",
    "estimated_total_gib",
    *fit_keys,
  ]
  with path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for point in points:
      row = {
        "cache_rate": point.cache_rate,
        "cache_slots": point.cache_slots,
        "effective_cache_rate": point.effective_cache_rate,
        "cache_gib": point.cache_gib,
        "lower_bound_total_gib": point.lower_bound_total_gib,
        "safety_margin_gib": max(0.0, point.estimated_total_gib - point.lower_bound_total_gib),
        "estimated_total_gib": point.estimated_total_gib,
      }
      row.update(fit_columns(point, gpu_sizes_gb))
      writer.writerow(row)


def write_plot(path: Path, points: list[CurvePoint], gpu_sizes_gb: list[float]) -> bool:
  try:
    import matplotlib.pyplot as plt
  except Exception:
    return False

  path.parent.mkdir(parents=True, exist_ok=True)
  x = [point.cache_rate for point in points]
  y = [point.estimated_total_gib for point in points]
  fig, ax = plt.subplots(figsize=(9, 5.2))
  ax.plot(x, y, marker="o", linewidth=2, label="total GPU memory lower bound" if all(point.estimated_total_gib == point.lower_bound_total_gib for point in points) else "estimated total GPU memory")
  for size in gpu_sizes_gb:
    ax.axhline(float(size), linestyle="--", linewidth=1, label=f"{size:g} GiB GPU")
  ax.set_xlabel("cache_rate")
  ax.set_ylabel("estimated total GPU memory (GiB)")
  ax.set_title("MoE cache GPU memory estimate")
  ax.grid(True, alpha=0.25)
  ax.legend()
  fig.tight_layout()
  fig.savefig(path, dpi=160)
  plt.close(fig)
  return True


def _is_under_prefix(name: str, prefixes: set[str]) -> bool:
  return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


def cuda_device_index(device: str | int) -> int:
  text = str(device).strip()
  if text.isdigit():
    return int(text)
  if text.startswith("cuda:"):
    suffix = text.split(":", 1)[1]
    if suffix.isdigit():
      return int(suffix)
  if text == "cuda":
    return 0
  raise ValueError(f"expected CUDA device like 'cuda:0' or '0', got {device!r}")


def calibrate_non_expert_gpu_baseline(model, adapter, device: str) -> dict[str, int | str]:
  import torch

  if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available; pass --base_runtime_gb instead")
  device_index = cuda_device_index(device)
  device_name = f"cuda:{device_index}"
  torch.cuda.set_device(device_index)
  expert_prefixes = {item.name for item in expert_sizes_from_model(model, adapter)}
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats(device_index)

  moved = 0
  with torch.no_grad():
    for name, module in model.named_modules():
      if name == "" or _is_under_prefix(name, expert_prefixes):
        continue
      if any(True for _ in module.children()):
        continue
      module.to(device_name)
      moved += 1
  torch.cuda.synchronize(device_index)
  return {
    "mode": "non_expert_leaf_modules_to_gpu",
    "device": device_name,
    "device_index": device_index,
    "moved_leaf_modules": moved,
    "allocated_bytes": int(torch.cuda.memory_allocated(device_index)),
    "reserved_bytes": int(torch.cuda.memory_reserved(device_index)),
    "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device_index)),
  }


def inspect_model(model_id: str, model_revision: str | None, allow_download: bool):
  from sparse_llm_cache.model_adapters import get_model_adapter

  model = load_model_for_sizing(model_id, model_revision, not allow_download)
  adapter = get_model_adapter(model, model_id)
  sizes = expert_sizes_from_model(model, adapter)
  summary = summarize_experts(sizes)
  return model, adapter, sizes, summary


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Build a GPU memory curve for Promoe MoE expert cache rates.",
  )
  parser.add_argument("--model_id", help="Hugging Face/local model id to inspect.")
  parser.add_argument("--model_revision")
  parser.add_argument("--allow_download", action="store_true")
  parser.add_argument("--expert_bytes", help="Single expert size, e.g. 128MiB. Skips model loading.")
  parser.add_argument("--num_moe_layer", type=int, help="Required with --expert_bytes.")
  parser.add_argument("--num_expert_per_layer", type=int, help="Required with --expert_bytes.")
  parser.add_argument("--cache_rates", required=True, help="Comma-separated cache rates, e.g. 0,0.125,0.25,0.375")
  parser.add_argument("--gpu_sizes_gb", default="24,48,80", help="Comma-separated GPU sizes for fit columns/plot lines.")
  parser.add_argument("--base_runtime_gb", type=float, help="Known non-cache runtime baseline. Skips GPU calibration.")
  parser.add_argument("--calibrate_gpu_baseline", action="store_true", help="Load non-expert leaf modules to GPU and measure baseline.")
  parser.add_argument("--baseline_metric", choices=["reserved", "allocated", "max_reserved"], default="reserved")
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--safety_margin_gb", type=float, default=0.0, help="Optional extra margin added above the computed lower bound. Defaults to 0.")
  parser.add_argument("--output_dir", default=str(Path(__file__).resolve().parent / "output"), help="Output root directory. A per-model subdirectory is created under it.")
  parser.add_argument("--output_prefix", help="Per-model output subdirectory name. Defaults to model owner-name or manual-expert-size.")
  parser.add_argument("--json", action="store_true", help="Also print JSON to stdout.")
  return parser


def main(argv: list[str] | None = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)

  cache_rates = parse_float_list(args.cache_rates)
  gpu_sizes_gb = parse_float_list(args.gpu_sizes_gb)
  if args.safety_margin_gb < 0:
    parser.error("--safety_margin_gb must be >= 0")

  model = None
  adapter = None
  calibration = None
  if args.expert_bytes:
    if args.num_moe_layer is None or args.num_expert_per_layer is None:
      parser.error("--expert_bytes requires --num_moe_layer and --num_expert_per_layer")
    max_expert_bytes = parse_bytes(args.expert_bytes)
    num_moe_layer = int(args.num_moe_layer)
    num_expert_per_layer = int(args.num_expert_per_layer)
    expert_summary = {"source": "expert_bytes", "max": {"bytes": max_expert_bytes}, "all_same_size": None}
    model_label = args.output_prefix or "manual-expert-size"
  else:
    if not args.model_id:
      parser.error("pass either --model_id or --expert_bytes")
    model, adapter, _, expert_summary = inspect_model(args.model_id, args.model_revision, args.allow_download)
    max_expert_bytes = int(expert_summary["max"]["bytes"])
    num_moe_layer = int(adapter.num_moe_layer)
    num_expert_per_layer = int(adapter.num_expert_per_layer)
    model_label = args.output_prefix or default_model_output_name(args.model_id)

  if args.base_runtime_gb is not None:
    if args.base_runtime_gb < 0:
      parser.error("--base_runtime_gb must be >= 0")
    base_runtime_bytes = int(args.base_runtime_gb * GIB)
    baseline_source = "base_runtime_gb"
  elif args.calibrate_gpu_baseline:
    if model is None or adapter is None:
      parser.error("--calibrate_gpu_baseline requires --model_id")
    calibration = calibrate_non_expert_gpu_baseline(model, adapter, args.device)
    metric_key = {
      "reserved": "reserved_bytes",
      "allocated": "allocated_bytes",
      "max_reserved": "max_reserved_bytes",
    }[args.baseline_metric]
    base_runtime_bytes = int(calibration[metric_key])
    baseline_source = f"calibrate_gpu_baseline:{args.baseline_metric}"
  else:
    parser.error("pass --base_runtime_gb or --calibrate_gpu_baseline")

  safety_margin_bytes = int(args.safety_margin_gb * GIB)
  points = build_curve_points(
    cache_rates=cache_rates,
    max_single_expert_bytes=max_expert_bytes,
    num_moe_layer=num_moe_layer,
    num_expert_per_layer=num_expert_per_layer,
    base_runtime_bytes=base_runtime_bytes,
    safety_margin_bytes=safety_margin_bytes,
  )

  output_root = Path(args.output_dir)
  output_prefix = sanitize_filename(model_label)
  output_dir = output_root / output_prefix
  csv_path = output_dir / "gpu-curve.csv"
  json_path = output_dir / "gpu-curve.json"
  png_path = output_dir / "gpu-curve.png"

  write_csv(csv_path, points, gpu_sizes_gb)
  plot_written = write_plot(png_path, points, gpu_sizes_gb)

  result = {
    "model_id": args.model_id,
    "num_moe_layer": num_moe_layer,
    "num_expert_per_layer": num_expert_per_layer,
    "total_experts": num_moe_layer * num_expert_per_layer,
    "max_single_expert_bytes": max_expert_bytes,
    "max_single_expert_human": format_bytes(max_expert_bytes),
    "base_runtime_bytes": base_runtime_bytes,
    "base_runtime_human": format_bytes(base_runtime_bytes),
    "baseline_source": baseline_source,
    "safety_margin_bytes": safety_margin_bytes,
    "safety_margin_human": format_bytes(safety_margin_bytes),
    "gpu_sizes_gb": gpu_sizes_gb,
    "expert_summary": expert_summary,
    "calibration": calibration,
    "points": [asdict(point) | fit_columns(point, gpu_sizes_gb) for point in points],
    "outputs": {
      "root_dir": str(output_root),
      "model_dir": str(output_dir),
      "csv": str(csv_path),
      "json": str(json_path),
      "png": str(png_path) if plot_written else None,
      "plot_written": plot_written,
    },
  }
  json_path.parent.mkdir(parents=True, exist_ok=True)
  json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

  print(f"num_moe_layer: {num_moe_layer}")
  print(f"num_expert_per_layer: {num_expert_per_layer}")
  print(f"total_experts: {num_moe_layer * num_expert_per_layer}")
  print(f"max_single_expert: {format_bytes(max_expert_bytes)}")
  print(f"base_runtime: {format_bytes(base_runtime_bytes)} ({baseline_source})")
  print(f"safety_margin: {format_bytes(safety_margin_bytes)}")
  print("")
  print("cache_rate,cache_slots,cache_gib,lower_bound_total_gib,safety_margin_gib,estimated_total_gib," + ",".join(f"fit_{size:g}g" for size in gpu_sizes_gb))
  for point in points:
    fits = fit_columns(point, gpu_sizes_gb)
    fit_values = ",".join("yes" if fits[f"fit_{size:g}g"] else "no" for size in gpu_sizes_gb)
    margin_gib = max(0.0, point.estimated_total_gib - point.lower_bound_total_gib)
    print(f"{point.cache_rate:g},{point.cache_slots},{point.cache_gib:.4f},{point.lower_bound_total_gib:.4f},{margin_gib:.4f},{point.estimated_total_gib:.4f},{fit_values}")
  print("")
  print(f"output_dir: {output_dir}")
  print(f"csv: {csv_path}")
  print(f"json: {json_path}")
  if plot_written:
    print(f"png: {png_path}")
  else:
    print("png: skipped (matplotlib is not available)")
  if args.json:
    print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
