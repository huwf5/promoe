#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


SUMMARY_FIELDS = [
  "run_id",
  "baseline",
  "backend_mode",
  "model_id",
  "gpu_profile",
  "gpu_id",
  "gpu_mem_gb",
  "cache_rate",
  "dataset",
  "task",
  "split",
  "n_total",
  "n_valid",
  "benchmark_ttft_ms_avg",
  "benchmark_decode_tpot_ms_avg",
  "benchmark_decode_tokens_per_second_excl_first",
  "benchmark_e2e_tokens_per_second",
  "benchmark_e2e_ms_avg",
  "benchmark_ttft_ms_p50",
  "benchmark_ttft_ms_p90",
  "benchmark_ttft_ms_p95",
  "benchmark_ttft_ms_p99",
  "benchmark_decode_tpot_ms_p50",
  "benchmark_decode_tpot_ms_p90",
  "benchmark_decode_tpot_ms_p95",
  "benchmark_decode_tpot_ms_p99",
  "benchmark_e2e_ms_p50",
  "benchmark_e2e_ms_p90",
  "benchmark_e2e_ms_p95",
  "benchmark_e2e_ms_p99",
]


def _float_or_none(value):
  if value is None:
    return None
  text = str(value).strip()
  if text == "" or text.lower() == "nan":
    return None
  return float(text)


def _mean(values):
  values = [v for v in values if v is not None]
  if not values:
    return None
  return sum(values) / len(values)


def _round(value):
  if value is None:
    return None
  return round(float(value), 6)


def _percentile(values, percentile):
  values = sorted(v for v in values if v is not None)
  if not values:
    return None
  if len(values) == 1:
    return values[0]
  rank = (len(values) - 1) * percentile / 100.0
  lower = int(rank)
  upper = min(lower + 1, len(values) - 1)
  weight = rank - lower
  return values[lower] * (1.0 - weight) + values[upper] * weight


def _first(rows, field, default=""):
  for row in rows:
    value = row.get(field, "")
    if value != "":
      return value
  return default


def collect_summary(run_dir):
  samples_path = run_dir / "samples.csv"
  if not samples_path.is_file():
    raise FileNotFoundError(f"samples.csv not found: {samples_path}")

  with samples_path.open(newline="") as f:
    rows = list(csv.DictReader(f))

  valid_rows = [row for row in rows if str(row.get("is_valid", "")).strip() == "1"]
  ttft_ms = [_float_or_none(row.get("ttft_ms")) for row in valid_rows]
  tpot_ms = [_float_or_none(row.get("tpot_ms")) for row in valid_rows]
  e2e_ms = [_float_or_none(row.get("e2e_ms")) for row in valid_rows]

  decode_tokens = 0
  decode_time_ms = 0.0
  generate_tokens = 0
  e2e_time_ms = 0.0
  for row in valid_rows:
    new_tokens = int(float(row.get("new_tokens") or 0))
    tpot = _float_or_none(row.get("tpot_ms"))
    e2e = _float_or_none(row.get("e2e_ms"))
    sample_decode_tokens = max(new_tokens - 1, 0)
    if tpot is not None:
      decode_tokens += sample_decode_tokens
      decode_time_ms += tpot * sample_decode_tokens
    if e2e is not None:
      generate_tokens += new_tokens
      e2e_time_ms += e2e

  decode_tps = None
  if decode_time_ms > 0:
    decode_tps = decode_tokens / (decode_time_ms / 1000.0)
  e2e_tps = None
  if e2e_time_ms > 0:
    e2e_tps = generate_tokens / (e2e_time_ms / 1000.0)

  summary = {
    "run_id": _first(rows, "run_id"),
    "baseline": _first(rows, "baseline", "promoe"),
    "backend_mode": _first(rows, "backend_mode", "default"),
    "model_id": _first(rows, "model_id"),
    "gpu_profile": _first(rows, "gpu_profile"),
    "gpu_id": _first(rows, "gpu_id"),
    "gpu_mem_gb": _first(rows, "gpu_mem_gb"),
    "cache_rate": _first(rows, "cache_rate"),
    "dataset": _first(rows, "dataset"),
    "task": _first(rows, "task"),
    "split": _first(rows, "split"),
    "n_total": len(rows),
    "n_valid": len(valid_rows),
    "benchmark_ttft_ms_avg": _round(_mean(ttft_ms)),
    "benchmark_decode_tpot_ms_avg": _round(_mean(tpot_ms)),
    "benchmark_decode_tokens_per_second_excl_first": _round(decode_tps),
    "benchmark_e2e_tokens_per_second": _round(e2e_tps),
    "benchmark_e2e_ms_avg": _round(_mean(e2e_ms)),
    "benchmark_ttft_ms_p50": _round(_percentile(ttft_ms, 50)),
    "benchmark_ttft_ms_p90": _round(_percentile(ttft_ms, 90)),
    "benchmark_ttft_ms_p95": _round(_percentile(ttft_ms, 95)),
    "benchmark_ttft_ms_p99": _round(_percentile(ttft_ms, 99)),
    "benchmark_decode_tpot_ms_p50": _round(_percentile(tpot_ms, 50)),
    "benchmark_decode_tpot_ms_p90": _round(_percentile(tpot_ms, 90)),
    "benchmark_decode_tpot_ms_p95": _round(_percentile(tpot_ms, 95)),
    "benchmark_decode_tpot_ms_p99": _round(_percentile(tpot_ms, 99)),
    "benchmark_e2e_ms_p50": _round(_percentile(e2e_ms, 50)),
    "benchmark_e2e_ms_p90": _round(_percentile(e2e_ms, 90)),
    "benchmark_e2e_ms_p95": _round(_percentile(e2e_ms, 95)),
    "benchmark_e2e_ms_p99": _round(_percentile(e2e_ms, 99)),
    "percentiles": {
      "ttft_ms": {
        "p50": _round(_percentile(ttft_ms, 50)),
        "p90": _round(_percentile(ttft_ms, 90)),
        "p95": _round(_percentile(ttft_ms, 95)),
        "p99": _round(_percentile(ttft_ms, 99)),
      },
      "tpot_ms": {
        "p50": _round(_percentile(tpot_ms, 50)),
        "p90": _round(_percentile(tpot_ms, 90)),
        "p95": _round(_percentile(tpot_ms, 95)),
        "p99": _round(_percentile(tpot_ms, 99)),
      },
      "e2e_ms": {
        "p50": _round(_percentile(e2e_ms, 50)),
        "p90": _round(_percentile(e2e_ms, 90)),
        "p95": _round(_percentile(e2e_ms, 95)),
        "p99": _round(_percentile(e2e_ms, 99)),
      },
    },
  }

  (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
  with (run_dir / "summary.tsv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, delimiter="\t", extrasaction="ignore")
    writer.writeheader()
    writer.writerow(summary)
  return summary


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--run-dir", type=Path, required=True)
  args = parser.parse_args()
  summary = collect_summary(args.run_dir)
  print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
