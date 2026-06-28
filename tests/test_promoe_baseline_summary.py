import csv
import json
import subprocess
import sys
from pathlib import Path


def test_collect_summary_writes_promoe_benchmark_fields(tmp_path: Path):
  run_dir = tmp_path / "run"
  run_dir.mkdir()
  samples_path = run_dir / "samples.csv"
  with samples_path.open("w", newline="") as f:
    writer = csv.DictWriter(
      f,
      fieldnames=[
        "run_id",
        "baseline",
        "backend_mode",
        "model_id",
        "dataset",
        "task",
        "split",
        "sample_idx",
        "batch_idx",
        "prompt_tokens",
        "new_tokens",
        "input_ms",
        "cache_init_ms",
        "ttft_ms",
        "tpot_ms",
        "e2e_ms",
        "decode_tokens_per_second_excl_first",
        "e2e_tokens_per_second",
        "warmup",
        "is_valid",
      ],
    )
    writer.writeheader()
    writer.writerow(
      {
        "run_id": "20260616_120000",
        "baseline": "promoe",
        "backend_mode": "base",
        "model_id": "google/switch-base-128",
        "dataset": "mmlu",
        "task": "professional_law",
        "split": "validation",
        "sample_idx": 0,
        "batch_idx": 0,
        "prompt_tokens": 12,
        "new_tokens": 3,
        "input_ms": 1.0,
        "cache_init_ms": 10.0,
        "ttft_ms": 999.0,
        "tpot_ms": 99.0,
        "e2e_ms": 1197.0,
        "decode_tokens_per_second_excl_first": 10.10101,
        "e2e_tokens_per_second": 2.506266,
        "warmup": 1,
        "is_valid": 0,
      }
    )
    writer.writerow(
      {
        "run_id": "20260616_120000",
        "baseline": "promoe",
        "backend_mode": "base",
        "model_id": "google/switch-base-128",
        "dataset": "mmlu",
        "task": "professional_law",
        "split": "validation",
        "sample_idx": 1,
        "batch_idx": 1,
        "prompt_tokens": 12,
        "new_tokens": 3,
        "input_ms": 1.0,
        "cache_init_ms": 10.0,
        "ttft_ms": 100.0,
        "tpot_ms": 10.0,
        "e2e_ms": 120.0,
        "decode_tokens_per_second_excl_first": 100.0,
        "e2e_tokens_per_second": 25.0,
        "warmup": 0,
        "is_valid": 1,
      }
    )
    writer.writerow(
      {
        "run_id": "20260616_120000",
        "baseline": "promoe",
        "backend_mode": "base",
        "model_id": "google/switch-base-128",
        "dataset": "mmlu",
        "task": "professional_law",
        "split": "validation",
        "sample_idx": 2,
        "batch_idx": 2,
        "prompt_tokens": 12,
        "new_tokens": 5,
        "input_ms": 2.0,
        "cache_init_ms": 20.0,
        "ttft_ms": 200.0,
        "tpot_ms": 20.0,
        "e2e_ms": 280.0,
        "decode_tokens_per_second_excl_first": 50.0,
        "e2e_tokens_per_second": 17.857143,
        "warmup": 0,
        "is_valid": 1,
      }
    )

  script = Path("experiment/baseline/LSP/promoe/scripts/collect_summary.py")
  result = subprocess.run(
    [sys.executable, str(script), "--run-dir", str(run_dir)],
    cwd=Path(__file__).resolve().parents[1],
    text=True,
    capture_output=True,
  )

  assert result.returncode == 0, result.stderr
  summary = json.loads((run_dir / "summary.json").read_text())
  assert summary["baseline"] == "promoe"
  assert summary["backend_mode"] == "base"
  assert summary["n_total"] == 3
  assert summary["n_valid"] == 2
  assert summary["benchmark_ttft_ms_avg"] == 150.0
  assert summary["benchmark_decode_tpot_ms_avg"] == 15.0
  assert summary["benchmark_e2e_ms_avg"] == 200.0
  assert summary["benchmark_decode_tokens_per_second_excl_first"] == 60.0
  assert summary["benchmark_e2e_tokens_per_second"] == 20.0
  assert summary["percentiles"]["ttft_ms"]["p50"] == 150.0

  summary_tsv = (run_dir / "summary.tsv").read_text().splitlines()
  assert summary_tsv[0].startswith("run_id\tbaseline\tbackend_mode\tmodel_id")
  assert "benchmark_ttft_ms_avg" in summary_tsv[0]
  assert "benchmark_e2e_ms_avg" in summary_tsv[0]
  assert len(summary_tsv) == 2
