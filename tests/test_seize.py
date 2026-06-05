import pytest

from seize import cli as seize_cli
from seize.cli import build_parser, validate_args
from seize.worker import (
  MIB,
  WorkerConfig,
  choose_compute_step,
  choose_probe_size_mib,
  next_probe_size_mib,
  parse_gpu_ids,
)


def test_parse_gpu_ids_all():
  assert parse_gpu_ids("all", 4) == [0, 1, 2, 3]


def test_parse_gpu_ids_preserves_order_and_deduplicates():
  assert parse_gpu_ids("2, 0,2", 4) == [2, 0]


def test_parse_gpu_ids_rejects_invalid_token():
  with pytest.raises(ValueError, match="Invalid GPU id"):
    parse_gpu_ids("0,x", 4)


def test_parse_gpu_ids_rejects_out_of_range_id():
  with pytest.raises(ValueError, match="out of range"):
    parse_gpu_ids("3", 2)


def test_next_probe_size_mib_halves_until_minimum():
  assert next_probe_size_mib(1024, 64) == 512
  assert next_probe_size_mib(512, 64) == 256
  assert next_probe_size_mib(96, 64) == 64
  assert next_probe_size_mib(64, 64) == 0


def test_choose_probe_size_mib_tracks_remaining_free_memory():
  assert choose_probe_size_mib(6239 * MIB, 1024, 1) == 1024
  assert choose_probe_size_mib(777 * MIB, 1024, 1) == 777
  assert choose_probe_size_mib(1 * MIB, 1024, 1) == 1
  assert choose_probe_size_mib(MIB - 1, 1024, 1) == 0


def test_build_parser_defaults():
  args = build_parser().parse_args([])

  assert args.gpus == "all"
  assert args.duration == 0
  assert args.initial_chunk_mb == 1024
  assert args.min_chunk_mb == 1
  assert args.touch is True
  assert args.compute is False
  assert args.compute_min_dim == 1024
  assert args.compute_max_dim == 4096
  assert args.compute_burst_min == 1
  assert args.compute_burst_max == 8
  assert args.compute_sleep_min_ms == 0
  assert args.compute_sleep_max_ms == 80


def test_build_parser_custom_values():
  args = build_parser().parse_args(
    ["--gpus", "0,1", "--duration", "30", "--initial-chunk-mb", "2048", "--min-chunk-mb", "32", "--no-touch"]
  )

  assert args.gpus == "0,1"
  assert args.duration == 30
  assert args.initial_chunk_mb == 2048
  assert args.min_chunk_mb == 32
  assert args.touch is False


def test_validate_args_rejects_min_chunk_above_initial():
  args = build_parser().parse_args(["--initial-chunk-mb", "64", "--min-chunk-mb", "128"])

  with pytest.raises(ValueError, match="min_chunk_mb"):
    validate_args(args)


def test_build_parser_compute_values():
  args = build_parser().parse_args(
    [
      "--compute",
      "--compute-min-dim",
      "512",
      "--compute-max-dim",
      "2048",
      "--compute-burst-min",
      "2",
      "--compute-burst-max",
      "9",
      "--compute-sleep-min-ms",
      "3",
      "--compute-sleep-max-ms",
      "17",
    ]
  )

  assert args.compute is True
  assert args.compute_min_dim == 512
  assert args.compute_max_dim == 2048
  assert args.compute_burst_min == 2
  assert args.compute_burst_max == 9
  assert args.compute_sleep_min_ms == 3
  assert args.compute_sleep_max_ms == 17


def test_validate_args_rejects_invalid_compute_ranges():
  args = build_parser().parse_args(["--compute", "--compute-min-dim", "4096", "--compute-max-dim", "1024"])

  with pytest.raises(ValueError, match="compute_min_dim"):
    validate_args(args)


def test_choose_compute_step_varies_workload():
  config = WorkerConfig(
    gpu_id=0,
    initial_chunk_mb=1024,
    min_chunk_mb=1,
    touch=True,
    duration=0,
    compute=True,
    compute_min_dim=1024,
    compute_max_dim=4096,
    compute_burst_min=1,
    compute_burst_max=8,
    compute_sleep_min_ms=0,
    compute_sleep_max_ms=80,
  )

  steps = [choose_compute_step(config) for _ in range(32)]

  assert {step.dim for step in steps} <= {1024, 2048, 3072, 4096}
  assert len({step.dim for step in steps}) > 1
  assert len({step.bursts for step in steps}) > 1
  assert all(0 <= step.sleep_seconds <= 0.080 for step in steps)


class FakeProcess:
  def __init__(self, exitcode):
    self.exitcode = exitcode


def test_process_exit_status_reports_worker_failure():
  assert seize_cli._process_exit_status([FakeProcess(0), FakeProcess(None)]) == 0
  assert seize_cli._process_exit_status([FakeProcess(0), FakeProcess(1)]) == 1
