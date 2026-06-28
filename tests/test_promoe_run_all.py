import os
import subprocess
from pathlib import Path


def test_run_all_dry_run_writes_direct_transformers_command(tmp_path: Path):
  root = Path(__file__).resolve().parents[1]
  run_root = tmp_path / "runs"
  env = os.environ.copy()
  env.update(
    {
      "PROMOE_DRY_RUN": "1",
      "RUN_ROOT": str(run_root),
      "RUN_ID": "20260616_120000",
      "MODELS": "switch-base-128",
      "PYTHON_BIN": "/usr/bin/python3",
      "GPU_ID": "7",
      "GPU_CONFIGS": "default",
      "MAX_NEW_TOKENS": "16",
      "PROMOE_BENCHMARK_WARMUP": "2",
    }
  )

  script = root / "experiment/baseline/LSP/promoe/scripts/run_all.sh"
  result = subprocess.run(
    ["bash", str(script)],
    cwd=root,
    env=env,
    text=True,
    capture_output=True,
  )

  assert result.returncode == 0, result.stderr
  run_dir = run_root / "mmlu/professional_law/validation/default/google_switch-base-128/20260616_120000"
  assert run_dir.is_dir()

  config = (run_dir / "config.env").read_text()
  assert "BASELINE=promoe\n" in config
  assert "MODEL_ALIAS=switch-base-128\n" in config
  assert "MODEL_ID=google/switch-base-128\n" in config
  assert "GPU_PROFILE=default\n" in config
  assert "GPU_ID=7\n" in config
  assert "MAX_NEW_TOKENS=16\n" in config
  assert "PROMOE_BENCHMARK_WARMUP=2\n" in config

  command = (run_dir / "command.txt").read_text()
  assert "examples/small-demo/transformers-app.py" in command
  assert "performance/run_switch_mmlu_validation.sh" not in command
  assert "--model_id google/switch-base-128" in command
  assert "--max_new_tokens 16" in command
  assert "--do_sample False" in command
  assert "--num_beams 1" in command
  assert "--gpu_mem_limit_gb 4" in command
  assert "PROMOE_BENCHMARK_OUTPUT_DIR=" in command

  assert (run_dir / "DRY_RUN").read_text().strip() == "1"



def test_run_all_records_transformers_exit_status_on_failure(tmp_path: Path):
  root = Path(__file__).resolve().parents[1]
  run_root = tmp_path / "runs"
  fake_python = tmp_path / "fake-python"
  fake_python.write_text(
    "#!/usr/bin/env bash\n"
    "script=\"$1\"\n"
    "shift\n"
    "case \"${script}\" in\n"
    "  */transformers-app.py) echo fake-transformers-failed >&2; exit 42 ;;\n"
    "  */collect_summary.py) echo summary-should-not-run >&2; exit 99 ;;\n"
    "  *) exit 98 ;;\n"
    "esac\n"
  )
  fake_python.chmod(0o755)
  env = os.environ.copy()
  env.update(
    {
      "RUN_ROOT": str(run_root),
      "RUN_ID": "20260616_120001",
      "MODELS": "switch-base-128",
      "GPU_CONFIGS": "default",
      "PYTHON_BIN": str(fake_python),
      "PROMOE_DRY_RUN": "0",
    }
  )

  script = root / "experiment/baseline/LSP/promoe/scripts/run_all.sh"
  result = subprocess.run(
    ["bash", str(script)],
    cwd=root,
    env=env,
    text=True,
    capture_output=True,
  )

  run_dir = run_root / "mmlu/professional_law/validation/default/google_switch-base-128/20260616_120001"
  assert result.returncode == 1
  assert (run_dir / "EXIT_STATUS").read_text().strip() == "42"
  assert "fake-transformers-failed" in (run_dir / "run.log").read_text()
  assert "summary-should-not-run" not in result.stderr



def test_run_all_ours_defaults_to_per_layer_cache_without_initial_cache(tmp_path: Path):
  root = Path(__file__).resolve().parents[1]
  run_root = tmp_path / "runs"
  env = os.environ.copy()
  env.update(
    {
      "PROMOE_DRY_RUN": "1",
      "RUN_ROOT": str(run_root),
      "RUN_ID": "20260616_120000",
      "GPU_CONFIGS": "default",
      "MODELS": "switch-base-128",
      "PYTHON_BIN": "/usr/bin/python3",
    }
  )

  script = root / "experiment/baseline/LSP/promoe/scripts/run_all.sh"
  result = subprocess.run(
    ["bash", str(script)],
    cwd=root,
    env=env,
    text=True,
    capture_output=True,
  )

  assert result.returncode == 0, result.stderr
  run_dir = run_root / "mmlu/professional_law/validation/default/google_switch-base-128/20260616_120000"
  config = (run_dir / "config.env").read_text()
  command = (run_dir / "command.txt").read_text()

  assert "BACKEND_MODE=ours\n" in config
  assert "PER_LAYER_CACHE=True\n" in config
  assert "DETERMINISTIC_INIT=0\n" in config
  assert "--per_layer_cache True" in command
  assert "--initial_cache_policy" not in command
  assert "--initial_hot_expert_file" not in command


def test_run_all_dry_run_uses_default_profile_for_switch_base_128(tmp_path: Path):
  root = Path(__file__).resolve().parents[1]
  run_root = tmp_path / "runs"
  env = os.environ.copy()
  env.update(
    {
      "PROMOE_DRY_RUN": "1",
      "RUN_ROOT": str(run_root),
      "RUN_ID": "20260616_120000",
      "MODELS": "switch-base-128",
      "PYTHON_BIN": "/usr/bin/python3",
    }
  )

  script = root / "experiment/baseline/LSP/promoe/scripts/run_all.sh"
  result = subprocess.run(
    ["bash", str(script)],
    cwd=root,
    env=env,
    text=True,
    capture_output=True,
  )

  assert result.returncode == 0, result.stderr
  base128 = run_root / "mmlu/professional_law/validation/default/google_switch-base-128/20260616_120000/config.env"
  base128_text = base128.read_text()

  assert "MODEL_ALIAS=switch-base-128\n" in base128_text
  assert "GPU_PROFILE=default\n" in base128_text
  assert "GPU_MEM_GB=4\n" in base128_text
  assert "CACHE_RATE=0.125\n" in base128_text



def test_run_all_dry_run_applies_known_switch_base_128_gpu_cache_rates(tmp_path: Path):
  root = Path(__file__).resolve().parents[1]
  run_root = tmp_path / "runs"
  env = os.environ.copy()
  env.update(
    {
      "PROMOE_DRY_RUN": "1",
      "RUN_ROOT": str(run_root),
      "RUN_ID": "20260616_120000",
      "GPU_CONFIGS": "gpu4gb gpu8gb gpu12gb gpu24gb gpu48gb",
      "MODELS": "switch-base-128",
      "PYTHON_BIN": "/usr/bin/python3",
    }
  )

  script = root / "experiment/baseline/LSP/promoe/scripts/run_all.sh"
  result = subprocess.run(
    ["bash", str(script)],
    cwd=root,
    env=env,
    text=True,
    capture_output=True,
  )

  assert result.returncode == 0, result.stderr
  gpu4 = (run_root / "mmlu/professional_law/validation/gpu4gb/google_switch-base-128/20260616_120000/config.env").read_text()
  gpu8 = (run_root / "mmlu/professional_law/validation/gpu8gb/google_switch-base-128/20260616_120000/config.env").read_text()
  gpu12 = (run_root / "mmlu/professional_law/validation/gpu12gb/google_switch-base-128/20260616_120000/config.env").read_text()
  gpu24 = (run_root / "mmlu/professional_law/validation/gpu24gb/google_switch-base-128/20260616_120000/config.env").read_text()
  gpu48 = (run_root / "mmlu/professional_law/validation/gpu48gb/google_switch-base-128/20260616_120000/config.env").read_text()

  assert "GPU_MEM_GB=4\n" in gpu4 and "CACHE_RATE=0.125\n" in gpu4
  assert "GPU_MEM_GB=8\n" in gpu8 and "CACHE_RATE=0.25\n" in gpu8
  assert "GPU_MEM_GB=12\n" in gpu12 and "CACHE_RATE=0.375\n" in gpu12
  assert "GPU_MEM_GB=24\n" in gpu24 and "CACHE_RATE=0.375\n" in gpu24
  assert "GPU_MEM_GB=48\n" in gpu48 and "CACHE_RATE=0.375\n" in gpu48



def test_run_all_dry_run_errors_for_unmapped_model_gpu_pair(tmp_path: Path):
  root = Path(__file__).resolve().parents[1]
  run_root = tmp_path / "runs"
  env = os.environ.copy()
  env.update(
    {
      "PROMOE_DRY_RUN": "1",
      "RUN_ROOT": str(run_root),
      "RUN_ID": "20260616_120000",
      "GPU_CONFIGS": "gpu4gb",
      "MODELS": "switch-base-256",
      "PYTHON_BIN": "/usr/bin/python3",
    }
  )

  script = root / "experiment/baseline/LSP/promoe/scripts/run_all.sh"
  result = subprocess.run(
    ["bash", str(script)],
    cwd=root,
    env=env,
    text=True,
    capture_output=True,
  )

  assert result.returncode != 0
  assert "no CACHE_RATE mapping for GPU/model pair: gpu4gb/switch-base-256" in result.stderr
