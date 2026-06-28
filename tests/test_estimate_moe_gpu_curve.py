import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiment/scripts/tools/estimate_moe_gpu_curve.py"


def _load_tool():
  spec = importlib.util.spec_from_file_location("estimate_moe_gpu_curve", SCRIPT)
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def test_build_curve_points_reports_lower_bound_and_optional_safety_margin():
  tool = _load_tool()

  points = tool.build_curve_points(
    cache_rates=[0.0, 0.5],
    max_single_expert_bytes=10,
    num_moe_layer=1,
    num_expert_per_layer=4,
    base_runtime_bytes=100,
    safety_margin_bytes=20,
  )

  assert points[0].cache_slots == 0
  assert points[0].lower_bound_total_bytes == 100
  assert points[0].estimated_total_bytes == 120
  assert points[1].cache_slots == 2
  assert points[1].lower_bound_total_bytes == 120
  assert points[1].estimated_total_bytes == 140


def test_fit_columns_uses_estimated_total_gib():
  tool = _load_tool()
  point = tool.CurvePoint(
    cache_rate=0.5,
    cache_slots=1,
    effective_cache_rate=0.5,
    cache_bytes=0,
    cache_gib=0.0,
    lower_bound_total_bytes=10 * tool.GIB,
    lower_bound_total_gib=10.0,
    estimated_total_bytes=10 * tool.GIB,
    estimated_total_gib=10.0,
  )

  assert tool.fit_columns(point, [8, 10, 12]) == {
    "fit_8g": False,
    "fit_10g": True,
    "fit_12g": True,
  }


def test_cli_manual_mode_writes_csv_and_json(tmp_path):
  tool = _load_tool()

  rc = tool.main([
    "--expert_bytes", "10B",
    "--num_moe_layer", "1",
    "--num_expert_per_layer", "4",
    "--cache_rates", "0,0.5",
    "--gpu_sizes_gb", "1,2",
    "--base_runtime_gb", "0",
    "--safety_margin_gb", "0",
    "--output_dir", str(tmp_path),
    "--output_prefix", "unit-test",
  ])

  assert rc == 0
  csv_path = tmp_path / "unit-test" / "gpu-curve.csv"
  json_path = tmp_path / "unit-test" / "gpu-curve.json"
  assert csv_path.exists()
  csv_text = csv_path.read_text()
  assert "cache_rate,cache_slots" in csv_text
  assert "lower_bound_total_gib" in csv_text
  data = json.loads(json_path.read_text())
  assert data["points"][1]["cache_slots"] == 2
  assert data["points"][1]["cache_bytes"] == 20
  assert data["points"][1]["lower_bound_total_bytes"] == 20


def test_default_model_output_name_uses_owner_and_model_for_local_path():
  tool = _load_tool()

  assert (
    tool.default_model_output_name("/mnt/huwf5/promoe/experiment/models/google/switch-base-128")
    == "google-switch-base-128"
  )


def test_cuda_device_index_accepts_common_cuda_forms():
  tool = _load_tool()

  assert tool.cuda_device_index("cuda:0") == 0
  assert tool.cuda_device_index("cuda") == 0
  assert tool.cuda_device_index("1") == 1
  assert tool.cuda_device_index(2) == 2
