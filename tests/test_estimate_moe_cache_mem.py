import importlib.util
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiment/scripts/tools/estimate_moe_cache_mem.py"


def _load_tool():
  spec = importlib.util.spec_from_file_location("estimate_moe_cache_mem", SCRIPT)
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def test_estimate_cache_memory_uses_cpp_rounding():
  tool = _load_tool()

  estimate = tool.estimate_cache_memory(
    max_single_expert_bytes=10,
    num_moe_layer=1,
    num_expert_per_layer=5,
    cache_rate=0.5,
  )

  assert estimate.cache_slots == 3
  assert estimate.cache_bytes == 30
  assert estimate.effective_cache_rate == 0.6


def test_max_cache_rate_for_budget_caps_to_total_experts():
  tool = _load_tool()

  max_slots, max_rate = tool.max_cache_rate_for_budget(
    budget_bytes=999,
    max_single_expert_bytes=10,
    num_moe_layer=2,
    num_expert_per_layer=4,
  )

  assert max_slots == 8
  assert max_rate == 1.0


def test_parse_bytes_accepts_binary_units():
  tool = _load_tool()

  assert tool.parse_bytes("1GiB") == 1024 ** 3
  assert tool.parse_bytes("1.5MiB") == int(1.5 * 1024 ** 2)


def test_module_prefetch_nbytes_includes_params_and_selected_quant_buffers():
  tool = _load_tool()

  class Expert(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.weight = torch.nn.Parameter(torch.zeros(2, 3, dtype=torch.float16))
      self.register_buffer("qweight", torch.zeros(4, dtype=torch.int32))
      self.register_buffer("ignored", torch.zeros(100, dtype=torch.float32))

  assert tool.module_prefetch_nbytes(Expert()) == (2 * 3 * 2) + (4 * 4)
