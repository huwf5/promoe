import inspect
from pathlib import Path

import pytest

from sparse_llm_cache.model_adapters.base import ModelAdapter
from sparse_llm_cache.utils.runner_util import parse_args
from sparse_llm_cache.utils import inject_model, wrap_generate_with_initial_cache, _resolve_initial_cache_inputs


class DummyPrefetchMngr:
  def __init__(self, events=None):
    self.calls = 0
    self.events = events

  def reset_for_generate(self):
    self.calls += 1
    if self.events is not None:
      self.events.append("reset_for_generate")


class DummyModel:
  def __init__(self, events=None):
    self.generate_calls = 0
    self.events = events

  def generate(self, *args, **kwargs):
    self.generate_calls += 1
    if self.events is not None:
      self.events.append("generate")
    return {"args": args, "kwargs": kwargs}


def test_generate_wrapper_resets_before_generate():
  events = []
  model = DummyModel(events)
  mngr = DummyPrefetchMngr(events)

  wrap_generate_with_initial_cache(model, mngr)
  result = model.generate(1, x=2)

  assert mngr.calls == 1
  assert model.generate_calls == 1
  assert events == ["reset_for_generate", "generate"]
  assert result == {"args": (1,), "kwargs": {"x": 2}}


def test_generate_wrapper_is_idempotent():
  model = DummyModel()
  mngr = DummyPrefetchMngr()

  wrap_generate_with_initial_cache(model, mngr)
  wrap_generate_with_initial_cache(model, mngr)
  model.generate()

  assert mngr.calls == 1
  assert model.generate_calls == 1


class DummyAdapter(ModelAdapter):
  @property
  def num_moe_layer(self) -> int:
    return 7


class DummyMeta:
  pass


def test_base_adapter_configures_decoder_only_stage_counts():
  adapter = DummyAdapter(model=None, model_id="dummy")
  meta = DummyMeta()

  adapter.configure_module_meta(meta)

  assert meta.num_encoder_moe_layer == 0
  assert meta.num_decoder_moe_layer == 7


def test_inject_model_exposes_initial_cache_config_parameters():
  signature = inspect.signature(inject_model)

  assert signature.parameters["initial_cache_policy"].default is None
  assert signature.parameters["initial_layer_budgets"].default is None
  assert signature.parameters["initial_hot_expert_file"].default is None
  assert "reset_cache_on_generate_start" not in signature.parameters
  assert "initial_expert_order" not in signature.parameters
  assert "initial_cache_ready_barrier" not in signature.parameters


def test_runner_util_parses_initial_cache_cli_args():
  parsed = parse_args([
    "--initial_cache_policy", "manual",
    "--initial_layer_budgets", "0:2,1:2",
  ])

  assert parsed["initial_cache_policy"] == "manual"
  assert parsed["initial_layer_budgets"] == "0:2,1:2"


def test_runner_util_parses_hot_expert_policy():
  parsed = parse_args([
    "--initial_cache_policy", "hot_expert",
    "--initial_hot_expert_file", "/tmp/hot.json",
  ])

  assert parsed["initial_cache_policy"] == "hot_expert"
  assert parsed["initial_hot_expert_file"] == "/tmp/hot.json"


def test_runner_util_rejects_removed_initial_cache_knobs():
  with pytest.raises(SystemExit):
    parse_args(["--reset_cache_on_generate_start", "True"])
  with pytest.raises(SystemExit):
    parse_args(["--initial_expert_order", "sequential"])
  with pytest.raises(SystemExit):
    parse_args(["--initial_cache_ready_barrier", "False"])


def test_resolve_initial_cache_inputs_validates_policy_specific_args():
  manual = _resolve_initial_cache_inputs("manual", "0:2", None, None)
  assert manual["reset_cache_on_generate_start"] is True
  assert manual["initial_layer_budgets"] == "0:2"
  assert manual["initial_hot_expert_file"] is None
  assert manual["per_layer_cache"] is False

  hot = _resolve_initial_cache_inputs("hot_expert", None, "/tmp/hot.json", None)
  assert hot["reset_cache_on_generate_start"] is True
  assert hot["initial_layer_budgets"] is None
  assert hot["initial_hot_expert_file"] == "/tmp/hot.json"
  assert hot["per_layer_cache"] is False

  with pytest.raises(ValueError, match="manual initial cache requires initial_layer_budgets"):
    _resolve_initial_cache_inputs("manual", None, None, None)
  with pytest.raises(ValueError, match="hot_expert initial cache requires initial_hot_expert_file"):
    _resolve_initial_cache_inputs("hot_expert", None, None, None)
  with pytest.raises(ValueError, match="hot_expert initial cache does not use initial_layer_budgets"):
    _resolve_initial_cache_inputs("hot_expert", "0:2", "/tmp/hot.json", None)
  with pytest.raises(ValueError, match="initial_hot_expert_file requires initial_cache_policy=hot_expert"):
    _resolve_initial_cache_inputs(None, None, "/tmp/hot.json", None)


def test_resolve_initial_cache_inputs_accepts_hot_encoder_coverage():
  hot = _resolve_initial_cache_inputs("hot_encoder_coverage", None, "/tmp/hot.json", None)

  assert hot["initial_cache_policy"] == "manual"
  assert hot["initial_hot_expert_file"] == "/tmp/hot.json"
  assert "initial_hot_expert_policy" not in hot
  assert hot["per_layer_cache"] is False

def test_adapter_pybind_does_not_expose_internal_initial_cache_knobs():
  adapter_cpp = Path(__file__).resolve().parents[1] / "src" / "cpp_worker" / "adapter.cpp"
  contents = adapter_cpp.read_text()

  assert '.def_readwrite("reset_cache_on_generate_start"' not in contents
  assert '.def_readwrite("initial_expert_order"' not in contents
  assert '.def_readwrite("initial_cache_ready_barrier"' not in contents


def test_python_does_not_send_internal_initial_cache_knobs_to_cpp():
  utils_py = Path(__file__).resolve().parents[1] / "src" / "sparse_llm_cache" / "utils" / "__init__.py"
  contents = utils_py.read_text()

  assert "'initial_expert_order'" not in contents
  assert "'initial_cache_ready_barrier'" not in contents
  assert "meta.reset_cache_on_generate_start" not in contents


def test_cpp_does_not_parse_internal_initial_cache_knobs_from_map():
  utils_cpp = Path(__file__).resolve().parents[1] / "src" / "cpp_worker" / "utils.cpp"
  contents = utils_cpp.read_text()

  assert 'optional_str("initial_expert_order"' not in contents
  assert 'optional_bool("initial_cache_ready_barrier"' not in contents


def test_cpp_initial_cache_has_no_order_or_barrier_config_fields():
  utils_hpp = Path(__file__).resolve().parents[1] / "src" / "cpp_worker" / "utils.hpp"
  cache_cpp = Path(__file__).resolve().parents[1] / "src" / "cpp_worker" / "cache.cpp"
  utils_contents = utils_hpp.read_text()
  cache_contents = cache_cpp.read_text()

  assert "initial_expert_order" not in utils_contents
  assert "initial_cache_ready_barrier" not in utils_contents
  assert "initial_expert_order" not in cache_contents
  assert "initial_cache_ready_barrier" not in cache_contents
  assert "expert->expert_status.get() == kReady" in cache_contents


def test_performance_script_uses_current_initial_cache_cli():
  script = Path(__file__).resolve().parents[1] / "performance" / "run_switch_mmlu_validation.sh"
  contents = script.read_text()

  assert "--reset_cache_on_generate_start" not in contents
  assert "--initial_expert_order" not in contents
  assert "--initial_cache_ready_barrier" not in contents
  assert "--initial_cache_policy" in contents
  assert "--initial_layer_budgets" in contents
  assert "--initial_hot_expert_file" in contents
