import json
import inspect
from types import SimpleNamespace

import pytest

from sparse_llm_cache.model_adapters.switch import SwitchAdapter
from sparse_llm_cache.utils import inject_model, round_like_cpp
from sparse_llm_cache.utils.hot_experts import (
  build_decoder_warmup_overlap_plan,
  build_encoder_balanced_hot_initial_plan,
  build_encoder_coverage_initial_plan,
  build_encoder_hot_initial_plan,
  build_encoder_l0_priority_hot_initial_plan,
  build_hot_initial_plan,
)
from sparse_llm_cache.utils.runner_util import parse_args


def _switch_config(**overrides):
  values = {
    "model_type": "switch_transformers",
    "num_experts": 128,
    "num_selected_experts": 1,
    "num_sparse_encoder_layers": 2,
    "num_sparse_decoder_layers": 2,
    "encoder_sparse_step": 2,
    "decoder_sparse_step": 2,
    "num_layers": 4,
    "num_decoder_layers": 4,
  }
  values.update(overrides)
  return SimpleNamespace(**values)


def test_build_encoder_hot_initial_plan_allocates_by_token_count_and_uses_hot_eids(tmp_path):
  payload = {
    "expert_usage_summary": {
      "encoder.block.1.layer.1.mlp.router.classifier": {
        "top_token_counts": [
          {"eid": 10, "count": 100},
          {"eid": 11, "count": 90},
          {"eid": 12, "count": 80},
        ],
        "frozen_top_eids": [10, 11, 12],
      },
      "encoder.block.3.layer.1.mlp.router.classifier": {
        "top_token_counts": [
          {"eid": 20, "count": 50},
          {"eid": 21, "count": 40},
          {"eid": 22, "count": 30},
        ],
        "frozen_top_eids": [20, 21, 22],
      },
      "decoder.block.1.layer.2.mlp.router.classifier": {
        "top_token_counts": [{"eid": 99, "count": 1000}],
        "frozen_top_eids": [99],
      },
    },
  }
  path = tmp_path / "hot.json"
  path.write_text(json.dumps(payload))
  adapter = SwitchAdapter(SimpleNamespace(config=_switch_config()), "google/switch-base-128")

  plan = build_encoder_hot_initial_plan(path, adapter, total_slots=4)

  assert plan == [(1, 20), (0, 10), (0, 11), (0, 12)]


def test_build_encoder_hot_initial_plan_falls_back_to_frozen_order_without_counts(tmp_path):
  payload = {
    "frozen_hot_experts": {
      "encoder.block.1.layer.1.mlp.router.classifier": [7, 6],
      "encoder.block.3.layer.1.mlp.router.classifier": [5, 4],
    }
  }
  path = tmp_path / "hot.json"
  path.write_text(json.dumps(payload))
  adapter = SwitchAdapter(SimpleNamespace(config=_switch_config()), "google/switch-base-128")

  plan = build_encoder_hot_initial_plan(path, adapter, total_slots=3)

  assert plan == [(1, 5), (0, 7), (0, 6)]


def test_build_encoder_hot_initial_plan_rejects_short_hot_snapshot_by_default(tmp_path):
  payload = {
    "frozen_hot_experts": {
      "encoder.block.1.layer.1.mlp.router.classifier": [7],
    }
  }
  path = tmp_path / "hot.json"
  path.write_text(json.dumps(payload))
  adapter = SwitchAdapter(SimpleNamespace(config=_switch_config()), "google/switch-base-128")

  with pytest.raises(ValueError, match="has 1 hot entries, expected 2"):
    build_encoder_hot_initial_plan(path, adapter, total_slots=2)


def test_build_encoder_hot_initial_plan_uses_frozen_to_fill_layers_missing_from_summary(tmp_path):
  payload = {
    "frozen_hot_experts": {
      "encoder.block.1.layer.1.mlp.router.classifier": [10],
      "encoder.block.3.layer.1.mlp.router.classifier": [20],
    },
    "expert_usage_summary": {
      "encoder.block.1.layer.1.mlp.router.classifier": {
        "top_token_counts": [{"eid": 11, "count": 100}],
      },
    },
  }
  path = tmp_path / "hot.json"
  path.write_text(json.dumps(payload))
  adapter = SwitchAdapter(SimpleNamespace(config=_switch_config()), "google/switch-base-128")

  plan = build_encoder_hot_initial_plan(path, adapter, total_slots=2)

  assert plan == [(1, 20), (0, 11)]


def test_build_encoder_coverage_initial_plan_selects_max_fit_and_fills_global_hot(tmp_path):
  payload = {
    "expert_usage_summary": {
      "encoder.block.1.layer.1.mlp.router.classifier": {
        "token_total_hits": 100,
        "top_token_counts": [
          {"eid": 10, "count": 50},
          {"eid": 11, "count": 30},
          {"eid": 12, "count": 10},
          {"eid": 13, "count": 10},
        ],
      },
      "encoder.block.3.layer.1.mlp.router.classifier": {
        "token_total_hits": 100,
        "top_token_counts": [
          {"eid": 20, "count": 40},
          {"eid": 21, "count": 30},
          {"eid": 22, "count": 20},
          {"eid": 23, "count": 10},
        ],
      },
      "decoder.block.1.layer.2.mlp.router.classifier": {
        "top_token_counts": [{"eid": 99, "count": 1000}],
      },
    },
  }
  path = tmp_path / "hot.json"
  path.write_text(json.dumps(payload))
  adapter = SwitchAdapter(SimpleNamespace(config=_switch_config()), "google/switch-base-128")

  result = build_encoder_coverage_initial_plan(
    path,
    adapter,
    total_slots=5,
    coverage_step=0.10,
  )

  assert result.coverage == pytest.approx(0.80)
  assert result.coverage_slots == 5
  assert result.plan == [(1, 20), (1, 21), (1, 22), (0, 10), (0, 11)]
  assert all(layer_idx < adapter.num_encoder_sparse_layers for layer_idx, _eid in result.plan)

def test_build_encoder_coverage_initial_plan_validates_step_and_encoder_entries_for_zero_slots(tmp_path):
  payload = {"expert_usage_summary": {}}
  path = tmp_path / "hot.json"
  path.write_text(json.dumps(payload))
  adapter = SwitchAdapter(SimpleNamespace(config=_switch_config()), "google/switch-base-128")

  with pytest.raises(ValueError, match=r"coverage_step must be in \(0, 1\]"):
    build_encoder_coverage_initial_plan(path, adapter, total_slots=0, coverage_step=2.0)

  with pytest.raises(ValueError, match="has no encoder expert entries"):
    build_encoder_coverage_initial_plan(path, adapter, total_slots=0)


def test_build_encoder_coverage_initial_plan_rejects_short_snapshot_without_fallback(tmp_path):
  payload = {
    "expert_usage_summary": {
      "encoder.block.1.layer.1.mlp.router.classifier": {
        "token_total_hits": 10,
        "top_token_counts": [{"eid": 2, "count": 10}],
      },
    },
  }
  path = tmp_path / "hot.json"
  path.write_text(json.dumps(payload))
  adapter = SwitchAdapter(SimpleNamespace(config=_switch_config(num_experts=4)), "google/switch-base-128")

  with pytest.raises(ValueError, match="has 1 entries, expected 3"):
    build_encoder_coverage_initial_plan(path, adapter, total_slots=3)

def test_build_encoder_coverage_initial_plan_can_fill_missing_encoder_slots_sequentially(tmp_path):
  payload = {
    "expert_usage_summary": {
      "encoder.block.1.layer.1.mlp.router.classifier": {
        "token_total_hits": 10,
        "top_token_counts": [{"eid": 2, "count": 10}],
      },
    },
  }
  path = tmp_path / "hot.json"
  path.write_text(json.dumps(payload))
  adapter = SwitchAdapter(SimpleNamespace(config=_switch_config(num_experts=4)), "google/switch-base-128")

  result = build_encoder_coverage_initial_plan(
    path,
    adapter,
    total_slots=3,
    allow_sequential_fallback=True,
  )

  assert result.plan == [(0, 2), (0, 0), (0, 1)]
  assert all(layer_idx < adapter.num_encoder_sparse_layers for layer_idx, _eid in result.plan)

def test_build_encoder_coverage_initial_plan_missing_encoder_layer_caps_coverage_at_zero(tmp_path):
  payload = {
    "expert_usage_summary": {
      "encoder.block.1.layer.1.mlp.router.classifier": {
        "token_total_hits": 100,
        "top_token_counts": [
          {"eid": 0, "count": 90},
          {"eid": 1, "count": 10},
        ],
      },
    },
  }
  path = tmp_path / "hot.json"
  path.write_text(json.dumps(payload))
  adapter = SwitchAdapter(SimpleNamespace(config=_switch_config(num_experts=4)), "google/switch-base-128")

  result = build_encoder_coverage_initial_plan(
    path,
    adapter,
    total_slots=3,
    allow_sequential_fallback=True,
  )

  assert result.coverage == 0.0
  assert result.coverage_slots == 0
  assert result.plan == [(0, 0), (0, 1), (0, 2)]
  assert all(layer_idx < adapter.num_encoder_sparse_layers for layer_idx, _eid in result.plan)


def test_build_encoder_coverage_initial_plan_rejects_slots_above_encoder_capacity(tmp_path):
  payload = {
    "expert_usage_summary": {
      "encoder.block.1.layer.1.mlp.router.classifier": {
        "token_total_hits": 10,
        "top_token_counts": [{"eid": 0, "count": 10}],
      },
    },
  }
  path = tmp_path / "hot.json"
  path.write_text(json.dumps(payload))
  adapter = SwitchAdapter(SimpleNamespace(config=_switch_config(num_experts=2)), "google/switch-base-128")

  with pytest.raises(ValueError, match="encoder capacity 4 is smaller than initial slots 5"):
    build_encoder_coverage_initial_plan(path, adapter, total_slots=5)

def test_build_encoder_balanced_hot_initial_plan_spreads_slots_by_layer_then_hot_rank(tmp_path):
  payload = {
    "expert_usage_summary": {
      "encoder.block.1.layer.1.mlp.router.classifier": {
        "top_token_counts": [
          {"eid": 10, "count": 100},
          {"eid": 11, "count": 90},
          {"eid": 12, "count": 80},
        ],
      },
      "encoder.block.3.layer.1.mlp.router.classifier": {
        "top_token_counts": [
          {"eid": 20, "count": 1000},
          {"eid": 21, "count": 10},
          {"eid": 22, "count": 1},
        ],
      },
    },
  }
  path = tmp_path / "hot.json"
  path.write_text(json.dumps(payload))
  adapter = SwitchAdapter(SimpleNamespace(config=_switch_config()), "google/switch-base-128")

  plan = build_encoder_balanced_hot_initial_plan(path, adapter, total_slots=4)

  assert plan == [(1, 20), (1, 21), (0, 10), (0, 11)]


def test_build_encoder_balanced_hot_initial_plan_assigns_remainder_by_marginal_count(tmp_path):
  payload = {
    "expert_usage_summary": {
      "encoder.block.1.layer.1.mlp.router.classifier": {
        "top_token_counts": [
          {"eid": 10, "count": 100},
          {"eid": 11, "count": 90},
          {"eid": 12, "count": 80},
        ],
      },
      "encoder.block.3.layer.1.mlp.router.classifier": {
        "top_token_counts": [
          {"eid": 20, "count": 1000},
          {"eid": 21, "count": 10},
          {"eid": 22, "count": 1},
        ],
      },
    },
  }
  path = tmp_path / "hot.json"
  path.write_text(json.dumps(payload))
  adapter = SwitchAdapter(SimpleNamespace(config=_switch_config()), "google/switch-base-128")

  plan = build_encoder_balanced_hot_initial_plan(path, adapter, total_slots=5)

  assert plan == [(1, 20), (1, 21), (0, 10), (0, 11), (0, 12)]


def test_build_encoder_balanced_hot_initial_plan_can_fill_missing_slots_sequentially(tmp_path):
  payload = {
    "expert_usage_summary": {
      "encoder.block.1.layer.1.mlp.router.classifier": {
        "top_token_counts": [{"eid": 2, "count": 10}],
      },
      "encoder.block.3.layer.1.mlp.router.classifier": {
        "top_token_counts": [{"eid": 3, "count": 10}],
      },
    },
  }
  path = tmp_path / "hot.json"
  path.write_text(json.dumps(payload))
  adapter = SwitchAdapter(SimpleNamespace(config=_switch_config(num_experts=4)), "google/switch-base-128")

  plan = build_encoder_balanced_hot_initial_plan(
    path,
    adapter,
    total_slots=6,
    allow_sequential_fallback=True,
  )

  assert plan == [(1, 3), (1, 0), (1, 1), (0, 2), (0, 0), (0, 1)]

def test_build_encoder_l0_priority_hot_initial_plan_prefers_l0_then_splits_remaining(tmp_path):
  payload = {
    "expert_usage_summary": {
      "encoder.block.1.layer.1.mlp.router.classifier": {
        "top_token_counts": [
          {"eid": eid, "count": 100 - eid}
          for eid in range(8)
        ],
      },
      "encoder.block.3.layer.1.mlp.router.classifier": {
        "top_token_counts": [
          {"eid": eid, "count": 60 - eid}
          for eid in range(8)
        ],
      },
    },
  }
  path = tmp_path / "hot.json"
  path.write_text(json.dumps(payload))
  adapter = SwitchAdapter(SimpleNamespace(config=_switch_config(num_experts=8)), "google/switch-base-128")

  plan = build_encoder_l0_priority_hot_initial_plan(path, adapter, total_slots=10)

  assert plan == [
    (1, 0), (1, 1), (1, 2), (1, 3),
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
  ]


def test_build_hot_initial_plan_uses_encoder_k90_then_even_decoder_slots(tmp_path):
  payload = {
    "expert_usage_summary": {
      "encoder.block.1.layer.1.mlp.router.classifier": {
        "token_total_hits": 100,
        "top_token_counts": [
          {"eid": 10, "count": 60},
          {"eid": 11, "count": 30},
          {"eid": 12, "count": 10},
        ],
      },
      "encoder.block.3.layer.1.mlp.router.classifier": {
        "token_total_hits": 100,
        "top_token_counts": [
          {"eid": 20, "count": 80},
          {"eid": 21, "count": 10},
          {"eid": 22, "count": 10},
        ],
      },
      "decoder.block.1.layer.2.mlp.router.classifier": {
        "top_token_counts": [
          {"eid": 30, "count": 50},
          {"eid": 31, "count": 40},
        ],
      },
      "decoder.block.3.layer.2.mlp.router.classifier": {
        "top_token_counts": [
          {"eid": 40, "count": 70},
          {"eid": 41, "count": 20},
        ],
      },
    }
  }
  path = tmp_path / "hot.json"
  path.write_text(json.dumps(payload))
  adapter = SwitchAdapter(SimpleNamespace(config=_switch_config()), "google/switch-base-128")

  plan = build_hot_initial_plan(path, adapter, total_slots=6)

  assert plan == [
    (3, 40),
    (2, 30),
    (1, 20),
    (1, 21),
    (0, 10),
    (0, 11),
  ]


def test_build_hot_initial_plan_errors_when_encoder_k90_exceeds_total_slots(tmp_path):
  payload = {
    "expert_usage_summary": {
      "encoder.block.1.layer.1.mlp.router.classifier": {
        "token_total_hits": 100,
        "top_token_counts": [
          {"eid": 10, "count": 50},
          {"eid": 11, "count": 40},
        ],
      },
      "encoder.block.3.layer.1.mlp.router.classifier": {
        "token_total_hits": 100,
        "top_token_counts": [
          {"eid": 20, "count": 50},
          {"eid": 21, "count": 40},
        ],
      },
    }
  }
  path = tmp_path / "hot.json"
  path.write_text(json.dumps(payload))
  adapter = SwitchAdapter(SimpleNamespace(config=_switch_config()), "google/switch-base-128")

  with pytest.raises(ValueError, match="encoder k90 requires 4 slots, but total_slots=3"):
    build_hot_initial_plan(path, adapter, total_slots=3)


def test_build_hot_initial_plan_skips_non_sparse_router_keys(tmp_path):
  payload = {
    "frozen_hot_experts": {
      "encoder.block.0.layer.1.mlp.router.classifier": [99],
      "encoder.block.1.layer.1.mlp.router.classifier": [10],
      "encoder.block.3.layer.1.mlp.router.classifier": [20],
    },
    "expert_usage_summary": {
      "encoder.block.1.layer.1.mlp.router.classifier": {
        "token_total_hits": 10,
        "top_token_counts": [{"eid": 0, "count": 9}],
      },
      "encoder.block.3.layer.1.mlp.router.classifier": {
        "token_total_hits": 10,
        "top_token_counts": [{"eid": 1, "count": 9}],
      },
    },
  }
  path = tmp_path / "hot.json"
  path.write_text(json.dumps(payload))
  adapter = SwitchAdapter(SimpleNamespace(config=_switch_config()), "google/switch-base-128")

  plan = build_hot_initial_plan(path, adapter, total_slots=2)

  assert plan == [(1, 1), (0, 0)]


def test_build_hot_initial_plan_errors_when_decoder_capacity_cannot_fill_remaining_slots(tmp_path):
  payload = {
    "expert_usage_summary": {
      "encoder.block.1.layer.1.mlp.router.classifier": {
        "token_total_hits": 10,
        "top_token_counts": [{"eid": 0, "count": 9}],
      },
      "encoder.block.3.layer.1.mlp.router.classifier": {
        "token_total_hits": 10,
        "top_token_counts": [{"eid": 1, "count": 9}],
      },
    },
  }
  path = tmp_path / "hot.json"
  path.write_text(json.dumps(payload))
  adapter = SwitchAdapter(
    SimpleNamespace(config=_switch_config(num_experts=2)),
    "google/switch-base-128",
  )

  with pytest.raises(ValueError, match="decoder capacity 4 is smaller than remaining slots 5"):
    build_hot_initial_plan(path, adapter, total_slots=7)


def test_build_hot_initial_plan_is_available_for_inject_model_default():
  assert callable(build_hot_initial_plan)


def test_build_decoder_warmup_overlap_plan_excludes_initial_and_interleaves_depth(tmp_path):
  payload = {
    "expert_usage_summary": {
      "decoder.block.1.layer.2.mlp.router.classifier": {
        "top_token_counts": [
          {"eid": 1, "count": 100},
          {"eid": 2, "count": 90},
          {"eid": 3, "count": 80},
        ],
      },
      "decoder.block.3.layer.2.mlp.router.classifier": {
        "top_token_counts": [
          {"eid": 4, "count": 70},
          {"eid": 5, "count": 60},
        ],
      },
    }
  }
  path = tmp_path / "hot.json"
  path.write_text(json.dumps(payload))
  adapter = SwitchAdapter(SimpleNamespace(config=_switch_config()), "google/switch-base-128")

  plan = build_decoder_warmup_overlap_plan(
    path,
    adapter,
    initial_plan={(2, 1), (3, 4)},
  )

  assert plan == [(2, 2), (3, 5), (2, 3)]


def test_round_like_cpp_matches_half_away_from_zero_cache_slot_count():
  assert round_like_cpp(2.5) == 3
  assert round_like_cpp(3.5) == 4


def test_runner_util_parses_initial_hot_expert_file_arg():
  parsed = parse_args([
    "--initial_cache_policy", "hot_expert",
    "--initial_hot_expert_file", "/tmp/hot.json",
  ])

  assert parsed["initial_cache_policy"] == "hot_expert"
  assert parsed["initial_hot_expert_file"] == "/tmp/hot.json"


def test_runner_util_parses_hot_encoder_coverage_policy():
  parsed = parse_args([
    "--initial_cache_policy", "hot_encoder_coverage",
    "--initial_hot_expert_file", "/tmp/hot.json",
  ])

  assert parsed["initial_cache_policy"] == "hot_encoder_coverage"
  assert parsed["initial_hot_expert_file"] == "/tmp/hot.json"


def test_runner_util_parses_hot_encoder_balanced_coverage_policy():
  parsed = parse_args([
    "--initial_cache_policy", "hot_encoder_balanced_coverage",
    "--initial_hot_expert_file", "/tmp/hot.json",
  ])

  assert parsed["initial_cache_policy"] == "hot_encoder_balanced_coverage"
  assert parsed["initial_hot_expert_file"] == "/tmp/hot.json"

def test_inject_model_imports_hot_encoder_l0_priority_builder():
  source = inspect.getsource(inject_model)

  assert "build_encoder_l0_priority_hot_initial_plan" in source
  assert "from sparse_llm_cache.utils.hot_experts import" in source


def test_runner_util_parses_hot_encoder_l0_priority_policy():
  parsed = parse_args([
    "--initial_cache_policy", "hot_encoder_l0_priority_coverage",
    "--initial_hot_expert_file", "/tmp/hot.json",
  ])

  assert parsed["initial_cache_policy"] == "hot_encoder_l0_priority_coverage"
  assert parsed["initial_hot_expert_file"] == "/tmp/hot.json"

def test_runner_util_parses_decoder_warmup_overlap_and_scheduler_aware_policy():
  parsed = parse_args([
    "--enable_decoder_warmup_overlap", "True",
    "--cache_policy", "scheduler_aware",
  ])

  assert parsed["enable_decoder_warmup_overlap"] is True
  assert parsed["cache_policy"] == "scheduler_aware"


def test_inject_model_exposes_initial_hot_expert_file_parameter():
  assert "initial_hot_expert_file" in inspect.signature(inject_model).parameters


def test_inject_model_exposes_decoder_warmup_overlap_parameter():
  assert "enable_decoder_warmup_overlap" in inspect.signature(inject_model).parameters
