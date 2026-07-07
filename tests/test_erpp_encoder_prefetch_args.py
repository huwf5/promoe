from pathlib import Path

import pytest
import torch

from sparse_llm_cache import cpp_worker
from sparse_llm_cache.utils.runner_util import prepare_argparser


def test_production_erpp_prefetch_args_exist_and_validation_args_do_not():
  parser = prepare_argparser()
  args = parser.parse_args(
    [
      "--enable_erpp_encoder_prefetch",
      "true",
      "--erpp_encoder_model_path",
      "erpp_encoder_predictor.ts",
      "--erpp_encoder_budgets",
      "fixed_mean",
    ]
  )

  assert args.enable_erpp_encoder_prefetch is True
  assert args.erpp_encoder_model_path == "erpp_encoder_predictor.ts"
  assert args.erpp_encoder_budgets == "fixed_mean"
  assert not hasattr(args, "enable_erpp_encoder_validation")
  assert not hasattr(args, "erpp_encoder_validation_mode")


def test_erpp_prefetch_args_pass_through_to_hack_transformers_parser():
  parser = prepare_argparser()
  args = vars(
    parser.parse_args(
      [
        "--enable_erpp_encoder_prefetch",
        "true",
        "--erpp_encoder_model_path",
        "/tmp/erpp.ts",
        "--erpp_encoder_budgets",
        "70,49,56,55,55,51",
      ]
    )
  )

  assert args["enable_erpp_encoder_prefetch"] is True
  assert args["erpp_encoder_model_path"] == "/tmp/erpp.ts"
  assert args["erpp_encoder_budgets"] == "70,49,56,55,55,51"


def test_erpp_encoder_layers_arg_exists_and_defaults_are_not_required():
  parser = prepare_argparser()
  args = parser.parse_args(
    [
      "--enable_erpp_encoder_prefetch",
      "true",
      "--erpp_encoder_model_path",
      "erpp_encoder_predictor.ts",
      "--erpp_encoder_budgets",
      "fixed_p90",
      "--erpp_encoder_layers",
      "-1,-2",
    ]
  )

  assert args.erpp_encoder_layers == "-1,-2"


def test_erpp_encoder_jit_refill_args_parse():
  parser = prepare_argparser()
  args = parser.parse_args(
    [
      "--enable_erpp_encoder_jit_refill",
      "true",
      "--erpp_encoder_jit_refill_window",
      "1",
      "--erpp_encoder_jit_refill_floor_mode",
      "avg",
      "--erpp_encoder_jit_refill_floor_value",
      "-1",
      "--erpp_encoder_jit_refill_low_watermark_ratio",
      "0.90",
      "--erpp_encoder_jit_refill_layers",
      "all",
      "--erpp_encoder_jit_refill_per_idle",
      "1",
      "--enable_erpp_encoder_jit_topk_cover",
      "true",
    ]
  )

  assert args.enable_erpp_encoder_jit_refill is True
  assert args.erpp_encoder_jit_refill_window == 1
  assert args.erpp_encoder_jit_refill_floor_mode == "avg"
  assert args.erpp_encoder_jit_refill_floor_value == -1
  assert args.erpp_encoder_jit_refill_low_watermark_ratio == 0.90
  assert args.erpp_encoder_jit_refill_layers == "all"
  assert args.erpp_encoder_jit_refill_per_idle == 1
  assert args.enable_erpp_encoder_jit_topk_cover is True


def test_erpp_encoder_jit_refill_budget_floor_mode_arg_parse():
  parser = prepare_argparser()
  args = parser.parse_args(
    [
      "--enable_erpp_encoder_jit_refill",
      "true",
      "--erpp_encoder_jit_refill_floor_mode",
      "budget",
    ]
  )

  assert args.erpp_encoder_jit_refill_floor_mode == "budget"


def test_enable_encoder_reclaim_arg_parse():
  parser = prepare_argparser()
  args = parser.parse_args(["--enable_encoder_reclaim", "false"])

  assert args.enable_encoder_reclaim is False


def test_erpp_prefetch_args_parse_into_module_meta():
  meta = cpp_worker.ModuleMeta(12, 128)
  meta.init_param_list([])
  meta.init_from_map(
    {
      "model_arch_string": "google/switch-base-128",
      "num_expert_per_token": "1",
      "num_encoder_moe_layer": "6",
      "num_decoder_moe_layer": "6",
      "per_layer_cache": "false",
      "cache_policy": "scheduler_aware",
      "enable_erpp_encoder_prefetch": "true",
      "erpp_encoder_model_path": "/tmp/erpp.ts",
      "erpp_encoder_budgets": "70,49,56,55,55,51",
      "erpp_encoder_layers": "-1,-2",
    }
  )
  meta.handle_uninited_configs()

  assert meta.enable_erpp_encoder_prefetch is True
  assert meta.erpp_encoder_model_path == "/tmp/erpp.ts"
  assert meta.erpp_encoder_budgets == "70,49,56,55,55,51"
  assert meta.erpp_encoder_layers == "-1,-2"


def test_enable_encoder_reclaim_parses_into_module_meta_and_defaults_true():
  default_meta = cpp_worker.ModuleMeta(12, 128)
  default_meta.init_param_list([])
  default_meta.init_from_map(
    {
      "model_arch_string": "google/switch-base-128",
      "num_expert_per_token": "1",
      "num_encoder_moe_layer": "6",
      "num_decoder_moe_layer": "6",
    }
  )
  default_meta.handle_uninited_configs()
  assert default_meta.enable_encoder_reclaim is True

  disabled_meta = cpp_worker.ModuleMeta(12, 128)
  disabled_meta.init_param_list([])
  disabled_meta.init_from_map(
    {
      "model_arch_string": "google/switch-base-128",
      "num_expert_per_token": "1",
      "num_encoder_moe_layer": "6",
      "num_decoder_moe_layer": "6",
      "enable_encoder_reclaim": "false",
    }
  )
  disabled_meta.handle_uninited_configs()
  assert disabled_meta.enable_encoder_reclaim is False


def test_erpp_encoder_dynamic_noisy_or_budget_arg_passes_to_module_meta():
  parser = prepare_argparser()
  args = parser.parse_args(
    [
      "--enable_erpp_encoder_prefetch",
      "true",
      "--erpp_encoder_model_path",
      "sida_noisy_or.ts",
      "--erpp_encoder_budgets",
      "dynamic_noisy_or_sum",
    ]
  )

  assert args.erpp_encoder_budgets == "dynamic_noisy_or_sum"

  meta = cpp_worker.ModuleMeta(12, 128)
  meta.init_param_list([])
  meta.init_from_map(
    {
      "model_arch_string": "google/switch-base-128",
      "num_expert_per_token": "1",
      "num_encoder_moe_layer": "6",
      "num_decoder_moe_layer": "6",
      "per_layer_cache": "false",
      "cache_policy": "scheduler_aware",
      "enable_erpp_encoder_prefetch": "true",
      "erpp_encoder_model_path": "sida_noisy_or.ts",
      "erpp_encoder_budgets": "dynamic_noisy_or_sum",
    }
  )
  meta.handle_uninited_configs()

  assert meta.erpp_encoder_budgets == "dynamic_noisy_or_sum"


def test_erpp_noisy_or_dynamic_budget_helper_ceil_and_clamp():
  scores = torch.tensor([0.2, 0.7, 0.1, 0.0])

  assert cpp_worker.erpp_noisy_or_sum_budget_for_test(scores, 4) == 1
  assert cpp_worker.erpp_noisy_or_sum_budget_for_test(scores + 0.3, 4) == 3
  assert cpp_worker.erpp_noisy_or_sum_budget_for_test(scores + 10.0, 4) == 4


def test_erpp_encoder_jit_refill_args_parse_into_module_meta():
  meta = cpp_worker.ModuleMeta(12, 128)
  meta.init_param_list([])
  meta.init_from_map(
    {
      "model_arch_string": "google/switch-base-128",
      "num_expert_per_token": "1",
      "num_encoder_moe_layer": "6",
      "num_decoder_moe_layer": "6",
      "per_layer_cache": "false",
      "cache_policy": "scheduler_aware",
      "enable_erpp_encoder_prefetch": "true",
      "erpp_encoder_model_path": "/tmp/erpp.ts",
      "erpp_encoder_budgets": "70,49,56,55,55,51",
      "enable_erpp_encoder_jit_refill": "true",
      "erpp_encoder_jit_refill_window": "1",
      "erpp_encoder_jit_refill_floor_mode": "avg",
      "erpp_encoder_jit_refill_floor_value": "-1",
      "erpp_encoder_jit_refill_low_watermark_ratio": "0.90",
      "erpp_encoder_jit_refill_layers": "all",
      "erpp_encoder_jit_refill_per_idle": "1",
      "enable_erpp_encoder_jit_topk_cover": "true",
    }
  )
  meta.handle_uninited_configs()

  assert meta.enable_erpp_encoder_jit_refill is True
  assert meta.erpp_encoder_jit_refill_window == 1
  assert meta.erpp_encoder_jit_refill_floor_mode == "avg"
  assert meta.erpp_encoder_jit_refill_floor_value == -1
  assert meta.erpp_encoder_jit_refill_low_watermark_ratio == 0.90
  assert meta.erpp_encoder_jit_refill_layers == "all"
  assert meta.erpp_encoder_jit_refill_per_idle == 1
  assert meta.enable_erpp_encoder_jit_topk_cover is True


def test_erpp_encoder_jit_refill_accepts_budget_floor_mode():
  meta = cpp_worker.ModuleMeta(12, 128)
  meta.init_param_list([])
  meta.init_from_map(
    {
      "model_arch_string": "google/switch-base-128",
      "num_expert_per_token": "1",
      "num_encoder_moe_layer": "6",
      "num_decoder_moe_layer": "6",
      "per_layer_cache": "false",
      "cache_policy": "scheduler_aware",
      "enable_erpp_encoder_prefetch": "true",
      "erpp_encoder_model_path": "/tmp/erpp.ts",
      "erpp_encoder_budgets": "dynamic_noisy_or_sum",
      "enable_erpp_encoder_jit_refill": "true",
      "erpp_encoder_jit_refill_window": "1",
      "erpp_encoder_jit_refill_floor_mode": "budget",
      "erpp_encoder_jit_refill_floor_value": "-1",
      "erpp_encoder_jit_refill_low_watermark_ratio": "0.90",
      "erpp_encoder_jit_refill_per_idle": "-1",
    }
  )
  meta.handle_uninited_configs()

  assert meta.erpp_encoder_jit_refill_floor_mode == "budget"


def test_erpp_encoder_jit_refill_requires_global_cache_before_cache_policy():
  meta = cpp_worker.ModuleMeta(12, 128)
  meta.init_param_list([])
  meta.init_from_map(
    {
      "model_arch_string": "google/switch-base-128",
      "num_expert_per_token": "1",
      "num_encoder_moe_layer": "6",
      "num_decoder_moe_layer": "6",
      "per_layer_cache": "true",
      "cache_policy": "lru",
      "enable_erpp_encoder_prefetch": "true",
      "erpp_encoder_model_path": "/tmp/erpp.ts",
      "erpp_encoder_budgets": "70,49,56,55,55,51",
      "enable_erpp_encoder_jit_refill": "true",
    }
  )

  with pytest.raises(
    RuntimeError,
    match="ERPP encoder JIT refill requires per_layer_cache=false",
  ):
    meta.handle_uninited_configs()


def test_erpp_prefetch_requires_scheduler_aware_cache_policy():
  meta = cpp_worker.ModuleMeta(12, 128)
  meta.init_param_list([])
  meta.init_from_map(
    {
      "model_arch_string": "google/switch-base-128",
      "num_expert_per_token": "1",
      "num_encoder_moe_layer": "6",
      "num_decoder_moe_layer": "6",
      "per_layer_cache": "false",
      "cache_policy": "lru",
      "enable_erpp_encoder_prefetch": "true",
      "erpp_encoder_model_path": "/tmp/erpp.ts",
      "erpp_encoder_budgets": "70,49,56,55,55,51",
    }
  )

  with pytest.raises(
    RuntimeError,
    match="ERPP encoder prefetch requires cache_policy=scheduler_aware",
  ):
    meta.handle_uninited_configs()



def _jit_refill_meta_with_per_idle(value):
  meta = cpp_worker.ModuleMeta(12, 128)
  meta.init_param_list([])
  meta.init_from_map(
    {
      "model_arch_string": "google/switch-base-128",
      "num_expert_per_token": "1",
      "num_encoder_moe_layer": "6",
      "num_decoder_moe_layer": "6",
      "per_layer_cache": "false",
      "cache_policy": "scheduler_aware",
      "enable_erpp_encoder_prefetch": "true",
      "erpp_encoder_model_path": "/tmp/erpp.ts",
      "erpp_encoder_budgets": "70,49,56,55,55,51",
      "enable_erpp_encoder_jit_refill": "true",
      "erpp_encoder_jit_refill_window": "1",
      "erpp_encoder_jit_refill_floor_mode": "avg",
      "erpp_encoder_jit_refill_floor_value": "-1",
      "erpp_encoder_jit_refill_low_watermark_ratio": "0.90",
      "erpp_encoder_jit_refill_layers": "all",
      "erpp_encoder_jit_refill_per_idle": str(value),
    }
  )
  return meta


def test_erpp_encoder_jit_refill_per_idle_minus_one_means_unlimited():
  parser = prepare_argparser()
  args = parser.parse_args(["--erpp_encoder_jit_refill_per_idle", "-1"])
  assert args.erpp_encoder_jit_refill_per_idle == -1

  meta = _jit_refill_meta_with_per_idle(-1)
  meta.handle_uninited_configs()
  assert meta.erpp_encoder_jit_refill_per_idle == -1


@pytest.mark.parametrize("value", [0, -2])
def test_erpp_encoder_jit_refill_per_idle_rejects_zero_and_below_minus_one(value):
  meta = _jit_refill_meta_with_per_idle(value)

  with pytest.raises(
    RuntimeError,
    match="ERPP encoder JIT refill requires erpp_encoder_jit_refill_per_idle == -1 or >= 1",
  ):
    meta.handle_uninited_configs()


def test_erpp_predictor_uses_computed_budget_for_ranking_limit():
  source = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "cpp_worker"
    / "erpp_encoder_predictor.cpp"
  ).read_text()

  assert "const int limit = encoder_jit_ranking_limit(layer, budget);" in source
  assert "const int limit = encoder_jit_ranking_limit(layer);" not in source
