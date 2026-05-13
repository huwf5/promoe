from types import SimpleNamespace

import pytest
import torch

from sparse_llm_cache.model_adapters import get_model_adapter
from sparse_llm_cache.model_adapters.switch import SwitchAdapter


def _switch_config(**overrides):
    values = {
        "model_type": "switch_transformers",
        "num_experts": 128,
        "num_selected_experts": 1,
        "num_sparse_encoder_layers": 6,
        "num_sparse_decoder_layers": 6,
        "encoder_sparse_step": 2,
        "decoder_sparse_step": 2,
        "num_layers": 12,
        "num_decoder_layers": 12,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_get_model_adapter_returns_switch_adapter_for_switch_config():
    model = SimpleNamespace(config=_switch_config())
    adapter = get_model_adapter(model, model_id="google/switch-base-128")
    assert isinstance(adapter, SwitchAdapter)


def test_switch_layer_mapping_uses_global_cache_layers_and_stage_local_predictor_layers():
    adapter = SwitchAdapter(SimpleNamespace(config=_switch_config()), "google/switch-base-128")

    assert adapter.encoder_sparse_layer_ids == [1, 3, 5, 7, 9, 11]
    assert adapter.decoder_sparse_layer_ids == [1, 3, 5, 7, 9, 11]
    assert adapter.global_layer_id("encoder", 0) == 0
    assert adapter.global_layer_id("encoder", 5) == 5
    assert adapter.global_layer_id("decoder", 0) == 6
    assert adapter.global_layer_id("decoder", 5) == 11
    assert adapter.predictor_layer_id("decoder", 6) == 0
    assert adapter.predictor_layer_id("decoder", 11) == 5


def test_switch_layer_mapping_is_config_driven_for_other_expert_counts():
    adapter = SwitchAdapter(
        SimpleNamespace(config=_switch_config(num_experts=64, num_sparse_decoder_layers=4, num_decoder_layers=8)),
        "google/switch-base-64",
    )

    assert adapter.num_expert_per_layer == 64
    assert adapter.num_moe_layer == 10
    assert adapter.decoder_sparse_layer_ids == [1, 3, 5, 7]
    assert adapter.global_layer_id("decoder", 0) == 6
    assert adapter.global_layer_id("decoder", 3) == 9


def test_switch_adapter_rejects_non_top1_switch():
    model = SimpleNamespace(config=_switch_config(num_selected_experts=2))
    with pytest.raises(ValueError, match="top-1"):
        SwitchAdapter(model, "google/switch-base-128")


class FakeModule:
    pass


def test_switch_add_metadata_sets_stage_and_global_layer_for_expert():
    adapter = SwitchAdapter(SimpleNamespace(config=_switch_config()), "google/switch-base-128")
    module = FakeModule()

    adapter.add_metadata_to_module(module, "decoder.block.3.layer.2.mlp.experts.expert_17")

    assert module._prefix == "decoder.block.3.layer.2.mlp.experts.expert_17"
    assert module._stage == "decoder"
    assert module._stage_layer_id == 1
    assert module._layer_id == 7
    assert module._expert_id == 17


def test_switch_add_metadata_sets_stage_and_global_layer_for_sparse_mlp():
    adapter = SwitchAdapter(SimpleNamespace(config=_switch_config()), "google/switch-base-128")
    module = FakeModule()

    adapter.add_metadata_to_module(module, "encoder.block.5.layer.1.mlp")

    assert module._prefix == "encoder.block.5.layer.1.mlp"
    assert module._stage == "encoder"
    assert module._stage_layer_id == 2
    assert module._layer_id == 2


def test_switch_moe_layer_filter_ignores_dense_mlp_blocks():
    adapter = SwitchAdapter(SimpleNamespace(config=_switch_config()), "google/switch-base-128")
    dense_module = FakeModule()

    assert adapter.moe_layer_name_filter("encoder.block.0.layer.1.mlp") is False
    assert adapter.moe_layer_name_filter("decoder.block.0.layer.2.mlp") is False
    assert adapter.moe_layer_name_filter("encoder.block.1.layer.1.mlp") is True
    assert adapter.moe_layer_name_filter("decoder.block.1.layer.1.mlp") is False
    assert adapter.moe_layer_name_filter("decoder.block.1.layer.2.mlp") is True
    assert adapter.parse_moe_layer_name("encoder.block.0.layer.1.mlp") is None
    adapter.add_metadata_to_module(dense_module, "encoder.block.0.layer.1.mlp")
    assert not hasattr(dense_module, "_layer_id")


def test_default_and_switch_adapters_expose_required_inject_fields():
    adapter = SwitchAdapter(SimpleNamespace(config=_switch_config()), "google/switch-base-128")

    assert adapter.num_moe_layer == 12
    assert adapter.num_expert_per_layer == 128
    assert adapter.num_expert_per_token == 1
    assert adapter.expert_name_filter("decoder.block.1.layer.2.mlp.experts.expert_0")
    assert adapter.moe_layer_name_filter("decoder.block.1.layer.2.mlp")
    assert adapter.expert_meta_parser("decoder.block.1.layer.2.mlp.experts.expert_0") == (6, 0)


def test_switch_predictor_reporting_skips_encoder_and_maps_decoder_layer():
    adapter = SwitchAdapter(SimpleNamespace(config=_switch_config()), "google/switch-base-128")

    assert adapter.should_report_moe_layer_to_predictor("encoder", 0) is False
    assert adapter.should_report_moe_layer_to_predictor("decoder", 6) is True
    assert adapter.report_layer_id_for_predictor("decoder", 6) == 0
    assert adapter.report_layer_id_for_predictor("decoder", 11) == 5


def test_switch_adapter_can_skip_report_experts_patch_for_modules_without_method():
    adapter = SwitchAdapter(SimpleNamespace(config=_switch_config()), "google/switch-base-128")
    module = FakeModule()
    module._stage = "encoder"
    module._layer_id = 0

    assert adapter.should_patch_report_experts(module) is False


def test_switch_configures_decoder_stage_local_predictor_meta():
    adapter = SwitchAdapter(SimpleNamespace(config=_switch_config()), "google/switch-base-128")
    meta = SimpleNamespace(
        num_encoder_moe_layer=0,
        num_decoder_moe_layer=-1,
        predictor_num_layer=-1,
        predictor_layer_offset=0,
        layer_predict_replace_first_input_with_last_output=True,
    )

    adapter.configure_module_meta(meta)

    assert meta.num_encoder_moe_layer == 6
    assert meta.num_decoder_moe_layer == 6
    assert meta.predictor_num_layer == 6
    assert meta.predictor_layer_offset == 6
    assert meta.layer_predict_replace_first_input_with_last_output is False


def test_switch_extracts_hidden_states_from_sparse_mlp_output_for_predictor():
    adapter = SwitchAdapter(SimpleNamespace(config=_switch_config()), "google/switch-base-128")
    hidden_states = torch.zeros(1, 2, 4)
    router_logits = torch.ones(1, 2, 128)
    expert_index = torch.zeros(1, 2, dtype=torch.long)

    extracted = adapter.extract_moe_layer_output_for_predictor(
        (hidden_states, (router_logits, expert_index))
    )

    assert extracted is hidden_states


def test_switch_extracts_hidden_states_from_flat_sparse_mlp_output_for_predictor():
    adapter = SwitchAdapter(SimpleNamespace(config=_switch_config()), "google/switch-base-128")
    hidden_states = torch.zeros(1, 2, 4)
    router_logits = torch.ones(1, 2, 128)

    extracted = adapter.extract_moe_layer_output_for_predictor((hidden_states, router_logits))

    assert extracted is hidden_states
