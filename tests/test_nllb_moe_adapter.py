from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sparse_llm_cache.model_adapters import get_model_adapter


def _nllb_config(**overrides):
    values = {
        "model_type": "nllb-moe",
        "num_experts": 128,
        "encoder_layers": 24,
        "decoder_layers": 24,
        "encoder_sparse_step": 4,
        "decoder_sparse_step": 4,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeModule:
    pass


def test_get_model_adapter_returns_nllb_moe_adapter_for_nllb_config():
    from sparse_llm_cache.model_adapters.nllb_moe import NllbMoeAdapter

    model = SimpleNamespace(config=_nllb_config())
    adapter = get_model_adapter(model, model_id="facebook/nllb-moe-54b")

    assert isinstance(adapter, NllbMoeAdapter)


def test_nllb_moe_layer_mapping_uses_global_encoder_decoder_layers():
    from sparse_llm_cache.model_adapters.nllb_moe import NllbMoeAdapter

    adapter = NllbMoeAdapter(SimpleNamespace(config=_nllb_config()), "facebook/nllb-moe-54b")

    assert adapter.encoder_sparse_layer_ids == [3, 7, 11, 15, 19, 23]
    assert adapter.decoder_sparse_layer_ids == [3, 7, 11, 15, 19, 23]
    assert adapter.num_moe_layer == 12
    assert adapter.num_expert_per_layer == 128
    assert adapter.num_expert_per_token == 2
    assert adapter.global_layer_id("encoder", 0) == 0
    assert adapter.global_layer_id("encoder", 5) == 5
    assert adapter.global_layer_id("decoder", 0) == 6
    assert adapter.global_layer_id("decoder", 5) == 11


def test_nllb_moe_filters_and_metadata_match_transformers_module_names():
    from sparse_llm_cache.model_adapters.nllb_moe import NllbMoeAdapter

    adapter = NllbMoeAdapter(SimpleNamespace(config=_nllb_config()), "facebook/nllb-moe-54b")
    expert = FakeModule()
    sparse_mlp = FakeModule()
    dense_mlp = FakeModule()

    expert_name = "model.decoder.layers.7.ffn.experts.expert_17"
    sparse_mlp_name = "model.encoder.layers.11.ffn"
    dense_mlp_name = "model.encoder.layers.10.ffn"

    assert adapter.expert_name_filter(expert_name)
    assert adapter.moe_mlp_name_filter(sparse_mlp_name)
    assert adapter.moe_layer_name_filter(sparse_mlp_name)
    assert not adapter.moe_layer_name_filter(dense_mlp_name)
    assert adapter.expert_meta_parser(expert_name) == (7, 17)

    adapter.add_metadata_to_module(expert, expert_name)
    assert expert._prefix == expert_name
    assert expert._stage == "decoder"
    assert expert._stage_layer_id == 1
    assert expert._layer_id == 7
    assert expert._expert_id == 17

    adapter.add_metadata_to_module(sparse_mlp, sparse_mlp_name)
    assert sparse_mlp._stage == "encoder"
    assert sparse_mlp._stage_layer_id == 2
    assert sparse_mlp._layer_id == 2

    adapter.add_metadata_to_module(dense_mlp, dense_mlp_name)
    assert not hasattr(dense_mlp, "_layer_id")


def test_nllb_moe_rejects_invalid_sparse_step():
    from sparse_llm_cache.model_adapters.nllb_moe import NllbMoeAdapter

    with pytest.raises(ValueError, match="sparse step must be positive"):
        NllbMoeAdapter(
            SimpleNamespace(config=_nllb_config(encoder_sparse_step=0)),
            "facebook/nllb-moe-54b",
        )


class FixedNllbRouter(nn.Module):
    def forward(self, hidden_states, padding_mask=False):
        top_1_mask = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.int64,
            device=hidden_states.device,
        )
        router_probs = torch.tensor(
            [
                [0.75, 0.00, 0.25, 0.00],
                [0.00, 1.00, 0.00, 0.00],
                [0.00, 0.00, 0.50, 0.50],
            ],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        return top_1_mask, router_probs


class RecordingNllbExpert(nn.Module):
    def __init__(self, config, ffn_dim, calls=None):
        super().__init__()
        self.expert_id = None
        self.calls = calls

    def forward(self, hidden_states):
        self.calls.append(self.expert_id)
        return hidden_states


def test_nllb_sparse_mlp_forward_uses_report_experts_returned_order():
    from transformers.models.nllb_moe.configuration_nllb_moe import NllbMoeConfig
    from transformers.models.nllb_moe.modeling_nllb_moe import NllbMoeSparseMLP

    calls = []

    class ExpertFactory(RecordingNllbExpert):
        def __init__(self, config, ffn_dim):
            super().__init__(config, ffn_dim, calls)

    config = NllbMoeConfig(
        d_model=2,
        encoder_ffn_dim=4,
        decoder_ffn_dim=4,
        num_experts=4,
        moe_token_dropout=0.0,
        router_bias=False,
    )
    module = NllbMoeSparseMLP(config, ffn_dim=4, expert_class=ExpertFactory)
    module.router = FixedNllbRouter()
    for expert_id in range(4):
        module.experts[f"expert_{expert_id}"].expert_id = expert_id
    reported = []

    def reverse_report_experts(experts):
        reported.append(experts)
        return torch.flip(experts, dims=[0])

    module.report_experts = reverse_report_experts

    output = module(torch.ones(1, 3, 2))

    assert reported
    assert reported[0].dtype == torch.int64
    assert reported[0].device.type == "cpu"
    assert reported[0].tolist() == [0, 1, 2, 3]
    assert calls == [3, 2, 1, 0]
    assert isinstance(output, tuple)
    assert output[0].shape == (1, 3, 2)


def test_nllb_erpp_encoder_prefetch_module_uses_actual_encoder_layers():
    from sparse_llm_cache.model_adapters.nllb_moe import NllbMoeAdapter

    layer0 = object()
    model = SimpleNamespace(
        config=_nllb_config(),
        model=SimpleNamespace(encoder=SimpleNamespace(layers=[layer0])),
    )
    adapter = NllbMoeAdapter(model, "facebook/nllb-moe-54b")

    assert adapter.erpp_encoder_prefetch_module() is layer0


def test_nllb_adapter_matches_real_transformers_sparse_mlp_names():
    from sparse_llm_cache.model_adapters.nllb_moe import NllbMoeAdapter
    from transformers.models.nllb_moe.configuration_nllb_moe import NllbMoeConfig
    from transformers.models.nllb_moe.modeling_nllb_moe import NllbMoeForConditionalGeneration, NllbMoeSparseMLP

    config = NllbMoeConfig(
        vocab_size=32,
        d_model=8,
        encoder_layers=4,
        decoder_layers=4,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=16,
        decoder_ffn_dim=16,
        num_experts=4,
        encoder_sparse_step=2,
        decoder_sparse_step=2,
        pad_token_id=1,
        decoder_start_token_id=1,
    )
    model = NllbMoeForConditionalGeneration(config)
    adapter = NllbMoeAdapter(model, "facebook/nllb-moe-54b")
    sparse_names = [name for name, module in model.named_modules() if isinstance(module, NllbMoeSparseMLP)]

    assert sparse_names == [
        "model.encoder.layers.1.ffn",
        "model.encoder.layers.3.ffn",
        "model.decoder.layers.1.ffn",
        "model.decoder.layers.3.ffn",
    ]
    assert [adapter.parse_moe_layer_name(name) for name in sparse_names] == [
        ("encoder", 0, 0),
        ("encoder", 1, 1),
        ("decoder", 0, 2),
        ("decoder", 1, 3),
    ]
