from types import SimpleNamespace

from sparse_llm_cache.model_adapters.nllb_moe import NllbMoeAdapter
from sparse_llm_cache.utils import hot_experts


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


def test_router_stage_block_accepts_nllb_layers_key():
    assert hot_experts._router_stage_block("encoder.layers.3.ffn.router") == ("encoder", 3)
    assert hot_experts._router_stage_block("decoder.layers.7.ffn.router") == ("decoder", 7)


def test_router_stage_block_keeps_switch_block_key_support():
    assert hot_experts._router_stage_block("encoder.block.1.layer.1.mlp.router") == ("encoder", 1)
    assert hot_experts._router_stage_block("decoder.block.3.layer.2.mlp.router") == ("decoder", 3)


def test_hot_pairs_by_layer_maps_nllb_sparse_layer_to_global_id():
    adapter = NllbMoeAdapter(SimpleNamespace(config=_nllb_config()), "facebook/nllb-moe-54b")
    payload = {
        "frozen_hot_experts": {
            "encoder.layers.3.ffn.router": [5, 2],
            "decoder.layers.3.ffn.router": [9, 1],
        },
        "expert_usage_summary": {
            "encoder.layers.3.ffn.router": {
                "token_total_hits": 10,
                "top_token_eids": [5, 2],
                "top_token_counts": [
                    {"eid": 5, "count": 7},
                    {"eid": 2, "count": 3},
                ],
            },
            "decoder.layers.3.ffn.router": {
                "token_total_hits": 12,
                "top_token_eids": [9, 1],
                "top_token_counts": [
                    {"eid": 9, "count": 8},
                    {"eid": 1, "count": 4},
                ],
            },
        },
    }

    encoder_by_layer, encoder_totals = hot_experts._hot_pairs_by_layer(payload, adapter, "encoder")
    decoder_by_layer, decoder_totals = hot_experts._hot_pairs_by_layer(payload, adapter, "decoder")

    assert encoder_by_layer == {0: [(5, 7), (2, 3)]}
    assert encoder_totals == {0: 10}
    assert decoder_by_layer == {6: [(9, 8), (1, 4)]}
    assert decoder_totals == {6: 12}


def test_hot_pairs_by_layer_ignores_nllb_dense_layer_keys():
    adapter = NllbMoeAdapter(SimpleNamespace(config=_nllb_config()), "facebook/nllb-moe-54b")
    payload = {
        "frozen_hot_experts": {
            "encoder.layers.2.ffn.router": [5, 2],
        },
        "expert_usage_summary": {
            "encoder.layers.2.ffn.router": {
                "token_total_hits": 10,
                "top_token_eids": [5, 2],
                "top_token_counts": [
                    {"eid": 5, "count": 7},
                    {"eid": 2, "count": 3},
                ],
            },
        },
    }

    by_layer, token_totals = hot_experts._hot_pairs_by_layer(payload, adapter, "encoder")

    assert by_layer == {}
    assert token_totals == {}
