from types import SimpleNamespace

from sida.performance import (
    _build_hot_encoder_layer_quota_map,
    _parse_layer_quota_arg,
)


class DummyPredictorManager:
    encoder_module_keys = ["encoder.block.1.router", "encoder.block.3.router"]
    _key2layer_idx = {
        "encoder.block.1.router": 0,
        "encoder.block.3.router": 1,
    }


def test_parse_layer_quota_arg_accepts_empty_and_pairs():
    assert _parse_layer_quota_arg("") == {}
    assert _parse_layer_quota_arg("0:2, 3:5") == {0: 2, 3: 5}


def test_build_hot_encoder_layer_quota_map_uses_token_count_marginal_gain():
    payload = {
        "expert_usage_summary": {
            "encoder.block.1.router": {
                "top_token_counts": [
                    {"eid": 10, "count": 100},
                    {"eid": 11, "count": 90},
                    {"eid": 12, "count": 80},
                ],
            },
            "encoder.block.3.router": {
                "top_token_counts": [
                    {"eid": 20, "count": 50},
                    {"eid": 21, "count": 40},
                    {"eid": 22, "count": 30},
                ],
            },
            "decoder.block.1.router": {
                "top_token_counts": [
                    {"eid": 30, "count": 1000},
                ],
            },
        },
    }

    quota = _build_hot_encoder_layer_quota_map(
        payload=payload,
        predictor_manager=DummyPredictorManager(),
        total_quota=4,
        experts_per_layer=None,
    )

    assert quota == {0: 3, 1: 1}


def test_build_hot_encoder_layer_quota_map_respects_experts_per_layer_cap():
    payload = {
        "expert_usage_summary": {
            "encoder.block.1.router": {
                "top_token_counts": [
                    {"eid": 10, "count": 100},
                    {"eid": 11, "count": 90},
                ],
            },
            "encoder.block.3.router": {
                "top_token_counts": [
                    {"eid": 20, "count": 80},
                    {"eid": 21, "count": 70},
                ],
            },
        },
    }

    quota = _build_hot_encoder_layer_quota_map(
        payload=payload,
        predictor_manager=DummyPredictorManager(),
        total_quota=4,
        experts_per_layer=1,
    )

    assert quota == {0: 1, 1: 1}
