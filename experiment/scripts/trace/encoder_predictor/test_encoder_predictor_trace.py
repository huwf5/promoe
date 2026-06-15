from pathlib import Path
import sys
import types

import torch

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from export_sparse_cache_encoder_trace import (
    SplitAccumulator,
    SparseCacheTraceRunner,
    TraceWriter,
    _model_config_metadata,
    _require_integer_tensor,
    build_parser,
    build_sparse_cache_config,
    derive_model_name,
    is_transformers_one_token_timing_zero_division,
    json_safe,
    load_saved_tensor,
    routed_topk_from_router_probs,
    resolve_cache_rate,
    resolve_default_output_dir,
    resolve_model_torch_dtype,
    resolve_storage_dtype,
    selected_experts_for_config,
    storage_dtype,
    validate_sparse_cache_encoder_trace_config,
    validate_sparse_cache_switch_top1,
    verify_trace,
)
from compare_encoder_traces import expert_diff_summary


def build_args(*, model_type="switch_transformers", cache_rate=None):
    del model_type
    return build_parser().parse_args(
        [
            "--model-path",
            "experiment/models/google/switch-base-128",
            "--dataset",
            "mmlu",
            "--task-name",
            "professional_law",
            "--device",
            "cpu",
            *([] if cache_rate is None else ["--cache-rate", str(cache_rate)]),
        ]
    )


def _fill_dummy_switch_trace_outputs(runner):
    runner.layer0_attn_out = torch.ones((1, 2, 3))
    runner.router_layer_to_model_block = [1]
    router_logits = torch.ones((1, 2, 4))
    expert_selection = torch.zeros((1, 2, 1), dtype=torch.int64)
    router_probs = torch.softmax(router_logits.float(), dim=-1)
    expert_weights = torch.gather(router_probs, dim=-1, index=expert_selection)
    runner.router_logits_by_layer[0] = router_logits
    runner.router_probs_by_layer[0] = router_probs
    runner.expert_selection_by_layer[0] = expert_selection
    runner.expert_weights_by_layer[0] = expert_weights
    runner.expert_selection_mask_by_layer[0] = expert_weights > 0


def test_derive_model_name_from_local_path():
    assert derive_model_name(Path("experiment/models/google/switch-base-128")) == "switch-base-128"


def test_resolve_default_output_dir():
    out = resolve_default_output_dir(
        repo_root=Path("/repo"),
        model_path=Path("experiment/models/google/switch-base-128"),
        dataset="mmlu",
        task_name="professional_law",
    )
    assert out == Path("/repo/experiment/traces/switch-base-128-mmlu-professional_law/encoder_predictor_sparse_cache_trace")


def test_storage_dtype_names():
    assert storage_dtype("float32") is torch.float32
    assert storage_dtype("float16") is torch.float16
    assert storage_dtype("bfloat16") is torch.bfloat16


def test_storage_dtype_default_auto_uses_model_config_dtype():
    class DummyConfig:
        torch_dtype = torch.bfloat16

    args = build_parser().parse_args(
        [
            "--model-path",
            "experiment/models/google/switch-base-128",
            "--dataset",
            "mmlu",
            "--task-name",
            "professional_law",
        ]
    )

    assert args.storage_dtype == "auto"
    assert resolve_storage_dtype(args.storage_dtype, DummyConfig()) is torch.bfloat16


def test_storage_dtype_auto_accepts_string_config_dtype():
    class DummyConfig:
        torch_dtype = "float16"

    assert resolve_storage_dtype("auto", DummyConfig()) is torch.float16


def test_model_torch_dtype_auto_uses_model_config_dtype():
    class DummyConfig:
        torch_dtype = "float32"

    assert resolve_model_torch_dtype("auto", DummyConfig()) is torch.float32


def test_model_torch_dtype_explicit_value_overrides_config():
    class DummyConfig:
        torch_dtype = "float32"

    assert resolve_model_torch_dtype("float16", DummyConfig()) is torch.float16
    assert resolve_model_torch_dtype("bfloat16", DummyConfig()) is torch.bfloat16


def test_sparse_cache_switch_top1_validation_accepts_top1_config():
    class DummyConfig:
        model_type = "switch_transformers"
        num_selected_experts = 1

    assert validate_sparse_cache_switch_top1(DummyConfig(), Path("experiment/models/google/switch-base-128")) == 1


def test_sparse_cache_switch_top1_validation_defaults_missing_num_selected_experts_to_top1():
    class DummyConfig:
        model_type = "switch_transformers"

    selected = validate_sparse_cache_switch_top1(DummyConfig(), Path("experiment/models/google/switch-large-128"))

    assert selected == 1


def test_model_config_metadata_defaults_missing_num_selected_experts_for_switch_transformers():
    class DummyConfig:
        model_type = "switch_transformers"
        hidden_size = 1024
        num_experts = 128
        num_sparse_encoder_layers = 12
        num_sparse_decoder_layers = 12
        encoder_sparse_step = 2
        decoder_sparse_step = 2

    metadata = _model_config_metadata(DummyConfig())

    assert metadata["num_selected_experts"] == 1


def test_sparse_cache_switch_top1_validation_rejects_top2_config():
    class DummyConfig:
        model_type = "switch_transformers"
        num_selected_experts = 2

    try:
        validate_sparse_cache_switch_top1(DummyConfig(), Path("experiment/models/google/switch-top2"))
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected top2 Switch config to be rejected")

    assert "num_selected_experts=2" in message



def test_nllb_selected_experts_defaults_to_top2():
    class Config:
        model_type = "nllb-moe"
        num_experts = 128
        d_model = 1024
        encoder_layers = 24
        decoder_layers = 24
        encoder_sparse_step = 4
        decoder_sparse_step = 4

    assert selected_experts_for_config(Config()) == 2


def test_selected_experts_for_config_uses_explicit_value():
    class Config:
        model_type = "nllb-moe"
        num_selected_experts = 3

    assert selected_experts_for_config(Config()) == 3


def test_sparse_cache_encoder_trace_validation_accepts_switch_top1_and_nllb_top2():
    class SwitchConfig:
        model_type = "switch_transformers"
        num_selected_experts = 1

    class NllbConfig:
        model_type = "nllb-moe"

    assert validate_sparse_cache_encoder_trace_config(SwitchConfig(), Path("switch-base-128")) == 1
    assert validate_sparse_cache_encoder_trace_config(NllbConfig(), Path("nllb-moe-54b")) == 2


def test_topk_selection_from_sparse_router_probs_masks_unrouted_slots():
    router_probs = torch.tensor([[[0.0, 0.7, 0.0, 0.3], [0.0, 0.0, 0.0, 0.0]]])

    selection, weights, mask = routed_topk_from_router_probs(router_probs, top_k=2)

    assert selection.shape == (1, 2, 2)
    assert weights.shape == (1, 2, 2)
    assert mask.tolist() == [[[True, True], [False, False]]]
    assert set(selection[0, 0].tolist()) == {1, 3}
    assert torch.all(selection[0, 1] == 0)
    assert torch.all(weights[0, 1] == 0)


def test_sparse_cache_config_is_minimal_for_on_demand_trace():
    args = build_parser().parse_args(
        [
            "--model-path",
            "experiment/models/google/switch-base-128",
            "--dataset",
            "mmlu",
            "--task-name",
            "professional_law",
        ]
    )

    config = build_sparse_cache_config(args)

    assert config == {
        "model_id": "experiment/models/google/switch-base-128",
        "cache_rate": 0.375,
        "cache_policy": "lru",
        "per_layer_cache": False,
        "num_predict_expert_per_layer": 0,
        "reorder_experts": False,
        "early_preempt": False,
        "chunk_prefetch": False,
        "predict_input_mode": "no_predict",
    }


def test_default_cache_rate_is_001_for_nllb_and_switch_default_stays_0375():
    nllb_args = build_args(model_type="nllb-moe", cache_rate=None)
    switch_args = build_args(model_type="switch_transformers", cache_rate=None)

    assert nllb_args.cache_rate is None
    assert resolve_cache_rate(nllb_args, model_type="nllb-moe") == 0.01
    assert resolve_cache_rate(switch_args, model_type="switch_transformers") == 0.375


def test_explicit_cache_rate_overrides_model_type_defaults():
    args = build_args(model_type="nllb-moe", cache_rate=0.2)

    assert resolve_cache_rate(args, model_type="nllb-moe") == 0.2
    assert build_sparse_cache_config(args, model_type="nllb-moe")["cache_rate"] == 0.2


def test_prefetch_initial_cache_options_are_not_cli_options():
    parser = build_parser()

    for option in (
        "--predictor-model-path",
        "--initial-cache-policy",
        "--enable-erpp-encoder-prefetch",
        "--cache-trace-path",
    ):
        try:
            parser.parse_args(
                [
                    "--model-path",
                    "experiment/models/google/switch-base-128",
                    "--dataset",
                    "mmlu",
                    "--task-name",
                    "professional_law",
                    option,
                    "unused",
                ]
            )
        except SystemExit as exc:
            assert exc.code != 0
        else:
            raise AssertionError(f"{option} should not be accepted")


def test_split_accumulator_pads_sequence_dimension():
    acc = SplitAccumulator()
    acc.append(
        input_ids=torch.tensor([[1, 2, 3]], dtype=torch.int64),
        attention_mask=torch.tensor([[1, 1, 1]], dtype=torch.int64),
        layer0_attn_out=torch.ones((1, 3, 2)),
        router_logits=torch.ones((1, 2, 3, 4)),
        expert_selection=torch.ones((1, 2, 3, 1), dtype=torch.int64),
        seq_ids=torch.tensor([0], dtype=torch.int64),
        prompts=["a"],
    )
    acc.append(
        input_ids=torch.tensor([[4, 5]], dtype=torch.int64),
        attention_mask=torch.tensor([[1, 1]], dtype=torch.int64),
        layer0_attn_out=2 * torch.ones((1, 2, 2)),
        router_logits=2 * torch.ones((1, 2, 2, 4)),
        expert_selection=2 * torch.ones((1, 2, 2, 1), dtype=torch.int64),
        seq_ids=torch.tensor([1], dtype=torch.int64),
        prompts=["b"],
    )

    input_ids = acc.cat("input_ids")
    router_logits = acc.cat("router_logits")
    expert_selection = acc.cat("expert_selection")

    assert input_ids.shape == (2, 3)
    assert input_ids[1, 2].item() == 0
    assert router_logits.shape == (2, 2, 3, 4)
    assert torch.all(router_logits[1, :, 2, :] == 0)
    assert expert_selection.shape == (2, 2, 3, 1)
    assert torch.all(expert_selection[1, :, 2, :] == 0)



def test_trace_writer_writes_expert_selection_mask(tmp_path):
    acc = SplitAccumulator()
    acc.append(
        input_ids=torch.ones((1, 2), dtype=torch.long),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
        layer0_attn_out=torch.ones((1, 2, 4)),
        router_logits=torch.ones((1, 1, 2, 4)),
        router_probs=torch.softmax(torch.ones((1, 1, 2, 4)), dim=-1),
        expert_selection=torch.zeros((1, 1, 2, 2), dtype=torch.long),
        expert_weights=torch.ones((1, 1, 2, 2)),
        expert_selection_mask=torch.ones((1, 1, 2, 2), dtype=torch.bool),
        seq_ids=torch.tensor([0]),
        prompts=["x"],
    )

    TraceWriter(tmp_path, torch.float32).write_split("train", acc)

    assert (tmp_path / "train" / "expert_selection_mask.pt").exists()


def test_trace_writer_requires_explicit_router_fields_when_configured_for_nllb(tmp_path):
    acc = SplitAccumulator()
    acc.append(
        input_ids=torch.ones((1, 2), dtype=torch.long),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
        layer0_attn_out=torch.ones((1, 2, 4)),
        router_logits=torch.ones((1, 1, 2, 4)),
        expert_selection=torch.zeros((1, 1, 2, 2), dtype=torch.long),
        seq_ids=torch.tensor([0]),
        prompts=["x"],
    )

    try:
        TraceWriter(tmp_path, torch.float32, require_explicit_router_probs=True).write_split("train", acc)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected strict TraceWriter to reject missing explicit router tensors")

    assert "router_probs" in message


def test_routed_topk_rejects_top_k_larger_than_num_experts_and_invalid_top_k():
    router_probs = torch.ones((1, 2, 3))

    for top_k in (0, -1, 4):
        try:
            routed_topk_from_router_probs(router_probs, top_k=top_k)
        except ValueError as exc:
            message = str(exc)
        else:
            raise AssertionError(f"expected top_k={top_k} to be rejected")

        assert "top_k" in message


def test_require_integer_tensor_rejects_bool_except_selection_mask():
    try:
        _require_integer_tensor("train", "expert_selection.pt", torch.ones((1,), dtype=torch.bool))
    except AssertionError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected bool expert_selection to be rejected")

    assert "bool" in message
    _require_integer_tensor("train", "expert_selection_mask.pt", torch.ones((1,), dtype=torch.bool))


def test_expert_diff_summary_splits_valid_and_padding():
    left = torch.tensor([[[[1], [2], [3]]]], dtype=torch.int64)
    right = torch.tensor([[[[1], [7], [9]]]], dtype=torch.int64)
    attention_mask = torch.tensor([[1, 0, 0]], dtype=torch.int64)

    summary = expert_diff_summary(left, right, attention_mask)

    assert summary["total_diff"] == 2
    assert summary["valid_diff"] == 0
    assert summary["padding_diff"] == 2


def test_json_safe_converts_paths_and_torch_values():
    payload = {
        "path": Path("experiment/models/google/switch-base-128"),
        "device": torch.device("cpu"),
        "dtype": torch.float32,
        "nested": [Path("prompt_list.txt")],
    }

    import json

    encoded = json.dumps(json_safe(payload))

    assert "switch-base-128" in encoded
    assert "cpu" in encoded
    assert "float32" in encoded


def test_verify_trace_accepts_writer_output(tmp_path):
    writer = TraceWriter(tmp_path, torch.float32)
    metadata = {"model_config": {"num_experts": 4, "num_selected_experts": 1}}

    for split in ("train", "validation"):
        acc = SplitAccumulator()
        acc.append(
            input_ids=torch.tensor([[1, 2]], dtype=torch.int64),
            attention_mask=torch.tensor([[1, 1]], dtype=torch.int64),
            layer0_attn_out=torch.ones((1, 2, 3)),
            router_logits=torch.ones((1, 2, 2, 4)),
            expert_selection=torch.zeros((1, 2, 2, 1), dtype=torch.int64),
            seq_ids=torch.tensor([0], dtype=torch.int64),
            prompts=[f"{split} prompt"],
        )
        writer.write_split(split, acc)

    result = verify_trace(tmp_path, metadata)
    saved = load_saved_tensor(tmp_path / "train" / "input_ids.pt")

    assert result["ok"] is True
    assert result["splits"]["train"]["num_samples"] == 1
    assert saved.dtype is torch.int64


def test_verify_trace_accepts_legacy_switch_without_mask_and_checks_weights(tmp_path):
    writer = TraceWriter(tmp_path, torch.float32)
    metadata = {"model_config": {"model_type": "switch_transformers", "num_experts": 4, "num_selected_experts": 1}}

    for split in ("train", "validation"):
        acc = SplitAccumulator()
        acc.append(
            input_ids=torch.tensor([[1, 2]], dtype=torch.int64),
            attention_mask=torch.tensor([[1, 1]], dtype=torch.int64),
            layer0_attn_out=torch.ones((1, 2, 3)),
            router_logits=torch.zeros((1, 1, 2, 4)),
            router_probs=torch.tensor([[[[0.1, 0.7, 0.1, 0.1], [0.2, 0.2, 0.5, 0.1]]]]),
            expert_selection=torch.tensor([[[[1], [2]]]], dtype=torch.int64),
            expert_weights=torch.tensor([[[[0.7], [0.5]]]]),
            expert_selection_mask=torch.ones((1, 1, 2, 1), dtype=torch.bool),
            seq_ids=torch.tensor([0], dtype=torch.int64),
            prompts=[f"{split} prompt"],
        )
        writer.write_split(split, acc)
        (tmp_path / split / "expert_selection_mask.pt").unlink()

    result = verify_trace(tmp_path, metadata)

    assert result["ok"] is True

    bad_weights = load_saved_tensor(tmp_path / "train" / "expert_weights.pt")
    bad_weights[0, 0, 0, 0] = 0.123
    torch.save(bad_weights, tmp_path / "train" / "expert_weights.pt")
    try:
        verify_trace(tmp_path, metadata)
    except AssertionError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected legacy Switch weight mismatch to be rejected")

    assert "expert_weights" in message


def test_verify_trace_rejects_nllb_without_selection_mask(tmp_path):
    writer = TraceWriter(tmp_path, torch.float32)
    metadata = {"model_config": {"model_type": "nllb-moe", "num_experts": 4, "num_selected_experts": 2}}

    for split in ("train", "validation"):
        acc = SplitAccumulator()
        acc.append(
            input_ids=torch.tensor([[1, 2]], dtype=torch.int64),
            attention_mask=torch.tensor([[1, 1]], dtype=torch.int64),
            layer0_attn_out=torch.ones((1, 2, 3)),
            router_logits=torch.zeros((1, 1, 2, 4)),
            router_probs=torch.tensor([[[[0.1, 0.7, 0.0, 0.2], [0.0, 0.0, 0.0, 0.0]]]]),
            expert_selection=torch.tensor([[[[1, 3], [0, 0]]]], dtype=torch.int64),
            expert_weights=torch.tensor([[[[0.7, 0.2], [0.0, 0.0]]]]),
            expert_selection_mask=torch.tensor([[[[True, True], [False, False]]]]),
            seq_ids=torch.tensor([0], dtype=torch.int64),
            prompts=[f"{split} prompt"],
        )
        writer.write_split(split, acc)
        (tmp_path / split / "expert_selection_mask.pt").unlink()

    try:
        verify_trace(tmp_path, metadata)
    except AssertionError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected NLLB trace without selection mask to be rejected")

    assert "expert_selection_mask.pt" in message


def test_verify_trace_rejects_dense_nllb_router_probs_outside_selection(tmp_path):
    writer = TraceWriter(tmp_path, torch.float32)
    metadata = {"model_config": {"model_type": "nllb-moe", "num_experts": 4, "num_selected_experts": 2}}

    for split in ("train", "validation"):
        acc = SplitAccumulator()
        acc.append(
            input_ids=torch.tensor([[1, 2]], dtype=torch.int64),
            attention_mask=torch.tensor([[1, 1]], dtype=torch.int64),
            layer0_attn_out=torch.ones((1, 2, 3)),
            router_logits=torch.zeros((1, 1, 2, 4)),
            router_probs=torch.tensor([[[[0.1, 0.7, 0.05, 0.15], [0.25, 0.25, 0.25, 0.25]]]]),
            expert_selection=torch.tensor([[[[1, 3], [0, 1]]]], dtype=torch.int64),
            expert_weights=torch.tensor([[[[0.7, 0.15], [0.25, 0.25]]]]),
            expert_selection_mask=torch.tensor([[[[True, True], [True, True]]]]),
            seq_ids=torch.tensor([0], dtype=torch.int64),
            prompts=[f"{split} prompt"],
        )
        writer.write_split(split, acc)

    try:
        verify_trace(tmp_path, metadata)
    except AssertionError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected dense NLLB router_probs to be rejected")

    assert "NLLB router_probs" in message


def test_model_config_metadata_uses_d_model_as_hidden_size_fallback():
    class DummyConfig:
        model_type = "switch_transformers"
        d_model = 768
        num_experts = 16
        num_selected_experts = 1
        num_sparse_encoder_layers = 6
        num_sparse_decoder_layers = 6
        encoder_sparse_step = 2
        decoder_sparse_step = 2

    metadata = _model_config_metadata(DummyConfig())

    assert metadata["hidden_size"] == 768


def test_model_config_metadata_derives_nllb_sparse_layer_counts_and_top2():
    class DummyConfig:
        model_type = "nllb-moe"
        d_model = 1024
        num_experts = 128
        encoder_layers = 24
        decoder_layers = 24
        encoder_sparse_step = 4
        decoder_sparse_step = 4

    metadata = _model_config_metadata(DummyConfig())

    assert metadata["hidden_size"] == 1024
    assert metadata["num_sparse_encoder_layers"] == 6
    assert metadata["num_sparse_decoder_layers"] == 6
    assert metadata["num_selected_experts"] == 2


def test_attach_nllb_trace_hooks_uses_nested_generation_model_encoder():
    from transformers.models.nllb_moe.configuration_nllb_moe import NllbMoeConfig
    from transformers.models.nllb_moe.modeling_nllb_moe import NllbMoeForConditionalGeneration

    config = NllbMoeConfig(
        vocab_size=16,
        d_model=4,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=1,
        decoder_attention_heads=1,
        encoder_ffn_dim=8,
        decoder_ffn_dim=8,
        encoder_sparse_step=1,
        decoder_sparse_step=1,
        num_experts=2,
        expert_capacity=4,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        decoder_start_token_id=2,
    )
    model = NllbMoeForConditionalGeneration(config)
    assert not hasattr(model, "encoder")

    runner = SparseCacheTraceRunner(build_args(model_type="nllb-moe"))
    runner.model = model

    runner._attach_nllb_trace_hooks()

    assert runner.router_layer_to_model_block == [0]
    assert runner.handles


def test_nllb_router_classifier_hook_uses_ffn_batch_shape_for_flattened_inputs():
    runner = SparseCacheTraceRunner(build_args())
    layer = types.SimpleNamespace(_encoder_predictor_trace_layer_id=0)
    hidden = torch.ones((2, 3, 4))
    flat_hidden = hidden.reshape(6, 4)
    flat_logits = torch.arange(6 * 5, dtype=torch.float32).reshape(6, 5)

    runner._nllb_sparse_mlp_pre_hook(layer, (hidden,))
    runner._nllb_router_classifier_hook(layer, (flat_hidden,), flat_logits)

    assert runner.router_logits_by_layer[0].shape == (2, 3, 5)
    assert torch.equal(runner.router_logits_by_layer[0], flat_logits.reshape(2, 3, 5))


def test_nllb_sparse_hook_records_logits_probs_selection_weights_and_mask():
    runner = SparseCacheTraceRunner(build_args())
    mlp = types.SimpleNamespace(_encoder_predictor_trace_layer_id=0)
    hidden = torch.ones((1, 2, 4))
    router_logits = torch.tensor([[0.0, 4.0, 0.0, 1.0], [1.0, 0.0, 3.0, 0.0]])
    router_probs = torch.tensor([[0.0, 0.8, 0.0, 0.2], [0.0, 0.0, 0.0, 0.0]])

    runner._nllb_router_classifier_hook(types.SimpleNamespace(_encoder_predictor_trace_layer_id=0), (hidden,), router_logits)
    runner._nllb_sparse_mlp_hook(mlp, (hidden,), (hidden, (router_probs, torch.tensor([1, 0]))))

    assert runner.router_logits_by_layer[0].shape == (1, 2, 4)
    assert runner.router_probs_by_layer[0].shape == (1, 2, 4)
    assert runner.expert_selection_by_layer[0].shape == (1, 2, 2)
    assert torch.allclose(runner.expert_weights_by_layer[0], torch.tensor([[[0.8, 0.2], [0.0, 0.0]]]))
    assert runner.expert_selection_mask_by_layer[0].tolist() == [[[True, True], [False, False]]]


def test_nllb_sparse_mlp_style_output_uses_routed_top2_post_capacity_probs():
    runner = SparseCacheTraceRunner(build_args())
    mlp = types.SimpleNamespace(_encoder_predictor_trace_layer_id=0)
    hidden = torch.ones((1, 2, 4))
    router_probs = torch.tensor([[0.0, 0.6, 0.4], [0.0, 0.0, 0.0]])

    runner.router_logits_by_layer[0] = torch.ones((1, 2, 3))
    runner._nllb_sparse_mlp_hook(mlp, (hidden,), (hidden, (router_probs, torch.tensor([1, 0]))))

    assert runner.expert_selection_by_layer[0].tolist() == [[[1, 2], [0, 0]]]
    assert torch.allclose(runner.expert_weights_by_layer[0], torch.tensor([[[0.6, 0.4], [0.0, 0.0]]]))
    assert runner.expert_selection_mask_by_layer[0].tolist() == [[[True, True], [False, False]]]

def test_run_batch_uses_model_generate_without_manual_prefetch_reset():
    class DummyTokenizer:
        def __call__(self, prompts, **kwargs):
            return {
                "input_ids": torch.tensor([[1, 2]], dtype=torch.int64),
                "attention_mask": torch.tensor([[1, 1]], dtype=torch.int64),
            }

    class DummyPrefetchManager:
        def __init__(self):
            self.reset_calls = 0

        def reset_and_load_initial_cache(self):
            self.reset_calls += 1

    class DummyModel:
        def __init__(self, runner):
            self.runner = runner
            self._prefetch_mngr = DummyPrefetchManager()
            self._sparse_cache_old_generate_called = False
            self.generate_calls = 0

        def _fill_trace_outputs(self):
            _fill_dummy_switch_trace_outputs(self.runner)

        def _sparse_cache_old_generate(self, **kwargs):
            self._sparse_cache_old_generate_called = True
            self._fill_trace_outputs()

        def generate(self, **kwargs):
            self.generate_calls += 1
            self._fill_trace_outputs()

    args = build_parser().parse_args(
        [
            "--model-path",
            "experiment/models/google/switch-base-128",
            "--dataset",
            "mmlu",
            "--task-name",
            "professional_law",
            "--device",
            "cpu",
        ]
    )
    runner = SparseCacheTraceRunner(args)
    runner.model = DummyModel(runner)
    runner.tokenizer = DummyTokenizer()
    runner.input_device = torch.device("cpu")

    batch = runner.run_batch(["prompt"])

    assert runner.model.generate_calls == 1
    assert runner.model._prefetch_mngr.reset_calls == 0
    assert runner.model._sparse_cache_old_generate_called is False
    assert batch["router_logits"].shape == (1, 1, 2, 4)

def test_run_batch_and_run_split_carry_router_probs_weights_and_mask():
    class DummyTokenizer:
        def __call__(self, prompts, **kwargs):
            return {
                "input_ids": torch.tensor([[1, 2]], dtype=torch.int64),
                "attention_mask": torch.tensor([[1, 1]], dtype=torch.int64),
            }

    class DummySparseMlp:
        _encoder_predictor_trace_layer_id = 0

    class DummyModel:
        def __init__(self, runner):
            self.runner = runner

        def generate(self, **kwargs):
            self.runner.layer0_attn_out = torch.ones((1, 2, 3))
            self.runner.router_layer_to_model_block = [1]
            hidden = torch.ones((1, 2, 3))
            router_logits = torch.tensor([[[0.0, 2.0, 0.0, -1.0], [1.0, 0.0, 3.0, 0.0]]])
            expert_index = torch.tensor([[1, 2]], dtype=torch.int64)
            self.runner._sparse_mlp_hook(
                DummySparseMlp(),
                (hidden,),
                (hidden, (router_logits, expert_index)),
            )

    args = build_parser().parse_args(
        [
            "--model-path",
            "experiment/models/google/switch-base-128",
            "--dataset",
            "mmlu",
            "--task-name",
            "professional_law",
            "--device",
            "cpu",
        ]
    )
    runner = SparseCacheTraceRunner(args)
    runner.model = DummyModel(runner)
    runner.tokenizer = DummyTokenizer()
    runner.input_device = torch.device("cpu")

    batch = runner.run_batch(["prompt"])

    assert batch["router_probs"].shape == (1, 1, 2, 4)
    assert batch["expert_weights"].shape == (1, 1, 2, 1)
    assert batch["expert_selection_mask"].shape == (1, 1, 2, 1)
    assert batch["expert_selection_mask"].dtype is torch.bool
    assert torch.allclose(
        batch["expert_weights"],
        torch.gather(batch["router_probs"], dim=-1, index=batch["expert_selection"]),
    )

    acc = runner.run_split(["prompt"], batch_size=1)

    assert acc.cat("router_probs").shape == (1, 1, 2, 4)
    assert acc.cat("expert_weights").shape == (1, 1, 2, 1)
    assert acc.cat("expert_selection_mask").shape == (1, 1, 2, 1)


def test_run_batch_prints_generate_progress_when_status_enabled(capsys):
    class DummyTokenizer:
        def __call__(self, prompts, **kwargs):
            return {
                "input_ids": torch.tensor([[1, 2]], dtype=torch.int64),
                "attention_mask": torch.tensor([[1, 1]], dtype=torch.int64),
            }

    class DummyModel:
        def __init__(self, runner):
            self.runner = runner

        def generate(self, **kwargs):
            _fill_dummy_switch_trace_outputs(self.runner)

    args = build_parser().parse_args(
        [
            "--model-path",
            "experiment/models/google/switch-base-128",
            "--dataset",
            "mmlu",
            "--task-name",
            "professional_law",
            "--device",
            "cpu",
            "--print-status",
        ]
    )
    runner = SparseCacheTraceRunner(args)
    runner.model = DummyModel(runner)
    runner.tokenizer = DummyTokenizer()
    runner.input_device = torch.device("cpu")

    runner.run_batch(["prompt"])

    out = capsys.readouterr().out
    assert "tokenizing batch" in out
    assert "starting model.generate" in out
    assert "model.generate finished" in out
    assert "batch trace tensors ready" in out

def test_run_batch_tolerates_transformers_one_token_timing_zero_division(capsys):
    class DummyTokenizer:
        def __call__(self, prompts, **kwargs):
            return {
                "input_ids": torch.tensor([[1, 2]], dtype=torch.int64),
                "attention_mask": torch.tensor([[1, 1]], dtype=torch.int64),
            }

    class DummyModel:
        def __init__(self, runner):
            self.runner = runner

        def generate(self, **kwargs):
            _fill_dummy_switch_trace_outputs(self.runner)
            code = compile(
                "def _sample():\n    1 / 0\n_sample()",
                "/repo/deps/transformers/src/transformers/generation/utils.py",
                "exec",
            )
            exec(code, {})

    args = build_parser().parse_args(
        [
            "--model-path",
            "experiment/models/google/switch-base-128",
            "--dataset",
            "mmlu",
            "--task-name",
            "professional_law",
            "--device",
            "cpu",
            "--max-new-tokens",
            "1",
            "--print-status",
        ]
    )
    runner = SparseCacheTraceRunner(args)
    runner.model = DummyModel(runner)
    runner.tokenizer = DummyTokenizer()
    runner.input_device = torch.device("cpu")

    batch = runner.run_batch(["prompt"])

    out = capsys.readouterr().out
    assert "one-token timing bug" in out
    assert batch["router_logits"].shape == (1, 1, 2, 4)


def test_transformers_timing_zero_division_detector_rejects_unrelated_error():
    try:
        1 / 0
    except ZeroDivisionError as exc:
        assert is_transformers_one_token_timing_zero_division(exc) is False
    else:
        raise AssertionError("expected ZeroDivisionError")

