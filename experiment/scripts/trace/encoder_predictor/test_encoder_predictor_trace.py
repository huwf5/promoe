from pathlib import Path
import sys

import torch

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from export_sparse_cache_encoder_trace import (
    SplitAccumulator,
    SparseCacheTraceRunner,
    TraceWriter,
    _model_config_metadata,
    build_parser,
    build_sparse_cache_config,
    derive_model_name,
    is_transformers_one_token_timing_zero_division,
    json_safe,
    load_saved_tensor,
    resolve_default_output_dir,
    resolve_model_torch_dtype,
    resolve_storage_dtype,
    storage_dtype,
    validate_sparse_cache_switch_top1,
    verify_trace,
)
from compare_encoder_traces import expert_diff_summary


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
            self.runner.layer0_attn_out = torch.ones((1, 2, 3))
            self.runner.router_layer_to_model_block = [1]
            self.runner.router_logits_by_layer[0] = torch.ones((1, 2, 4))
            self.runner.expert_selection_by_layer[0] = torch.zeros((1, 2, 1), dtype=torch.int64)

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
            self.runner.layer0_attn_out = torch.ones((1, 2, 3))
            self.runner.router_layer_to_model_block = [1]
            self.runner.router_logits_by_layer[0] = torch.ones((1, 2, 4))
            self.runner.expert_selection_by_layer[0] = torch.zeros((1, 2, 1), dtype=torch.int64)

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
            self.runner.layer0_attn_out = torch.ones((1, 2, 3))
            self.runner.router_layer_to_model_block = [1]
            self.runner.router_logits_by_layer[0] = torch.ones((1, 2, 4))
            self.runner.expert_selection_by_layer[0] = torch.zeros((1, 2, 1), dtype=torch.int64)
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

