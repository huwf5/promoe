from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "experiment/scripts/trace/run_encoder_erpp_trace.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("experiment_run_encoder_erpp_trace", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_output_dir_uses_model_dataset_task_layout():
    module = _load_module()

    output_dir = module.default_output_dir(
        repo_root=Path("/tmp/repo"),
        model_name="switch-base-128",
        dataset_name="mmlu",
        task_name="professional_law",
    )

    assert output_dir == Path(
        "/tmp/repo/experiment/traces/switch-base-128-mmlu-professional_law/encoder_erpp_trace"
    )


def test_default_prompt_split_dir_uses_experiment_dataset_layout():
    module = _load_module()

    split_dir = module.default_prompt_split_dir(
        repo_root=Path("/tmp/repo"),
        dataset_name="mmlu",
        task_name="professional_law",
        split="test",
    )

    assert split_dir == Path("/tmp/repo/experiment/datasets/mmlu/professional_law/test")


def test_resolve_prompt_file_prefers_txt_file(tmp_path: Path):
    module = _load_module()
    split_dir = tmp_path / "test"
    split_dir.mkdir()
    torch.save(["from pt"], split_dir / "prompt_list.pt")
    txt = split_dir / "prompt_list.txt"
    txt.write_text("from txt\n", encoding="utf-8")

    assert module.resolve_prompt_file(split_dir) == txt


def test_materialize_prompt_txt_converts_pt_when_txt_missing(tmp_path: Path):
    module = _load_module()
    split_dir = tmp_path / "validation"
    split_dir.mkdir()
    torch.save(["line1\n", "line2"], split_dir / "prompt_list.pt")
    output_dir = tmp_path / "trace"

    prompt_file = module.materialize_prompt_txt(split_dir, output_dir, split="validation")

    assert prompt_file == output_dir / "_inputs" / "validation_prompt_list.txt"
    assert prompt_file.read_text(encoding="utf-8").splitlines() == ["line1\\n", "line2"]


def test_build_export_command_uses_small_demo_defaults(tmp_path: Path):
    module = _load_module()
    command = module.build_export_command(
        python_executable=Path("/env/bin/python"),
        exporter_path=Path("/repo/performance_predictor/encoder/ERPP/implement/trace/export_erpp_encoder_trace.py"),
        model_path=Path("/repo/experiment/models/google/switch-base-128"),
        train_prompt_file=Path("/tmp/train.txt"),
        validation_prompt_file=Path("/tmp/validation.txt"),
        output_dir=tmp_path / "trace",
        max_input_tokens=512,
        batch_size=1,
        device="cuda:0",
        seed=42,
        padding="longest",
        storage_dtype="float32",
        verify=True,
        print_status=True,
    )

    assert [str(x) for x in command[:2]] == ["/env/bin/python", "/repo/performance_predictor/encoder/ERPP/implement/trace/export_erpp_encoder_trace.py"]
    assert "--padding" in command
    assert command[command.index("--padding") + 1] == "longest"
    assert "--batch-size" in command
    assert command[command.index("--batch-size") + 1] == "1"
    assert "--max-input-tokens" in command
    assert command[command.index("--max-input-tokens") + 1] == "512"
    assert "--verify" in command
    assert "--print-status" in command


def test_annotate_metadata_records_attention_mask_training_rule(tmp_path: Path):
    module = _load_module()
    output_dir = tmp_path / "trace"
    output_dir.mkdir()
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps({"schema": "erpp_encoder_trace"}), encoding="utf-8")

    module.annotate_metadata(
        output_dir=output_dir,
        dataset_name="mmlu",
        task_name="professional_law",
        train_split="test",
        validation_split="validation",
        wrapper_padding="longest",
        wrapper_batch_size=1,
        wrapper_max_input_tokens=512,
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    wrapper = metadata["experiment_wrapper"]
    assert wrapper["small_demo_input口径"] is True
    assert wrapper["padding"] == "longest"
    assert wrapper["batch_size"] == 1
    assert metadata["training_notes"]["must_filter_padding_with_attention_mask"] is True
    assert "attention_mask.bool()" in metadata["training_notes"]["valid_token_rule"]
