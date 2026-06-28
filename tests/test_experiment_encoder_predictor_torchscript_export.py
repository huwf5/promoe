from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "experiment/scripts/train/export_encoder_predictor_torchscript.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("experiment_export_encoder_predictor_torchscript", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _minimal_src_config() -> dict:
    return {
        "model": "src-simplenn-token",
        "objective": "hard-ce",
        "output_name": "src-tiny",
        "trace_dir": "experiment/traces/tiny/encoder_predictor_sparse_cache_trace",
        "workload_task": "mmlu-professional_law",
        "base_model": "switch-tiny",
        "trace_id": "sparse-cache-b1-longest-v1",
        "hidden_dim": 4,
        "src_layers": 1,
        "dropout": 0.0,
        "metadata": {
            "hidden_size": 3,
            "num_encoder_moe_layers": 2,
            "num_experts": 5,
            "num_selected_experts": 1,
            "max_input_tokens": 4,
        },
    }


def test_infer_ble_dir_from_blte_dir():
    module = _load_module()
    blte_dir = Path(
        "experiment/models/predictors/encoder_expert_prefetch/mmlu-professional_law/"
        "switch-base-256/sparse-cache-b1-longest-v1/blte/src-run"
    )

    assert module.infer_ble_dir(blte_dir) == Path(
        "experiment/models/predictors/encoder_expert_prefetch/mmlu-professional_law/"
        "switch-base-256/sparse-cache-b1-longest-v1/ble/noisyor-from-src-run"
    )


def test_hidden_only_noisy_or_wrapper_matches_softmax_noisy_or():
    module = _load_module()

    class FixedTokenModel(torch.nn.Module):
        def __init__(self, logits: torch.Tensor) -> None:
            super().__init__()
            self.register_buffer("logits", logits)

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            return self.logits.expand(hidden.shape[0], -1, -1, -1)

    logits = torch.tensor(
        [[[[2.0, 0.0, -1.0], [0.5, 1.0, -0.5]], [[-1.0, 3.0, 0.0], [0.0, 0.0, 2.0]]]],
        dtype=torch.float32,
    )
    hidden = torch.zeros(1, 2, 4)
    wrapper = module.HiddenOnlyNoisyOrWrapper(FixedTokenModel(logits))

    actual = wrapper(hidden)
    probs = torch.softmax(logits, dim=-1)
    expected = 1.0 - torch.exp(torch.log1p(-probs.clamp(max=1.0 - 1e-6)).sum(dim=2))

    assert actual.shape == (1, 2, 3)
    assert torch.allclose(actual, expected)


def test_hidden_only_noisy_or_wrapper_can_use_sigmoid_for_bce_logits():
    module = _load_module()

    class FixedTokenModel(torch.nn.Module):
        def __init__(self, logits: torch.Tensor) -> None:
            super().__init__()
            self.register_buffer("logits", logits)

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            return self.logits.expand(hidden.shape[0], -1, -1, -1)

    logits = torch.tensor(
        [[[[2.0, 0.0, -1.0], [0.5, 1.0, -0.5]], [[-1.0, 3.0, 0.0], [0.0, 0.0, 2.0]]]],
        dtype=torch.float32,
    )
    hidden = torch.zeros(1, 2, 4)
    wrapper = module.HiddenOnlyNoisyOrWrapper(FixedTokenModel(logits), score_activation="sigmoid")

    actual = wrapper(hidden)
    probs = torch.sigmoid(logits)
    expected = 1.0 - torch.exp(torch.log1p(-probs.clamp(max=1.0 - 1e-6)).sum(dim=2))

    assert actual.shape == (1, 2, 3)
    assert torch.allclose(actual, expected)


def test_export_manifest_records_sigmoid_for_multi_label_bce(tmp_path: Path):
    module = _load_module()
    blte_dir = tmp_path / "experiment/models/predictors/encoder_expert_prefetch/task/model/trace/blte/src-tiny"
    blte_dir.mkdir(parents=True)
    config = _minimal_src_config()
    config["loss_type"] = "multi_label_bce"
    (blte_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    model = module.build_model_from_config(config)
    torch.save({"model_state_dict": model.state_dict(), "config": config}, blte_dir / "best_model.pt")

    module.export_torchscript_artifacts(
        blte_dir=blte_dir,
        checkpoint_name="best_model.pt",
        example_tokens=4,
        check_tokens=3,
    )

    ble_dir = tmp_path / "experiment/models/predictors/encoder_expert_prefetch/task/model/trace/ble/noisyor-from-src-tiny"
    ble_manifest = json.loads((ble_dir / "ble_manifest.json").read_text(encoding="utf-8"))
    export_manifest = json.loads((ble_dir / "export_manifest.json").read_text(encoding="utf-8"))

    assert ble_manifest["probability_transform"] == "sigmoid"
    assert export_manifest["probability_transform"] == "sigmoid"


def test_export_torchscript_artifacts_writes_blte_and_ble_outputs(tmp_path: Path):
    module = _load_module()
    blte_dir = tmp_path / "experiment/models/predictors/encoder_expert_prefetch/task/model/trace/blte/src-tiny"
    blte_dir.mkdir(parents=True)
    config = _minimal_src_config()
    (blte_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    model = module.build_model_from_config(config)
    torch.save({"model_state_dict": model.state_dict(), "config": config}, blte_dir / "best_model.pt")

    result = module.export_torchscript_artifacts(
        blte_dir=blte_dir,
        checkpoint_name="best_model.pt",
        example_tokens=4,
        check_tokens=3,
    )

    blte_ts = blte_dir / "encoder_predictor_blte.ts"
    ble_dir = tmp_path / "experiment/models/predictors/encoder_expert_prefetch/task/model/trace/ble/noisyor-from-src-tiny"
    ble_ts = ble_dir / "encoder_predictor_ble.ts"
    assert result["blte"]["torchscript_path"] == str(blte_ts)
    assert result["ble"]["torchscript_path"] == str(ble_ts)
    assert blte_ts.exists()
    assert ble_ts.exists()
    assert (blte_dir / "export_manifest.json").exists()
    assert (ble_dir / "export_manifest.json").exists()
    assert (ble_dir / "ble_manifest.json").exists()

    hidden = torch.randn(1, 3, 3)
    blte_model = torch.jit.load(str(blte_ts), map_location="cpu")
    ble_model = torch.jit.load(str(ble_ts), map_location="cpu")
    blte_out = blte_model(hidden)
    ble_out = ble_model(hidden)

    assert tuple(blte_out.shape) == (1, 2, 3, 5)
    assert tuple(ble_out.shape) == (1, 2, 5)
    manifest = json.loads((ble_dir / "export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_level"] == "ble"
    assert manifest["aggregation"] == "noisy_or"
    assert manifest["runtime_padding_contract"] == "hidden-only-no-padding"
