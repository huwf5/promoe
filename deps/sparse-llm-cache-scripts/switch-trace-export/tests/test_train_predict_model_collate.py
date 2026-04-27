from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


def load_train_predict_model():
    script_dir = Path(__file__).resolve().parents[2] / "train-predict-model"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location(
        "train_predict_model_under_test", script_dir / "train_predict_model.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_trace() -> SimpleNamespace:
    num_tokens = 12
    num_layers = 2
    num_experts = 4
    token_idx = torch.arange(num_tokens)
    return SimpleNamespace(
        num_moe_layer=num_layers,
        num_expert=num_experts,
        per_token_expert=1,
        decode_stage_moe_layer_logits_per_token=torch.randn(
            num_tokens, num_layers, num_experts
        ),
        decode_stage_moe_layer_gate_logits_per_token=torch.randn(
            num_tokens, num_layers, num_experts
        ),
        decode_stage_expert_freq_per_token=torch.randn(
            num_tokens, num_layers, num_experts
        ),
        decode_stage_seq_id_of_token=torch.zeros(num_tokens, dtype=torch.long),
        decode_stage_token_idx_in_seq=token_idx,
        decode_stage_token_idx_in_seq_flip=torch.flip(token_idx, dims=(0,)),
    )


def make_train_config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        predict_output="freq",
        input_norm_method="max1",
        loss_func="l1",
        window=1,
        batch_size=4,
        hidden_size=4,
        n_layer=1,
        dropout=0.0,
        lr=0.001,
        train_log_path=str(tmp_path / "train_log"),
        predict_model_path=str(tmp_path / "models"),
    )


def test_train_model_loaders_batch_custom_dataset_tuples(monkeypatch, tmp_path):
    train_predict_model = load_train_predict_model()
    trace = make_trace()
    train_config = make_train_config(tmp_path)

    (tmp_path / "train_log").mkdir()
    (tmp_path / "models").mkdir()

    def assert_loader_batches_custom_dataset(_, train_loader, test_loader, __):
        for loader in (train_loader, test_loader):
            inputs, labels, metadata = next(iter(loader))
            assert isinstance(inputs, torch.Tensor)
            assert isinstance(labels, torch.Tensor)
            assert isinstance(metadata, torch.Tensor)
            assert inputs.shape[0] > 1
            assert labels.shape[0] == inputs.shape[0]
            assert metadata.shape[0] == inputs.shape[0]
        return [0.0], [0.0]

    class FakeScriptedModel:
        def save(self, _):
            return None

    monkeypatch.setattr(train_predict_model, "train_loop", assert_loader_batches_custom_dataset)
    monkeypatch.setattr(train_predict_model, "save_loss_logs", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        train_predict_model, "save_accuracy_logs", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        train_predict_model.torch.jit, "script", lambda _: FakeScriptedModel()
    )

    train_predict_model.train_one_model(trace, train_config, 0)
    train_predict_model.train_single_target_layer_model(trace, train_config, 0, 1)
