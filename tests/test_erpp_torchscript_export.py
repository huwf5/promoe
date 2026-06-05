from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from performance_predictor.encoder.ERPP.implement.model.export_erpp_encoder_torchscript import (
  build_merged_router_token_model,
  SampleLogitsWrapper,
  TokenNoisyOrWrapper,
  build_model_from_config,
  load_checkpoint_state,
)
from performance_predictor.encoder.ERPP.implement.train.baseline.train_baseline import (
  build_model as build_baseline_model,
)
from performance_predictor.encoder.ERPP.implement.train.sample_level.models import (
  build_sample_model,
)


def explicit_noisy_or(token_logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
  probs = torch.softmax(token_logits.float(), dim=-1)
  valid = attention_mask.to(device=probs.device, dtype=torch.bool)[:, None, :, None]
  probs = probs.masked_fill(~valid, 0.0)
  log_no_hit = torch.log1p(-probs.clamp(max=1.0 - 1e-6)).sum(dim=2)
  return 1.0 - torch.exp(log_no_hit)


def write_tiny_erpp_model_dir(root: Path) -> Path:
  model_dir = root / "erpp-tiny"
  model_dir.mkdir()
  config = {
    "model": "erpp-hybrid-tokenset",
    "hidden_dim": 8,
    "dropout": 0.0,
    "set_transformer_seeds": 2,
    "metadata": {
      "hidden_size": 4,
      "num_encoder_moe_layers": 2,
      "num_experts": 5,
      "max_input_tokens": 3,
    },
  }
  model = build_sample_model(
    model_name=config["model"],
    input_dim=config["metadata"]["hidden_size"],
    hidden_dim=config["hidden_dim"],
    num_router_layers=config["metadata"]["num_encoder_moe_layers"],
    num_experts=config["metadata"]["num_experts"],
    dropout=config["dropout"],
    set_transformer_seeds=config["set_transformer_seeds"],
  )
  torch.save({"model_state_dict": model.state_dict(), "config": config}, model_dir / "best_model.pt")
  (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
  return model_dir


def write_tiny_budgetadaptive_model_dir(root: Path) -> Path:
  model_dir = root / "erpp-budgetadaptive-tiny"
  model_dir.mkdir()
  config = {
    "model": "erpp-budgetadaptive-set",
    "hidden_dim": 8,
    "dropout": 0.0,
    "max_budget": 5,
    "metadata": {
      "hidden_size": 4,
      "num_encoder_moe_layers": 2,
      "num_experts": 5,
      "max_input_tokens": 3,
    },
  }
  model = build_sample_model(
    model_name=config["model"],
    input_dim=config["metadata"]["hidden_size"],
    hidden_dim=config["hidden_dim"],
    num_router_layers=config["metadata"]["num_encoder_moe_layers"],
    num_experts=config["metadata"]["num_experts"],
    dropout=config["dropout"],
    max_budget=config["max_budget"],
  )
  torch.save({"model_state_dict": model.state_dict(), "config": config}, model_dir / "best_model.pt")
  (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
  return model_dir


def write_tiny_sample_model_dir(root: Path, model_name: str) -> Path:
  model_dir = root / f"{model_name}-tiny"
  model_dir.mkdir()
  config = {
    "model": model_name,
    "hidden_dim": 8,
    "dropout": 0.0,
    "set_transformer_seeds": 2,
    "max_budget": 5,
    "metadata": {
      "hidden_size": 4,
      "num_encoder_moe_layers": 2,
      "num_experts": 5,
      "max_input_tokens": 3,
    },
  }
  model = build_sample_model(
    model_name=config["model"],
    input_dim=config["metadata"]["hidden_size"],
    hidden_dim=config["hidden_dim"],
    num_router_layers=config["metadata"]["num_encoder_moe_layers"],
    num_experts=config["metadata"]["num_experts"],
    dropout=config["dropout"],
    max_budget=config["max_budget"],
    set_transformer_seeds=config["set_transformer_seeds"],
  )
  torch.save({"model_state_dict": model.state_dict(), "config": config}, model_dir / "best_model.pt")
  (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
  return model_dir


def build_eager_sample_logits(model_dir: Path) -> SampleLogitsWrapper:
  config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
  wrapper = build_model_from_config(config)
  assert isinstance(wrapper, SampleLogitsWrapper)
  wrapper.sample_model.load_state_dict(load_checkpoint_state(model_dir / "best_model.pt"))
  wrapper.eval()
  return wrapper


def test_export_erpp_encoder_torchscript_writes_loadable_artifact(tmp_path: Path) -> None:
  model_dir = write_tiny_erpp_model_dir(tmp_path)
  output_path = model_dir / "predictor.ts"

  subprocess.run(
    [
      sys.executable,
      "performance_predictor/encoder/ERPP/implement/model/export_erpp_encoder_torchscript.py",
      "--model-dir",
      str(model_dir),
      "--output",
      str(output_path),
      "--example-tokens",
      "3",
      "--check-tokens",
      "2",
    ],
    check=True,
  )

  loaded = torch.jit.load(str(output_path), map_location="cpu")
  loaded.eval()
  with torch.no_grad():
    logits = loaded(torch.zeros(1, 2, 4), torch.ones(1, 2, dtype=torch.long))
  assert tuple(logits.shape) == (1, 2, 5)


def test_export_budgetadaptive_erpp_torchscript_writes_loadable_logits_artifact(tmp_path: Path) -> None:
  model_dir = write_tiny_budgetadaptive_model_dir(tmp_path)
  output_path = model_dir / "budgetadaptive_predictor.ts"

  subprocess.run(
    [
      sys.executable,
      "performance_predictor/encoder/ERPP/implement/model/export_erpp_encoder_torchscript.py",
      "--model-dir",
      str(model_dir),
      "--output",
      str(output_path),
      "--example-tokens",
      "3",
      "--check-tokens",
      "2",
    ],
    check=True,
  )

  loaded = torch.jit.load(str(output_path), map_location="cpu")
  loaded.eval()
  with torch.no_grad():
    logits = loaded(torch.zeros(1, 2, 4), torch.ones(1, 2, dtype=torch.long))
  assert tuple(logits.shape) == (1, 2, 5)


@pytest.mark.parametrize(
  "model_name",
  ["erpp-setpool-mlp", "erpp-budgetadaptive-set", "erpp-settransformer"],
)
def test_export_sample_level_torchscript_matches_eager_logits(tmp_path: Path, model_name: str) -> None:
  model_dir = write_tiny_sample_model_dir(tmp_path, model_name)
  output_path = model_dir / "sample_predictor.ts"

  subprocess.run(
    [
      sys.executable,
      "performance_predictor/encoder/ERPP/implement/model/export_erpp_encoder_torchscript.py",
      "--model-dir",
      str(model_dir),
      "--output",
      str(output_path),
      "--example-tokens",
      "3",
      "--check-tokens",
      "2",
    ],
    check=True,
  )

  loaded = torch.jit.load(str(output_path), map_location="cpu")
  loaded.eval()
  eager = build_eager_sample_logits(model_dir)
  hidden = torch.tensor(
    [
      [
        [0.10, -0.20, 0.30, -0.40],
        [0.50, 0.60, -0.70, -0.80],
        [0.90, -1.00, 1.10, -1.20],
      ]
    ],
    dtype=torch.float32,
  )
  mask = torch.tensor([[1, 0, 1]], dtype=torch.long)

  with torch.no_grad():
    eager_logits = eager(hidden, mask)
    loaded_logits = loaded(hidden, mask)

  assert torch.allclose(loaded_logits, eager_logits, atol=1e-4, rtol=1e-4)


def test_export_settransformer_empty_mask_matches_eager_and_is_finite(tmp_path: Path) -> None:
  model_dir = write_tiny_sample_model_dir(tmp_path, "erpp-settransformer")
  output_path = model_dir / "settransformer_predictor.ts"

  subprocess.run(
    [
      sys.executable,
      "performance_predictor/encoder/ERPP/implement/model/export_erpp_encoder_torchscript.py",
      "--model-dir",
      str(model_dir),
      "--output",
      str(output_path),
      "--example-tokens",
      "3",
      "--check-tokens",
      "2",
    ],
    check=True,
  )

  loaded = torch.jit.load(str(output_path), map_location="cpu")
  loaded.eval()
  eager = build_eager_sample_logits(model_dir)
  hidden = torch.tensor(
    [
      [
        [0.10, -0.20, 0.30, -0.40],
        [0.50, 0.60, -0.70, -0.80],
        [0.90, -1.00, 1.10, -1.20],
      ]
    ],
    dtype=torch.float32,
  )
  empty_mask = torch.zeros(1, 3, dtype=torch.long)

  with torch.no_grad():
    eager_logits = eager(hidden, empty_mask)
    loaded_logits = loaded(hidden, empty_mask)

  assert torch.isfinite(loaded_logits).all()
  assert torch.allclose(loaded_logits, eager_logits, atol=1e-4, rtol=1e-4)


def write_tiny_token_baseline_model_dir(root: Path, model_name: str) -> Path:
  model_dir = root / f"{model_name}-tiny"
  model_dir.mkdir()
  config = {
    "model": model_name,
    "hidden_dim": 8,
    "recurrent_layers": 1,
    "src_layers": 2,
    "dropout": 0.0,
    "metadata": {
      "hidden_size": 4,
      "num_encoder_moe_layers": 2,
      "num_experts": 5,
      "max_input_tokens": 3,
    },
  }
  torch.manual_seed(0)
  model = build_baseline_model(
    model_name=config["model"],
    input_dim=config["metadata"]["hidden_size"],
    hidden_dim=config["hidden_dim"],
    num_layers=config["recurrent_layers"],
    num_experts=config["metadata"]["num_experts"],
    num_router_layers=config["metadata"]["num_encoder_moe_layers"],
    src_layers=config["src_layers"],
    dropout=config["dropout"],
    tokens=config["metadata"]["max_input_tokens"],
  )
  torch.save({"model_state_dict": model.state_dict(), "config": config}, model_dir / "model.pt")
  (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
  return model_dir


def write_tiny_sida_model_dir(root: Path) -> Path:
  return write_tiny_token_baseline_model_dir(root, "sida-gru-sa")


def build_eager_sida_noisy_or(model_dir: Path) -> TokenNoisyOrWrapper:
  config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
  metadata = config["metadata"]
  assert isinstance(metadata, dict)
  token_model = build_baseline_model(
    model_name=config["model"],
    input_dim=metadata["hidden_size"],
    hidden_dim=config["hidden_dim"],
    num_layers=config["recurrent_layers"],
    num_experts=metadata["num_experts"],
    num_router_layers=metadata["num_encoder_moe_layers"],
    src_layers=config["src_layers"],
    dropout=config["dropout"],
    tokens=metadata["max_input_tokens"],
  )
  wrapper = TokenNoisyOrWrapper(token_model)
  wrapper.token_model.load_state_dict(load_checkpoint_state(model_dir / "model.pt"))
  wrapper.eval()
  return wrapper


@pytest.mark.parametrize("model_name", ["sida-gru-sa", "sida-lstm-sa", "src-simplenn-token"])
def test_export_token_level_baseline_noisy_or_support_matrix(tmp_path: Path, model_name: str) -> None:
  model_dir = write_tiny_token_baseline_model_dir(tmp_path, model_name)
  output_path = model_dir / "noisy_or.ts"

  subprocess.run(
    [
      sys.executable,
      "performance_predictor/encoder/ERPP/implement/model/export_erpp_encoder_torchscript.py",
      "--model-dir",
      str(model_dir),
      "--output",
      str(output_path),
      "--example-tokens",
      "3",
      "--check-tokens",
      "2",
    ],
    check=True,
  )

  loaded = torch.jit.load(str(output_path), map_location="cpu")
  loaded.eval()
  with torch.no_grad():
    scores = loaded(torch.zeros(1, 2, 4), torch.ones(1, 2, dtype=torch.long))
  assert tuple(scores.shape) == (1, 2, 5)
  assert torch.all(scores >= 0.0)
  assert torch.all(scores <= 1.0)


def test_export_sida_noisy_or_torchscript_writes_loadable_artifact(tmp_path: Path) -> None:
  model_dir = write_tiny_sida_model_dir(tmp_path)
  output_path = model_dir / "sida_noisy_or.ts"

  subprocess.run(
    [
      sys.executable,
      "performance_predictor/encoder/ERPP/implement/model/export_erpp_encoder_torchscript.py",
      "--model-dir",
      str(model_dir),
      "--output",
      str(output_path),
      "--example-tokens",
      "3",
      "--check-tokens",
      "2",
    ],
    check=True,
  )

  loaded = torch.jit.load(str(output_path), map_location="cpu")
  loaded.eval()
  eager = build_eager_sida_noisy_or(model_dir)
  hidden = torch.tensor(
    [
      [
        [0.10, -0.20, 0.30, -0.40],
        [0.50, 0.60, -0.70, -0.80],
        [0.90, -1.00, 1.10, -1.20],
      ]
    ],
    dtype=torch.float32,
  )
  masks = [
    torch.tensor([[1, 1, 0]], dtype=torch.long),
    torch.tensor([[0, 1, 1]], dtype=torch.long),
    torch.tensor([[1, 0, 1]], dtype=torch.long),
  ]
  empty_mask = torch.zeros(1, 3, dtype=torch.long)

  with torch.no_grad():
    token_logits = eager.token_model(hidden)
    for mask in masks:
      scores = loaded(hidden, mask)
      expected = explicit_noisy_or(token_logits, mask)
      assert tuple(scores.shape) == (1, 2, 5)
      assert torch.all(scores >= 0.0)
      assert torch.all(scores <= 1.0)
      assert torch.allclose(scores, expected, atol=1e-5, rtol=1e-5)

    empty_scores = loaded(hidden, empty_mask)
    empty_expected = explicit_noisy_or(token_logits, empty_mask)

  assert torch.allclose(empty_scores, torch.zeros_like(empty_scores))
  assert torch.allclose(empty_scores, empty_expected, atol=1e-5, rtol=1e-5)


def test_merged_router_token_model_matches_original_sida_token_logits(tmp_path: Path) -> None:
  model_dir = write_tiny_sida_model_dir(tmp_path)
  eager = build_eager_sida_noisy_or(model_dir)
  merged_token_model = build_merged_router_token_model(eager.token_model)
  merged = TokenNoisyOrWrapper(merged_token_model)
  merged.eval()
  hidden = torch.tensor(
    [
      [
        [0.10, -0.20, 0.30, -0.40],
        [0.50, 0.60, -0.70, -0.80],
        [0.90, -1.00, 1.10, -1.20],
      ]
    ],
    dtype=torch.float32,
  )
  masks = [
    torch.tensor([[1, 1, 0]], dtype=torch.long),
    torch.tensor([[0, 1, 1]], dtype=torch.long),
    torch.zeros(1, 3, dtype=torch.long),
  ]

  with torch.no_grad():
    eager_token_logits = eager.token_model(hidden)
    merged_token_logits = merged_token_model(hidden)
    assert torch.allclose(merged_token_logits, eager_token_logits, atol=1e-6, rtol=1e-6)
    for mask in masks:
      assert torch.allclose(merged(hidden, mask), eager(hidden, mask), atol=1e-5, rtol=1e-5)


def test_export_merged_router_v2_uses_v2_default_output_without_overwrite(tmp_path: Path) -> None:
  model_dir = write_tiny_sida_model_dir(tmp_path)
  original_output = model_dir / "erpp_encoder_predictor.ts"
  v2_output = model_dir / "erpp_encoder_predictor_v2.ts"

  subprocess.run(
    [
      sys.executable,
      "performance_predictor/encoder/ERPP/implement/model/export_erpp_encoder_torchscript.py",
      "--model-dir",
      str(model_dir),
      "--merged-router-v2",
      "--example-tokens",
      "3",
      "--check-tokens",
      "2",
    ],
    check=True,
  )

  assert v2_output.exists()
  assert not original_output.exists()

  loaded = torch.jit.load(str(v2_output), map_location="cpu")
  loaded.eval()
  eager = build_eager_sida_noisy_or(model_dir)
  hidden = torch.tensor(
    [
      [
        [0.10, -0.20, 0.30, -0.40],
        [0.50, 0.60, -0.70, -0.80],
        [0.90, -1.00, 1.10, -1.20],
      ]
    ],
    dtype=torch.float32,
  )
  mask = torch.tensor([[1, 0, 1]], dtype=torch.long)

  with torch.no_grad():
    assert torch.allclose(loaded(hidden, mask), eager(hidden, mask), atol=1e-4, rtol=1e-4)


def test_load_checkpoint_state_rejects_non_tensor_mapping(tmp_path: Path) -> None:
  checkpoint_path = tmp_path / "bad.pt"
  torch.save({"config": {"model": "erpp-hybrid-tokenset"}}, checkpoint_path)

  with pytest.raises(ValueError, match="model_state_dict or a tensor state_dict"):
    load_checkpoint_state(checkpoint_path)


def test_export_rejects_sample_level_baseline_for_noisy_or() -> None:
  config = {
    "model": "src-simplenn-sample",
    "hidden_dim": 8,
    "src_layers": 2,
    "dropout": 0.0,
    "metadata": {
      "hidden_size": 4,
      "num_encoder_moe_layers": 2,
      "num_experts": 5,
      "max_input_tokens": 3,
    },
  }

  with pytest.raises(ValueError, match="only supports ERPP sample-level and token-level baseline noisy-or"):
    build_model_from_config(config)
