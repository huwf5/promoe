from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from performance_predictor.encoder.ERPP.implement.train.sample_level.models import (
  build_sample_model,
)


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
