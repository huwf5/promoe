from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping
import json
import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from performance_predictor.encoder.ERPP.implement.train.sample_level.models import (
  build_sample_model,
)
from performance_predictor.encoder.ERPP.implement.train.baseline.train_baseline import (
  build_model as build_baseline_model,
)


TOKEN_LEVEL_BASELINE_MODELS = {"sida-gru-sa", "sida-lstm-sa", "src-simplenn-token"}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Export an ERPP sample-level or token-level baseline noisy-or encoder predictor to TorchScript."
  )
  parser.add_argument("--model-dir", required=True, type=Path)
  parser.add_argument("--checkpoint", default="best_model.pt")
  parser.add_argument("--output", type=Path)
  parser.add_argument("--example-tokens", type=int)
  parser.add_argument("--check-tokens", type=int, default=8)
  parser.add_argument(
    "--merged-router-v2",
    action="store_true",
    help="Export a v2 TorchScript artifact with SIDA router Linear heads merged into one Linear.",
  )
  return parser.parse_args()


class SampleLogitsWrapper(torch.nn.Module):
  def __init__(self, sample_model: torch.nn.Module) -> None:
    super().__init__()
    self.sample_model = sample_model

  def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    output = self.sample_model(hidden, attention_mask)
    if isinstance(output, dict):
      return output["sample_logits"]
    return output


class TokenNoisyOrWrapper(torch.nn.Module):
  def __init__(self, token_model: torch.nn.Module) -> None:
    super().__init__()
    self.token_model = token_model

  def _noisy_or(self, token_logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(token_logits.float(), dim=-1)
    valid = attention_mask.to(device=probs.device, dtype=torch.bool)[:, None, :, None]
    probs = probs.masked_fill(~valid, 0.0)
    log_no_hit = torch.log1p(-probs.clamp(max=1.0 - 1e-6)).sum(dim=2)
    return 1.0 - torch.exp(log_no_hit)

  def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    token_logits = self.token_model(hidden)
    return self._noisy_or(token_logits, attention_mask)


class MergedSidaRouterTokenModel(torch.nn.Module):
  def __init__(self, token_model: torch.nn.Module) -> None:
    super().__init__()
    self.compression_fc = copy.deepcopy(token_model.compression_fc)
    self.residual_fc = copy.deepcopy(token_model.residual_fc)
    self.relu = copy.deepcopy(token_model.relu)
    self.recurrent = copy.deepcopy(token_model.recurrent)
    self.attention = copy.deepcopy(token_model.attention)
    self.y_keys = tuple(token_model.y_keys)
    self.num_router_layers = len(self.y_keys)

    first_fc = token_model.fc[self.y_keys[0]]
    self.num_experts = int(first_fc.out_features)
    self.merged_fc = torch.nn.Linear(
      int(first_fc.in_features),
      self.num_router_layers * self.num_experts,
      bias=first_fc.bias is not None,
    )
    with torch.no_grad():
      self.merged_fc.weight.copy_(
        torch.cat([token_model.fc[key].weight.detach() for key in self.y_keys], dim=0)
      )
      if first_fc.bias is not None:
        self.merged_fc.bias.copy_(
          torch.cat([token_model.fc[key].bias.detach() for key in self.y_keys], dim=0)
        )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.compression_fc(x)
    x = self.relu(x)
    recurrent_out, _ = self.recurrent(x)
    context, _ = self.attention(recurrent_out, recurrent_out, recurrent_out)
    context = context + self.residual_fc(x)
    batch_size = context.size(0)
    tokens = context.size(1)
    out = self.merged_fc(context)
    out = out.reshape(batch_size, tokens, self.num_router_layers, self.num_experts)
    return out.permute(0, 2, 1, 3).contiguous()


def build_merged_router_token_model(token_model: torch.nn.Module) -> torch.nn.Module:
  required_attrs = ("compression_fc", "residual_fc", "relu", "recurrent", "attention", "fc", "y_keys")
  missing = [attr for attr in required_attrs if not hasattr(token_model, attr)]
  if missing:
    raise ValueError(
      "merged-router v2 export only supports SIDA token models with separate router Linear heads; "
      f"missing {', '.join(missing)}"
    )
  if len(tuple(token_model.y_keys)) == 0:
    raise ValueError("merged-router v2 export requires at least one router head")
  return MergedSidaRouterTokenModel(token_model)


def build_model_from_config(config: dict[str, object]) -> torch.nn.Module:
  metadata = config["metadata"]
  if not isinstance(metadata, dict):
    raise ValueError("config metadata must be an object")

  model_name = str(config["model"])
  if model_name.startswith("erpp-"):
    sample_model = build_sample_model(
      model_name=model_name,
      input_dim=int(metadata["hidden_size"]),
      hidden_dim=int(config["hidden_dim"]),
      num_router_layers=int(metadata["num_encoder_moe_layers"]),
      num_experts=int(metadata["num_experts"]),
      dropout=float(config.get("dropout", 0.0)),
      max_budget=config.get("max_budget"),
      set_transformer_seeds=int(config.get("set_transformer_seeds", 2)),
      hybrid_fusion=str(config.get("hybrid_fusion", "fixed")),
    )
    return SampleLogitsWrapper(sample_model)

  if model_name not in TOKEN_LEVEL_BASELINE_MODELS:
    supported = ", ".join(sorted(TOKEN_LEVEL_BASELINE_MODELS))
    raise ValueError(
      "export_erpp_encoder_torchscript only supports ERPP sample-level and "
      f"token-level baseline noisy-or models ({supported}); got {model_name}"
    )

  baseline = build_baseline_model(
    model_name=model_name,
    input_dim=int(metadata["hidden_size"]),
    hidden_dim=int(config["hidden_dim"]),
    num_layers=int(config.get("recurrent_layers", 2)),
    num_experts=int(metadata["num_experts"]),
    num_router_layers=int(metadata["num_encoder_moe_layers"]),
    src_layers=int(config.get("src_layers", 2)),
    dropout=float(config.get("dropout", 0.5)),
    tokens=int(metadata.get("max_input_tokens", 512)),
  )
  return TokenNoisyOrWrapper(baseline)


def _is_tensor_state_dict(value: object) -> bool:
  return (
    isinstance(value, Mapping)
    and bool(value)
    and all(isinstance(tensor, torch.Tensor) for tensor in value.values())
  )


def load_checkpoint_state(checkpoint_path: Path) -> dict[str, torch.Tensor]:
  try:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
  except TypeError:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
  if isinstance(checkpoint, Mapping):
    if "model_state_dict" in checkpoint:
      state_dict = checkpoint["model_state_dict"]
      if _is_tensor_state_dict(state_dict):
        return dict(state_dict)
      raise ValueError(f"checkpoint model_state_dict must be a tensor state_dict: {checkpoint_path}")
    if _is_tensor_state_dict(checkpoint):
      return dict(checkpoint)
    keys = ", ".join(str(key) for key in list(checkpoint.keys())[:5])
    raise ValueError(
      "unsupported checkpoint payload: expected a dict with model_state_dict "
      f"or a tensor state_dict at {checkpoint_path}; got keys [{keys}]"
    )
  raise ValueError(
    "unsupported checkpoint payload: expected a dict with model_state_dict "
    f"or a tensor state_dict at {checkpoint_path}; got {type(checkpoint).__name__}"
  )


def _compile_sample_wrapper(
    model: SampleLogitsWrapper,
    model_name: str,
    example_hidden: torch.Tensor,
    example_mask: torch.Tensor,
    check_hidden: torch.Tensor,
    check_mask: torch.Tensor,
) -> torch.jit.ScriptModule:
  if model_name == "erpp-settransformer":
    try:
      return torch.jit.script(model)
    except RuntimeError:
      # SampleLogitsWrapper contains a dict-output guard for budget-adaptive models;
      # the raw set transformer returns a tensor and scripts without tracing the mask branch.
      return torch.jit.script(model.sample_model)
  return torch.jit.trace(
    model,
    (example_hidden, example_mask),
    check_inputs=[(check_hidden, check_mask)],
    strict=True,
  )


def _resolve_checkpoint_path(model_dir: Path, checkpoint_name: str) -> Path:
  checkpoint_path = model_dir / checkpoint_name
  if not checkpoint_path.exists() and checkpoint_name == "best_model.pt":
    fallback = model_dir / "model.pt"
    if fallback.exists():
      checkpoint_path = fallback
  return checkpoint_path


def _load_config(model_dir: Path) -> dict[str, object]:
  return json.loads((model_dir / "config.json").read_text(encoding="utf-8"))


def _validate_metadata(config: dict[str, object]) -> dict[str, object]:
  metadata = config["metadata"]
  if not isinstance(metadata, dict):
    raise ValueError("config metadata must be an object")
  return metadata


def export_torchscript(
    model_dir: Path,
    checkpoint_name: str,
    output_path: Path | None,
    example_tokens: int | None,
    check_tokens: int,
) -> Path:
  checkpoint_path = _resolve_checkpoint_path(model_dir, checkpoint_name)
  if output_path is None:
    output_path = model_dir / "erpp_encoder_predictor.ts"

  config = _load_config(model_dir)
  metadata = _validate_metadata(config)

  hidden_size = int(metadata["hidden_size"])
  num_layers = int(metadata["num_encoder_moe_layers"])
  num_experts = int(metadata["num_experts"])
  if example_tokens is None:
    example_tokens = int(metadata.get("max_input_tokens", 512))

  model = build_model_from_config(config)
  state_dict = load_checkpoint_state(checkpoint_path)
  if isinstance(model, SampleLogitsWrapper):
    model.sample_model.load_state_dict(state_dict)
  elif isinstance(model, TokenNoisyOrWrapper):
    model.token_model.load_state_dict(state_dict)
  else:
    model.load_state_dict(state_dict)
  model.eval()

  example_hidden = torch.zeros(1, example_tokens, hidden_size, dtype=torch.float32)
  example_mask = torch.ones(1, example_tokens, dtype=torch.long)
  check_hidden = torch.randn(1, check_tokens, hidden_size, dtype=torch.float32)
  check_mask = torch.ones(1, check_tokens, dtype=torch.long)
  model_name = str(config["model"])
  if model_name == "erpp-settransformer":
    check_mask = torch.zeros(1, check_tokens, dtype=torch.long)

  with torch.no_grad():
    eager_check_logits = model(check_hidden, check_mask)
    if isinstance(model, TokenNoisyOrWrapper):
      traced_token_model = torch.jit.trace(
        model.token_model,
        (example_hidden,),
        check_inputs=[(check_hidden,)],
        strict=True,
      )
      traced = torch.jit.script(TokenNoisyOrWrapper(traced_token_model))
    else:
      traced = _compile_sample_wrapper(
        model,
        model_name,
        example_hidden,
        example_mask,
        check_hidden,
        check_mask,
      )
    example_logits = traced(example_hidden, example_mask)
    check_logits = traced(check_hidden, check_mask)

  expected_shape = (1, num_layers, num_experts)
  if tuple(example_logits.shape) != expected_shape:
    raise RuntimeError(f"example logits shape {tuple(example_logits.shape)} != {expected_shape}")
  if tuple(check_logits.shape) != expected_shape:
    raise RuntimeError(f"check logits shape {tuple(check_logits.shape)} != {expected_shape}")

  output_path.parent.mkdir(parents=True, exist_ok=True)
  traced.save(str(output_path))

  loaded = torch.jit.load(str(output_path), map_location="cpu")
  loaded.eval()
  with torch.no_grad():
    loaded_logits = loaded(check_hidden, check_mask)
  if tuple(loaded_logits.shape) != expected_shape:
    raise RuntimeError(f"loaded logits shape {tuple(loaded_logits.shape)} != {expected_shape}")
  if not torch.allclose(loaded_logits, eager_check_logits, atol=1e-4, rtol=1e-4):
    max_diff = (loaded_logits - eager_check_logits).abs().max().item()
    raise RuntimeError(f"loaded logits do not match eager output; max diff {max_diff:.6g}")

  return output_path


def export_torchscript_v2_merged_router(
    model_dir: Path,
    checkpoint_name: str,
    output_path: Path | None,
    example_tokens: int | None,
    check_tokens: int,
) -> Path:
  checkpoint_path = _resolve_checkpoint_path(model_dir, checkpoint_name)
  if output_path is None:
    output_path = model_dir / "erpp_encoder_predictor_v2.ts"

  config = _load_config(model_dir)
  metadata = _validate_metadata(config)

  hidden_size = int(metadata["hidden_size"])
  num_layers = int(metadata["num_encoder_moe_layers"])
  num_experts = int(metadata["num_experts"])
  if example_tokens is None:
    example_tokens = int(metadata.get("max_input_tokens", 512))

  model = build_model_from_config(config)
  if not isinstance(model, TokenNoisyOrWrapper):
    raise ValueError("merged-router v2 export only supports token-level SIDA noisy-or models")

  state_dict = load_checkpoint_state(checkpoint_path)
  model.token_model.load_state_dict(state_dict)
  model.eval()

  merged_model = TokenNoisyOrWrapper(build_merged_router_token_model(model.token_model))
  merged_model.eval()

  example_hidden = torch.zeros(1, example_tokens, hidden_size, dtype=torch.float32)
  example_mask = torch.ones(1, example_tokens, dtype=torch.long)
  check_hidden = torch.randn(1, check_tokens, hidden_size, dtype=torch.float32)
  check_mask = torch.ones(1, check_tokens, dtype=torch.long)

  with torch.no_grad():
    eager_check_logits = model(check_hidden, check_mask)
    merged_check_logits = merged_model(check_hidden, check_mask)
    if not torch.allclose(merged_check_logits, eager_check_logits, atol=1e-5, rtol=1e-5):
      max_diff = (merged_check_logits - eager_check_logits).abs().max().item()
      raise RuntimeError(f"merged-router v2 eager output does not match original; max diff {max_diff:.6g}")

    traced_token_model = torch.jit.trace(
      merged_model.token_model,
      (example_hidden,),
      check_inputs=[(check_hidden,)],
      strict=True,
    )
    traced = torch.jit.script(TokenNoisyOrWrapper(traced_token_model))
    example_logits = traced(example_hidden, example_mask)
    check_logits = traced(check_hidden, check_mask)

  expected_shape = (1, num_layers, num_experts)
  if tuple(example_logits.shape) != expected_shape:
    raise RuntimeError(f"example logits shape {tuple(example_logits.shape)} != {expected_shape}")
  if tuple(check_logits.shape) != expected_shape:
    raise RuntimeError(f"check logits shape {tuple(check_logits.shape)} != {expected_shape}")

  output_path.parent.mkdir(parents=True, exist_ok=True)
  traced.save(str(output_path))

  loaded = torch.jit.load(str(output_path), map_location="cpu")
  loaded.eval()
  with torch.no_grad():
    loaded_logits = loaded(check_hidden, check_mask)
  if tuple(loaded_logits.shape) != expected_shape:
    raise RuntimeError(f"loaded logits shape {tuple(loaded_logits.shape)} != {expected_shape}")
  if not torch.allclose(loaded_logits, eager_check_logits, atol=1e-4, rtol=1e-4):
    max_diff = (loaded_logits - eager_check_logits).abs().max().item()
    raise RuntimeError(f"loaded v2 logits do not match original eager output; max diff {max_diff:.6g}")

  return output_path


def main() -> None:
  args = parse_args()
  export_fn = export_torchscript_v2_merged_router if args.merged_router_v2 else export_torchscript
  output_path = export_fn(
    model_dir=args.model_dir,
    checkpoint_name=args.checkpoint,
    output_path=args.output,
    example_tokens=args.example_tokens,
    check_tokens=args.check_tokens,
  )
  print(output_path)


if __name__ == "__main__":
  main()
