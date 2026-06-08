#!/usr/bin/env python3
"""Export experiment encoder predictor BLTE/BLE artifacts to hidden-only TorchScript."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


BLTE_TS_NAME = "encoder_predictor_blte.ts"
BLE_TS_NAME = "encoder_predictor_ble.ts"
EXPORT_MANIFEST_NAME = "export_manifest.json"
BLE_MANIFEST_NAME = "ble_manifest.json"


class HiddenOnlyNoisyOrWrapper(nn.Module):
    """Convert token-level BLTE logits to hidden-only BLE noisy-or scores."""

    def __init__(self, token_model: nn.Module, eps: float = 1e-6) -> None:
        super().__init__()
        self.token_model = token_model
        self.eps = float(eps)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        token_logits = self.token_model(hidden)
        probs = torch.softmax(token_logits.float(), dim=-1)
        log_no_hit = torch.log1p(-probs.clamp(max=1.0 - self.eps)).sum(dim=2)
        return 1.0 - torch.exp(log_no_hit)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export experiment encoder predictor BLTE and BLE TorchScript artifacts.")
    parser.add_argument("--blte-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", default="best_model.pt")
    parser.add_argument("--example-tokens", type=int, default=None)
    parser.add_argument("--check-tokens", type=int, default=8)
    parser.add_argument("--blte-output", type=Path, default=None)
    parser.add_argument("--ble-output", type=Path, default=None)
    parser.add_argument("--skip-blte", action="store_true")
    parser.add_argument("--skip-ble", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def infer_ble_dir(blte_dir: Path) -> Path:
    blte_dir = Path(blte_dir)
    if blte_dir.parent.name != "blte":
        raise ValueError(f"expected a .../blte/<run-name> directory, got {blte_dir}")
    trace_dir = blte_dir.parent.parent
    return trace_dir / "ble" / f"noisyor-from-{blte_dir.name}"


def resolve_checkpoint_path(model_dir: Path, checkpoint_name: str) -> Path:
    checkpoint_path = model_dir / checkpoint_name
    if not checkpoint_path.exists() and checkpoint_name == "best_model.pt":
        fallback = model_dir / "model.pt"
        if fallback.exists():
            checkpoint_path = fallback
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    return checkpoint_path


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
            "unsupported checkpoint payload: expected model_state_dict or tensor state_dict "
            f"at {checkpoint_path}; got keys [{keys}]"
        )
    raise ValueError(f"unsupported checkpoint payload at {checkpoint_path}: {type(checkpoint).__name__}")


def _metadata(config: dict[str, Any]) -> dict[str, Any]:
    metadata = config.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("config metadata must be an object")
    return metadata


def build_model_from_config(config: dict[str, Any]) -> nn.Module:
    metadata = _metadata(config)
    model_name = str(config.get("model"))
    if model_name == "src-simplenn-token":
        from experiment.scripts.train.encoder_predictor_src_simplenn_token_hard_ce import SrcSimpleNNTokenPredictor

        return SrcSimpleNNTokenPredictor(
            input_dim=int(metadata["hidden_size"]),
            hidden_dim=int(config["hidden_dim"]),
            num_router_layers=int(metadata["num_encoder_moe_layers"]),
            num_experts=int(metadata["num_experts"]),
            src_layers=int(config.get("src_layers", 1)),
            dropout=float(config.get("dropout", 0.5)),
        )
    if model_name == "sida-gru-sa":
        from experiment.scripts.train.encoder_predictor_sida_gru_sa_hard_ce import SidaGRUSparseAttentionPredictor

        return SidaGRUSparseAttentionPredictor(
            input_dim=int(metadata["hidden_size"]),
            hidden_dim=int(config["hidden_dim"]),
            num_router_layers=int(metadata["num_encoder_moe_layers"]),
            num_experts=int(metadata["num_experts"]),
            recurrent_layers=int(config.get("recurrent_layers", 2)),
        )
    raise ValueError(f"unsupported encoder predictor model: {model_name}")


def load_trained_model(blte_dir: Path, checkpoint_name: str, config: dict[str, Any]) -> tuple[nn.Module, Path]:
    checkpoint_path = resolve_checkpoint_path(blte_dir, checkpoint_name)
    model = build_model_from_config(config)
    model.load_state_dict(load_checkpoint_state(checkpoint_path), strict=True)
    model.eval()
    return model, checkpoint_path


def _shape_values(config: dict[str, Any]) -> tuple[int, int, int, int]:
    metadata = _metadata(config)
    hidden_size = int(metadata["hidden_size"])
    num_layers = int(metadata["num_encoder_moe_layers"])
    num_experts = int(metadata["num_experts"])
    max_input_tokens = int(metadata.get("max_input_tokens", 512))
    return hidden_size, num_layers, num_experts, max_input_tokens


def _check_close(loaded: torch.Tensor, eager: torch.Tensor, label: str) -> None:
    if not torch.allclose(loaded, eager, atol=1e-4, rtol=1e-4):
        max_diff = (loaded - eager).abs().max().item()
        raise RuntimeError(f"{label} loaded TorchScript output does not match eager output; max diff {max_diff:.6g}")


def _trace_hidden_only_model(model: nn.Module, example_hidden: torch.Tensor, check_hidden: torch.Tensor) -> torch.jit.ScriptModule:
    return torch.jit.trace(model, (example_hidden,), check_inputs=[(check_hidden,)], strict=True)


def export_blte_torchscript(
    model: nn.Module,
    output_path: Path,
    example_hidden: torch.Tensor,
    check_hidden: torch.Tensor,
    num_layers: int,
    num_experts: int,
) -> None:
    with torch.no_grad():
        eager_check = model(check_hidden)
        traced = _trace_hidden_only_model(model, example_hidden, check_hidden)
        example_out = traced(example_hidden)
        check_out = traced(check_hidden)

    expected_example = (1, num_layers, int(example_hidden.shape[1]), num_experts)
    expected_check = (1, num_layers, int(check_hidden.shape[1]), num_experts)
    if tuple(example_out.shape) != expected_example:
        raise RuntimeError(f"BLTE example output shape {tuple(example_out.shape)} != {expected_example}")
    if tuple(check_out.shape) != expected_check:
        raise RuntimeError(f"BLTE check output shape {tuple(check_out.shape)} != {expected_check}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output_path))
    loaded = torch.jit.load(str(output_path), map_location="cpu")
    loaded.eval()
    with torch.no_grad():
        loaded_check = loaded(check_hidden)
    if tuple(loaded_check.shape) != expected_check:
        raise RuntimeError(f"loaded BLTE output shape {tuple(loaded_check.shape)} != {expected_check}")
    _check_close(loaded_check, eager_check, "BLTE")


def export_ble_torchscript(
    model: nn.Module,
    output_path: Path,
    example_hidden: torch.Tensor,
    check_hidden: torch.Tensor,
    num_layers: int,
    num_experts: int,
) -> None:
    wrapper = HiddenOnlyNoisyOrWrapper(model)
    wrapper.eval()
    with torch.no_grad():
        eager_check = wrapper(check_hidden)
        traced = _trace_hidden_only_model(wrapper, example_hidden, check_hidden)
        example_out = traced(example_hidden)
        check_out = traced(check_hidden)

    expected = (1, num_layers, num_experts)
    if tuple(example_out.shape) != expected:
        raise RuntimeError(f"BLE example output shape {tuple(example_out.shape)} != {expected}")
    if tuple(check_out.shape) != expected:
        raise RuntimeError(f"BLE check output shape {tuple(check_out.shape)} != {expected}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output_path))
    loaded = torch.jit.load(str(output_path), map_location="cpu")
    loaded.eval()
    with torch.no_grad():
        loaded_check = loaded(check_hidden)
    if tuple(loaded_check.shape) != expected:
        raise RuntimeError(f"loaded BLE output shape {tuple(loaded_check.shape)} != {expected}")
    _check_close(loaded_check, eager_check, "BLE")


def _base_manifest(
    *,
    artifact_level: str,
    torchscript_path: Path,
    blte_dir: Path,
    checkpoint_path: Path,
    config: dict[str, Any],
    aggregation: str,
    src_compatible: bool,
) -> dict[str, Any]:
    hidden_size, num_layers, num_experts, max_input_tokens = _shape_values(config)
    return {
        "aggregation": aggregation,
        "artifact_level": artifact_level,
        "base_model": config.get("base_model"),
        "forward_signature": "forward(hidden: Tensor[B,T,H]) -> Tensor[B,L,T,E]"
        if artifact_level == "blte"
        else "forward(hidden: Tensor[B,T,H]) -> Tensor[B,L,E]",
        "hidden_size": hidden_size,
        "max_input_tokens": max_input_tokens,
        "model": config.get("model"),
        "num_encoder_moe_layers": num_layers,
        "num_experts": num_experts,
        "objective": config.get("objective"),
        "runtime_padding_contract": "hidden-only-no-padding",
        "source_blte_artifact": str(blte_dir),
        "source_checkpoint": str(checkpoint_path),
        "src_compatible": bool(src_compatible),
        "torchscript_path": str(torchscript_path),
        "trace_id": config.get("trace_id"),
        "workload_task": config.get("workload_task"),
    }


def _minimal_ble_manifest(ble_dir: Path, blte_dir: Path, checkpoint_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "aggregation": "noisy_or",
        "artifact_dir": str(ble_dir),
        "artifact_level": "ble",
        "base_model": config.get("base_model"),
        "ble_view_name": ble_dir.name,
        "budget_clamp_max": "num_experts",
        "budget_clamp_min": 1,
        "budget_rounding": "ceil",
        "budget_source": "sum_ble_score",
        "output_layout": "BLE",
        "predictor_task": "encoder_expert_prefetch",
        "probability_transform": "softmax",
        "selection_rule": "topk_by_ble_score",
        "source_blte_artifact": str(blte_dir),
        "source_blte_run_name": blte_dir.name,
        "source_checkpoint": checkpoint_path.name,
        "trace_dir": config.get("trace_dir"),
        "trace_id": config.get("trace_id"),
        "valid_token_handling": "hidden-only-no-padding",
        "workload_task": config.get("workload_task"),
    }


def ensure_ble_manifest(ble_dir: Path, blte_dir: Path, checkpoint_path: Path, config: dict[str, Any]) -> None:
    manifest_path = ble_dir / BLE_MANIFEST_NAME
    if manifest_path.exists():
        return
    write_json(manifest_path, _minimal_ble_manifest(ble_dir, blte_dir, checkpoint_path, config))


def export_torchscript_artifacts(
    *,
    blte_dir: Path,
    checkpoint_name: str = "best_model.pt",
    example_tokens: int | None = None,
    check_tokens: int = 8,
    blte_output: Path | None = None,
    ble_output: Path | None = None,
    export_blte: bool = True,
    export_ble: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    blte_dir = Path(blte_dir)
    if not export_blte and not export_ble:
        raise ValueError("nothing to export: both BLTE and BLE were skipped")
    config = read_json(blte_dir / "config.json")
    hidden_size, num_layers, num_experts, max_input_tokens = _shape_values(config)
    example_tokens = max_input_tokens if example_tokens is None else int(example_tokens)
    if example_tokens <= 0 or check_tokens <= 0:
        raise ValueError("example_tokens and check_tokens must be positive")

    ble_dir = infer_ble_dir(blte_dir)
    checkpoint_path = resolve_checkpoint_path(blte_dir, checkpoint_name)
    blte_ts = Path(blte_output) if blte_output is not None else blte_dir / BLTE_TS_NAME
    ble_ts = Path(ble_output) if ble_output is not None else ble_dir / BLE_TS_NAME
    result: dict[str, Any] = {
        "blte_dir": str(blte_dir),
        "ble_dir": str(ble_dir),
        "checkpoint_path": str(checkpoint_path),
        "example_tokens": example_tokens,
        "check_tokens": int(check_tokens),
        "hidden_size": hidden_size,
        "num_encoder_moe_layers": num_layers,
        "num_experts": num_experts,
    }

    if export_blte:
        result["blte"] = _base_manifest(
            artifact_level="blte",
            torchscript_path=blte_ts,
            blte_dir=blte_dir,
            checkpoint_path=checkpoint_path,
            config=config,
            aggregation="none",
            src_compatible=False,
        )
    if export_ble:
        result["ble"] = _base_manifest(
            artifact_level="ble",
            torchscript_path=ble_ts,
            blte_dir=blte_dir,
            checkpoint_path=checkpoint_path,
            config=config,
            aggregation="noisy_or",
            src_compatible=True,
        )
    if dry_run:
        return result

    model, loaded_checkpoint_path = load_trained_model(blte_dir, checkpoint_name, config)
    checkpoint_path = loaded_checkpoint_path
    example_hidden = torch.zeros(1, example_tokens, hidden_size, dtype=torch.float32)
    generator = torch.Generator().manual_seed(0)
    check_hidden = torch.randn(1, int(check_tokens), hidden_size, dtype=torch.float32, generator=generator)

    if export_blte:
        export_blte_torchscript(model, blte_ts, example_hidden, check_hidden, num_layers, num_experts)
        manifest = _base_manifest(
            artifact_level="blte",
            torchscript_path=blte_ts,
            blte_dir=blte_dir,
            checkpoint_path=checkpoint_path,
            config=config,
            aggregation="none",
            src_compatible=False,
        )
        write_json(blte_dir / EXPORT_MANIFEST_NAME, manifest)
        result["blte"] = manifest
    if export_ble:
        ble_dir.mkdir(parents=True, exist_ok=True)
        ensure_ble_manifest(ble_dir, blte_dir, checkpoint_path, config)
        export_ble_torchscript(model, ble_ts, example_hidden, check_hidden, num_layers, num_experts)
        manifest = _base_manifest(
            artifact_level="ble",
            torchscript_path=ble_ts,
            blte_dir=blte_dir,
            checkpoint_path=checkpoint_path,
            config=config,
            aggregation="noisy_or",
            src_compatible=True,
        )
        write_json(ble_dir / EXPORT_MANIFEST_NAME, manifest)
        result["ble"] = manifest
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = export_torchscript_artifacts(
        blte_dir=args.blte_dir,
        checkpoint_name=args.checkpoint,
        example_tokens=args.example_tokens,
        check_tokens=args.check_tokens,
        blte_output=args.blte_output,
        ble_output=args.ble_output,
        export_blte=not args.skip_blte,
        export_ble=not args.skip_ble,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
