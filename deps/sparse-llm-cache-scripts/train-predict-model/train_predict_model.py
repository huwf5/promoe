#!/usr/bin/env python3
"""Minimal train_predict_model smoke trainer for Switch trace exports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--logits_path", required=True, type=Path)
    p.add_argument("--predict_model_path", required=True, type=Path)
    p.add_argument("--train_log_path", type=Path, default=None)
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--window_begin", type=int, default=0)
    p.add_argument("--window", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--n_layer", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--threshold", type=float, default=1.0)
    p.add_argument("--threshold_window", type=int, default=1)
    p.add_argument("--model_index", nargs="+", type=int, default=[0])
    p.add_argument("--predict_output", choices=["gate", "freq"], default="freq")
    p.add_argument("--predict_input", choices=["moe-layer-logits", "token-id"], default="moe-layer-logits")
    p.add_argument("--input_norm_method", choices=["max1", "std", "replace"], default="max1")
    p.add_argument("--loss_func", choices=["l1", "smooth-l1"], default="l1")
    p.add_argument("--print_loss", default=False, action="store_true")
    p.add_argument("--model_type", choices=["single", "split"], default="split")
    return p.parse_args()


class SmokeNet(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, n_layer: int, dropout: float):
        super().__init__()
        layers: list[torch.nn.Module] = []
        width = input_size
        for _ in range(max(1, n_layer)):
            layers.append(torch.nn.Linear(width, hidden_size))
            layers.append(torch.nn.ReLU())
            if dropout:
                layers.append(torch.nn.Dropout(dropout))
            width = hidden_size
        layers.append(torch.nn.Linear(width, output_size))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _load_trace(trace_dir: Path, predict_output: str) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.load(trace_dir / "decode_stage_moe_layer_logits_per_token.pt", map_location="cpu")
    if predict_output == "gate":
        labels = torch.load(trace_dir / "decode_stage_moe_layer_gate_logits_per_token.pt", map_location="cpu")
    else:
        labels = torch.load(trace_dir / "decode_stage_expert_freq_per_token.pt", map_location="cpu")
    if inputs.ndim != 3 or labels.ndim != 3:
        raise ValueError("expected trace tensors shaped [N, num_layers, num_experts]")
    if inputs.shape != labels.shape:
        raise ValueError(f"input/label shape mismatch: {tuple(inputs.shape)} vs {tuple(labels.shape)}")
    if inputs.shape[0] == 0:
        raise ValueError("cannot train smoke model on an empty trace")
    return inputs.float(), labels.float()


def main() -> int:
    args = parse_args()
    if args.predict_input != "moe-layer-logits":
        raise ValueError("smoke trainer supports --predict_input moe-layer-logits")
    if args.model_type != "split":
        raise ValueError("smoke trainer supports --model_type split")

    args.predict_model_path.mkdir(parents=True, exist_ok=True)
    train_log_path = args.train_log_path or (args.predict_model_path / "train_log")
    train_log_path.mkdir(parents=True, exist_ok=True)

    inputs, labels = _load_trace(args.logits_path, args.predict_output)
    layer = args.model_index[0] if args.model_index else 0
    if layer >= inputs.shape[1]:
        raise ValueError(f"model_index {layer} outside trace layer count {inputs.shape[1]}")

    x = inputs[:, layer, :]
    y = labels[:, layer, :]
    model = SmokeNet(x.shape[-1], args.hidden_size, y.shape[-1], args.n_layer, args.dropout)
    loss_fn = torch.nn.SmoothL1Loss() if args.loss_func == "smooth-l1" else torch.nn.L1Loss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    opt.zero_grad()
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    opt.step()

    scripted = torch.jit.script(model.eval())
    scripted.save(str(args.predict_model_path / f"{layer}-{layer}.pt"))
    (train_log_path / "metas.json").write_text(
        json.dumps(
            {
                "num_expert": int(inputs.shape[-1]),
                "num_moe_layer": int(inputs.shape[1]),
                "predict_output": args.predict_output,
                "model_type": args.model_type,
                "loss": float(loss.detach()),
            },
            indent=2,
        )
    )
    print(f"Smoke training wrote {args.predict_model_path / f'{layer}-{layer}.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
