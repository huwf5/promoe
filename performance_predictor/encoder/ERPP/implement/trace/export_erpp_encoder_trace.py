#!/usr/bin/env python3
"""Export ERPP encoder trace data for Switch Transformers.

The ERPP trace contract is sequence-first:
  layer0_attn_out [S, T, H] -> router_logits [S, L, T, E]

This script is intentionally independent from deps/sparse-llm-cache-scripts/
switch-trace-export and does not emit train_predict_model's decode_stage_* files.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
TRANSFORMERS_SRC = REPO_ROOT / "deps" / "transformers" / "src"
if TRANSFORMERS_SRC.is_dir():
    sys.path.insert(0, str(TRANSFORMERS_SRC))


def _storage_dtype(name: str) -> torch.dtype:
    table = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return table[name]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"unsupported dtype {name!r}; choose {sorted(table)}") from exc


def _load_prompts(path: Path) -> list[str]:
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"prompt file has no non-empty lines: {path}")
    return prompts


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


@dataclass
class ERPPSplitAccumulator:
    input_ids: list[torch.Tensor] = field(default_factory=list)
    attention_mask: list[torch.Tensor] = field(default_factory=list)
    layer0_attn_out: list[torch.Tensor] = field(default_factory=list)
    router_logits: list[torch.Tensor] = field(default_factory=list)
    expert_selection: list[torch.Tensor] = field(default_factory=list)
    seq_ids: list[torch.Tensor] = field(default_factory=list)
    prompt_records: list[dict] = field(default_factory=list)

    def total_samples(self) -> int:
        return sum(int(t.shape[0]) for t in self.input_ids)

    def append(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layer0_attn_out: torch.Tensor,
        router_logits: torch.Tensor,
        expert_selection: torch.Tensor,
        seq_ids: torch.Tensor,
        prompts: list[str],
    ) -> None:
        self.input_ids.append(input_ids.cpu())
        self.attention_mask.append(attention_mask.cpu())
        self.layer0_attn_out.append(layer0_attn_out.cpu())
        self.router_logits.append(router_logits.cpu())
        self.expert_selection.append(expert_selection.cpu())
        self.seq_ids.append(seq_ids.cpu())
        for seq_id, text in zip(seq_ids.tolist(), prompts):
            self.prompt_records.append({"seq_id": int(seq_id), "text": text})

    def cat(self, name: str) -> torch.Tensor:
        values = getattr(self, name)
        if not values:
            raise RuntimeError(f"split accumulator has no tensors for {name}")
        return torch.cat(values, dim=0)


class ERPPTraceWriter:
    def __init__(self, output_dir: Path, storage_dtype: torch.dtype):
        self.output_dir = Path(output_dir)
        self.storage_dtype = storage_dtype

    def write_split(
        self,
        split: str,
        acc: ERPPSplitAccumulator,
        *,
        store_router_probs: bool,
        store_expert_weights: bool,
        store_prompt_texts: bool,
    ) -> dict:
        if acc.total_samples() == 0:
            raise ValueError(f"refusing to write empty split {split!r}")

        split_dir = self.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        input_ids = acc.cat("input_ids").to(torch.int64)
        attention_mask = acc.cat("attention_mask").to(torch.int64)
        layer0_attn_out = acc.cat("layer0_attn_out").to(self.storage_dtype)
        router_logits = acc.cat("router_logits").to(self.storage_dtype)
        expert_selection = acc.cat("expert_selection").to(torch.int64)
        seq_ids = acc.cat("seq_ids").to(torch.int64)

        torch.save(input_ids, split_dir / "input_ids.pt")
        torch.save(attention_mask, split_dir / "attention_mask.pt")
        torch.save(layer0_attn_out, split_dir / "layer0_attn_out.pt")
        torch.save(router_logits, split_dir / "router_logits.pt")
        torch.save(expert_selection, split_dir / "expert_selection.pt")
        torch.save(seq_ids, split_dir / "seq_ids.pt")

        if store_router_probs:
            router_probs = torch.softmax(router_logits.float(), dim=-1).to(self.storage_dtype)
            torch.save(router_probs, split_dir / "router_probs.pt")
        if store_expert_weights:
            router_probs_for_weights = torch.softmax(router_logits.float(), dim=-1)
            expert_weights = torch.gather(router_probs_for_weights, dim=-1, index=expert_selection)
            torch.save(expert_weights.to(self.storage_dtype), split_dir / "expert_weights.pt")
        if store_prompt_texts:
            with (split_dir / "prompt_texts.jsonl").open("w", encoding="utf-8") as f:
                for record in acc.prompt_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return {
            "num_samples": int(input_ids.shape[0]),
            "max_input_tokens": int(input_ids.shape[1]),
            "input_ids_shape": list(input_ids.shape),
            "layer0_attn_out_shape": list(layer0_attn_out.shape),
            "router_logits_shape": list(router_logits.shape),
            "expert_selection_shape": list(expert_selection.shape),
        }


class ERPPEncoderTraceRunner:
    def __init__(
        self,
        *,
        model_path: Path,
        device: str,
        seed: int,
        max_input_tokens: int,
        padding: str,
        verbose: bool,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.seed = int(seed)
        self.max_input_tokens = int(max_input_tokens)
        self.padding = padding
        self.verbose = verbose
        self.model = None
        self.tokenizer = None
        self._hook_handles: list = []
        self._layer0_attn_out: Optional[torch.Tensor] = None
        self._router_logits_by_layer: dict[int, torch.Tensor] = {}
        self._expert_selection_by_layer: dict[int, torch.Tensor] = {}
        self.router_layer_to_model_block: list[int] = []

    def _status(self, message: str) -> None:
        if self.verbose:
            print(f"[erpp-trace] {message}", flush=True)

    def _seed_all(self) -> None:
        from transformers import set_seed as hf_set_seed

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        hf_set_seed(self.seed)

    def load(self) -> None:
        from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration

        self._seed_all()
        self._status(f"loading model from {self.model_path} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = SwitchTransformersForConditionalGeneration.from_pretrained(
            str(self.model_path),
            torch_dtype=torch.float32,
        )
        self.model.eval()
        self.model.to(self.device)
        self._install_hooks()
        self._status("model loaded and hooks installed")

    def close(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def _install_hooks(self) -> None:
        from transformers.models.switch_transformers.modeling_switch_transformers import (
            SwitchTransformersLayerSelfAttention,
            SwitchTransformersSparseMLP,
        )

        layer0_attn = self.model.encoder.block[0].layer[0]
        if not isinstance(layer0_attn, SwitchTransformersLayerSelfAttention):
            raise RuntimeError(f"encoder.block[0].layer[0] is {type(layer0_attn).__name__}, expected SwitchTransformersLayerSelfAttention")
        self._hook_handles.append(layer0_attn.register_forward_hook(self._layer0_attention_hook))

        router_layer_id = 0
        self.router_layer_to_model_block.clear()
        for block_id, block in enumerate(self.model.encoder.block):
            for layer in block.layer:
                mlp = getattr(layer, "mlp", None)
                if isinstance(mlp, SwitchTransformersSparseMLP):
                    mlp._erpp_router_layer_id = router_layer_id
                    mlp._erpp_model_block_id = block_id
                    self.router_layer_to_model_block.append(int(block_id))
                    self._hook_handles.append(mlp.register_forward_hook(self._sparse_mlp_hook))
                    router_layer_id += 1
        if router_layer_id == 0:
            raise RuntimeError("found no encoder SwitchTransformersSparseMLP modules")

    def _layer0_attention_hook(self, module, inputs, output) -> None:
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(hidden, torch.Tensor) or hidden.dim() != 3:
            raise RuntimeError(f"layer0 attention hook expected [B,T,H], got {type(hidden).__name__} {getattr(hidden, 'shape', None)}")
        self._layer0_attn_out = hidden.detach().cpu()

    def _sparse_mlp_hook(self, module, inputs, output) -> None:
        layer_id = int(module._erpp_router_layer_id)
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError("SparseMLP hook expected hidden states in inputs[0]")
        input_hidden = inputs[0]
        if input_hidden.dim() != 3:
            raise RuntimeError(f"SparseMLP input hidden expected [B,T,H], got {tuple(input_hidden.shape)}")
        bsz, seq_len = int(input_hidden.shape[0]), int(input_hidden.shape[1])

        if not isinstance(output, (tuple, list)) or len(output) < 2:
            raise RuntimeError("SparseMLP output must contain router tuple in output[1]")
        router_tuple = output[1]
        if not isinstance(router_tuple, (tuple, list)) or len(router_tuple) < 2:
            raise RuntimeError("SparseMLP router tuple must contain (router_logits, expert_index)")
        router_logits = router_tuple[0]
        expert_index = router_tuple[1]
        if router_logits.dim() == 2:
            router_logits = router_logits.view(bsz, seq_len, -1)
        if router_logits.dim() != 3:
            raise RuntimeError(f"router logits expected [B,T,E], got {tuple(router_logits.shape)}")
        if expert_index.dim() == 2:
            expert_index = expert_index.unsqueeze(-1)
        if expert_index.dim() != 3:
            raise RuntimeError(f"expert index expected [B,T,K], got {tuple(expert_index.shape)}")
        if router_logits.shape[:2] != input_hidden.shape[:2] or expert_index.shape[:2] != input_hidden.shape[:2]:
            raise RuntimeError("router/expert tensors do not align with SparseMLP input hidden shape")
        self._router_logits_by_layer[layer_id] = router_logits.detach().cpu()
        self._expert_selection_by_layer[layer_id] = expert_index.detach().to(torch.int64).cpu()

    def run_split(self, prompts: list[str], *, batch_size: int, seq_id_start: int = 0) -> ERPPSplitAccumulator:
        if self.model is None:
            self.load()
        acc = ERPPSplitAccumulator()
        total_batches = (len(prompts) + batch_size - 1) // batch_size
        for batch_idx, start in enumerate(range(0, len(prompts), batch_size), start=1):
            batch_prompts = prompts[start:start + batch_size]
            seq_ids = torch.arange(seq_id_start + start, seq_id_start + start + len(batch_prompts), dtype=torch.int64)
            self._status(f"batch {batch_idx}/{total_batches}: prompts={len(batch_prompts)} seq_id={int(seq_ids[0])}..{int(seq_ids[-1])}")
            batch = self._run_batch(batch_prompts)
            acc.append(seq_ids=seq_ids, prompts=batch_prompts, **batch)
        return acc

    def _run_batch(self, prompts: list[str]) -> dict[str, torch.Tensor]:
        self._layer0_attn_out = None
        self._router_logits_by_layer.clear()
        self._expert_selection_by_layer.clear()

        token_kwargs = {
            "padding": self.padding,
            "truncation": True,
            "max_length": self.max_input_tokens,
            "return_tensors": "pt",
        }
        enc_inputs = self.tokenizer(prompts, **token_kwargs)
        input_ids = enc_inputs["input_ids"].to(self.device)
        attention_mask = enc_inputs["attention_mask"].to(self.device)

        with torch.inference_mode():
            self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_router_logits=True,
                return_dict=True,
            )

        if self._layer0_attn_out is None:
            raise RuntimeError("missing layer0 attention output; hook did not fire")
        num_layers = len(self.router_layer_to_model_block)
        missing = [i for i in range(num_layers) if i not in self._router_logits_by_layer]
        if missing:
            raise RuntimeError(f"missing router logits for layers {missing}")

        per_layer_logits = [self._router_logits_by_layer[i] for i in range(num_layers)]
        per_layer_experts = [self._expert_selection_by_layer[i] for i in range(num_layers)]
        router_logits = torch.stack(per_layer_logits, dim=1)  # [B,L,T,E]
        expert_selection = torch.stack(per_layer_experts, dim=1)  # [B,L,T,K]
        layer0_attn_out = self._layer0_attn_out

        expected_bt = tuple(input_ids.shape)
        if tuple(layer0_attn_out.shape[:2]) != expected_bt:
            raise RuntimeError(f"layer0 attention shape {tuple(layer0_attn_out.shape)} does not align with input_ids {expected_bt}")
        if tuple(router_logits.shape[:1] + router_logits.shape[2:3]) != expected_bt:
            raise RuntimeError(f"router logits shape {tuple(router_logits.shape)} does not align with input_ids {expected_bt}")
        if tuple(expert_selection.shape[:1] + expert_selection.shape[2:3]) != expected_bt:
            raise RuntimeError(f"expert selection shape {tuple(expert_selection.shape)} does not align with input_ids {expected_bt}")

        return {
            "input_ids": input_ids.detach().cpu(),
            "attention_mask": attention_mask.detach().cpu(),
            "layer0_attn_out": layer0_attn_out,
            "router_logits": router_logits,
            "expert_selection": expert_selection,
        }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export ERPP encoder trace for Switch Transformers.")
    p.add_argument("--model-path", required=True, type=Path)
    p.add_argument("--train-prompt-file", required=True, type=Path)
    p.add_argument("--validation-prompt-file", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--max-input-tokens", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--padding", choices=["max_length", "longest"], default="max_length")
    p.add_argument("--storage-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    p.add_argument("--no-router-probs", action="store_true", help="Do not write router_probs.pt")
    p.add_argument("--no-expert-weights", action="store_true", help="Do not write expert_weights.pt")
    p.add_argument("--no-prompt-texts", action="store_true", help="Do not write prompt_texts.jsonl")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--print-status", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    train_prompts = _load_prompts(args.train_prompt_file)
    val_prompts = _load_prompts(args.validation_prompt_file)
    t0 = time.time()

    runner = ERPPEncoderTraceRunner(
        model_path=args.model_path,
        device=args.device,
        seed=args.seed,
        max_input_tokens=args.max_input_tokens,
        padding=args.padding,
        verbose=args.print_status,
    )
    try:
        runner.load()
        train_acc = runner.run_split(train_prompts, batch_size=args.batch_size, seq_id_start=0)
        val_acc = runner.run_split(val_prompts, batch_size=args.batch_size, seq_id_start=0)
    finally:
        runner.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    writer = ERPPTraceWriter(args.output_dir, _storage_dtype(args.storage_dtype))
    split_meta = {
        "train": writer.write_split(
            "train",
            train_acc,
            store_router_probs=not args.no_router_probs,
            store_expert_weights=not args.no_expert_weights,
            store_prompt_texts=not args.no_prompt_texts,
        ),
        "validation": writer.write_split(
            "validation",
            val_acc,
            store_router_probs=not args.no_router_probs,
            store_expert_weights=not args.no_expert_weights,
            store_prompt_texts=not args.no_prompt_texts,
        ),
    }

    config = runner.model.config
    metadata = {
        "schema": "erpp_encoder_trace",
        "schema_version": 1,
        "model_name": getattr(config, "_name_or_path", "switch_transformers"),
        "model_path": str(args.model_path),
        "stage": "encoder",
        "execution_mode": "encoder_forward",
        "canonical_input": "encoder.block.0.layer.0.self_attention.output_hidden_states_after_residual",
        "tensor_layout": "packed_sequence_first",
        "torch_dtype": str(getattr(config, "torch_dtype", "float32")),
        "router_dtype": str(getattr(config, "router_dtype", "float32")),
        "router_jitter_noise": float(getattr(config, "router_jitter_noise", 0.0)),
        "router_jitter_active": False,
        "router_jitter_note": "model.eval() disables Switch router jitter; exporter preserves config value and does not mutate router noise",
        "storage_dtype": args.storage_dtype,
        "max_input_tokens": int(args.max_input_tokens),
        "padding": args.padding,
        "hidden_size": int(config.d_model),
        "num_experts": int(config.num_experts),
        "routing_top_k": int(getattr(config, "num_selected_experts", 1)),
        "num_encoder_layers": int(config.num_layers),
        "num_encoder_moe_layers": len(runner.router_layer_to_model_block),
        "target_router_layers": list(range(len(runner.router_layer_to_model_block))),
        "router_layer_to_model_block": runner.router_layer_to_model_block,
        "has_router_logits": True,
        "has_router_probs": not args.no_router_probs,
        "has_expert_selection": True,
        "has_expert_weights": not args.no_expert_weights,
        "expert_selection_source": "sparse_mlp_router_tuple_expert_index",
        "splits": split_meta,
    }
    _write_json(args.output_dir / "metadata.json", metadata)
    run_log = {
        "seed": args.seed,
        "batch_size": args.batch_size,
        "device": args.device,
        "train_prompts": len(train_prompts),
        "validation_prompts": len(val_prompts),
        "elapsed_sec": round(time.time() - t0, 3),
    }
    _write_json(args.output_dir / "run_log.json", run_log)
    print(json.dumps({"metadata": metadata, "run_log": run_log}, indent=2, ensure_ascii=False))

    if args.verify:
        from verify_erpp_trace import verify_trace

        verify_trace(args.output_dir)
        print("ERPP verifier: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
