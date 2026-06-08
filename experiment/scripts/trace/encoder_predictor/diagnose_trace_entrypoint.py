#!/usr/bin/env python3
"""Diagnostic trace exporter for isolating entrypoint/runtime differences.

This file is intentionally separate from the production exporters.  It can run
the same encoder trace hooks through either an unpatched single-GPU model or the
sparse cache patched model, and through either encoder(...) or generate(...).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from export_sparse_cache_encoder_trace import (  # noqa: E402
    REPO_ROOT,
    SplitAccumulator,
    TraceWriter,
    _model_config_metadata,
    build_sparse_cache_config,
    bool_arg,
    is_transformers_one_token_timing_zero_division,
    json_safe,
    load_prompts,
    resolve_prompt_file,
    storage_dtype,
    verify_trace,
    write_json,
)


def _normalize_dtype_name(value) -> str:
    return str(value).strip().lower().replace("torch.", "")


def _model_torch_dtype(name: str, config) -> torch.dtype:
    requested = _normalize_dtype_name(name)
    if requested == "auto":
        requested = _normalize_dtype_name(getattr(config, "torch_dtype", None) or "float32")
    table = {
        "float": torch.float32,
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if requested not in table:
        raise argparse.ArgumentTypeError(f"unsupported model torch dtype {name!r}")
    return table[requested]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose encoder trace differences by runtime and entrypoint.")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--train-split", default="test")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--train-prompt-file", type=Path)
    parser.add_argument("--validation-prompt-file", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--runtime", choices=["single", "sparse"], required=True)
    parser.add_argument("--entrypoint", choices=["encoder", "generate"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--padding", choices=["longest", "max_length"], default="longest")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--storage-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--model-torch-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-rate", type=float, default=0.375)
    parser.add_argument("--cache-policy", default="lru")
    parser.add_argument("--per-layer-cache", type=bool_arg, default=False)
    parser.add_argument("--gpu-mem-limit-gb", type=float)
    parser.add_argument("--no-verify", dest="verify", action="store_false", default=True)
    parser.add_argument("--print-status", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


class DiagnosticTraceRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model = None
        self.tokenizer = None
        self.input_device = torch.device(args.device)
        self.layer0_attn_out: torch.Tensor | None = None
        self.router_logits_by_layer: dict[int, torch.Tensor] = {}
        self.expert_selection_by_layer: dict[int, torch.Tensor] = {}
        self.router_layer_to_model_block: list[int] = []
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.resolved_model_torch_dtype: str | None = None

    def status(self, message: str) -> None:
        if self.args.print_status:
            print(f"[trace-entrypoint-diagnostic] {message}", flush=True)

    def _seed_all(self) -> None:
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)

    def load(self) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HUGGINGFACE_OFFLINE", "1")
        self._seed_all()
        if self.input_device.type == "cuda":
            torch.cuda.set_device(self.input_device)
            if self.args.gpu_mem_limit_gb is not None:
                total_mem_gb = torch.cuda.get_device_properties(self.input_device).total_memory / (1024 ** 3)
                fraction = min(float(self.args.gpu_mem_limit_gb) / total_mem_gb, 1.0)
                torch.cuda.set_per_process_memory_fraction(fraction, device=self.input_device)
                self.status(f"set per-process CUDA memory fraction {fraction:.6f}")

        from transformers import AutoConfig, AutoTokenizer, SwitchTransformersForConditionalGeneration

        config = AutoConfig.from_pretrained(self.args.model_path, local_files_only=True)
        torch_dtype = _model_torch_dtype(self.args.model_torch_dtype, config)
        self.resolved_model_torch_dtype = str(torch_dtype).replace("torch.", "")

        if self.args.runtime == "sparse":
            import sparse_llm_cache

            sparse_llm_cache.utils.hack_transformers(
                **build_sparse_cache_config(self.args),
                pin_memory=True,
                enable_model_timer=False,
            )

        self.status(f"loading {self.args.runtime} model from {self.args.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.args.runtime == "sparse":
            self.model = SwitchTransformersForConditionalGeneration.from_pretrained(
                self.args.model_path,
                torch_dtype=torch_dtype,
                local_files_only=True,
                device_map=0,
                trust_remote_code=True,
            )
        else:
            self.model = SwitchTransformersForConditionalGeneration.from_pretrained(
                self.args.model_path,
                torch_dtype=torch_dtype,
                local_files_only=True,
                trust_remote_code=True,
            )
            self.model.to(self.input_device)

        if getattr(self.model.config, "is_encoder_decoder", False) and self.model.config.decoder_start_token_id is None:
            self.model.config.decoder_start_token_id = self.tokenizer.pad_token_id
            self.model.generation_config.decoder_start_token_id = self.tokenizer.pad_token_id
        self.model.eval()
        self.attach_trace_hooks()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def attach_trace_hooks(self) -> None:
        from transformers.models.switch_transformers.modeling_switch_transformers import SwitchTransformersSparseMLP

        encoder_blocks = getattr(self.model.encoder, "block", None)
        if encoder_blocks is None:
            raise RuntimeError("model has no encoder.block")
        self.handles.append(encoder_blocks[0].layer[0].register_forward_hook(self._layer0_attention_hook))

        router_layer_id = 0
        self.router_layer_to_model_block.clear()
        for block_id, block in enumerate(self.model.encoder.block):
            for layer in block.layer:
                mlp = getattr(layer, "mlp", None)
                if isinstance(mlp, SwitchTransformersSparseMLP):
                    mlp._diagnostic_trace_layer_id = router_layer_id
                    self.router_layer_to_model_block.append(int(block_id))
                    self.handles.append(mlp.register_forward_hook(self._sparse_mlp_hook))
                    router_layer_id += 1
        if router_layer_id == 0:
            raise RuntimeError("found no encoder SwitchTransformersSparseMLP modules")

    def _layer0_attention_hook(self, module, inputs, output) -> None:
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(hidden, torch.Tensor) or hidden.dim() != 3:
            raise RuntimeError(f"layer0 attention hook expected [B,T,H], got {type(hidden).__name__}")
        self.layer0_attn_out = hidden.detach().cpu()

    def _sparse_mlp_hook(self, module, inputs, output) -> None:
        layer_id = int(module._diagnostic_trace_layer_id)
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError("SparseMLP hook expected hidden states in inputs[0]")
        hidden = inputs[0]
        bsz, seq_len = int(hidden.shape[0]), int(hidden.shape[1])
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            raise RuntimeError("SparseMLP output must contain router tuple")
        router_tuple = output[1]
        if not isinstance(router_tuple, (tuple, list)) or len(router_tuple) < 2:
            raise RuntimeError("SparseMLP router tuple must contain (router_logits, expert_index)")
        router_logits = router_tuple[0]
        expert_index = router_tuple[1]
        if router_logits.dim() == 2:
            router_logits = router_logits.view(bsz, seq_len, -1)
        if expert_index.dim() == 2:
            expert_index = expert_index.unsqueeze(-1)
        self.router_logits_by_layer[layer_id] = router_logits.detach().cpu()
        self.expert_selection_by_layer[layer_id] = expert_index.detach().to(torch.int64).cpu()

    def run_split(self, prompts: list[str], *, batch_size: int, seq_id_start: int = 0) -> SplitAccumulator:
        if self.model is None:
            self.load()
        acc = SplitAccumulator()
        total_batches = (len(prompts) + batch_size - 1) // batch_size
        for batch_idx, start in enumerate(range(0, len(prompts), batch_size), start=1):
            batch_prompts = prompts[start : start + batch_size]
            seq_ids = torch.arange(seq_id_start + start, seq_id_start + start + len(batch_prompts), dtype=torch.int64)
            self.status(f"batch {batch_idx}/{total_batches}: prompts={len(batch_prompts)} seq_id={int(seq_ids[0])}..{int(seq_ids[-1])}")
            batch = self.run_batch(batch_prompts)
            acc.append(seq_ids=seq_ids, prompts=batch_prompts, **batch)
        return acc

    def run_batch(self, prompts: list[str]) -> dict[str, torch.Tensor]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("runner must be loaded before run_batch")
        self.layer0_attn_out = None
        self.router_logits_by_layer.clear()
        self.expert_selection_by_layer.clear()

        enc_inputs = self.tokenizer(
            prompts,
            padding=self.args.padding,
            truncation=True,
            max_length=self.args.max_input_tokens,
            return_tensors="pt",
        )
        input_ids = enc_inputs["input_ids"].to(self.input_device)
        attention_mask = enc_inputs["attention_mask"].to(self.input_device)

        self.status(f"starting {self.args.entrypoint}: input_shape={tuple(input_ids.shape)}")
        with torch.inference_mode():
            if self.args.entrypoint == "encoder":
                self.model.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_router_logits=True,
                    return_dict=True,
                )
            else:
                try:
                    self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.args.max_new_tokens,
                        do_sample=False,
                    )
                except ZeroDivisionError as exc:
                    if self.args.max_new_tokens != 1 or not is_transformers_one_token_timing_zero_division(exc):
                        raise
                    self.status("generate hit transformers one-token timing bug; continuing with hook traces")
        self.status(f"{self.args.entrypoint} finished")

        if self.layer0_attn_out is None:
            raise RuntimeError("missing layer0 attention output; hook did not fire")
        num_layers = len(self.router_layer_to_model_block)
        missing_logits = [i for i in range(num_layers) if i not in self.router_logits_by_layer]
        missing_selection = [i for i in range(num_layers) if i not in self.expert_selection_by_layer]
        if missing_logits:
            raise RuntimeError(f"missing encoder router logits for layers {missing_logits}")
        if missing_selection:
            raise RuntimeError(f"missing encoder expert selection for layers {missing_selection}")

        router_logits = torch.stack([self.router_logits_by_layer[i] for i in range(num_layers)], dim=1)
        expert_selection = torch.stack([self.expert_selection_by_layer[i] for i in range(num_layers)], dim=1)
        return {
            "input_ids": input_ids.detach().cpu(),
            "attention_mask": attention_mask.detach().cpu(),
            "layer0_attn_out": self.layer0_attn_out,
            "router_logits": router_logits,
            "expert_selection": expert_selection,
        }


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    train_prompt_file = args.train_prompt_file or resolve_prompt_file(REPO_ROOT, args.dataset, args.task_name, args.train_split)
    validation_prompt_file = args.validation_prompt_file or resolve_prompt_file(
        REPO_ROOT, args.dataset, args.task_name, args.validation_split
    )
    return Path(train_prompt_file), Path(validation_prompt_file)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.model_path = Path(args.model_path)
    args.output_dir = Path(args.output_dir)
    train_prompt_file, validation_prompt_file = _resolve_paths(args)
    resolved = {
        "repo_root": str(REPO_ROOT),
        "runtime": args.runtime,
        "entrypoint": args.entrypoint,
        "model_path": str(args.model_path),
        "train_prompt_file": str(train_prompt_file),
        "validation_prompt_file": str(validation_prompt_file),
        "output_dir": str(args.output_dir),
        "sparse_cache_config": json_safe(build_sparse_cache_config(args)) if args.runtime == "sparse" else None,
    }
    if args.dry_run:
        print(json.dumps(resolved, indent=2, ensure_ascii=False))
        return 0

    start = time.time()
    train_prompts = load_prompts(train_prompt_file)
    validation_prompts = load_prompts(validation_prompt_file)

    runner = DiagnosticTraceRunner(args)
    try:
        runner.load()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        writer = TraceWriter(args.output_dir, storage_dtype(args.storage_dtype))
        train_acc = runner.run_split(train_prompts, batch_size=args.batch_size, seq_id_start=0)
        validation_acc = runner.run_split(validation_prompts, batch_size=args.batch_size, seq_id_start=0)
        metadata = {
            "schema": "diagnostic_encoder_trace",
            "runtime": args.runtime,
            "entrypoint": args.entrypoint,
            "args": json_safe(vars(args)),
            "resolved": resolved,
            "sparse_cache_config": json_safe(build_sparse_cache_config(args)) if args.runtime == "sparse" else None,
            "device": str(args.device),
            "storage_dtype": args.storage_dtype,
            "model_torch_dtype": args.model_torch_dtype,
            "resolved_model_torch_dtype": runner.resolved_model_torch_dtype,
            "model_config": _model_config_metadata(runner.model.config),
            "router_layer_to_model_block": list(runner.router_layer_to_model_block),
            "splits": {
                "train": writer.write_split("train", train_acc),
                "validation": writer.write_split("validation", validation_acc),
            },
        }
        if args.verify:
            metadata["verification"] = verify_trace(args.output_dir, metadata)
        write_json(args.output_dir / "metadata.json", json_safe(metadata))
        write_json(
            args.output_dir / "run_log.json",
            json_safe(
                {
                    "start_time": start,
                    "end_time": time.time(),
                    "elapsed_seconds": time.time() - start,
                    "python": sys.executable,
                    "runtime": args.runtime,
                    "entrypoint": args.entrypoint,
                    "output_dir": str(args.output_dir),
                    "num_train_prompts": len(train_prompts),
                    "num_validation_prompts": len(validation_prompts),
                }
            ),
        )
    finally:
        runner.close()

    print(f"wrote diagnostic trace to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
