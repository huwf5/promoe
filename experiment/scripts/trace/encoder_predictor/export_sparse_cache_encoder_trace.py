#!/usr/bin/env python3
"""Export encoder predictor traces through sparse_llm_cache patched Transformers."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def find_repo_root(start: Optional[Path] = None) -> Path:
    current = Path.cwd() if start is None else Path(start).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "experiment").is_dir() and (candidate / "src").is_dir():
            return candidate
    return Path(__file__).resolve().parents[4]


REPO_ROOT = find_repo_root()
SRC_DIR = REPO_ROOT / "src"
TRANSFORMERS_SRC = REPO_ROOT / "deps" / "transformers" / "src"
for path in (SRC_DIR, TRANSFORMERS_SRC):
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


def storage_dtype(name: str) -> torch.dtype:
    table = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return table[name]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"unsupported storage dtype {name!r}") from exc


def dtype_name(dtype: torch.dtype) -> str:
    if dtype is torch.float32:
        return "float32"
    if dtype is torch.float16:
        return "float16"
    if dtype is torch.bfloat16:
        return "bfloat16"
    raise argparse.ArgumentTypeError(f"unsupported storage dtype {dtype!r}")


def _config_torch_dtype(config) -> torch.dtype | None:
    raw = getattr(config, "torch_dtype", None)
    if raw is None:
        return None
    if isinstance(raw, torch.dtype):
        return raw
    text = str(raw).replace("torch.", "").lower()
    aliases = {
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "half": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return aliases.get(text)


def resolve_model_torch_dtype(name: str, config) -> torch.dtype:
    if name != "auto":
        return storage_dtype(name)
    inferred = _config_torch_dtype(config)
    if inferred is None:
        raise ValueError(f"failed to resolve model torch dtype from config {config}")
    return storage_dtype(dtype_name(inferred))


def resolve_storage_dtype(name: str, config) -> torch.dtype:
    if name != "auto":
        return storage_dtype(name)
    inferred = _config_torch_dtype(config)
    if inferred is None:
        raise ValueError(f"failed to resolve storage dtype from config {config}")
    return storage_dtype(dtype_name(inferred))


def load_transformers_config(model_path: Path):
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )


def _is_nllb_moe_model_type(model_type: str | None) -> bool:
    return model_type in {"nllb-moe", "nllb_moe"}


def selected_experts_for_config(config) -> int:
    raw = getattr(config, "num_selected_experts", None)
    if raw is not None:
        return int(raw)
    model_type = getattr(config, "model_type", None)
    if model_type == "switch_transformers":
        return 1
    if _is_nllb_moe_model_type(model_type):
        return 2
    raise AttributeError("model config is missing required integer field: num_selected_experts")


def sparse_cache_switch_selected_experts(config) -> int:
    return selected_experts_for_config(config)


def validate_sparse_cache_encoder_trace_config(config, model_path: Path) -> int:
    model_type = getattr(config, "model_type", None)
    selected = selected_experts_for_config(config)
    if model_type == "switch_transformers":
        if selected != 1:
            raise ValueError(
                "encoder predictor sparse-cache trace currently supports Switch top-1 routing only "
                "because sparse_llm_cache.model_adapters.SwitchAdapter requires "
                "config.num_selected_experts == 1. "
                f"{model_path} has num_selected_experts={selected!r}. "
                "Use a top-1 Switch model, or add top-k support to sparse_llm_cache before exporting this trace."
            )
        return selected
    if _is_nllb_moe_model_type(model_type):
        if selected != 2:
            raise ValueError(
                "encoder predictor sparse-cache trace supports NLLB-MoE top-2 routing only; "
                f"{model_path} has num_selected_experts={selected!r}."
            )
        return selected
    raise ValueError(
        "encoder predictor sparse-cache trace requires a SwitchTransformers or NLLB-MoE config "
        f"(model_type='switch_transformers' or 'nllb-moe'); {model_path} has model_type={model_type!r}"
    )


def validate_sparse_cache_switch_top1(config, model_path: Path) -> int:
    model_type = getattr(config, "model_type", None)
    if model_type != "switch_transformers":
        raise ValueError(
            "encoder predictor sparse-cache trace requires a SwitchTransformers config "
            f"(model_type='switch_transformers'); {model_path} has model_type={model_type!r}"
        )
    return validate_sparse_cache_encoder_trace_config(config, model_path)


def routed_topk_from_router_probs(router_probs: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if router_probs.dim() < 1:
        raise ValueError("router_probs must have an expert dimension")
    num_experts = int(router_probs.shape[-1])
    if num_experts <= 0:
        raise ValueError("router_probs expert dimension must be non-empty")
    if top_k > num_experts:
        raise ValueError(f"top_k must be <= num_experts ({num_experts}), got {top_k}")

    weights, selection = torch.topk(router_probs, k=int(top_k), dim=-1)
    mask = weights > 0
    selection = torch.where(mask, selection.to(torch.int64), torch.zeros_like(selection, dtype=torch.int64))
    weights = torch.where(mask, weights, torch.zeros_like(weights))
    return selection.to(torch.int64), weights, mask.to(torch.bool)


def derive_model_name(model_path: Path) -> str:
    return Path(model_path).name


def nllb_encoder_module(model: torch.nn.Module) -> torch.nn.Module:
    get_encoder = getattr(model, "get_encoder", None)
    if callable(get_encoder):
        encoder = get_encoder()
        if encoder is not None:
            return encoder
    nested_model = getattr(model, "model", None)
    nested_encoder = getattr(nested_model, "encoder", None) if nested_model is not None else None
    if nested_encoder is not None:
        return nested_encoder
    encoder = getattr(model, "encoder", None)
    if encoder is not None:
        return encoder
    raise RuntimeError(f"NLLB model {type(model).__name__} has no encoder module")


def resolve_default_output_dir(repo_root: Path, model_path: Path, dataset: str, task_name: str) -> Path:
    model_name = derive_model_name(model_path)
    return (
        repo_root
        / "experiment"
        / "traces"
        / f"{model_name}-{dataset}-{task_name}"
        / "encoder_predictor_sparse_cache_trace"
    )


def resolve_prompt_file(repo_root: Path, dataset: str, task_name: str, split: str) -> Path:
    return repo_root / "experiment" / "datasets" / dataset / task_name / split / "prompt_list.txt"


def load_prompts(path: Path) -> list[str]:
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"prompt file has no non-empty lines: {path}")
    return prompts


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


@dataclass
class SplitAccumulator:
    input_ids: list[torch.Tensor] = field(default_factory=list)
    attention_mask: list[torch.Tensor] = field(default_factory=list)
    layer0_attn_out: list[torch.Tensor] = field(default_factory=list)
    router_logits: list[torch.Tensor] = field(default_factory=list)
    router_probs: list[torch.Tensor] = field(default_factory=list)
    expert_selection: list[torch.Tensor] = field(default_factory=list)
    expert_weights: list[torch.Tensor] = field(default_factory=list)
    expert_selection_mask: list[torch.Tensor] = field(default_factory=list)
    seq_ids: list[torch.Tensor] = field(default_factory=list)
    prompt_records: list[dict] = field(default_factory=list)

    def append(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layer0_attn_out: torch.Tensor,
        router_logits: torch.Tensor,
        expert_selection: torch.Tensor,
        seq_ids: torch.Tensor,
        router_probs: torch.Tensor | None = None,
        expert_weights: torch.Tensor | None = None,
        expert_selection_mask: torch.Tensor | None = None,
        prompts: list[str],
    ) -> None:
        self.input_ids.append(input_ids.detach().cpu())
        self.attention_mask.append(attention_mask.detach().cpu())
        self.layer0_attn_out.append(layer0_attn_out.detach().cpu())
        self.router_logits.append(router_logits.detach().cpu())
        if router_probs is not None:
            self.router_probs.append(router_probs.detach().cpu())
        self.expert_selection.append(expert_selection.detach().to(torch.int64).cpu())
        if expert_weights is not None:
            self.expert_weights.append(expert_weights.detach().cpu())
        if expert_selection_mask is not None:
            self.expert_selection_mask.append(expert_selection_mask.detach().to(torch.bool).cpu())
        self.seq_ids.append(seq_ids.detach().to(torch.int64).cpu())
        for seq_id, text in zip(seq_ids.tolist(), prompts):
            self.prompt_records.append({"seq_id": int(seq_id), "text": text})

    def total_samples(self) -> int:
        return sum(int(t.shape[0]) for t in self.input_ids)

    @staticmethod
    def _sequence_dim(name: str) -> Optional[int]:
        if name in {"input_ids", "attention_mask", "layer0_attn_out"}:
            return 1
        if name in {"router_logits", "router_probs", "expert_selection", "expert_weights", "expert_selection_mask"}:
            return 2
        return None

    def cat(self, name: str) -> torch.Tensor:
        values = getattr(self, name)
        if not values:
            raise RuntimeError(f"split accumulator has no tensors for {name}")
        seq_dim = self._sequence_dim(name)
        if seq_dim is None:
            return torch.cat(values, dim=0)
        max_len = max(int(t.shape[seq_dim]) for t in values)
        padded: list[torch.Tensor] = []
        for tensor in values:
            current_len = int(tensor.shape[seq_dim])
            if current_len == max_len:
                padded.append(tensor)
                continue
            shape = list(tensor.shape)
            shape[seq_dim] = max_len
            out = tensor.new_zeros(shape)
            index = [slice(None)] * tensor.dim()
            index[seq_dim] = slice(0, current_len)
            out[tuple(index)] = tensor
            padded.append(out)
        return torch.cat(padded, dim=0)

    def true_token_lengths(self) -> list[int]:
        lengths: list[int] = []
        for attention_mask in self.attention_mask:
            lengths.extend(int(x) for x in attention_mask.to(torch.int64).sum(dim=1).tolist())
        return lengths


class TraceWriter:
    def __init__(self, output_dir: Path, dtype: torch.dtype, *, require_explicit_router_probs: bool = False) -> None:
        self.output_dir = Path(output_dir)
        self.dtype = dtype
        self.require_explicit_router_probs = bool(require_explicit_router_probs)

    def write_split(self, split: str, acc: SplitAccumulator) -> dict:
        if acc.total_samples() == 0:
            raise ValueError(f"refusing to write empty split {split!r}")
        split_dir = self.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        input_ids = acc.cat("input_ids").to(torch.int64)
        attention_mask = acc.cat("attention_mask").to(torch.int64)
        layer0_attn_out = acc.cat("layer0_attn_out").to(self.dtype)
        router_logits = acc.cat("router_logits").to(self.dtype)
        expert_selection = acc.cat("expert_selection").to(torch.int64)
        seq_ids = acc.cat("seq_ids").to(torch.int64)

        missing_explicit = []
        if not acc.router_probs:
            missing_explicit.append("router_probs")
        if not acc.expert_weights:
            missing_explicit.append("expert_weights")
        if not acc.expert_selection_mask:
            missing_explicit.append("expert_selection_mask")
        if self.require_explicit_router_probs and missing_explicit:
            raise RuntimeError(
                "TraceWriter strict router schema requires explicit " + ", ".join(missing_explicit)
            )

        if acc.router_probs:
            router_probs = acc.cat("router_probs").to(self.dtype)
        else:
            router_probs = torch.softmax(router_logits.float(), dim=-1).to(self.dtype)
        if acc.expert_weights:
            expert_weights = acc.cat("expert_weights").to(self.dtype)
        else:
            expert_weights = torch.gather(
                router_probs.float(),
                dim=-1,
                index=expert_selection,
            ).to(self.dtype)
        if acc.expert_selection_mask:
            expert_selection_mask = acc.cat("expert_selection_mask").to(torch.bool)
        else:
            expert_selection_mask = expert_weights > 0

        torch.save(input_ids, split_dir / "input_ids.pt")
        torch.save(attention_mask, split_dir / "attention_mask.pt")
        torch.save(layer0_attn_out, split_dir / "layer0_attn_out.pt")
        torch.save(router_logits, split_dir / "router_logits.pt")
        torch.save(router_probs, split_dir / "router_probs.pt")
        torch.save(expert_selection, split_dir / "expert_selection.pt")
        torch.save(expert_weights, split_dir / "expert_weights.pt")
        torch.save(expert_selection_mask, split_dir / "expert_selection_mask.pt")
        torch.save(seq_ids, split_dir / "seq_ids.pt")

        with (split_dir / "prompt_texts.jsonl").open("w", encoding="utf-8") as f:
            for record in acc.prompt_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return {
            "num_samples": int(input_ids.shape[0]),
            "true_token_lengths": acc.true_token_lengths(),
            "input_ids_shape": list(input_ids.shape),
            "attention_mask_shape": list(attention_mask.shape),
            "layer0_attn_out_shape": list(layer0_attn_out.shape),
            "router_logits_shape": list(router_logits.shape),
            "router_probs_shape": list(router_probs.shape),
            "expert_selection_shape": list(expert_selection.shape),
            "expert_weights_shape": list(expert_weights.shape),
            "expert_selection_mask_shape": list(expert_selection_mask.shape),
            "seq_ids_shape": list(seq_ids.shape),
        }



def bool_arg(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {value!r}")


def resolve_cache_rate(args: argparse.Namespace, *, model_type: str | None) -> float:
    if getattr(args, "cache_rate", None) is not None:
        return float(args.cache_rate)
    if _is_nllb_moe_model_type(model_type):
        return 0.01
    return 0.375


def build_sparse_cache_config(args: argparse.Namespace, model_type: str | None = "switch_transformers") -> dict:
    return {
        "model_id": str(args.model_path),
        "cache_rate": resolve_cache_rate(args, model_type=model_type),
        "cache_policy": args.cache_policy,
        "per_layer_cache": args.per_layer_cache,
        "num_predict_expert_per_layer": 0,
        "reorder_experts": False,
        "early_preempt": False,
        "chunk_prefetch": False,
        "predict_input_mode": "no_predict",
    }


def is_transformers_one_token_timing_zero_division(exc: ZeroDivisionError) -> bool:
    for frame in traceback.extract_tb(exc.__traceback__):
        filename = frame.filename.replace("\\", "/")
        if filename.endswith("transformers/generation/utils.py") and frame.name == "_sample":
            return True
    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export sparse-cache-backed encoder predictor traces.")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--train-split", default="test")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--train-prompt-file", type=Path)
    parser.add_argument("--validation-prompt-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--padding", choices=["longest", "max_length"], default="longest")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--storage-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--model-torch-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-rate", type=float, default=None)
    parser.add_argument("--cache-policy", default="lru")
    parser.add_argument("--per-layer-cache", type=bool_arg, default=False)
    parser.add_argument("--gpu-mem-limit-gb", type=float)
    parser.add_argument("--assert-gpu-expert-forward", dest="assert_gpu_expert_forward", action="store_true", default=True)
    parser.add_argument("--no-assert-gpu-expert-forward", dest="assert_gpu_expert_forward", action="store_false")
    parser.add_argument("--no-verify", dest="verify", action="store_false", default=True)
    parser.add_argument("--print-status", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Resolve paths/config and exit without loading model")
    return parser


class GpuExpertForwardAsserter:
    def __init__(self, expected_device: str) -> None:
        self.expected_device = torch.device(expected_device)
        self.modules: list[torch.nn.Module] = []
        self.observed = 0

    def attach(self, model: torch.nn.Module) -> None:
        from sparse_llm_cache.utils import hooks as cache_hooks

        asserter = self

        class AssertGpuExpertForwardHook(cache_hooks.ModelHook):
            def pre_forward(self, module, *args, **kwargs):
                asserter._check_forward_state(module, args)
                return args, kwargs

        for module in model.modules():
            if hasattr(module, "_expert_id") and hasattr(module, "_layer_id"):
                cache_hooks.add_hook_to_module(module, AssertGpuExpertForwardHook(), append=True)
                self.modules.append(module)

    def remove(self) -> None:
        self.modules.clear()

    def _expected_device_label(self) -> str:
        if self.expected_device.type == "cuda" and self.expected_device.index is None:
            return "cuda:any"
        return str(self.expected_device)

    def _device_matches_expected(self, actual: torch.device) -> bool:
        if actual.type != self.expected_device.type:
            return False
        if self.expected_device.type == "cuda" and self.expected_device.index is None:
            return True
        return actual.index == self.expected_device.index

    def _check_forward_state(self, module: torch.nn.Module, inputs: tuple) -> None:
        self.observed += 1
        expected = self._expected_device_label()
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError("expert forward did not receive tensor input")
        actual_input_device = inputs[0].device
        if not self._device_matches_expected(actual_input_device):
            raise RuntimeError(f"expert input device mismatch: expected {expected}, actual {actual_input_device}")
        for param in module.parameters(recurse=True):
            actual_param_device = param.device
            if not self._device_matches_expected(actual_param_device):
                raise RuntimeError(
                    f"expert L{getattr(module, '_layer_id', '?')} E{getattr(module, '_expert_id', '?')} "
                    f"parameter device mismatch: expected {expected}, actual {actual_param_device}"
                )
            break


class SparseCacheTraceRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model = None
        self.tokenizer = None
        self.input_device = torch.device(args.device)
        self.layer0_attn_out: torch.Tensor | None = None
        self.router_logits_by_layer: dict[int, torch.Tensor] = {}
        self.router_probs_by_layer: dict[int, torch.Tensor] = {}
        self.expert_selection_by_layer: dict[int, torch.Tensor] = {}
        self.expert_weights_by_layer: dict[int, torch.Tensor] = {}
        self.expert_selection_mask_by_layer: dict[int, torch.Tensor] = {}
        self.nllb_layer_input_shape_by_layer: dict[int, tuple[int, int]] = {}
        self.router_layer_to_model_block: list[int] = []
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.gpu_asserter: GpuExpertForwardAsserter | None = None
        self.resolved_model_torch_dtype: torch.dtype | None = None

    def status(self, message: str) -> None:
        if self.args.print_status:
            print(f"[encoder-predictor-trace] {message}", flush=True)

    def load(self) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HUGGINGFACE_OFFLINE", "1")
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if self.input_device.type == "cuda":
            torch.cuda.set_device(self.input_device)
            if self.args.gpu_mem_limit_gb is not None:
                if self.args.gpu_mem_limit_gb <= 0:
                    raise ValueError(f"gpu_mem_limit_gb should be > 0, got {self.args.gpu_mem_limit_gb}")
                total_mem_gb = torch.cuda.get_device_properties(self.input_device).total_memory / (1024 ** 3)
                gpu_mem_fraction = min(float(self.args.gpu_mem_limit_gb) / total_mem_gb, 1.0)
                torch.cuda.set_per_process_memory_fraction(gpu_mem_fraction, device=self.input_device)
                self.status(
                    f"set torch per-process memory fraction to {gpu_mem_fraction:.6f} "
                    f"on {self.input_device} (limit {self.args.gpu_mem_limit_gb}GB / total {total_mem_gb:.2f}GB)"
                )

        import sparse_llm_cache
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, SwitchTransformersForConditionalGeneration

        config = load_transformers_config(self.args.model_path)
        validate_sparse_cache_encoder_trace_config(config, self.args.model_path)
        self.resolved_model_torch_dtype = resolve_model_torch_dtype(self.args.model_torch_dtype, config)
        self.status(
            f"resolved model torch dtype {dtype_name(self.resolved_model_torch_dtype)} "
            f"from --model-torch-dtype {self.args.model_torch_dtype}"
        )

        cache_config = build_sparse_cache_config(self.args, model_type=getattr(config, "model_type", None))
        sparse_llm_cache.utils.hack_transformers(
            **cache_config,
            pin_memory=True,
            enable_model_timer=False,
        )

        self.status(f"loading model from {self.args.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_type = getattr(config, "model_type", None)
        model_cls = AutoModelForSeq2SeqLM if _is_nllb_moe_model_type(model_type) else SwitchTransformersForConditionalGeneration
        self.model = model_cls.from_pretrained(
            self.args.model_path,
            torch_dtype=self.resolved_model_torch_dtype,
            local_files_only=True,
            device_map=0,
            trust_remote_code=True,
        )
        if getattr(self.model.config, "is_encoder_decoder", False) and self.model.config.decoder_start_token_id is None:
            self.model.config.decoder_start_token_id = self.tokenizer.pad_token_id
            self.model.generation_config.decoder_start_token_id = self.tokenizer.pad_token_id
        self.model.eval()
        self.attach_trace_hooks()
        if self.args.assert_gpu_expert_forward:
            self.gpu_asserter = GpuExpertForwardAsserter(self.args.device)
            self.gpu_asserter.attach(self.model)

    def attach_trace_hooks(self) -> None:
        model_type = getattr(getattr(self.model, "config", None), "model_type", None)
        if _is_nllb_moe_model_type(model_type):
            self._attach_nllb_trace_hooks()
        elif model_type == "switch_transformers":
            self._attach_switch_trace_hooks()
        else:
            raise RuntimeError(f"unsupported model_type for trace hooks: {model_type!r}")

    def _attach_switch_trace_hooks(self) -> None:
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
                    mlp._encoder_predictor_trace_layer_id = router_layer_id
                    self.router_layer_to_model_block.append(int(block_id))
                    self.handles.append(mlp.register_forward_hook(self._sparse_mlp_hook))
                    router_layer_id += 1
        if router_layer_id == 0:
            raise RuntimeError("found no encoder SwitchTransformersSparseMLP modules")

    def _attach_nllb_trace_hooks(self) -> None:
        from sparse_llm_cache.model_adapters.nllb_moe import NllbMoeAdapter
        from transformers.models.nllb_moe.modeling_nllb_moe import NllbMoeSparseMLP

        encoder = nllb_encoder_module(self.model)
        encoder_layers = getattr(encoder, "layers", None)
        if encoder_layers is None:
            raise RuntimeError("NLLB encoder has no layers")
        if not encoder_layers:
            raise RuntimeError("NLLB model encoder.layers is empty")
        self.handles.append(encoder_layers[0].self_attn.register_forward_hook(self._layer0_attention_hook))

        adapter = NllbMoeAdapter(self.model, str(self.args.model_path))
        self.router_layer_to_model_block.clear()
        for router_layer_id, block_id in enumerate(adapter.encoder_sparse_layer_ids):
            if block_id >= len(encoder_layers):
                raise RuntimeError(f"NLLB sparse encoder layer {block_id} is outside encoder.layers")
            ffn = getattr(encoder_layers[block_id], "ffn", None)
            if not isinstance(ffn, NllbMoeSparseMLP):
                raise RuntimeError(f"NLLB encoder layer {block_id} ffn is not NllbMoeSparseMLP")
            classifier = getattr(getattr(ffn, "router", None), "classifier", None)
            if classifier is None:
                raise RuntimeError(f"NLLB encoder layer {block_id} ffn.router has no classifier")
            ffn._encoder_predictor_trace_layer_id = router_layer_id
            classifier._encoder_predictor_trace_layer_id = router_layer_id
            self.router_layer_to_model_block.append(int(block_id))
            self.handles.append(ffn.register_forward_pre_hook(self._nllb_sparse_mlp_pre_hook))
            self.handles.append(classifier.register_forward_hook(self._nllb_router_classifier_hook))
            self.handles.append(ffn.register_forward_hook(self._nllb_sparse_mlp_hook))
        if not self.router_layer_to_model_block:
            raise RuntimeError("found no encoder NllbMoeSparseMLP modules")

    def _layer0_attention_hook(self, module, inputs, output) -> None:
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(hidden, torch.Tensor) or hidden.dim() != 3:
            raise RuntimeError(f"layer0 attention hook expected [B,T,H], got {type(hidden).__name__}")
        self.layer0_attn_out = hidden.detach().cpu()

    def _sparse_mlp_hook(self, module, inputs, output) -> None:
        layer_id = int(module._encoder_predictor_trace_layer_id)
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
        expert_index = expert_index.to(torch.int64)
        router_probs = torch.softmax(router_logits.float(), dim=-1)
        expert_weights = torch.gather(router_probs, dim=-1, index=expert_index)
        expert_selection_mask = expert_weights > 0
        self.router_logits_by_layer[layer_id] = router_logits.detach().cpu()
        self.router_probs_by_layer[layer_id] = router_probs.detach().cpu()
        self.expert_selection_by_layer[layer_id] = expert_index.detach().cpu()
        self.expert_weights_by_layer[layer_id] = expert_weights.detach().cpu()
        self.expert_selection_mask_by_layer[layer_id] = expert_selection_mask.detach().cpu()


    def _nllb_sparse_mlp_pre_hook(self, module, inputs) -> None:
        layer_id = int(module._encoder_predictor_trace_layer_id)
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError("NLLB SparseMLP pre-hook expected hidden states in inputs[0]")
        hidden = inputs[0]
        if hidden.dim() != 3:
            raise RuntimeError(f"NLLB SparseMLP pre-hook expected hidden states [B,T,H], got {tuple(hidden.shape)}")
        self.nllb_layer_input_shape_by_layer[layer_id] = (int(hidden.shape[0]), int(hidden.shape[1]))

    def _nllb_router_classifier_hook(self, module, inputs, output) -> None:
        layer_id = int(module._encoder_predictor_trace_layer_id)
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError("NLLB router classifier hook expected hidden states in inputs[0]")
        hidden = inputs[0]
        if hidden.dim() == 3:
            bsz, seq_len = int(hidden.shape[0]), int(hidden.shape[1])
        elif hidden.dim() == 2:
            if layer_id not in self.nllb_layer_input_shape_by_layer:
                raise RuntimeError(
                    "NLLB router classifier hook received flattened hidden states before SparseMLP pre-hook recorded B,T"
                )
            bsz, seq_len = self.nllb_layer_input_shape_by_layer[layer_id]
            if int(hidden.shape[0]) != bsz * seq_len:
                raise RuntimeError(
                    f"NLLB router classifier hidden states first dimension {tuple(hidden.shape)} does not match B*T={bsz * seq_len}"
                )
        else:
            raise RuntimeError(
                f"NLLB router classifier hook expected hidden states [B,T,H] or [B*T,H], got {tuple(hidden.shape)}"
            )
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("NLLB router classifier hook expected tensor output")
        router_logits = output
        if router_logits.dim() == 2:
            if int(router_logits.shape[0]) != bsz * seq_len:
                raise RuntimeError(
                    f"NLLB router logits first dimension {tuple(router_logits.shape)} does not match B*T={bsz * seq_len}"
                )
            router_logits = router_logits.view(bsz, seq_len, -1)
        elif router_logits.dim() != 3:
            raise RuntimeError(f"NLLB router logits expected [B*T,E] or [B,T,E], got {tuple(router_logits.shape)}")
        self.router_logits_by_layer[layer_id] = router_logits.detach().cpu()

    def _nllb_sparse_mlp_hook(self, module, inputs, output) -> None:
        layer_id = int(module._encoder_predictor_trace_layer_id)
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError("NLLB SparseMLP hook expected hidden states in inputs[0]")
        hidden = inputs[0]
        if hidden.dim() != 3:
            raise RuntimeError(f"NLLB SparseMLP hook expected hidden states [B,T,H], got {tuple(hidden.shape)}")
        bsz, seq_len = int(hidden.shape[0]), int(hidden.shape[1])
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            raise RuntimeError("NLLB SparseMLP output must contain router tuple")
        router_tuple = output[1]
        if not isinstance(router_tuple, (tuple, list)) or len(router_tuple) < 1:
            raise RuntimeError("NLLB SparseMLP router tuple must contain router_probs")
        router_probs = router_tuple[0]
        if not isinstance(router_probs, torch.Tensor):
            raise RuntimeError("NLLB SparseMLP router_probs must be a tensor")
        if router_probs.dim() == 2:
            if int(router_probs.shape[0]) != bsz * seq_len:
                raise RuntimeError(
                    f"NLLB router_probs first dimension {tuple(router_probs.shape)} does not match B*T={bsz * seq_len}"
                )
            router_probs = router_probs.view(bsz, seq_len, -1)
        elif router_probs.dim() != 3:
            raise RuntimeError(f"NLLB router_probs expected [B*T,E] or [B,T,E], got {tuple(router_probs.shape)}")
        selection, weights, mask = routed_topk_from_router_probs(router_probs.float(), top_k=2)
        self.router_probs_by_layer[layer_id] = router_probs.detach().cpu()
        self.expert_selection_by_layer[layer_id] = selection.detach().cpu()
        self.expert_weights_by_layer[layer_id] = weights.detach().cpu()
        self.expert_selection_mask_by_layer[layer_id] = mask.detach().cpu()


    def run_split(self, prompts: list[str], *, batch_size: int, seq_id_start: int = 0) -> SplitAccumulator:
        if self.model is None:
            self.load()
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        acc = SplitAccumulator()
        total_batches = (len(prompts) + batch_size - 1) // batch_size
        for batch_idx, start in enumerate(range(0, len(prompts), batch_size), start=1):
            batch_prompts = prompts[start:start + batch_size]
            seq_ids = torch.arange(
                seq_id_start + start,
                seq_id_start + start + len(batch_prompts),
                dtype=torch.int64,
            )
            self.status(
                f"batch {batch_idx}/{total_batches}: prompts={len(batch_prompts)} "
                f"seq_id={int(seq_ids[0])}..{int(seq_ids[-1])}"
            )
            batch = self.run_batch(batch_prompts)
            acc.append(seq_ids=seq_ids, prompts=batch_prompts, **batch)
        return acc

    def run_batch(self, prompts: list[str]) -> dict[str, torch.Tensor]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("runner must be loaded before run_batch")
        self.layer0_attn_out = None
        self.router_logits_by_layer.clear()
        self.router_probs_by_layer.clear()
        self.expert_selection_by_layer.clear()
        self.expert_weights_by_layer.clear()
        self.expert_selection_mask_by_layer.clear()
        self.nllb_layer_input_shape_by_layer.clear()

        self.status("tokenizing batch")
        enc_inputs = self.tokenizer(
            prompts,
            padding=self.args.padding,
            truncation=True,
            max_length=self.args.max_input_tokens,
            return_tensors="pt",
        )
        input_ids = enc_inputs["input_ids"].to(self.input_device)
        attention_mask = enc_inputs["attention_mask"].to(self.input_device)
        self.status(f"tokenized batch: input_shape={tuple(input_ids.shape)}")

        self.status("starting model.generate")
        with torch.inference_mode():
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
                self.status("model.generate hit transformers one-token timing bug; continuing with hook traces")
        self.status("model.generate finished")

        if self.layer0_attn_out is None:
            raise RuntimeError("missing layer0 attention output; hook did not fire")
        num_layers = len(self.router_layer_to_model_block)
        missing_logits = [i for i in range(num_layers) if i not in self.router_logits_by_layer]
        missing_probs = [i for i in range(num_layers) if i not in self.router_probs_by_layer]
        missing_selection = [i for i in range(num_layers) if i not in self.expert_selection_by_layer]
        missing_weights = [i for i in range(num_layers) if i not in self.expert_weights_by_layer]
        missing_selection_mask = [i for i in range(num_layers) if i not in self.expert_selection_mask_by_layer]
        if missing_logits:
            raise RuntimeError(f"missing encoder router logits for layers {missing_logits}")
        if missing_probs:
            raise RuntimeError(f"missing encoder router probs for layers {missing_probs}")
        if missing_selection:
            raise RuntimeError(f"missing encoder expert selection for layers {missing_selection}")
        if missing_weights:
            raise RuntimeError(f"missing encoder expert weights for layers {missing_weights}")
        if missing_selection_mask:
            raise RuntimeError(f"missing encoder expert selection mask for layers {missing_selection_mask}")

        router_logits = torch.stack([self.router_logits_by_layer[i] for i in range(num_layers)], dim=1)
        router_probs = torch.stack([self.router_probs_by_layer[i] for i in range(num_layers)], dim=1)
        expert_selection = torch.stack([self.expert_selection_by_layer[i] for i in range(num_layers)], dim=1)
        expert_weights = torch.stack([self.expert_weights_by_layer[i] for i in range(num_layers)], dim=1)
        expert_selection_mask = torch.stack([self.expert_selection_mask_by_layer[i] for i in range(num_layers)], dim=1)
        self.status(
            f"batch trace tensors ready: layer0_attn_out={tuple(self.layer0_attn_out.shape)} "
            f"router_logits={tuple(router_logits.shape)} expert_selection={tuple(expert_selection.shape)}"
        )

        expected_bt = tuple(input_ids.shape)
        if tuple(self.layer0_attn_out.shape[:2]) != expected_bt:
            raise RuntimeError(
                f"layer0 attention shape {tuple(self.layer0_attn_out.shape)} does not align with input_ids {expected_bt}"
            )
        if tuple(router_logits.shape[:1] + router_logits.shape[2:3]) != expected_bt:
            raise RuntimeError(f"router logits shape {tuple(router_logits.shape)} does not align with input_ids {expected_bt}")
        if router_probs.shape != router_logits.shape:
            raise RuntimeError(f"router probs shape {tuple(router_probs.shape)} does not match router logits {tuple(router_logits.shape)}")
        if tuple(expert_selection.shape[:1] + expert_selection.shape[2:3]) != expected_bt:
            raise RuntimeError(
                f"expert selection shape {tuple(expert_selection.shape)} does not align with input_ids {expected_bt}"
            )
        if expert_weights.shape != expert_selection.shape:
            raise RuntimeError(
                f"expert weights shape {tuple(expert_weights.shape)} does not align with expert_selection {tuple(expert_selection.shape)}"
            )
        if expert_selection_mask.shape != expert_selection.shape:
            raise RuntimeError(
                "expert selection mask shape "
                f"{tuple(expert_selection_mask.shape)} does not align with expert_selection {tuple(expert_selection.shape)}"
            )

        return {
            "input_ids": input_ids.detach().cpu(),
            "attention_mask": attention_mask.detach().cpu(),
            "layer0_attn_out": self.layer0_attn_out,
            "router_logits": router_logits,
            "router_probs": router_probs,
            "expert_selection": expert_selection,
            "expert_weights": expert_weights,
            "expert_selection_mask": expert_selection_mask,
        }


def load_saved_tensor(path: Path) -> torch.Tensor:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _require_integer_tensor(split: str, name: str, tensor: torch.Tensor) -> None:
    if name == "expert_selection_mask.pt":
        if tensor.dtype is not torch.bool:
            raise AssertionError(f"{split}: {name} must be a bool tensor, got {tensor.dtype}")
        return
    if tensor.dtype is torch.bool or tensor.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
        raise AssertionError(f"{split}: {name} must be a non-bool integer tensor, got {tensor.dtype}")


def _require_floating_tensor(split: str, name: str, tensor: torch.Tensor) -> None:
    if not torch.is_floating_point(tensor):
        raise AssertionError(f"{split}: {name} must be a floating tensor, got {tensor.dtype}")


def verify_trace(output_dir: Path, metadata: dict) -> dict:
    required = [
        "attention_mask.pt",
        "expert_selection.pt",
        "expert_weights.pt",
        "input_ids.pt",
        "layer0_attn_out.pt",
        "prompt_texts.jsonl",
        "router_logits.pt",
        "router_probs.pt",
        "seq_ids.pt",
    ]
    result = {"ok": True, "splits": {}}
    model_config = metadata["model_config"]
    num_experts = int(model_config["num_experts"])
    top_k = int(model_config["num_selected_experts"])
    requires_selection_mask = _is_nllb_moe_model_type(model_config.get("model_type"))
    for split in ("train", "validation"):
        split_dir = Path(output_dir) / split
        for name in required:
            path = split_dir / name
            if not path.exists():
                raise AssertionError(f"missing {split}/{name}")
        mask_path = split_dir / "expert_selection_mask.pt"
        if requires_selection_mask and not mask_path.exists():
            raise AssertionError(f"missing {split}/expert_selection_mask.pt")

        input_ids = load_saved_tensor(split_dir / "input_ids.pt")
        attention_mask = load_saved_tensor(split_dir / "attention_mask.pt")
        router_logits = load_saved_tensor(split_dir / "router_logits.pt")
        router_probs = load_saved_tensor(split_dir / "router_probs.pt")
        expert_selection = load_saved_tensor(split_dir / "expert_selection.pt")
        expert_weights = load_saved_tensor(split_dir / "expert_weights.pt")
        expert_selection_mask = load_saved_tensor(mask_path) if mask_path.exists() else None
        layer0_attn_out = load_saved_tensor(split_dir / "layer0_attn_out.pt")
        seq_ids = load_saved_tensor(split_dir / "seq_ids.pt")

        for name, tensor in (
            ("input_ids.pt", input_ids),
            ("attention_mask.pt", attention_mask),
            ("expert_selection.pt", expert_selection),
            ("seq_ids.pt", seq_ids),
        ):
            _require_integer_tensor(split, name, tensor)
        for name, tensor in (
            ("layer0_attn_out.pt", layer0_attn_out),
            ("router_logits.pt", router_logits),
            ("router_probs.pt", router_probs),
            ("expert_weights.pt", expert_weights),
        ):
            _require_floating_tensor(split, name, tensor)

        if input_ids.shape != attention_mask.shape:
            raise AssertionError(f"{split}: input_ids shape does not match attention_mask")
        if layer0_attn_out.shape[:2] != input_ids.shape:
            raise AssertionError(f"{split}: layer0_attn_out shape does not align with input_ids")
        if router_logits.shape != router_probs.shape:
            raise AssertionError(f"{split}: router_logits shape does not match router_probs")
        if router_logits.shape[:3] != expert_selection.shape[:3]:
            raise AssertionError(f"{split}: router logits do not align with expert_selection")
        if expert_selection.shape != expert_weights.shape:
            raise AssertionError(f"{split}: expert_selection does not align with expert_weights")
        if expert_selection_mask is not None:
            if expert_selection_mask.dtype is not torch.bool:
                raise AssertionError(f"{split}: expert_selection_mask.pt must be a bool tensor, got {expert_selection_mask.dtype}")
            if expert_selection_mask.shape != expert_selection.shape:
                raise AssertionError(f"{split}: expert_selection_mask does not align with expert_selection")
            if torch.any(expert_selection[~expert_selection_mask] != 0):
                raise AssertionError(f"{split}: masked expert ids must be zero")
            if torch.any(expert_weights[~expert_selection_mask] != 0):
                raise AssertionError(f"{split}: masked expert weights must be zero")
            if torch.any(expert_weights[expert_selection_mask] <= 0):
                raise AssertionError(f"{split}: selected expert weights must be positive")
            weight_check_mask = expert_selection_mask
        else:
            weight_check_mask = torch.ones_like(expert_selection, dtype=torch.bool)
        expected_weights = torch.gather(router_probs.float(), dim=-1, index=expert_selection)
        if not torch.allclose(
            expected_weights[weight_check_mask],
            expert_weights.float()[weight_check_mask],
            rtol=1e-2,
            atol=1e-3,
        ):
            raise AssertionError(f"{split}: expert_weights do not align with router_probs and expert_selection")
        if requires_selection_mask:
            selected_probs = torch.zeros_like(router_probs, dtype=torch.bool)
            selected_probs.scatter_(-1, expert_selection, expert_selection_mask)
            nonzero_router_probs = router_probs.float() > 0
            if torch.any(nonzero_router_probs & ~selected_probs):
                raise AssertionError(f"{split}: NLLB router_probs must be sparse outside selected routed experts")
            routed_counts = nonzero_router_probs.sum(dim=-1)
            if torch.any(routed_counts > top_k):
                raise AssertionError(f"{split}: NLLB router_probs has more than top_k positive experts")
            if not torch.equal(expert_selection_mask, expert_weights.float() > 0):
                raise AssertionError(f"{split}: NLLB expert_selection_mask must match positive expert_weights")
        if expert_selection.shape[-1] != top_k:
            raise AssertionError(f"{split}: expert top_k mismatch")
        if expert_selection.numel() and (int(expert_selection.min()) < 0 or int(expert_selection.max()) >= num_experts):
            raise AssertionError(f"{split}: expert ids out of range")
        if router_logits.shape[-1] != num_experts:
            raise AssertionError(f"{split}: router num_experts mismatch")

        prompt_lines = (split_dir / "prompt_texts.jsonl").read_text(encoding="utf-8").splitlines()
        if len(prompt_lines) != int(input_ids.shape[0]):
            raise AssertionError(f"{split}: prompt line count mismatch")
        if seq_ids.shape[0] != input_ids.shape[0]:
            raise AssertionError(f"{split}: seq_ids length mismatch")

        result["splits"][split] = {
            "num_samples": int(input_ids.shape[0]),
            "seq_len": int(input_ids.shape[1]),
            "num_router_layers": int(router_logits.shape[1]),
            "num_experts": int(router_logits.shape[-1]),
        }

    gpu_assertion = metadata.get("gpu_expert_forward_assertion", {})
    if gpu_assertion.get("enabled") and int(gpu_assertion.get("observed_forwards", 0)) <= 0:
        raise AssertionError("GPU expert forward assertion was enabled but observed no expert forwards")
    return result


def _required_int_config(config, *names: str) -> int:
    for name in names:
        if hasattr(config, name):
            value = getattr(config, name)
            if value is not None:
                return int(value)
    joined = " or ".join(names)
    raise AttributeError(f"model config is missing required integer field: {joined}")


def _sparse_layer_count(config, stage: str) -> int:
    explicit_name = f"num_sparse_{stage}_layers"
    if hasattr(config, explicit_name) and getattr(config, explicit_name) is not None:
        return int(getattr(config, explicit_name))
    layers = _required_int_config(config, f"{stage}_layers")
    step = _required_int_config(config, f"{stage}_sparse_step")
    if step <= 0:
        raise ValueError(f"{stage}_sparse_step must be positive, got {step}")
    return len(range(step - 1, layers, step))


def _model_config_metadata(config) -> dict:
    return {
        "model_type": getattr(config, "model_type", None),
        "hidden_size": _required_int_config(config, "hidden_size", "d_model"),
        "num_experts": _required_int_config(config, "num_experts"),
        "num_selected_experts": selected_experts_for_config(config),
        "num_sparse_encoder_layers": _sparse_layer_count(config, "encoder"),
        "num_sparse_decoder_layers": _sparse_layer_count(config, "decoder"),
        "encoder_sparse_step": _required_int_config(config, "encoder_sparse_step"),
        "decoder_sparse_step": _required_int_config(config, "decoder_sparse_step"),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = REPO_ROOT
    args.model_path = Path(args.model_path)
    if args.train_prompt_file is not None:
        args.train_prompt_file = Path(args.train_prompt_file)
    if args.validation_prompt_file is not None:
        args.validation_prompt_file = Path(args.validation_prompt_file)
    if args.output_dir is not None:
        args.output_dir = Path(args.output_dir)

    train_prompt_file = args.train_prompt_file or resolve_prompt_file(
        repo_root, args.dataset, args.task_name, args.train_split
    )
    validation_prompt_file = args.validation_prompt_file or resolve_prompt_file(
        repo_root, args.dataset, args.task_name, args.validation_split
    )
    output_dir = args.output_dir or resolve_default_output_dir(repo_root, args.model_path, args.dataset, args.task_name)
    output_dir = Path(output_dir)
    model_config_for_dtype = load_transformers_config(args.model_path)
    validate_sparse_cache_encoder_trace_config(model_config_for_dtype, args.model_path)
    sparse_cache_config = build_sparse_cache_config(args, model_type=getattr(model_config_for_dtype, "model_type", None))
    resolved_model_torch_dtype = resolve_model_torch_dtype(args.model_torch_dtype, model_config_for_dtype)
    resolved_storage_dtype_for_config = resolve_storage_dtype(args.storage_dtype, model_config_for_dtype)

    resolved = {
        "repo_root": str(repo_root),
        "model_path": str(args.model_path),
        "train_prompt_file": str(train_prompt_file),
        "validation_prompt_file": str(validation_prompt_file),
        "output_dir": str(output_dir),
        "model_torch_dtype_request": args.model_torch_dtype,
        "model_torch_dtype": dtype_name(resolved_model_torch_dtype),
        "storage_dtype_request": args.storage_dtype,
        "storage_dtype": dtype_name(resolved_storage_dtype_for_config),
        "sparse_cache_config": json_safe(sparse_cache_config),
    }
    if args.dry_run:
        print(json.dumps(resolved, indent=2, ensure_ascii=False))
        return 0

    start = time.time()
    train_prompts = load_prompts(train_prompt_file)
    validation_prompts = load_prompts(validation_prompt_file)

    runner = SparseCacheTraceRunner(args)
    runner.load()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_storage_dtype = resolve_storage_dtype(args.storage_dtype, runner.model.config)
    writer = TraceWriter(
        output_dir,
        resolved_storage_dtype,
        require_explicit_router_probs=_is_nllb_moe_model_type(getattr(runner.model.config, "model_type", None)),
    )
    train_acc = runner.run_split(train_prompts, batch_size=args.batch_size, seq_id_start=0)
    validation_acc = runner.run_split(validation_prompts, batch_size=args.batch_size, seq_id_start=0)

    model_config = _model_config_metadata(runner.model.config)
    metadata = {
        "args": json_safe(vars(args)),
        "resolved": resolved,
        "sparse_cache_config": json_safe(sparse_cache_config),
        "device": str(args.device),
        "padding": args.padding,
        "batch_size": int(args.batch_size),
        "max_input_tokens": int(args.max_input_tokens),
        "max_new_tokens": int(args.max_new_tokens),
        "storage_dtype": dtype_name(resolved_storage_dtype),
        "storage_dtype_request": args.storage_dtype,
        "model_torch_dtype": dtype_name(runner.resolved_model_torch_dtype or resolved_model_torch_dtype),
        "model_torch_dtype_request": args.model_torch_dtype,
        "model_config": model_config,
        "router_layer_to_model_block": list(runner.router_layer_to_model_block),
        "router_layers": [
            {
                "encoder_sparse_layer_index": int(layer_id),
                "encoder_block_id": int(block_id),
                "global_sparse_layer_id": int(layer_id),
            }
            for layer_id, block_id in enumerate(runner.router_layer_to_model_block)
        ],
        "gpu_expert_forward_assertion": {
            "enabled": bool(args.assert_gpu_expert_forward),
            "observed_forwards": int(runner.gpu_asserter.observed) if runner.gpu_asserter else 0,
        },
        "splits": {
            "train": writer.write_split("train", train_acc),
            "validation": writer.write_split("validation", validation_acc),
        },
    }
    if args.verify:
        metadata["verification"] = verify_trace(output_dir, metadata)

    end = time.time()
    write_json(output_dir / "metadata.json", json_safe(metadata))
    write_json(
        output_dir / "run_log.json",
        json_safe(
            {
                "start_time": start,
                "end_time": end,
                "elapsed_seconds": end - start,
                "python": sys.executable,
                "cuda_available": torch.cuda.is_available(),
                "device": str(args.device),
                "cuda_device_name": torch.cuda.get_device_name(torch.device(args.device))
                if torch.cuda.is_available() and str(args.device).startswith("cuda")
                else None,
                "cache_config": sparse_cache_config,
                "output_dir": str(output_dir),
                "num_train_prompts": len(train_prompts),
                "num_validation_prompts": len(validation_prompts),
            }
        ),
    )
    print(f"wrote encoder predictor sparse cache trace to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

