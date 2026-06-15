#!/usr/bin/env python3
"""Export hot experts from experiment prompts.

By default this script is independent from project-specific predictor runtimes. It
loads a standard HuggingFace MoE model, hooks router modules, counts the experts
actually dispatched by the router mask after capacity/drop handling, and writes a
hot expert snapshot under the experiment trace layout. For very large models,
--load-backend sparse-cache patches Transformers through sparse_llm_cache before
loading and collects NLLB-MoE encoder/decoder router probabilities from sparse
MLP hooks while preserving the same hot expert payload schema.

Default input:
    experiment/datasets/<dataset>/<task>/<split>/prompt_list.pt

Default output:
    experiment/traces/<model>-<dataset>-<task>-<split>/hot_experts/<model>.<split>.json

Basic usage:
    python3 experiment/scripts/trace/export_hot_experts.py \
      --model-path experiment/models/google/switch-base-128 \
      --dataset mmlu \
      --task-name professional_law \
      --split test \
      --router-topk auto

By default, --torch-dtype auto uses config.torch_dtype. If that config dtype does not fit on the target GPU, override it explicitly:
    python3 experiment/scripts/trace/export_hot_experts.py \
      --model-path experiment/models/google/switch-base-128 \
      --dataset mmlu \
      --task-name professional_law \
      --split test \
      --router-topk auto \
      --torch-dtype float16

Use the project conda environment explicitly:
    /mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
      experiment/scripts/trace/export_hot_experts.py \
      --model-path experiment/models/google/switch-base-128 \
      --dataset mmlu \
      --task-name professional_law \
      --split test \
      --router-topk auto

For models that do not fit fully in GPU memory, let HuggingFace/Accelerate use one selected GPU and offload the rest:
    python3 experiment/scripts/trace/export_hot_experts.py \
      --model-path experiment/models/google/switch-base-256 \
      --dataset mmlu \
      --task-name professional_law \
      --split test \
      --router-topk auto \
      --torch-dtype float16 \
      --device-map auto \
      --device cuda:0 \
      --max-cpu-memory 256GiB \
      --offload-folder /tmp/promoe-hf-offload/switch-base-256

By default, --device-map auto only uses the GPU selected by --device. If automatic free-memory probing is too aggressive or too conservative, cap that GPU explicitly:
    --max-gpu-memory 70GiB

Multi-GPU sharding can be requested with --device-map-gpus all, but Switch router-logit export may fail when router tensors land on different GPUs.

Router top-k:
    --router-topk auto uses config.num_selected_experts when present.
    If that field is absent but config.second_expert_policy exists, auto uses 2.
    If neither field exists, auto fails and requires an explicit value such as
    --router-topk 1 or --router-topk 2.

Notes:
    - The script follows the small-demo input shape: dynamic padding only, with
      --enc-pad-to used as a truncation cap instead of fixed padding.
    - The script hooks modules whose names end with .router and counts nonzero
      entries in the returned dispatch mask, so dropped tokens are not counted.
    - The default HuggingFace backend does not use project-specific predictor
      checkpoints, scheduler, prefetch, GPU pool, or predictor runtime managers.
    - The sparse-cache backend uses sparse_llm_cache with no predictor
      prefetching and keeps the hot expert JSON schema unchanged.
    - Real export loads the model and should be run in the intended GPU
      environment.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def experiment_root(repo_root: Optional[Path] = None) -> Path:
    root = repo_root or globals()["repo_root"]()
    return root / "experiment"


def normalize_dataset_name(dataset_name: str) -> str:
    return dataset_name.strip().split("/")[-1]


def default_prompt_root(
    *,
    repo_root: Path,
    dataset_name: str,
    task_name: str,
    split: str,
) -> Path:
    return experiment_root(repo_root) / "datasets" / dataset_name / task_name / split


def default_hot_expert_output_path(
    *,
    repo_root: Path,
    model_name: str,
    dataset_name: str,
    task_name: str,
    split: str,
) -> Path:
    trace_dir = f"{model_name}-{dataset_name}-{task_name}-{split}"
    return (
        experiment_root(repo_root)
        / "traces"
        / trace_dir
        / "hot_experts"
        / f"{model_name}.{split}.json"
    )


def load_prompts(prompt_root: Path) -> List[str]:
    pt_path = prompt_root / "prompt_list.pt"
    txt_path = prompt_root / "prompt_list.txt"

    if pt_path.exists():
        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, list) or any(not isinstance(item, str) for item in payload):
            raise ValueError(f"Expected list[str] in {pt_path}")
        return payload

    if txt_path.exists():
        prompts: List[str] = []
        for line in txt_path.read_text(encoding="utf-8").splitlines():
            prompts.append(line.replace("\\n", "\n"))
        return prompts

    raise FileNotFoundError(
        f"Missing prompt_list.pt and prompt_list.txt under {prompt_root}"
    )


def resolve_router_topk(router_topk: str, config: Any) -> int:
    if router_topk != "auto":
        try:
            value = int(router_topk)
        except ValueError as exc:
            raise ValueError("--router-topk must be 'auto' or a positive integer") from exc
        if value <= 0:
            raise ValueError("--router-topk must be positive")
        return value

    selected = getattr(config, "num_selected_experts", None)
    if selected is not None:
        value = int(selected)
        if value <= 0:
            raise ValueError("config.num_selected_experts must be positive")
        return value

    if hasattr(config, "second_expert_policy"):
        return 2

    if getattr(config, "model_type", None) == "switch_transformers":
        return 1

    raise ValueError(
        "Cannot infer router_topk from config: missing num_selected_experts "
        "and second_expert_policy. Please pass --router-topk explicitly."
    )


def _infer_stage(router_key: str) -> str:
    if router_key.startswith("encoder."):
        return "encoder"
    if router_key.startswith("decoder."):
        return "decoder"
    raise ValueError(f"Cannot infer router stage from key: {router_key}")


def _sort_expert_counts(counts: Counter[int]) -> List[int]:
    return [eid for eid, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


class RouterHotExpertCollector:
    def __init__(self, router_topk: int):
        self.router_topk = int(router_topk)
        if self.router_topk <= 0:
            raise ValueError("router_topk must be positive")
        self._counts: Dict[str, Counter[int]] = {}
        self._stages: Dict[str, str] = {}

    def add_router_mask(self, router_key: str, router_mask: torch.Tensor) -> None:
        if router_mask.dim() not in (2, 3):
            raise ValueError(
                f"Expected router mask with shape [tokens, experts] or "
                f"[batch, tokens, experts] for {router_key}, got {tuple(router_mask.shape)}"
            )

        num_experts = int(router_mask.shape[-1])
        if self.router_topk > num_experts:
            raise ValueError(
                f"router_topk={self.router_topk} exceeds num_experts={num_experts} for {router_key}"
            )

        stage = _infer_stage(router_key)
        selected = torch.nonzero(router_mask.bool(), as_tuple=False)
        if selected.numel() == 0:
            self._stages[router_key] = stage
            return

        flat_ids = selected[:, -1].detach().cpu().tolist()
        counts = self._counts.setdefault(router_key, Counter())
        counts.update(int(eid) for eid in flat_ids)
        self._stages[router_key] = stage

    def frozen_hot_experts(self) -> Dict[str, List[int]]:
        return {
            router_key: _sort_expert_counts(counts)
            for router_key, counts in self._counts.items()
        }

    def usage_summary(self) -> Dict[str, Dict[str, Any]]:
        summary: Dict[str, Dict[str, Any]] = {}
        for router_key, counts in self._counts.items():
            ordered_eids = _sort_expert_counts(counts)
            summary[router_key] = {
                "token_total_hits": int(sum(counts.values())),
                "top_token_eids": [int(eid) for eid in ordered_eids],
                "token_hit_freq": {str(eid): int(counts[eid]) for eid in ordered_eids},
            }
        return summary

    def encoder_layers(self) -> Set[str]:
        return {router_key for router_key, stage in self._stages.items() if stage == "encoder"}

    def decoder_layers(self) -> Set[str]:
        return {router_key for router_key, stage in self._stages.items() if stage == "decoder"}


def build_hot_expert_payload(
    *,
    model_name: str,
    dataset_name: str,
    task_name: str,
    split: str,
    actual_samples: int,
    frozen: Dict[str, List[int]],
    usage_summary: Dict[str, Dict[str, Any]],
    encoder_layers: Set[str],
    decoder_layers: Set[str],
    router_topk: int,
    selection_rule: str,
    counting_unit: str,
    num_experts_per_layer: int = 0,
    manage_encoder_experts: bool = True,
    use_encoder_predictor: bool = False,
) -> Dict[str, Any]:
    compact: Dict[str, Dict[str, Any]] = {}
    kind_totals = {"encoder": 0, "decoder": 0}

    for router_key, info in usage_summary.items():
        if router_key not in encoder_layers and router_key not in decoder_layers:
            continue

        kind = "encoder" if router_key in encoder_layers else "decoder"
        token_hit_freq = info.get("token_hit_freq") or {}
        top_token_counts = []
        if isinstance(token_hit_freq, dict):
            for eid in info.get("top_token_eids", []) or []:
                eid_int = int(eid)
                top_token_counts.append(
                    {
                        "eid": eid_int,
                        "count": int(token_hit_freq.get(str(eid_int), 0)),
                    }
                )

        token_total_hits = int(info.get("token_total_hits", 0) or 0)
        kind_totals[kind] += token_total_hits
        compact[router_key] = {
            "kind": kind,
            "token_total_hits": token_total_hits,
            "top_token_eids": [int(eid) for eid in info.get("top_token_eids", []) or []],
            "top_token_counts": top_token_counts,
            "frozen_top_eids": [int(eid) for eid in frozen.get(router_key, [])],
        }

    return {
        "version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "dataset": dataset_name,
        "task_name": task_name,
        "split": split,
        "actual_samples": int(actual_samples),
        "source": f"{split}_prompt_list",
        "manage_encoder_experts": bool(manage_encoder_experts),
        "use_encoder_predictor": bool(use_encoder_predictor),
        "num_experts_per_layer": int(num_experts_per_layer or 0),
        "router_topk": int(router_topk),
        "selection_rule": selection_rule,
        "counting_unit": counting_unit,
        "frozen_hot_experts": {
            router_key: [int(eid) for eid in eids]
            for router_key, eids in frozen.items()
        },
        "token_hits_by_stage": kind_totals,
        "expert_usage_summary": compact,
    }


def save_hot_expert_snapshot(*, payload: Dict[str, Any], save_path: Path) -> Path:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return save_path


def _ensure_tokenizer_padding(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token_id", None) is not None:
        return
    if getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
        return
    raise ValueError("Tokenizer must provide pad_token_id or eos_token")


def _decoder_start_token_id(model: Any, tokenizer: Any) -> int:
    generation_config = getattr(model, "generation_config", None)
    model_config = getattr(model, "config", None)

    for owner in (generation_config, model_config):
        if owner is None:
            continue
        for attr in ("decoder_start_token_id", "bos_token_id"):
            value = getattr(owner, attr, None)
            if value is not None:
                return int(value)

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        return int(pad_token_id)

    raise RuntimeError("could not determine decoder_start_token_id for generation")


def _ensure_generation_decoder_start(model: Any, tokenizer: Any) -> int:
    decoder_start = _decoder_start_token_id(model, tokenizer)
    if hasattr(model, "generation_config"):
        model.generation_config.decoder_start_token_id = decoder_start
    return decoder_start


def _normalize_dtype_name(dtype_value: Any) -> str:
    if isinstance(dtype_value, torch.dtype):
        return str(dtype_value).replace("torch.", "")
    return str(dtype_value).strip().lower().replace("torch.", "")


def resolve_torch_dtype(dtype_name: str, config: Optional[Any] = None) -> torch.dtype:
    normalized = dtype_name.strip().lower()
    if normalized == "auto":
        if config is None or getattr(config, "torch_dtype", None) is None:
            raise ValueError(
                "Cannot infer torch dtype from config: missing torch_dtype. "
                "Please pass --torch-dtype explicitly."
            )
        normalized = _normalize_dtype_name(getattr(config, "torch_dtype"))

    if normalized in {"float32", "fp32"}:
        return torch.float32
    if normalized in {"float16", "fp16"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError("--torch-dtype must be one of: auto, float32, float16, bfloat16")


def normalize_device_map(device_map: Optional[str]) -> Optional[str]:
    if device_map is None:
        return None
    normalized = device_map.strip()
    if not normalized or normalized.lower() in {"none", "false", "off"}:
        return None
    return normalized


def _cuda_index_from_device(device: str) -> int:
    normalized = str(device).strip().lower()
    if normalized == "cuda":
        return 0
    if normalized.startswith("cuda:"):
        index_text = normalized.split(":", 1)[1]
        try:
            return int(index_text)
        except ValueError as exc:
            raise ValueError(f"Invalid CUDA device: {device!r}") from exc
    raise ValueError("--device-map-gpus=device requires --device to be cuda or cuda:<index>")


def _gpu_indices_for_device_map(device: str, device_map_gpus: str, gpu_count: int) -> List[int]:
    if gpu_count <= 0:
        return []
    normalized_scope = device_map_gpus.strip().lower()
    if normalized_scope == "all":
        return list(range(gpu_count))
    if normalized_scope != "device":
        raise ValueError("--device-map-gpus must be one of: device, all")

    index = _cuda_index_from_device(device)
    if index < 0 or index >= gpu_count:
        raise ValueError(f"CUDA device index {index} is outside visible range 0..{gpu_count - 1}")
    return [index]


def parse_max_gpu_memory(max_gpu_memory: Optional[str], gpu_count: int) -> Optional[Dict[int, str]]:
    if max_gpu_memory is None or not max_gpu_memory.strip():
        return None
    if gpu_count <= 0:
        raise ValueError("--max-gpu-memory requires at least one visible CUDA device")

    spec = max_gpu_memory.strip()
    parts = [part.strip() for part in spec.split(",") if part.strip()]
    if not parts:
        return None

    if len(parts) == 1 and ":" not in parts[0]:
        return {index: parts[0] for index in range(gpu_count)}

    parsed: Dict[int, str] = {}
    for part in parts:
        if ":" not in part:
            raise ValueError(
                "--max-gpu-memory must be a single value or comma-separated index:value pairs"
            )
        index_text, memory = [item.strip() for item in part.split(":", 1)]
        if not index_text or not memory:
            raise ValueError(f"Invalid --max-gpu-memory entry: {part!r}")
        try:
            index = int(index_text)
        except ValueError as exc:
            raise ValueError(f"Invalid GPU index in --max-gpu-memory: {index_text!r}") from exc
        if index < 0 or index >= gpu_count:
            raise ValueError(
                f"GPU index {index} in --max-gpu-memory is outside visible range 0..{gpu_count - 1}"
            )
        parsed[index] = memory
    return parsed


def build_max_memory(
    *,
    device_map: Optional[str],
    device: str,
    device_map_gpus: str,
    max_gpu_memory: Optional[str],
    max_cpu_memory: Optional[str],
    gpu_memory_reserve_mib: int,
) -> Optional[Dict[Any, str]]:
    if normalize_device_map(device_map) is None:
        return None

    max_memory: Dict[Any, str] = {}
    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_available else 0

    if max_gpu_memory is not None and max_gpu_memory.strip():
        gpu_memory_spec = max_gpu_memory.strip()
        if ":" not in gpu_memory_spec:
            for index in _gpu_indices_for_device_map(device, device_map_gpus, gpu_count):
                max_memory[index] = gpu_memory_spec
        else:
            parsed_gpu_memory = parse_max_gpu_memory(gpu_memory_spec, gpu_count)
            if parsed_gpu_memory is not None:
                max_memory.update(parsed_gpu_memory)
    elif cuda_available and normalize_device_map(device_map) != "cpu":
        reserve_mib = max(0, int(gpu_memory_reserve_mib))
        for index in _gpu_indices_for_device_map(device, device_map_gpus, gpu_count):
            free_bytes, _total_bytes = torch.cuda.mem_get_info(index)
            free_mib = int(free_bytes // (1024**2))
            usable_mib = max(0, free_mib - reserve_mib)
            if usable_mib > 0:
                max_memory[index] = f"{usable_mib}MiB"

    if max_cpu_memory is not None and max_cpu_memory.strip():
        max_memory["cpu"] = max_cpu_memory.strip()

    return max_memory or None


def load_hf_model(
    model_path: Path,
    device: str,
    dtype_name: str = "auto",
    device_map: Optional[str] = None,
    max_memory: Optional[Dict[Any, str]] = None,
    offload_folder: Optional[Path] = None,
) -> Tuple[Any, Any, Any]:
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    torch_dtype = resolve_torch_dtype(dtype_name, config)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    _ensure_tokenizer_padding(tokenizer)

    model_kwargs: Dict[str, Any] = {
        "config": config,
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
    }
    normalized_device_map = normalize_device_map(device_map)
    if normalized_device_map is not None:
        model_kwargs["device_map"] = normalized_device_map
        model_kwargs["offload_state_dict"] = True
        model_kwargs["low_cpu_mem_usage"] = True
        if max_memory is not None:
            model_kwargs["max_memory"] = max_memory
        if offload_folder is not None:
            offload_folder.mkdir(parents=True, exist_ok=True)
            model_kwargs["offload_folder"] = str(offload_folder)

    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path), **model_kwargs)
    _ensure_generation_decoder_start(model, tokenizer)
    if normalized_device_map is None:
        model.to(device)
    model.eval()
    return model, tokenizer, config


def _default_sparse_cache_rate(config: Any) -> float:
    if getattr(config, "model_type", None) in {"nllb-moe", "nllb_moe"}:
        return 0.01
    return 0.375


def _sparse_cache_config(
    *,
    model_path: Path,
    config: Any,
    cache_rate: Optional[float],
    cache_policy: str,
    per_layer_cache: bool,
) -> Dict[str, Any]:
    resolved_cache_rate = _default_sparse_cache_rate(config) if cache_rate is None else float(cache_rate)
    return {
        "model_id": str(model_path),
        "cache_rate": resolved_cache_rate,
        "cache_policy": cache_policy,
        "per_layer_cache": bool(per_layer_cache),
        "num_predict_expert_per_layer": 0,
        "reorder_experts": False,
        "early_preempt": False,
        "chunk_prefetch": False,
        "predict_input_mode": "no_predict",
    }


def load_sparse_cache_model(
    model_path: Path,
    device: str,
    dtype_name: str = "auto",
    cache_rate: Optional[float] = None,
    cache_policy: str = "lru",
    per_layer_cache: bool = False,
) -> Tuple[Any, Any, Any]:
    import sparse_llm_cache
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(device))

    config = AutoConfig.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    torch_dtype = resolve_torch_dtype(dtype_name, config)
    sparse_config = _sparse_cache_config(
        model_path=model_path,
        config=config,
        cache_rate=cache_rate,
        cache_policy=cache_policy,
        per_layer_cache=per_layer_cache,
    )
    sparse_llm_cache.utils.hack_transformers(
        **sparse_config,
        pin_memory=True,
        enable_model_timer=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    _ensure_tokenizer_padding(tokenizer)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(model_path),
        config=config,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        local_files_only=True,
        device_map=0,
    )
    _ensure_generation_decoder_start(model, tokenizer)
    model.eval()
    return model, tokenizer, config


def _num_experts(config: Any) -> Optional[int]:
    value = getattr(config, "num_experts", None)
    if value is None:
        return None
    return int(value)


def _looks_like_router_module(module_name: str) -> bool:
    return module_name.lower().endswith(".router")


def _stage_router_key(module_name: str) -> str:
    if module_name.startswith(("encoder.", "decoder.")):
        return module_name

    for stage in ("encoder", "decoder"):
        marker = f".{stage}."
        marker_index = module_name.find(marker)
        if marker_index >= 0:
            return module_name[marker_index + 1 :]

    return module_name


def _iter_tensors(value: Any) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensors(item)


def extract_router_mask(
    output: Any,
    num_experts: Optional[int] = None,
    model_type: Optional[str] = None,
) -> Optional[torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return None

    if model_type == "nllb-moe" and isinstance(output, (tuple, list)):
        if len(output) < 2:
            return None
        router_probs = output[1]
        if (
            isinstance(router_probs, torch.Tensor)
            and router_probs.dim() >= 2
            and (num_experts is None or int(router_probs.shape[-1]) == num_experts)
        ):
            return router_probs
        return None

    for tensor in _iter_tensors(output):
        if tensor.dim() >= 2 and (num_experts is None or int(tensor.shape[-1]) == num_experts):
            return tensor
    return None


def install_router_hooks(
    *,
    model: Any,
    config: Any,
    collector: RouterHotExpertCollector,
) -> List[Any]:
    expected_num_experts = _num_experts(config)
    if expected_num_experts is None:
        raise ValueError("Model config must define num_experts for router mask collection")
    model_type = getattr(config, "model_type", None)

    handles: List[Any] = []

    for module_name, module in model.named_modules():
        if not module_name or not _looks_like_router_module(module_name):
            continue

        router_key = _stage_router_key(module_name)

        def hook(_module: Any, _inputs: Tuple[Any, ...], output: Any, key: str = router_key) -> None:
            router_mask = extract_router_mask(output, expected_num_experts, model_type=model_type)
            if router_mask is not None:
                collector.add_router_mask(key, router_mask)

        handles.append(module.register_forward_hook(hook))

    if not handles:
        raise RuntimeError(
            "No router modules were found. Expected module names ending with '.router'."
        )
    return handles


_NLLB_SPARSE_MLP_PATTERN = re.compile(r"(?:^|\.)(encoder|decoder)\.layers\.(\d+)\.ffn$")


def _nllb_sparse_mlp_router_key(module_name: str, module: Any) -> Optional[str]:
    match = _NLLB_SPARSE_MLP_PATTERN.search(str(module_name))
    if match is None:
        return None
    stage = str(getattr(module, "_stage", match.group(1)))
    if stage not in {"encoder", "decoder"}:
        return None
    if stage != match.group(1):
        return None
    block_id = int(match.group(2))
    return f"{stage}.layers.{block_id}.ffn.router"


def extract_nllb_sparse_mlp_router_probs(
    output: Any,
    num_experts: Optional[int] = None,
) -> Optional[torch.Tensor]:
    if not isinstance(output, (tuple, list)) or len(output) < 2:
        return None
    router_tuple = output[1]
    if not isinstance(router_tuple, (tuple, list)) or not router_tuple:
        return None
    router_probs = router_tuple[0]
    if not isinstance(router_probs, torch.Tensor):
        return None
    if router_probs.dim() not in (2, 3):
        return None
    if num_experts is not None and int(router_probs.shape[-1]) != int(num_experts):
        return None
    return router_probs


def install_sparse_cache_nllb_router_hooks(
    *,
    model: Any,
    config: Any,
    collector: RouterHotExpertCollector,
) -> List[Any]:
    if getattr(config, "model_type", None) not in {"nllb-moe", "nllb_moe"}:
        raise ValueError("sparse-cache NLLB router hooks require model_type='nllb-moe'")
    expected_num_experts = _num_experts(config)
    if expected_num_experts is None:
        raise ValueError("Model config must define num_experts for router mask collection")

    handles: List[Any] = []
    for module_name, module in model.named_modules():
        router_key = _nllb_sparse_mlp_router_key(module_name, module)
        if router_key is None or not hasattr(module, "register_forward_hook"):
            continue

        def hook(_module: Any, _inputs: Tuple[Any, ...], output: Any, key: str = router_key) -> None:
            router_probs = extract_nllb_sparse_mlp_router_probs(output, expected_num_experts)
            if router_probs is not None:
                collector.add_router_mask(key, router_probs)

        handles.append(module.register_forward_hook(hook))

    if not handles:
        raise RuntimeError("No NLLB sparse MLP modules were found for sparse-cache router collection.")
    return handles


def _generation_kwargs(model: Any, tokenizer: Any, max_new_tokens: int) -> Dict[str, Any]:
    kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "decoder_start_token_id": _decoder_start_token_id(model, tokenizer),
    }
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id
    return kwargs


def collect_hot_experts(
    *,
    model: Any,
    tokenizer: Any,
    config: Any,
    prompts: Sequence[str],
    device: str,
    router_topk: int,
    max_new_tokens: int,
    enc_pad_to: int,
    max_samples: int,
    hook_backend: str = "hf",
) -> Tuple[int, Dict[str, List[int]], Dict[str, Dict[str, Any]], Set[str], Set[str], int]:
    collector = RouterHotExpertCollector(router_topk=router_topk)
    if hook_backend == "sparse-cache" and getattr(config, "model_type", None) in {"nllb-moe", "nllb_moe"}:
        handles = install_sparse_cache_nllb_router_hooks(model=model, config=config, collector=collector)
    else:
        handles = install_router_hooks(model=model, config=config, collector=collector)

    actual_samples = 0
    try:
        for prompt in prompts:
            if max_samples > 0 and actual_samples >= max_samples:
                break
            encoded = tokenizer(
                [prompt],
                truncation=True,
                padding=True,
                max_length=enc_pad_to,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                generation_kwargs = _generation_kwargs(model, tokenizer, max_new_tokens)
                if hook_backend == "sparse-cache":
                    model.generate(**encoded, **generation_kwargs)
                else:
                    try:
                        model.generate(
                            **encoded,
                            output_router_logits=True,
                            **generation_kwargs,
                        )
                    except ValueError as exc:
                        if "output_router_logits" not in str(exc):
                            raise
                        model.generate(**encoded, **generation_kwargs)
            actual_samples += 1
    finally:
        for handle in handles:
            handle.remove()

    usage_summary = collector.usage_summary()
    if not usage_summary:
        raise RuntimeError("No router dispatch masks were collected; refusing to write an empty hot expert snapshot.")
    frozen = collector.frozen_hot_experts()
    encoder_layers = collector.encoder_layers()
    decoder_layers = collector.decoder_layers()
    num_experts_per_layer = int(getattr(config, "num_experts", 0) or 0)
    return actual_samples, frozen, usage_summary, encoder_layers, decoder_layers, num_experts_per_layer


def bool_arg(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export hot experts from experiment prompt lists into experiment/traces."
    )
    parser.add_argument("--model-path", type=Path, required=True, help="Path to the local model directory")
    parser.add_argument("--model-name", type=str, default=None, help="Override model name; defaults to model-path basename")
    parser.add_argument("--dataset", type=str, default="mmlu", help="Experiment dataset name")
    parser.add_argument("--task-name", type=str, default="professional_law", help="Experiment task name")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to export")
    parser.add_argument("--prompt-root", type=Path, default=None, help="Override prompt directory")
    parser.add_argument("--output-path", type=Path, default=None, help="Override JSON output path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device for inputs")
    parser.add_argument(
        "--load-backend",
        type=str,
        default="hf",
        choices=["hf", "sparse-cache"],
        help="Model loading backend: hf keeps the standard HuggingFace path; sparse-cache patches Transformers first",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default=None,
        help="Optional HuggingFace device_map such as auto, balanced, balanced_low_0, sequential, or cpu",
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        default=None,
        help="GPU memory cap for device_map; use one value for selected scope, e.g. 70GiB, or index:value pairs",
    )
    parser.add_argument(
        "--device-map-gpus",
        type=str,
        default="device",
        choices=["device", "all"],
        help="GPU scope for automatic max_memory: selected --device only, or all visible GPUs",
    )
    parser.add_argument(
        "--max-cpu-memory",
        type=str,
        default=None,
        help="CPU memory cap passed to HuggingFace max_memory, e.g. 256GiB",
    )
    parser.add_argument(
        "--gpu-memory-reserve-mib",
        type=int,
        default=1024,
        help="MiB kept free on each GPU when --device-map is set and --max-gpu-memory is omitted",
    )
    parser.add_argument(
        "--offload-folder",
        type=Path,
        default=None,
        help="Optional folder for HuggingFace disk offload when CPU RAM is insufficient",
    )
    parser.add_argument(
        "--cache-rate",
        type=float,
        default=None,
        help="Sparse-cache cache_rate; defaults to 0.01 for NLLB-MoE and 0.375 otherwise",
    )
    parser.add_argument(
        "--cache-policy",
        type=str,
        default="lru",
        help="Sparse-cache cache policy used with --load-backend sparse-cache",
    )
    parser.add_argument(
        "--per-layer-cache",
        type=bool_arg,
        default=False,
        help="Whether sparse-cache uses per-layer cache partitioning",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Generation length per prompt")
    parser.add_argument("--enc-pad-to", type=int, default=512, help="Encoder-side truncation cap; dynamic padding follows small-demo")
    parser.add_argument("--max-samples", type=int, default=-1, help="Max prompts to process; <=0 means all")
    parser.add_argument(
        "--router-topk",
        type=str,
        default="auto",
        help="Router selections to count per token: auto or a positive integer",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "fp32", "float16", "fp16", "bfloat16", "bf16"],
        help="Dtype used when loading the HuggingFace model; auto reads config.torch_dtype",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root_path = repo_root()
    dataset_name = normalize_dataset_name(args.dataset)
    model_name = args.model_name.strip() if isinstance(args.model_name, str) and args.model_name.strip() else args.model_path.name
    prompt_root = args.prompt_root or default_prompt_root(
        repo_root=repo_root_path,
        dataset_name=dataset_name,
        task_name=args.task_name,
        split=args.split,
    )
    output_path = args.output_path or default_hot_expert_output_path(
        repo_root=repo_root_path,
        model_name=model_name,
        dataset_name=dataset_name,
        task_name=args.task_name,
        split=args.split,
    )

    prompts = load_prompts(prompt_root)
    device_map = normalize_device_map(args.device_map)
    max_memory = build_max_memory(
        device_map=device_map,
        device=args.device,
        device_map_gpus=args.device_map_gpus,
        max_gpu_memory=args.max_gpu_memory,
        max_cpu_memory=args.max_cpu_memory,
        gpu_memory_reserve_mib=int(args.gpu_memory_reserve_mib),
    )
    if args.load_backend == "sparse-cache":
        model, tokenizer, config = load_sparse_cache_model(
            args.model_path.resolve(),
            args.device,
            dtype_name=args.torch_dtype,
            cache_rate=args.cache_rate,
            cache_policy=args.cache_policy,
            per_layer_cache=args.per_layer_cache,
        )
    else:
        model, tokenizer, config = load_hf_model(
            args.model_path.resolve(),
            args.device,
            dtype_name=args.torch_dtype,
            device_map=device_map,
            max_memory=max_memory,
            offload_folder=args.offload_folder,
        )
    router_topk = resolve_router_topk(args.router_topk, config)

    actual_samples, frozen, usage_summary, encoder_layers, decoder_layers, num_experts_per_layer = collect_hot_experts(
        model=model,
        tokenizer=tokenizer,
        config=config,
        prompts=prompts,
        device=args.device,
        router_topk=router_topk,
        max_new_tokens=int(args.max_new_tokens),
        enc_pad_to=int(args.enc_pad_to),
        max_samples=int(args.max_samples),
        hook_backend=args.load_backend,
    )

    payload = build_hot_expert_payload(
        model_name=model_name,
        dataset_name=dataset_name,
        task_name=args.task_name,
        split=args.split,
        actual_samples=actual_samples,
        frozen=frozen,
        usage_summary=usage_summary,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        num_experts_per_layer=num_experts_per_layer,
        manage_encoder_experts=True,
        use_encoder_predictor=False,
        router_topk=router_topk,
        selection_rule="router_mask.nonzero",
        counting_unit="token_dispatched_expert",
    )

    written_path = save_hot_expert_snapshot(payload=payload, save_path=output_path)
    print(f"saved hot experts to {written_path}")


if __name__ == "__main__":
    main()
