#!/usr/bin/env python3
"""Run the existing ERPP encoder trace exporter under the experiment layout.

### Deprecated: use trace/encoder_predictor 's files

Usage:
    /mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
      experiment/scripts/trace/run_encoder_erpp_trace.py \
      --model-path experiment/models/google/switch-base-128 \
      --dataset mmlu \
      --task-name professional_law \
      --device cuda:0 \
      --model-device-map single-auto

Dry run without loading the model:
    /mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
      experiment/scripts/trace/run_encoder_erpp_trace.py \
      --model-path experiment/models/google/switch-base-128 \
      --dataset mmlu \
      --task-name professional_law \
      --dry-run

Default input:
    experiment/datasets/<dataset>/<task>/test/prompt_list.txt|pt
    experiment/datasets/<dataset>/<task>/validation/prompt_list.txt|pt

Default output:
    experiment/traces/<model>-<dataset>-<task>/encoder_erpp_trace/

This wrapper keeps the original ERPP trace format intact and only fixes the
experiment defaults:
  - prompts come from experiment/datasets/<dataset>/<task>/<split>/
  - output goes to experiment/traces/<model>-<dataset>-<task>/encoder_erpp_trace/
  - small-demo input shape is the default: padding=longest, batch_size=1,
    max_input_tokens=512
  - storage dtype defaults to auto, resolved from model config torch_dtype
  - model device map defaults to single-auto: one GPU from --device plus CPU offload
  - model load dtype defaults to auto and follows model config precision

Training code must use attention_mask.pt to exclude padded storage positions from
loss computation.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, List, Optional

import torch


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def normalize_dataset_name(dataset_name: str) -> str:
    return dataset_name.strip().split("/")[-1]


def default_model_name(model_path: Path) -> str:
    return Path(model_path).name


def default_prompt_split_dir(*, repo_root: Path, dataset_name: str, task_name: str, split: str) -> Path:
    return repo_root / "experiment" / "datasets" / dataset_name / task_name / split


def default_output_dir(*, repo_root: Path, model_name: str, dataset_name: str, task_name: str) -> Path:
    return repo_root / "experiment" / "traces" / f"{model_name}-{dataset_name}-{task_name}" / "encoder_erpp_trace"


def default_erpp_exporter_path(repo_root: Path) -> Path:
    return repo_root / "performance_predictor" / "encoder" / "ERPP" / "implement" / "trace" / "export_erpp_encoder_trace.py"


def resolve_prompt_file(split_dir: Path) -> Optional[Path]:
    txt_path = split_dir / "prompt_list.txt"
    if txt_path.is_file():
        return txt_path
    pt_path = split_dir / "prompt_list.pt"
    if pt_path.is_file():
        return pt_path
    return None


def _load_pt_prompts(path: Path) -> List[str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, list) or any(not isinstance(item, str) for item in payload):
        raise ValueError(f"Expected list[str] in {path}")
    return payload


def materialize_prompt_txt(split_dir: Path, output_dir: Path, *, split: str) -> Path:
    prompt_file = resolve_prompt_file(split_dir)
    if prompt_file is None:
        raise FileNotFoundError(f"Missing prompt_list.txt or prompt_list.pt under {split_dir}")
    if prompt_file.suffix == ".txt":
        return prompt_file

    prompts = _load_pt_prompts(prompt_file)
    inputs_dir = output_dir / "_inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    out_path = inputs_dir / f"{split}_prompt_list.txt"
    with out_path.open("w", encoding="utf-8") as f:
        for prompt in prompts:
            f.write(prompt.replace("\n", "\\n") + "\n")
    return out_path


_SUPPORTED_STORAGE_DTYPES = {"float32", "float16", "bfloat16"}
_DTYPE_ALIASES = {
    "float": "float32",
    "fp32": "float32",
    "float32": "float32",
    "fp16": "float16",
    "float16": "float16",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
}


def _normalize_dtype_name(value: Any) -> str:
    return str(value).strip().lower().replace("torch.", "")


def resolve_storage_dtype(storage_dtype: str, model_path: Path) -> str:
    requested = _normalize_dtype_name(storage_dtype)
    if requested != "auto":
        resolved = _DTYPE_ALIASES.get(requested)
        if resolved is None or resolved not in _SUPPORTED_STORAGE_DTYPES:
            raise ValueError("--storage-dtype must be auto, float32, float16, or bfloat16")
        return resolved

    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Cannot resolve --storage-dtype auto: missing {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    configured = config.get("torch_dtype")
    if configured is None:
        return "float32"
    resolved = _DTYPE_ALIASES.get(_normalize_dtype_name(configured))
    if resolved is None or resolved not in _SUPPORTED_STORAGE_DTYPES:
        raise ValueError(f"Cannot map config.torch_dtype={configured!r} to ERPP storage dtype")
    return resolved


def build_export_command(
    *,
    python_executable: Path,
    exporter_path: Path,
    model_path: Path,
    train_prompt_file: Path,
    validation_prompt_file: Path,
    output_dir: Path,
    max_input_tokens: int,
    batch_size: int,
    device: str,
    seed: int,
    padding: str,
    storage_dtype: str,
    model_torch_dtype: str,
    model_device_map: str,
    gpu_memory_gb: Optional[float],
    cpu_memory_gb: Optional[float],
    verify: bool,
    print_status: bool,
) -> List[str]:
    command = [
        str(python_executable),
        str(exporter_path),
        "--model-path",
        str(model_path),
        "--train-prompt-file",
        str(train_prompt_file),
        "--validation-prompt-file",
        str(validation_prompt_file),
        "--output-dir",
        str(output_dir),
        "--max-input-tokens",
        str(int(max_input_tokens)),
        "--batch-size",
        str(int(batch_size)),
        "--device",
        device,
        "--seed",
        str(int(seed)),
        "--padding",
        padding,
        "--storage-dtype",
        storage_dtype,
        "--model-torch-dtype",
        model_torch_dtype,
        "--model-device-map",
        model_device_map,
    ]
    if gpu_memory_gb is not None:
        command.extend(["--gpu-memory-gb", f"{float(gpu_memory_gb):g}"])
    if cpu_memory_gb is not None:
        command.extend(["--cpu-memory-gb", f"{float(cpu_memory_gb):g}"])
    if verify:
        command.append("--verify")
    if print_status:
        command.append("--print-status")
    return command


def annotate_metadata(
    *,
    output_dir: Path,
    dataset_name: str,
    task_name: str,
    train_split: str,
    validation_split: str,
    wrapper_padding: str,
    wrapper_batch_size: int,
    wrapper_max_input_tokens: int,
    wrapper_storage_dtype: str,
    wrapper_storage_dtype_arg: str,
    wrapper_model_torch_dtype: str,
    wrapper_model_device_map: str,
    wrapper_gpu_memory_gb: Optional[float],
    wrapper_cpu_memory_gb: Optional[float],
) -> None:
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"ERPP exporter did not write metadata.json under {output_dir}")
    metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["experiment_wrapper"] = {
        "script": "experiment/scripts/trace/run_encoder_erpp_trace.py",
        "dataset": dataset_name,
        "task_name": task_name,
        "train_split": train_split,
        "validation_split": validation_split,
        "small_demo_input口径": wrapper_padding == "longest" and int(wrapper_batch_size) == 1,
        "padding": wrapper_padding,
        "batch_size": int(wrapper_batch_size),
        "max_input_tokens": int(wrapper_max_input_tokens),
        "storage_dtype_arg": wrapper_storage_dtype_arg,
        "resolved_storage_dtype": wrapper_storage_dtype,
        "storage_dtype_note": "auto is resolved from model config torch_dtype before calling the ERPP exporter.",
        "model_torch_dtype": wrapper_model_torch_dtype,
        "model_torch_dtype_note": "Controls model loading dtype. Use float16/bfloat16 when fp32 does not fit GPU memory.",
        "model_device_map": wrapper_model_device_map,
        "gpu_memory_gb": wrapper_gpu_memory_gb,
        "cpu_memory_gb": wrapper_cpu_memory_gb,
        "model_device_map_note": "single-auto uses only --device GPU plus CPU offload; auto may use all visible devices; single moves the full model to --device.",
        "note": "Trace files keep the original ERPP schema. Padding is not removed from tensors; training must mask it out.",
    }
    metadata["training_notes"] = {
        "must_filter_padding_with_attention_mask": True,
        "valid_token_rule": "valid_tokens = attention_mask.bool(); broadcast to [S, L, T] before computing loss",
        "reason": "router_logits/expert_selection/router_probs contain storage padding positions when a split is tensorized; attention_mask.pt marks real tokens.",
        "drop_handling": "No extra drop filtering is applied by this wrapper; expert_selection.pt keeps the original ERPP exporter semantics.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Run ERPP encoder trace export with experiment defaults.")
    parser.add_argument("--model-path", type=Path, default=root / "experiment" / "models" / "google" / "switch-base-128")
    parser.add_argument("--model-name", type=str, default=None, help="Defaults to --model-path basename")
    parser.add_argument("--dataset", type=str, default="mmlu")
    parser.add_argument("--task-name", type=str, default="professional_law")
    parser.add_argument("--train-split", type=str, default="test")
    parser.add_argument("--validation-split", type=str, default="validation")
    parser.add_argument("--train-prompt-dir", type=Path, default=None)
    parser.add_argument("--validation-prompt-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--exporter-path", type=Path, default=default_erpp_exporter_path(root))
    parser.add_argument("--python", dest="python_executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--padding", choices=["longest", "max_length"], default="longest")
    parser.add_argument("--storage-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto", help="Trace storage dtype; auto resolves model config torch_dtype")
    parser.add_argument("--model-torch-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto", help="Dtype used to load model weights; use float16/bfloat16 to reduce GPU memory")
    parser.add_argument("--model-device-map", choices=["single-auto", "auto", "single"], default="single-auto", help="single-auto uses --device GPU plus CPU offload; auto may use all visible devices; single moves the whole model to --device")
    parser.add_argument("--gpu-memory-gb", type=float, default=None, help="GPU memory budget for single-auto; default uses most of --device GPU with a small reserve")
    parser.add_argument("--cpu-memory-gb", type=float, default=None, help="CPU memory budget for single-auto; default 128GiB")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--print-status", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without running it")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    raise NotImplementedError("Deprecated: use trace/encoder_predictor 's files")
    args = parse_args(argv)
    root = repo_root()
    dataset_name = normalize_dataset_name(args.dataset)
    model_name = args.model_name.strip() if isinstance(args.model_name, str) and args.model_name.strip() else default_model_name(args.model_path)
    output_dir = args.output_dir or default_output_dir(
        repo_root=root,
        model_name=model_name,
        dataset_name=dataset_name,
        task_name=args.task_name,
    )
    train_dir = args.train_prompt_dir or default_prompt_split_dir(
        repo_root=root,
        dataset_name=dataset_name,
        task_name=args.task_name,
        split=args.train_split,
    )
    validation_dir = args.validation_prompt_dir or default_prompt_split_dir(
        repo_root=root,
        dataset_name=dataset_name,
        task_name=args.task_name,
        split=args.validation_split,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    train_prompt_file = materialize_prompt_txt(train_dir, output_dir, split=args.train_split)
    validation_prompt_file = materialize_prompt_txt(validation_dir, output_dir, split=args.validation_split)
    resolved_storage_dtype = resolve_storage_dtype(args.storage_dtype, args.model_path)
    command = build_export_command(
        python_executable=args.python_executable,
        exporter_path=args.exporter_path,
        model_path=args.model_path,
        train_prompt_file=train_prompt_file,
        validation_prompt_file=validation_prompt_file,
        output_dir=output_dir,
        max_input_tokens=args.max_input_tokens,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        padding=args.padding,
        storage_dtype=resolved_storage_dtype,
        model_torch_dtype=args.model_torch_dtype,
        model_device_map=args.model_device_map,
        gpu_memory_gb=args.gpu_memory_gb,
        cpu_memory_gb=args.cpu_memory_gb,
        verify=not args.no_verify,
        print_status=args.print_status,
    )
    print(" ".join(command), flush=True)
    if args.dry_run:
        return 0
    subprocess.run(command, check=True)
    annotate_metadata(
        output_dir=output_dir,
        dataset_name=dataset_name,
        task_name=args.task_name,
        train_split=args.train_split,
        validation_split=args.validation_split,
        wrapper_padding=args.padding,
        wrapper_batch_size=args.batch_size,
        wrapper_max_input_tokens=args.max_input_tokens,
        wrapper_storage_dtype=resolved_storage_dtype,
        wrapper_storage_dtype_arg=args.storage_dtype,
        wrapper_model_torch_dtype=args.model_torch_dtype,
        wrapper_model_device_map=args.model_device_map,
        wrapper_gpu_memory_gb=args.gpu_memory_gb,
        wrapper_cpu_memory_gb=args.cpu_memory_gb,
    )
    print(f"ERPP encoder trace written to {output_dir}", flush=True)
    print("Training reminder: use attention_mask.pt to filter padding tokens before computing loss.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
