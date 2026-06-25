#!/usr/bin/env python3
"""Check whether one sparse cache ratio fits a requested GPU memory budget."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Sequence


@dataclass
class CompletedRun:
    returncode: int
    stdout: str
    stderr: str


@dataclass
class CheckResult:
    can_run: bool
    failure_kind: Optional[str]
    model_path: str
    gpu_mem_gb: float
    cache_rate: float
    returncode: int
    elapsed_seconds: float
    stdout_tail: str
    stderr_tail: str
    temp_dir: Optional[Path] = None
    output_dir: Optional[Path] = None

    def to_json_dict(self) -> dict:
        payload = asdict(self)
        if self.temp_dir is not None:
            payload["temp_dir"] = str(self.temp_dir)
        if self.output_dir is not None:
            payload["output_dir"] = str(self.output_dir)
        return payload


def find_repo_root(start: Optional[Path] = None) -> Path:
    current = Path.cwd() if start is None else Path(start).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "experiment").is_dir() and (candidate / "src").is_dir():
            return candidate
    return Path(__file__).resolve().parents[4]


def default_exporter_path(repo_root: Path) -> Path:
    return repo_root / "experiment" / "scripts" / "trace" / "encoder_predictor" / "export_sparse_cache_encoder_trace.py"


def default_result_dir(repo_root: Path) -> Path:
    return repo_root / "experiment" / "scripts" / "trace" / "cache_ratio" / "result"


def derive_model_name(model_path: str | Path) -> str:
    return Path(model_path).name or "model"


def _safe_filename(value: str) -> str:
    safe = []
    for char in value:
        if char.isalnum() or char in {"-", "_", "."}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("._") or "model"


def save_result_record(
    result: CheckResult,
    *,
    result_dir: Path,
    device: str,
    dataset: str,
    task_name: str,
    model_torch_dtype: str,
    storage_dtype: str,
    batch_size: int,
    max_input_tokens: int,
    max_new_tokens: int,
    cache_policy: str,
    per_layer_cache: bool,
) -> dict:
    result_dir = Path(result_dir)
    by_model_dir = result_dir / "by_model"
    by_model_dir.mkdir(parents=True, exist_ok=True)

    model_name = derive_model_name(result.model_path)
    record = {
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "can_run": bool(result.can_run),
        "failure_kind": result.failure_kind,
        "model_path": result.model_path,
        "model_name": model_name,
        "gpu_mem_gb": float(result.gpu_mem_gb),
        "cache_rate": float(result.cache_rate),
        "device": device,
        "dataset": dataset,
        "task_name": task_name,
        "model_torch_dtype": model_torch_dtype,
        "storage_dtype": storage_dtype,
        "batch_size": int(batch_size),
        "max_input_tokens": int(max_input_tokens),
        "max_new_tokens": int(max_new_tokens),
        "cache_policy": cache_policy,
        "per_layer_cache": bool(per_layer_cache),
        "returncode": int(result.returncode),
        "elapsed_seconds": float(result.elapsed_seconds),
        "stdout_tail": result.stdout_tail,
        "stderr_tail": result.stderr_tail,
    }
    if result.temp_dir is not None:
        record["temp_dir"] = str(result.temp_dir)
    if result.output_dir is not None:
        record["output_dir"] = str(result.output_dir)

    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    with (result_dir / "runs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    (result_dir / "latest.json").write_text(json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    model_file = by_model_dir / f"{_safe_filename(model_name)}.jsonl"
    with model_file.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    return record


def _tail(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def classify_failure(returncode: int, stdout: str, stderr: str) -> Optional[str]:
    if returncode == 0:
        return None
    combined = f"{stdout}\n{stderr}".lower()
    if "cuda out of memory" in combined or "outofmemoryerror" in combined or "out of memory" in combined:
        return "oom"
    if "timed out" in combined or returncode == 124:
        return "timeout"
    return "runtime_error"


def run_subprocess(command: Sequence[str], *, timeout_seconds: Optional[float] = None) -> CompletedRun:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return CompletedRun(completed.returncode, completed.stdout, completed.stderr)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        stderr = f"{stderr}\nCommand timed out after {timeout_seconds} seconds".strip()
        return CompletedRun(124, stdout, stderr)


def _write_prompt_files(temp_dir: Path, prompt: str) -> tuple[Path, Path]:
    train_prompt = temp_dir / "train_prompt.txt"
    validation_prompt = temp_dir / "validation_prompt.txt"
    content = prompt.rstrip("\n") + "\n"
    train_prompt.write_text(content, encoding="utf-8")
    validation_prompt.write_text(content, encoding="utf-8")
    return train_prompt, validation_prompt


def build_export_command(
    *,
    python_executable: str,
    exporter_path: Path,
    model_path: Path,
    dataset: str,
    task_name: str,
    train_prompt_file: Path,
    validation_prompt_file: Path,
    output_dir: Path,
    device: str,
    gpu_mem_gb: float,
    cache_rate: float,
    batch_size: int,
    max_input_tokens: int,
    max_new_tokens: int,
    model_torch_dtype: str,
    storage_dtype: str,
    cache_policy: str,
    per_layer_cache: bool,
    print_status: bool,
) -> List[str]:
    command = [
        python_executable,
        str(exporter_path),
        "--model-path",
        str(model_path),
        "--dataset",
        dataset,
        "--task-name",
        task_name,
        "--train-prompt-file",
        str(train_prompt_file),
        "--validation-prompt-file",
        str(validation_prompt_file),
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        "--batch-size",
        str(batch_size),
        "--max-input-tokens",
        str(max_input_tokens),
        "--max-new-tokens",
        str(max_new_tokens),
        "--gpu-mem-limit-gb",
        f"{gpu_mem_gb:g}",
        "--cache-rate",
        f"{cache_rate:g}",
        "--cache-policy",
        cache_policy,
        "--per-layer-cache",
        "True" if per_layer_cache else "False",
        "--model-torch-dtype",
        model_torch_dtype,
        "--storage-dtype",
        storage_dtype,
    ]
    if print_status:
        command.append("--print-status")
    return command


def check_cache_ratio(
    *,
    python_executable: str,
    exporter_path: Path,
    model_path: Path,
    dataset: str,
    task_name: str,
    device: str,
    gpu_mem_gb: float,
    cache_rate: float,
    work_root: Optional[Path] = None,
    keep_temp: bool = False,
    prompt: str = "What is the answer?",
    batch_size: int = 1,
    max_input_tokens: int = 512,
    max_new_tokens: int = 1,
    model_torch_dtype: str = "auto",
    storage_dtype: str = "auto",
    cache_policy: str = "lru",
    per_layer_cache: bool = False,
    print_status: bool = False,
    timeout_seconds: Optional[float] = None,
    run_command: Callable[..., CompletedRun] = run_subprocess,
) -> CheckResult:
    if gpu_mem_gb <= 0:
        raise ValueError(f"gpu_mem_gb must be > 0, got {gpu_mem_gb}")
    if cache_rate < 0:
        raise ValueError(f"cache_rate must be >= 0, got {cache_rate}")

    temp_parent = Path(work_root) if work_root is not None else None
    temp_parent.mkdir(parents=True, exist_ok=True) if temp_parent is not None else None
    temp_dir_obj = None
    if keep_temp:
        temp_dir = Path(tempfile.mkdtemp(prefix="promoe-cache-ratio-", dir=str(temp_parent) if temp_parent else None))
    else:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="promoe-cache-ratio-", dir=str(temp_parent) if temp_parent else None)
        temp_dir = Path(temp_dir_obj.name)

    output_dir = temp_dir / "trace_output"
    start = time.time()
    try:
        train_prompt_file, validation_prompt_file = _write_prompt_files(temp_dir, prompt)
        command = build_export_command(
            python_executable=python_executable,
            exporter_path=exporter_path,
            model_path=model_path,
            dataset=dataset,
            task_name=task_name,
            train_prompt_file=train_prompt_file,
            validation_prompt_file=validation_prompt_file,
            output_dir=output_dir,
            device=device,
            gpu_mem_gb=gpu_mem_gb,
            cache_rate=cache_rate,
            batch_size=batch_size,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
            model_torch_dtype=model_torch_dtype,
            storage_dtype=storage_dtype,
            cache_policy=cache_policy,
            per_layer_cache=per_layer_cache,
            print_status=print_status,
        )
        completed = run_command(command, timeout_seconds=timeout_seconds)
        failure_kind = classify_failure(completed.returncode, completed.stdout, completed.stderr)
        return CheckResult(
            can_run=completed.returncode == 0,
            failure_kind=failure_kind,
            model_path=str(model_path),
            gpu_mem_gb=float(gpu_mem_gb),
            cache_rate=float(cache_rate),
            returncode=int(completed.returncode),
            elapsed_seconds=time.time() - start,
            stdout_tail=_tail(completed.stdout),
            stderr_tail=_tail(completed.stderr),
            temp_dir=temp_dir if keep_temp else None,
            output_dir=output_dir if keep_temp else None,
        )
    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()


def build_parser() -> argparse.ArgumentParser:
    repo_root = find_repo_root()
    parser = argparse.ArgumentParser(description="Check whether one cache_rate fits one GPU memory budget.")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-mem-gb", required=True, type=float)
    parser.add_argument("--cache-rate", required=True, type=float)
    parser.add_argument("--model-torch-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--storage-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--cache-policy", default="lru")
    parser.add_argument("--per-layer-cache", action="store_true")
    parser.add_argument("--prompt", default="What is the answer?")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--exporter-path", type=Path, default=default_exporter_path(repo_root))
    parser.add_argument("--work-root", type=Path, default=None, help="Temporary parent directory; outputs are still cleaned by default")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary prompt and trace output directory for debugging")
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--result-dir", type=Path, default=default_result_dir(repo_root))
    parser.add_argument("--no-save-result", dest="save_result", action="store_false", help="Do not append the lightweight result record")
    parser.set_defaults(save_result=True)
    parser.add_argument("--print-status", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = check_cache_ratio(
        python_executable=args.python_executable,
        exporter_path=args.exporter_path,
        model_path=args.model_path,
        dataset=args.dataset,
        task_name=args.task_name,
        device=args.device,
        gpu_mem_gb=args.gpu_mem_gb,
        cache_rate=args.cache_rate,
        work_root=args.work_root,
        keep_temp=args.keep_temp,
        prompt=args.prompt,
        batch_size=args.batch_size,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        model_torch_dtype=args.model_torch_dtype,
        storage_dtype=args.storage_dtype,
        cache_policy=args.cache_policy,
        per_layer_cache=args.per_layer_cache,
        print_status=args.print_status,
        timeout_seconds=args.timeout_seconds,
    )
    payload = result.to_json_dict()
    if args.save_result:
        record = save_result_record(
            result,
            result_dir=args.result_dir,
            device=args.device,
            dataset=args.dataset,
            task_name=args.task_name,
            model_torch_dtype=args.model_torch_dtype,
            storage_dtype=args.storage_dtype,
            batch_size=args.batch_size,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            cache_policy=args.cache_policy,
            per_layer_cache=args.per_layer_cache,
        )
        payload["result_record"] = {
            "result_dir": str(args.result_dir),
            "runs_jsonl": str(args.result_dir / "runs.jsonl"),
            "latest_json": str(args.result_dir / "latest.json"),
            "model_jsonl": str(args.result_dir / "by_model" / f"{_safe_filename(record['model_name'])}.jsonl"),
        }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if result.can_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
