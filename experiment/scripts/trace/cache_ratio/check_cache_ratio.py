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


class GpuMemoryMonitor:
    def __init__(self, *, device: str, sampler: Callable[[str], Optional[int]]):
        self.device = device
        self.sampler = sampler
        self.peak_mb: Optional[int] = None
        self.current_mb: Optional[int] = None
        self.sample_count = 0
        self.error: Optional[str] = None
        self._disabled = False

    def sample(self) -> None:
        if self._disabled:
            return
        try:
            used_mb = self.sampler(self.device)
        except Exception as exc:  # nvidia-smi may be unavailable on non-GPU hosts.
            self.error = str(exc)
            self._disabled = True
            return
        if used_mb is None:
            return
        used_mb = int(used_mb)
        self.current_mb = used_mb
        self.sample_count += 1
        if self.peak_mb is None or used_mb > self.peak_mb:
            self.peak_mb = used_mb

    @property
    def peak_gb(self) -> Optional[float]:
        if self.peak_mb is None:
            return None
        return self.peak_mb / 1024.0

    @property
    def current_gb(self) -> Optional[float]:
        if self.current_mb is None:
            return None
        return self.current_mb / 1024.0

    def exceeds_limit(self, gpu_mem_gb: float) -> Optional[bool]:
        peak_gb = self.peak_gb
        if peak_gb is None:
            return None
        return peak_gb > gpu_mem_gb


def _parse_cuda_device_index(device: str) -> Optional[int]:
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        suffix = device.split(":", 1)[1]
        if suffix.isdigit():
            return int(suffix)
    return None


def query_gpu_memory_used_mb(device: str) -> Optional[int]:
    gpu_index = _parse_cuda_device_index(device)
    if gpu_index is None:
        return None
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "-i",
            str(gpu_index),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "nvidia-smi failed")
    first_line = completed.stdout.strip().splitlines()[0]
    return int(first_line.strip())


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
    peak_gpu_memory_mb: Optional[int] = None
    peak_gpu_memory_gb: Optional[float] = None
    gpu_memory_sample_count: int = 0
    peak_exceeds_limit: Optional[bool] = None
    gpu_memory_error: Optional[str] = None
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
        "peak_gpu_memory_mb": result.peak_gpu_memory_mb,
        "peak_gpu_memory_gb": result.peak_gpu_memory_gb,
        "gpu_memory_sample_count": int(result.gpu_memory_sample_count),
        "peak_exceeds_limit": result.peak_exceeds_limit,
        "gpu_memory_error": result.gpu_memory_error,
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


def run_subprocess(
    command: Sequence[str],
    *,
    timeout_seconds: Optional[float] = None,
    gpu_memory_monitor: Optional[GpuMemoryMonitor] = None,
    gpu_memory_poll_interval_seconds: float = 0.1,
    heartbeat_seconds: Optional[float] = 30.0,
) -> CompletedRun:
    if gpu_memory_monitor is None:
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

    start = time.time()
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    timed_out = False
    poll_interval = max(float(gpu_memory_poll_interval_seconds), 0.01)
    heartbeat_interval = None
    if heartbeat_seconds is not None and heartbeat_seconds > 0:
        heartbeat_interval = float(heartbeat_seconds)
    next_heartbeat = start + heartbeat_interval if heartbeat_interval is not None else None
    print(
        f"[cache_ratio] started child pid={process.pid} command={' '.join(command)}",
        file=sys.stderr,
        flush=True,
    )
    while process.poll() is None:
        gpu_memory_monitor.sample()
        now = time.time()
        if next_heartbeat is not None and now >= next_heartbeat:
            elapsed = now - start
            current_gb = gpu_memory_monitor.current_gb
            peak_gb = gpu_memory_monitor.peak_gb
            current_text = "n/a" if current_gb is None else f"{current_gb:.2f}GB"
            peak_text = "n/a" if peak_gb is None else f"{peak_gb:.2f}GB"
            print(
                f"[cache_ratio] still running pid={process.pid} elapsed={elapsed:.0f}s "
                f"gpu_current={current_text} gpu_peak={peak_text} "
                f"samples={gpu_memory_monitor.sample_count}",
                file=sys.stderr,
                flush=True,
            )
            next_heartbeat = now + heartbeat_interval
        if timeout_seconds is not None and time.time() - start >= timeout_seconds:
            timed_out = True
            process.kill()
            break
        time.sleep(poll_interval)
    gpu_memory_monitor.sample()
    stdout, stderr = process.communicate()
    if timed_out:
        stderr = f"{stderr}\nCommand timed out after {timeout_seconds} seconds".strip()
        return CompletedRun(124, stdout, stderr)
    return CompletedRun(process.returncode, stdout, stderr)


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
    monitor_gpu_memory: bool = True,
    gpu_memory_sampler: Callable[[str], Optional[int]] = query_gpu_memory_used_mb,
    gpu_memory_poll_interval_seconds: float = 0.1,
    heartbeat_seconds: Optional[float] = 30.0,
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
    gpu_memory_monitor = (
        GpuMemoryMonitor(device=device, sampler=gpu_memory_sampler) if monitor_gpu_memory else None
    )
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
        completed = run_command(
            command,
            timeout_seconds=timeout_seconds,
            gpu_memory_monitor=gpu_memory_monitor,
            gpu_memory_poll_interval_seconds=gpu_memory_poll_interval_seconds,
            heartbeat_seconds=heartbeat_seconds,
        )
        failure_kind = classify_failure(completed.returncode, completed.stdout, completed.stderr)
        peak_gpu_memory_mb = gpu_memory_monitor.peak_mb if gpu_memory_monitor is not None else None
        peak_gpu_memory_gb = gpu_memory_monitor.peak_gb if gpu_memory_monitor is not None else None
        gpu_memory_sample_count = gpu_memory_monitor.sample_count if gpu_memory_monitor is not None else 0
        peak_exceeds_limit = (
            gpu_memory_monitor.exceeds_limit(gpu_mem_gb) if gpu_memory_monitor is not None else None
        )
        gpu_memory_error = gpu_memory_monitor.error if gpu_memory_monitor is not None else None
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
            peak_gpu_memory_mb=peak_gpu_memory_mb,
            peak_gpu_memory_gb=peak_gpu_memory_gb,
            gpu_memory_sample_count=gpu_memory_sample_count,
            peak_exceeds_limit=peak_exceeds_limit,
            gpu_memory_error=gpu_memory_error,
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
    parser.add_argument("--no-gpu-memory-monitor", dest="monitor_gpu_memory", action="store_false", help="Disable nvidia-smi GPU memory peak sampling during the inference subprocess")
    parser.add_argument("--gpu-memory-poll-interval-seconds", type=float, default=0.1)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0, help="Print parent-process progress while the exporter subprocess is still running; set <=0 to disable")
    parser.set_defaults(monitor_gpu_memory=True)
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
        monitor_gpu_memory=args.monitor_gpu_memory,
        gpu_memory_poll_interval_seconds=args.gpu_memory_poll_interval_seconds,
        heartbeat_seconds=args.heartbeat_seconds,
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
