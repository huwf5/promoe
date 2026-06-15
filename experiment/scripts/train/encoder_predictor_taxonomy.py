"""Path and manifest helpers for encoder predictor BLTE/BLE artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


PREDICTOR_ROOT = Path("experiment/models/predictors")
PREDICTOR_TASK = "encoder_expert_prefetch"
DEFAULT_DATASET = "mmlu"
DEFAULT_TASK_NAME = "professional_law"
WORKLOAD_TASK = "mmlu-professional_law"
DEFAULT_MODEL_PATH = Path("experiment/models/google/switch-base-128")
BASE_MODEL = "switch-base-128"
TRACE_ID = "sparse-cache-b1-longest-v1"
DATA_MODE_VALIDTRIM = "validtrim"


@dataclass(frozen=True)
class TraceContext:
    workload_task: str = WORKLOAD_TASK
    base_model: str = BASE_MODEL
    trace_id: str = TRACE_ID


def normalize_dataset_name(dataset: str) -> str:
    return str(dataset).strip().split("/")[-1]


def workload_task_name(dataset: str = DEFAULT_DATASET, task_name: str = DEFAULT_TASK_NAME) -> str:
    return f"{normalize_dataset_name(dataset)}-{task_name}"


def model_name_from_path(model_path: Path | str | None = None, model_name: str | None = None) -> str:
    if model_name is not None and str(model_name).strip():
        return str(model_name).strip()
    if model_path is None:
        return BASE_MODEL
    return Path(model_path).name


def trace_dir_for_model_task(
    *,
    model_path: Path | str | None = DEFAULT_MODEL_PATH,
    dataset: str = DEFAULT_DATASET,
    task_name: str = DEFAULT_TASK_NAME,
    repo_root: Path | str | None = None,
    model_name: str | None = None,
) -> Path:
    root = Path(".") if repo_root is None else Path(repo_root)
    base_model = model_name_from_path(model_path, model_name=model_name)
    return root / "experiment" / "traces" / f"{base_model}-{workload_task_name(dataset, task_name)}" / "encoder_predictor_sparse_cache_trace"


def resolve_trace_context(
    *,
    trace_dir: Path | str | None = None,
    model_path: Path | str | None = None,
    dataset: str = DEFAULT_DATASET,
    task_name: str = DEFAULT_TASK_NAME,
    trace_id: str = TRACE_ID,
    model_name: str | None = None,
) -> TraceContext:
    workload = workload_task_name(dataset, task_name)
    base_model = model_name_from_path(model_path, model_name=model_name) if model_path is not None or model_name else BASE_MODEL
    if trace_dir is not None and model_path is None and model_name is None:
        parent_name = Path(trace_dir).parent.name
        suffix = f"-{workload}"
        if parent_name.endswith(suffix):
            base_model = parent_name[: -len(suffix)]
    return TraceContext(workload_task=workload, base_model=base_model, trace_id=trace_id)


def _float_token(value: float) -> str:
    text = f"{value:g}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        mantissa = mantissa.replace(".", "p").replace("-", "m")
        exponent = exponent.replace("+", "").replace("-", "m")
        return f"{mantissa}e{exponent}"
    return text.replace(".", "p").replace("-", "m")


def _lr_token(value: float) -> str:
    if value == 1e-4:
        return "lr1e4"
    if value == 1e-3:
        return "lr1e3"
    if value == 3e-4:
        return "lr3e4"
    if value == 3e-5:
        return "lr3e5"
    return f"lr{_float_token(value)}"


def _dropout_token(value: float) -> str:
    return f"drop{_float_token(value)}"


def src_simplenn_blte_run_name(
    *,
    hidden_dim: int = 384,
    src_layers: int = 1,
    dropout: float = 0.5,
    lr: float = 1e-4,
    batch_size: int = 2,
    seed: int = 0,
    data_mode: str = DATA_MODE_VALIDTRIM,
    loss_token: str = "hardce",
) -> str:
    return (
        f"src-simplenn-token-{loss_token}-h{hidden_dim}-l{src_layers}-"
        f"{_dropout_token(dropout)}-{_lr_token(lr)}-bs{batch_size}-seed{seed}-{data_mode}"
    )


def sida_gru_sa_blte_run_name(
    *,
    hidden_dim: int = 256,
    recurrent_layers: int = 2,
    dropout: float = 0.0,
    lr: float = 1e-4,
    batch_size: int = 2,
    seed: int = 0,
    data_mode: str = DATA_MODE_VALIDTRIM,
) -> str:
    return (
        f"sida-gru-sa-hardce-h{hidden_dim}-rnnl{recurrent_layers}-"
        f"{_dropout_token(dropout)}-{_lr_token(lr)}-bs{batch_size}-seed{seed}-{data_mode}"
    )


SRC_SIMPLENN_BLTE_RUN_NAME = src_simplenn_blte_run_name()
SIDA_GRU_SA_BLTE_RUN_NAME = sida_gru_sa_blte_run_name()


def taxonomy_base_dir(
    root: Path = PREDICTOR_ROOT,
    *,
    workload_task: str = WORKLOAD_TASK,
    base_model: str = BASE_MODEL,
    trace_id: str = TRACE_ID,
) -> Path:
    return root / PREDICTOR_TASK / workload_task / base_model / trace_id


def blte_artifact_dir(
    run_name: str,
    root: Path = PREDICTOR_ROOT,
    *,
    workload_task: str = WORKLOAD_TASK,
    base_model: str = BASE_MODEL,
    trace_id: str = TRACE_ID,
) -> Path:
    return taxonomy_base_dir(root, workload_task=workload_task, base_model=base_model, trace_id=trace_id) / "blte" / run_name


def ble_view_name(source_blte_run_name: str, aggregation: str = "noisyor") -> str:
    return f"{aggregation}-from-{source_blte_run_name}"


def ble_artifact_dir(
    source_blte_run_name: str,
    root: Path = PREDICTOR_ROOT,
    aggregation: str = "noisyor",
    *,
    workload_task: str = WORKLOAD_TASK,
    base_model: str = BASE_MODEL,
    trace_id: str = TRACE_ID,
) -> Path:
    return taxonomy_base_dir(root, workload_task=workload_task, base_model=base_model, trace_id=trace_id) / "ble" / ble_view_name(
        source_blte_run_name, aggregation=aggregation
    )


def report_dir(
    report_name: str,
    root: Path = PREDICTOR_ROOT,
    *,
    workload_task: str = WORKLOAD_TASK,
    base_model: str = BASE_MODEL,
    trace_id: str = TRACE_ID,
) -> Path:
    return taxonomy_base_dir(root, workload_task=workload_task, base_model=base_model, trace_id=trace_id) / "reports" / report_name


def default_ble_report_dir(root: Path = PREDICTOR_ROOT) -> Path:
    return report_dir("ble-noisyor-report", root=root)


def blte_manifest(
    *,
    run_name: str,
    model_arch: str,
    objective: str,
    output_dir: Path,
    trace_dir: Path,
    hidden_dim: int,
    batch_size: int,
    lr: float,
    seed: int,
    data_mode: str = DATA_MODE_VALIDTRIM,
    src_layers: int | None = None,
    recurrent_layers: int | None = None,
    dropout: float | None = None,
    workload_task: str = WORKLOAD_TASK,
    base_model: str = BASE_MODEL,
    trace_id: str = TRACE_ID,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "artifact_level": "blte",
        "output_layout": "BLTE",
        "predictor_task": PREDICTOR_TASK,
        "workload_task": workload_task,
        "base_model": base_model,
        "trace_id": trace_id,
        "trace_dir": str(trace_dir),
        "run_name": run_name,
        "artifact_dir": str(output_dir),
        "model_arch": model_arch,
        "objective": objective,
        "hidden_dim": hidden_dim,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "data_mode": data_mode,
    }
    if src_layers is not None:
        payload["src_layers"] = src_layers
    if recurrent_layers is not None:
        payload["recurrent_layers"] = recurrent_layers
    if dropout is not None:
        payload["dropout"] = dropout
    if extra:
        payload.update(extra)
    return payload


def ble_manifest(
    *,
    source_blte_artifact: Path,
    source_checkpoint: str,
    source_run_name: str,
    output_dir: Path,
    aggregation: str = "noisy_or",
    trace_dir: Path | None = None,
    report_dir_path: Path | None = None,
    workload_task: str = WORKLOAD_TASK,
    base_model: str = BASE_MODEL,
    trace_id: str = TRACE_ID,
) -> dict[str, Any]:
    return {
        "artifact_level": "ble",
        "output_layout": "BLE",
        "predictor_task": PREDICTOR_TASK,
        "workload_task": workload_task,
        "base_model": base_model,
        "trace_id": trace_id,
        "trace_dir": str(trace_dir) if trace_dir is not None else None,
        "ble_view_name": ble_view_name(source_run_name),
        "artifact_dir": str(output_dir),
        "source_blte_artifact": str(source_blte_artifact),
        "source_blte_run_name": source_run_name,
        "source_checkpoint": source_checkpoint,
        "probability_transform": "softmax",
        "aggregation": aggregation,
        "valid_token_handling": DATA_MODE_VALIDTRIM,
        "budget_source": "sum_ble_score",
        "budget_rounding": "ceil",
        "budget_clamp_min": 1,
        "budget_clamp_max": "num_experts",
        "selection_rule": "topk_by_ble_score",
        "report_dir": str(report_dir_path) if report_dir_path is not None else None,
    }
