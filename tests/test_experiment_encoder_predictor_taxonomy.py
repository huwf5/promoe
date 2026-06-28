from __future__ import annotations

from pathlib import Path

from experiment.scripts.train import encoder_predictor_sida_gru_sa_hard_ce as train_sida
from experiment.scripts.train import encoder_predictor_src_simplenn_token_hard_ce as train_src
from experiment.scripts.train import evaluate_encoder_predictor_prefetch as report
from experiment.scripts.train.encoder_predictor_taxonomy import (
    SRC_SIMPLENN_BLTE_RUN_NAME,
    ble_artifact_dir,
    ble_manifest,
    ble_view_name,
    blte_artifact_dir,
    blte_manifest,
    default_ble_report_dir,
    resolve_trace_context,
    trace_dir_for_model_task,
    sida_gru_sa_blte_run_name,
    src_simplenn_blte_run_name,
    taxonomy_base_dir,
)


def test_src_and_sida_run_names_are_parameterized() -> None:
    assert SRC_SIMPLENN_BLTE_RUN_NAME == "src-simplenn-token-hardce-h384-l1-drop0p5-lr1e4-bs2-seed0-validtrim"
    assert src_simplenn_blte_run_name(hidden_dim=512, src_layers=2, dropout=0.25, lr=3e-5, batch_size=4, seed=7) == (
        "src-simplenn-token-hardce-h512-l2-drop0p25-lr3e5-bs4-seed7-validtrim"
    )
    assert src_simplenn_blte_run_name(loss_token="bce-weighted") == (
        "src-simplenn-token-bce-weighted-h384-l1-drop0p5-lr1e4-bs2-seed0-validtrim"
    )
    assert src_simplenn_blte_run_name(loss_token="bce-equal-top2") == (
        "src-simplenn-token-bce-equal-top2-h384-l1-drop0p5-lr1e4-bs2-seed0-validtrim"
    )
    assert sida_gru_sa_blte_run_name() == "sida-gru-sa-hardce-h256-rnnl2-drop0-lr1e4-bs2-seed0-validtrim"


def test_taxonomy_paths_use_task_workload_model_trace_then_blte_ble() -> None:
    run_name = SRC_SIMPLENN_BLTE_RUN_NAME

    assert blte_artifact_dir(run_name) == Path(
        "experiment/models/predictors/encoder_expert_prefetch/mmlu-professional_law/"
        "switch-base-128/sparse-cache-b1-longest-v1/blte/"
        "src-simplenn-token-hardce-h384-l1-drop0p5-lr1e4-bs2-seed0-validtrim"
    )
    assert ble_view_name(run_name) == "noisyor-from-src-simplenn-token-hardce-h384-l1-drop0p5-lr1e4-bs2-seed0-validtrim"
    assert ble_artifact_dir(run_name) == Path(
        "experiment/models/predictors/encoder_expert_prefetch/mmlu-professional_law/"
        "switch-base-128/sparse-cache-b1-longest-v1/ble/"
        "noisyor-from-src-simplenn-token-hardce-h384-l1-drop0p5-lr1e4-bs2-seed0-validtrim"
    )
    assert default_ble_report_dir() == Path(
        "experiment/models/predictors/encoder_expert_prefetch/mmlu-professional_law/"
        "switch-base-128/sparse-cache-b1-longest-v1/reports/ble-noisyor-report"
    )


def test_taxonomy_paths_accept_switch_base_256_context() -> None:
    run_name = SRC_SIMPLENN_BLTE_RUN_NAME

    assert trace_dir_for_model_task(
        model_path=Path("experiment/models/google/switch-base-256"),
        dataset="mmlu",
        task_name="professional_law",
    ) == Path("experiment/traces/switch-base-256-mmlu-professional_law/encoder_predictor_sparse_cache_trace")
    assert taxonomy_base_dir(workload_task="mmlu-professional_law", base_model="switch-base-256") == Path(
        "experiment/models/predictors/encoder_expert_prefetch/mmlu-professional_law/"
        "switch-base-256/sparse-cache-b1-longest-v1"
    )
    assert blte_artifact_dir(run_name, workload_task="mmlu-professional_law", base_model="switch-base-256") == Path(
        "experiment/models/predictors/encoder_expert_prefetch/mmlu-professional_law/"
        "switch-base-256/sparse-cache-b1-longest-v1/blte/"
        "src-simplenn-token-hardce-h384-l1-drop0p5-lr1e4-bs2-seed0-validtrim"
    )


def test_trace_context_can_be_resolved_from_model_task_or_trace_dir() -> None:
    from_model = resolve_trace_context(
        model_path=Path("experiment/models/google/switch-base-256"),
        dataset="mmlu",
        task_name="professional_law",
    )
    from_trace_dir = resolve_trace_context(
        trace_dir=Path("experiment/traces/switch-base-256-mmlu-professional_law/encoder_predictor_sparse_cache_trace"),
        dataset="mmlu",
        task_name="professional_law",
    )

    assert from_model.workload_task == "mmlu-professional_law"
    assert from_model.base_model == "switch-base-256"
    assert from_trace_dir == from_model


def test_blte_and_ble_manifests_record_required_fields(tmp_path: Path) -> None:
    run_name = SRC_SIMPLENN_BLTE_RUN_NAME
    blte = blte_manifest(
        run_name=run_name,
        model_arch="src-simplenn-token",
        objective="hard-ce",
        output_dir=tmp_path / "blte" / run_name,
        trace_dir=tmp_path / "trace",
        hidden_dim=384,
        batch_size=2,
        lr=1e-4,
        seed=0,
        src_layers=1,
        dropout=0.5,
    )
    ble = ble_manifest(
        source_blte_artifact=tmp_path / "blte" / run_name,
        source_checkpoint="best_model.pt",
        source_run_name=run_name,
        output_dir=tmp_path / "ble" / ble_view_name(run_name),
        trace_dir=tmp_path / "trace",
        report_dir_path=tmp_path / "reports" / "ble-noisyor-report",
    )

    assert blte["artifact_level"] == "blte"
    assert blte["output_layout"] == "BLTE"
    assert blte["predictor_task"] == "encoder_expert_prefetch"
    assert blte["workload_task"] == "mmlu-professional_law"
    assert blte["base_model"] == "switch-base-128"
    assert blte["trace_id"] == "sparse-cache-b1-longest-v1"
    assert blte["data_mode"] == "validtrim"

    assert ble["artifact_level"] == "ble"
    assert ble["output_layout"] == "BLE"
    assert ble["source_blte_run_name"] == run_name
    assert ble["probability_transform"] == "softmax"
    assert ble["aggregation"] == "noisy_or"
    assert ble["budget_source"] == "sum_ble_score"
    assert ble["budget_rounding"] == "ceil"
    assert ble["selection_rule"] == "topk_by_ble_score"


def test_training_and_report_defaults_use_new_taxonomy() -> None:
    src_args = train_src.parse_args([])
    sida_args = train_sida.parse_args([])
    report_args = report.parse_args([])

    src_args.trace_dir = train_src.resolve_trace_dir(src_args)
    sida_args.trace_dir = train_sida.resolve_trace_dir(sida_args)

    assert train_src.resolve_output_dir(src_args) == blte_artifact_dir(SRC_SIMPLENN_BLTE_RUN_NAME)
    assert train_sida.resolve_output_dir(sida_args) == blte_artifact_dir("sida-gru-sa-hardce-h256-rnnl2-drop0-lr1e4-bs2-seed0-validtrim")
    assert report_args.model_dir == blte_artifact_dir(SRC_SIMPLENN_BLTE_RUN_NAME)
    assert report_args.output_dir == default_ble_report_dir()


def test_training_defaults_can_target_switch_base_256_from_model_task() -> None:
    src_args = train_src.parse_args(
        [
            "--model-path",
            "experiment/models/google/switch-base-256",
            "--dataset",
            "mmlu",
            "--task-name",
            "professional_law",
        ]
    )
    sida_args = train_sida.parse_args(
        [
            "--model-path",
            "experiment/models/google/switch-base-256",
            "--dataset",
            "mmlu",
            "--task-name",
            "professional_law",
        ]
    )

    src_args.trace_dir = train_src.resolve_trace_dir(src_args)
    sida_args.trace_dir = train_sida.resolve_trace_dir(sida_args)

    assert src_args.trace_dir == Path(
        "experiment/traces/switch-base-256-mmlu-professional_law/encoder_predictor_sparse_cache_trace"
    )
    assert train_src.resolve_output_dir(src_args) == blte_artifact_dir(
        SRC_SIMPLENN_BLTE_RUN_NAME,
        workload_task="mmlu-professional_law",
        base_model="switch-base-256",
    )
    assert train_sida.resolve_output_dir(sida_args) == blte_artifact_dir(
        "sida-gru-sa-hardce-h256-rnnl2-drop0-lr1e4-bs2-seed0-validtrim",
        workload_task="mmlu-professional_law",
        base_model="switch-base-256",
    )

