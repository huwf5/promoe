from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import torch


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiment/scripts/trace/motivation/analyze_encoder_active_experts_online.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("motivation_encoder_active_experts", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_seq_lens_accepts_comma_separated_values():
    module = _load_module()

    assert module.parse_seq_lens("32, 64,128") == [32, 64, 128]


def test_parse_seq_lens_rejects_non_positive_values():
    module = _load_module()

    try:
        module.parse_seq_lens("32,0,64")
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("expected parse_seq_lens to reject zero")


def test_default_seq_lens_cover_long_prefill_to_10240():
    module = _load_module()

    assert module.parse_seq_lens(module.DEFAULT_SEQ_LENS) == [
        16,
        32,
        64,
        128,
        256,
        512,
        768,
        1024,
        1536,
        2048,
        3072,
        4096,
        6144,
        8192,
        10240,
    ]


def test_resolve_cache_rate_uses_model_defaults_and_override():
    module = _load_module()

    assert module.resolve_cache_rate(None, model_type="switch_transformers") == 0.375
    assert module.resolve_cache_rate(None, model_type="nllb-moe") == 0.01
    assert module.resolve_cache_rate(0.25, model_type="switch_transformers") == 0.25


def test_build_sparse_cache_config_records_runtime_options():
    module = _load_module()

    config = module.build_sparse_cache_config(
        model_path=Path("/models/switch"),
        model_type="switch_transformers",
        cache_rate=0.5,
        cache_policy="lfu",
        per_layer_cache=True,
        device="cuda:2",
    )

    assert config["model_id"] == "/models/switch"
    assert config["cache_rate"] == 0.5
    assert config["cache_policy"] == "lfu"
    assert config["per_layer_cache"] is True
    assert config["num_predict_expert_per_layer"] == 0
    assert config["predict_input_mode"] == "no_predict"
    assert config["cache_device"] == "cuda:2"


def test_compute_layer_stats_counts_router_reported_and_forward_experts():
    module = _load_module()
    expert_selection = torch.tensor(
        [
            [
                [[1], [1], [2], [3], [0]],
                [[4], [4], [4], [5], [0]],
            ],
        ],
        dtype=torch.int64,
    )
    selection_mask = torch.tensor(
        [
            [
                [[True], [True], [True], [True], [False]],
                [[True], [True], [True], [True], [False]],
            ],
        ],
        dtype=torch.bool,
    )

    rows = module.compute_layer_stats(
        seq_len=4,
        sample_index=0,
        expert_selection=expert_selection,
        selection_mask=selection_mask,
        reported_used_by_layer={0: {1, 2}, 1: {4, 5}},
        expert_forward_used_by_layer={0: {1, 2, 3}, 1: {4}},
        num_experts=8,
        router_layer_to_model_block=[1, 3],
    )

    assert [row.router_selected_experts for row in rows] == [3, 2]
    assert [row.reported_used_experts for row in rows] == [2, 2]
    assert [row.expert_forward_used_experts for row in rows] == [3, 1]
    assert [row.expert_forward_used_ratio for row in rows] == [0.375, 0.125]
    assert [row.encoder_block for row in rows] == [1, 3]


def test_summarize_rows_groups_by_seq_len_and_layer():
    module = _load_module()
    rows = [
        module.LayerSampleStats(32, 0, 0, 1, 10, 0.1, 9, 0.09, 8, 0.08),
        module.LayerSampleStats(32, 1, 0, 1, 14, 0.14, 13, 0.13, 12, 0.12),
        module.LayerSampleStats(32, 0, 1, 3, 20, 0.2, 19, 0.19, 18, 0.18),
    ]

    summary = module.summarize_rows(rows, num_experts=100)

    assert summary == [
        {
            "seq_len": 32,
            "encoder_sparse_layer": 0,
            "encoder_block": 1,
            "samples": 2,
            "mean_router_selected_experts": 12.0,
            "min_router_selected_experts": 10,
            "max_router_selected_experts": 14,
            "mean_router_selected_ratio": 0.12,
            "mean_reported_used_experts": 11.0,
            "min_reported_used_experts": 9,
            "max_reported_used_experts": 13,
            "mean_reported_used_ratio": 0.11,
            "mean_expert_forward_used_experts": 10.0,
            "min_expert_forward_used_experts": 8,
            "max_expert_forward_used_experts": 12,
            "mean_expert_forward_used_ratio": 0.1,
        },
        {
            "seq_len": 32,
            "encoder_sparse_layer": 1,
            "encoder_block": 3,
            "samples": 1,
            "mean_router_selected_experts": 20.0,
            "min_router_selected_experts": 20,
            "max_router_selected_experts": 20,
            "mean_router_selected_ratio": 0.2,
            "mean_reported_used_experts": 19.0,
            "min_reported_used_experts": 19,
            "max_reported_used_experts": 19,
            "mean_reported_used_ratio": 0.19,
            "mean_expert_forward_used_experts": 18.0,
            "min_expert_forward_used_experts": 18,
            "max_expert_forward_used_experts": 18,
            "mean_expert_forward_used_ratio": 0.18,
        },
    ]


def test_summarize_rows_by_layer_aggregates_all_samples():
    module = _load_module()
    rows = [
        module.LayerSampleStats(16, 0, 0, 1, 3, 0.3, 2, 0.2, 2, 0.2),
        module.LayerSampleStats(32, 1, 0, 1, 5, 0.5, 4, 0.4, 3, 0.3),
        module.LayerSampleStats(16, 0, 1, 3, 6, 0.6, 5, 0.5, 4, 0.4),
    ]

    summary = module.summarize_rows_by_layer(rows, num_experts=10)

    assert summary == [
        {
            "encoder_sparse_layer": 0,
            "encoder_block": 1,
            "samples": 2,
            "mean_seq_len": 24.0,
            "min_seq_len": 16,
            "max_seq_len": 32,
            "mean_router_selected_experts": 4.0,
            "min_router_selected_experts": 3,
            "max_router_selected_experts": 5,
            "mean_router_selected_ratio": 0.4,
            "mean_reported_used_experts": 3.0,
            "min_reported_used_experts": 2,
            "max_reported_used_experts": 4,
            "mean_reported_used_ratio": 0.3,
            "mean_expert_forward_used_experts": 2.5,
            "min_expert_forward_used_experts": 2,
            "max_expert_forward_used_experts": 3,
            "mean_expert_forward_used_ratio": 0.25,
        },
        {
            "encoder_sparse_layer": 1,
            "encoder_block": 3,
            "samples": 1,
            "mean_seq_len": 16.0,
            "min_seq_len": 16,
            "max_seq_len": 16,
            "mean_router_selected_experts": 6.0,
            "min_router_selected_experts": 6,
            "max_router_selected_experts": 6,
            "mean_router_selected_ratio": 0.6,
            "mean_reported_used_experts": 5.0,
            "min_reported_used_experts": 5,
            "max_reported_used_experts": 5,
            "mean_reported_used_ratio": 0.5,
            "mean_expert_forward_used_experts": 4.0,
            "min_expert_forward_used_experts": 4,
            "max_expert_forward_used_experts": 4,
            "mean_expert_forward_used_ratio": 0.4,
        },
    ]


def test_load_prompts_from_txt_unescapes_newlines(tmp_path: Path):
    module = _load_module()
    path = tmp_path / "prompt_list.txt"
    path.write_text("hello\nline\\nfeed\n", encoding="utf-8")

    assert module.load_prompts(path) == ["hello", "line\nfeed"]


def test_write_outputs_creates_csv_markdown_and_metric_plots(tmp_path: Path):
    module = _load_module()
    summary = []
    for seq_len, layer0, layer1 in [(32, 10, 12), (64, 15, 17)]:
        for layer_id, value in [(0, layer0), (1, layer1)]:
            summary.append(
                {
                    "seq_len": seq_len,
                    "encoder_sparse_layer": layer_id,
                    "encoder_block": layer_id * 2 + 1,
                    "samples": 2,
                    "mean_router_selected_experts": float(value + 2),
                    "min_router_selected_experts": value,
                    "max_router_selected_experts": value + 4,
                    "mean_router_selected_ratio": round((value + 2) / 100, 4),
                    "mean_reported_used_experts": float(value + 1),
                    "min_reported_used_experts": value,
                    "max_reported_used_experts": value + 2,
                    "mean_reported_used_ratio": round((value + 1) / 100, 4),
                    "mean_expert_forward_used_experts": float(value),
                    "min_expert_forward_used_experts": value,
                    "max_expert_forward_used_experts": value,
                    "mean_expert_forward_used_ratio": round(value / 100, 4),
                }
            )

    module.write_outputs(
        output_dir=tmp_path,
        summary_rows=summary,
        detail_rows=[],
        metadata={"model_path": "model", "num_experts": 100},
    )

    with (tmp_path / "active_experts_by_seq_len.csv").open(newline="", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    assert csv_rows[0]["seq_len"] == "32"
    assert csv_rows[0]["mean_expert_forward_used_experts"] == "10.0000"

    markdown = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "| seq_len | encoder_sparse_layer | encoder_block | samples | mean_router_selected | mean_reported_used | mean_expert_forward_used |" in markdown
    assert "active_experts_by_seq_len_router_selected.png" in markdown
    assert "active_experts_by_seq_len_reported_used.png" in markdown
    assert "active_experts_by_seq_len_expert_forward_used.png" in markdown
    assert (tmp_path / "active_experts_by_seq_len_router_selected.png").stat().st_size > 0
    assert (tmp_path / "active_experts_by_seq_len_reported_used.png").stat().st_size > 0
    assert (tmp_path / "active_experts_by_seq_len_expert_forward_used.png").stat().st_size > 0
    assert (tmp_path / "active_experts_by_seq_len.png").stat().st_size > 0
    assert (tmp_path / "active_experts_by_layer.csv").exists()
