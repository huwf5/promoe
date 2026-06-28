from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiment/scripts/trace/motivation/analyze_prompt_length_distribution.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("motivation_prompt_length_distribution", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_prompts_from_txt_unescapes_newlines(tmp_path: Path):
    module = _load_module()
    path = tmp_path / "prompt_list.txt"
    path.write_text("hello\nworld\nline\\nfeed\n", encoding="utf-8")

    assert module.load_prompts(path) == ["hello", "world", "line\nfeed"]


def test_bucket_lengths_uses_right_open_bins_and_overflow():
    module = _load_module()

    rows = module.bucket_lengths([1, 32, 33, 64, 65, 200], buckets=[32, 64, 128])

    assert rows == [
        {"bucket": "1-32", "count": 2, "ratio": 2 / 6},
        {"bucket": "33-64", "count": 2, "ratio": 2 / 6},
        {"bucket": "65-128", "count": 1, "ratio": 1 / 6},
        {"bucket": ">128", "count": 1, "ratio": 1 / 6},
    ]


def test_summarize_lengths_reports_percentiles():
    module = _load_module()

    summary = module.summarize_lengths([10, 20, 30, 40])

    assert summary["count"] == 4
    assert summary["min"] == 10
    assert summary["max"] == 40
    assert summary["mean"] == 25.0
    assert summary["p50"] == 25.0
    assert summary["p90"] == 37.0


def test_write_outputs_creates_tables_and_histogram(tmp_path: Path):
    module = _load_module()
    records = [
        {"sample_id": 0, "char_len": 10, "word_len": 2, "token_len": 5, "prompt": "aa bb"},
        {"sample_id": 1, "char_len": 20, "word_len": 4, "token_len": 15, "prompt": "cc dd ee ff"},
    ]
    payload = module.build_report_payload(
        records=records,
        length_key="token_len",
        buckets=[8, 16],
        source_path=Path("prompt_list.txt"),
        tokenizer_path=Path("model"),
    )

    module.write_outputs(output_dir=tmp_path, payload=payload)

    assert (tmp_path / "prompt_length_distribution.png").stat().st_size > 0
    assert (tmp_path / "summary.md").read_text(encoding="utf-8").startswith("# Prompt Length Distribution")
    with (tmp_path / "prompt_lengths.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["token_len"] == "5"
