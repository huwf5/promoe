"""Shared test fixtures."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

SCRIPTS_ROOT = Path(__file__).resolve().parents[2]
SWITCH_BASE_128_PATH = (
    SCRIPTS_ROOT
    / "huggingface-modules/modules/transformers_modules/google/switch-base-128"
)
MAIN_WORKSPACE_SWITCH_BASE_128_PATH = Path(
    "/mnt/huwf5/promoe/deps/sparse-llm-cache-scripts/"
    "huggingface-modules/modules/transformers_modules/google/switch-base-128"
)


@pytest.fixture
def switch_model_path() -> Path:
    if SWITCH_BASE_128_PATH.exists():
        return SWITCH_BASE_128_PATH
    if MAIN_WORKSPACE_SWITCH_BASE_128_PATH.exists():
        return MAIN_WORKSPACE_SWITCH_BASE_128_PATH
    if os.environ.get("PROMOE_SWITCH_TRACE_ALLOW_REMOTE_MODEL") == "1":
        return Path("google/switch-base-128")
    pytest.skip(
        "Local switch-base-128 not found. Set PROMOE_SWITCH_TRACE_ALLOW_REMOTE_MODEL=1 "
        "to opt in to loading from Hugging Face."
    )


@pytest.fixture
def tmp_out(tmp_path: Path) -> Path:
    out = tmp_path / "out"
    out.mkdir()
    return out
