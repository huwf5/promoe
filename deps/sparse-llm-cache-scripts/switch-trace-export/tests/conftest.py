"""Shared test fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest

SCRIPTS_ROOT = Path(__file__).resolve().parents[2]
SWITCH_BASE_128_PATH = (
    SCRIPTS_ROOT
    / "huggingface-modules/modules/transformers_modules/google/switch-base-128"
)


@pytest.fixture
def switch_model_path() -> Path:
    if not SWITCH_BASE_128_PATH.exists():
        pytest.skip(f"Local switch-base-128 not found at {SWITCH_BASE_128_PATH}")
    return SWITCH_BASE_128_PATH


@pytest.fixture
def tmp_out(tmp_path: Path) -> Path:
    out = tmp_path / "out"
    out.mkdir()
    return out
