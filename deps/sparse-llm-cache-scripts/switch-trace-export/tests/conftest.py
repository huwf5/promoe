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


@pytest.fixture
def switch_model_path() -> Path:
    env_model = os.environ.get("PROMOE_SWITCH_TRACE_MODEL_PATH")
    if env_model:
        p = Path(env_model).expanduser().resolve()
        if p.is_dir():
            return p
        pytest.skip(f"PROMOE_SWITCH_TRACE_MODEL_PATH is set but not a directory: {p}")
    if SWITCH_BASE_128_PATH.exists():
        return SWITCH_BASE_128_PATH
    if os.environ.get("PROMOE_SWITCH_TRACE_ALLOW_REMOTE_MODEL") == "1":
        return Path("google/switch-base-128")
    pytest.skip(
        "Local switch-base-128 not found under sparse-llm-cache-scripts. "
        "Clone or place the model there, set PROMOE_SWITCH_TRACE_MODEL_PATH to a local dir, "
        "or set PROMOE_SWITCH_TRACE_ALLOW_REMOTE_MODEL=1 to load from Hugging Face."
    )


@pytest.fixture
def tmp_out(tmp_path: Path) -> Path:
    out = tmp_path / "out"
    out.mkdir()
    return out
