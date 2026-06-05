from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UTILS = ROOT / "src" / "sparse_llm_cache" / "utils" / "__init__.py"


def test_runtime_expert_set_diagnostics_env_and_fields_present():
    source = UTILS.read_text()

    assert "SPARSE_CACHE_LOG_RUNTIME_EXPERT_SET" in source
    assert "runtime_expert_set:" in source
    assert "stage_layer=" in source
    assert "global_layer=" in source
    assert "training=" in source
    assert "experts=" in source
