from pathlib import Path

from sparse_llm_cache.utils.model_loading import (
    is_nllb_moe_model_id,
    is_switch_model_id,
    resolve_transformers_module_model_path,
)


def test_model_id_detection_covers_switch_and_nllb():
    assert is_switch_model_id("google/switch-base-128")
    assert is_switch_model_id("/models/google/switch-large-128")
    assert is_nllb_moe_model_id("facebook/nllb-moe-54b")
    assert is_nllb_moe_model_id("/models/facebook/nllb-moe-54b")
    assert not is_nllb_moe_model_id("deepseek-ai/deepseek-moe-16b-chat")


def test_resolve_transformers_module_model_path_uses_local_repo_copy(tmp_path):
    local_model = (
        tmp_path
        / "deps"
        / "sparse-llm-cache-scripts"
        / "huggingface-modules"
        / "modules"
        / "transformers_modules"
        / "facebook"
        / "nllb-moe-54b"
    )
    local_model.mkdir(parents=True)

    resolved = resolve_transformers_module_model_path("facebook/nllb-moe-54b", repo_root=tmp_path)

    assert resolved == str(local_model)


def test_resolve_transformers_module_model_path_keeps_absolute_paths(tmp_path):
    model_path = tmp_path / "experiment" / "models" / "facebook" / "nllb-moe-54b"
    model_path.mkdir(parents=True)

    resolved = resolve_transformers_module_model_path(str(model_path), repo_root=tmp_path)

    assert resolved == str(model_path)
