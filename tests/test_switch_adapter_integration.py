from pathlib import Path

import pytest
from transformers import SwitchTransformersForConditionalGeneration

from sparse_llm_cache.model_adapters.switch import SwitchAdapter


SWITCH_ROOT = Path(
    "/mnt/huwf5/promoe/deps/sparse-llm-cache-scripts/huggingface-modules/modules/transformers_modules/google"
)


@pytest.mark.parametrize("model_name", ["switch-base-128"])
def test_switch_adapter_scans_local_switch_structure(model_name):
    model_path = SWITCH_ROOT / model_name
    if not model_path.exists():
        pytest.skip(f"local Switch model not found: {model_path}")

    model = SwitchTransformersForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=True,
        device_map=None,
    )
    adapter = SwitchAdapter(model, str(model_path))

    expert_count = 0
    sparse_mlp_count = 0
    for name, module in model.named_modules():
        adapter.add_metadata_to_module(module, name)
        if adapter.expert_name_filter(name):
            expert_count += 1
            assert hasattr(module, "_layer_id")
            assert hasattr(module, "_expert_id")
        if adapter.moe_mlp_name_filter(name):
            sparse_mlp_count += 1
            assert hasattr(module, "_stage")
            assert hasattr(module, "_stage_layer_id")

    assert sparse_mlp_count == adapter.num_moe_layer
    assert expert_count == adapter.num_moe_layer * adapter.num_expert_per_layer
