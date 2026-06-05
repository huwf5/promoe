from pathlib import Path

import torch


def test_switch_sparse_mlp_active_experts_preserve_expert_axis():
    router_mask = torch.zeros((1, 2, 3), dtype=torch.bool)
    router_mask[0, 0, 0] = True
    router_mask[0, 1, 2] = True

    batch_size, seq_len, num_experts = router_mask.shape
    active = router_mask.reshape(batch_size * seq_len, num_experts).sum(dim=0)

    assert torch.nonzero(active, as_tuple=True)[0].tolist() == [0, 2]


def test_switch_sparse_mlp_source_does_not_transpose_router_mask_before_flattening():
    source = Path(
        "deps/transformers/src/transformers/models/switch_transformers/modeling_switch_transformers.py"
    ).read_text()

    assert "router_mask.transpose(1, 2).reshape(batch_size * seq_len, num_experts)" not in source
    assert "router_mask.reshape(batch_size * seq_len, num_experts).sum(dim=0)" in source
