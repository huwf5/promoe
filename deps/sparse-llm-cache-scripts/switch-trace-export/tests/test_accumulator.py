import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
MODULE_DIR = THIS_DIR.parent
sys.path.insert(0, str(MODULE_DIR))
from utils import NUM_SPARSE_LAYERS, TraceAccumulator  # noqa: E402


def _logits(value: float) -> torch.Tensor:
    return torch.full((128,), value, dtype=torch.float32)


def test_add_two_tokens_in_one_seq():
    acc = TraceAccumulator()
    layers0 = [_logits(float(i)) for i in range(NUM_SPARSE_LAYERS)]
    layers1 = [_logits(float(i + 10)) for i in range(NUM_SPARSE_LAYERS)]
    acc.add_token(seq_id=0, token_idx_in_seq=0, token_id=42, per_layer_logits=layers0)
    acc.add_token(seq_id=0, token_idx_in_seq=1, token_id=43, per_layer_logits=layers1)
    assert acc.total_tokens() == 2
    assert acc.seq_ids() == [0, 0]
    assert acc.token_idx_in_seq() == [0, 1]
    assert acc.token_ids() == [42, 43]
    stacked = acc.stacked_logits()  # [N=2, L=6, V=128]
    assert stacked.shape == (2, 6, 128)
    assert stacked.dtype == torch.float32
    assert torch.equal(stacked[0, 0], _logits(0.0))
    assert torch.equal(stacked[1, 5], _logits(15.0))


def test_two_seqs_keep_order():
    acc = TraceAccumulator()
    z = [_logits(0.0) for _ in range(NUM_SPARSE_LAYERS)]
    acc.add_token(0, 0, 1, z)
    acc.add_token(1, 0, 2, z)
    acc.add_token(0, 1, 3, z)
    assert acc.seq_ids() == [0, 1, 0]
    assert acc.token_idx_in_seq() == [0, 0, 1]


def test_wrong_layer_count_raises():
    acc = TraceAccumulator()
    import pytest
    with pytest.raises(ValueError):
        acc.add_token(0, 0, 0, [_logits(0.0)] * 3)


def test_wrong_logit_shape_raises():
    acc = TraceAccumulator()
    import pytest
    bad = [_logits(0.0) for _ in range(NUM_SPARSE_LAYERS)]
    bad[0] = torch.zeros(64)
    with pytest.raises(ValueError):
        acc.add_token(0, 0, 0, bad)


def test_all_layers_wrong_expert_dim_raises():
    acc = TraceAccumulator()
    import pytest
    all64 = [torch.zeros(64, dtype=torch.float32) for _ in range(NUM_SPARSE_LAYERS)]
    with pytest.raises(ValueError):
        acc.add_token(0, 0, 0, all64)
