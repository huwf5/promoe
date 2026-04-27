from pathlib import Path
import sys

import pytest
import torch

THIS_DIR = Path(__file__).resolve().parent
MODULE_DIR = THIS_DIR.parent
sys.path.insert(0, str(MODULE_DIR))
from utils import NUM_SPARSE_LAYERS, ContractWriter, TraceAccumulator, Verifier  # noqa: E402


def _build_good_dir(d: Path):
    acc = TraceAccumulator()
    for s in range(2):
        for k in range(64):
            flat = s * 64 + k
            layers = []
            for i in range(NUM_SPARSE_LAYERS):
                logits = torch.full((128,), -10.0, dtype=torch.float32)
                peak = (flat + i) % 128
                logits[peak] = 10.0
                layers.append(logits)
            acc.add_token(s, k, 100 + k, layers)
    ContractWriter(d, stage="decoder").write(acc)


def test_verifier_passes_on_good_dir(tmp_out: Path):
    d = tmp_out / "decoder"
    _build_good_dir(d)
    Verifier.verify(d)  # should not raise


def test_verifier_catches_freq_not_normalized(tmp_out: Path):
    d = tmp_out / "decoder"
    _build_good_dir(d)
    bad_freq = torch.full((128, 6, 128), 0.5, dtype=torch.float32)
    torch.save(bad_freq, d / "decode_stage_expert_freq_per_token.pt")
    with pytest.raises(AssertionError, match="freq.*sum"):
        Verifier.verify(d)


def test_verifier_catches_wrong_num_expert(tmp_out: Path):
    d = tmp_out / "decoder"
    _build_good_dir(d)
    bad_sel = torch.zeros((128, 6, 1), dtype=torch.int64)
    bad_sel[0, 0, 0] = 200  # > 128
    torch.save(bad_sel, d / "expert_selection.pt")
    with pytest.raises(AssertionError, match="expert.*range|expert.*128"):
        Verifier.verify(d)


def test_verifier_catches_token_idx_not_monotonic(tmp_out: Path):
    d = tmp_out / "decoder"
    _build_good_dir(d)
    bad_idx = torch.tensor(list(range(64)) + [0, 2] + list(range(2, 64)), dtype=torch.int64)
    torch.save(bad_idx, d / "decode_stage_token_idx_in_seq.pt")
    with pytest.raises(AssertionError, match="monotonic|token_idx"):
        Verifier.verify(d)
