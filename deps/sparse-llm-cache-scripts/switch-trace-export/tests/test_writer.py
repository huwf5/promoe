import json
import sys
from pathlib import Path

import pytest
import torch

THIS_DIR = Path(__file__).resolve().parent
MODULE_DIR = THIS_DIR.parent
sys.path.insert(0, str(MODULE_DIR))
from utils import NUM_SPARSE_LAYERS, ContractWriter, TraceAccumulator  # noqa: E402

CONTRACT_PT_FILES = [
    "expert_selection.pt",
    "decode_stage_moe_layer_logits_per_token.pt",
    "decode_stage_moe_layer_gate_logits_per_token.pt",
    "decode_stage_expert_freq_per_token.pt",
    "decode_stage_token_ids_per_token.pt",
    "decode_stage_seq_id_of_token.pt",
    "decode_stage_token_idx_in_seq.pt",
]


def _make_layer_logits(peak: int, vocab: int) -> torch.Tensor:
    t = torch.arange(vocab, dtype=torch.float32) * 1e-3
    t[peak] = 100.0
    return t


def _make_acc(n_seqs: int = 2, tokens_per_seq: int = 3, vocab: int = 128) -> TraceAccumulator:
    """Build logits so argmax hits expert vocab-1 at least once; other peaks stay below that."""
    acc = TraceAccumulator(num_experts=vocab)
    tok = 1000
    for s in range(n_seqs):
        for k in range(tokens_per_seq):
            layers = []
            for i in range(NUM_SPARSE_LAYERS):
                flat = s * (tokens_per_seq * NUM_SPARSE_LAYERS) + k * NUM_SPARSE_LAYERS + i
                if flat == 0:
                    peak = vocab - 1
                else:
                    peak = flat % (vocab - 1)
                layers.append(_make_layer_logits(peak, vocab))
            acc.add_token(seq_id=s, token_idx_in_seq=k, token_id=tok, per_layer_logits=layers)
            tok += 1
    return acc


def test_write_files_present(tmp_out: Path):
    acc = _make_acc()
    writer = ContractWriter(tmp_out / "decoder", stage="decoder")
    writer.write(acc, extra_metadata={"max_new_tokens": 4})
    d = tmp_out / "decoder"
    for fn in [*CONTRACT_PT_FILES, "metadata.json"]:
        assert (d / fn).exists(), fn


def test_shapes_and_dtypes(tmp_out: Path):
    acc = _make_acc(n_seqs=2, tokens_per_seq=3, vocab=128)
    writer = ContractWriter(tmp_out / "decoder", stage="decoder")
    writer.write(acc)

    sel = torch.load(tmp_out / "decoder/expert_selection.pt", weights_only=True)
    assert sel.shape == (6, 6, 1) and sel.dtype == torch.int64

    feat = torch.load(tmp_out / "decoder/decode_stage_moe_layer_logits_per_token.pt", weights_only=True)
    gate = torch.load(tmp_out / "decoder/decode_stage_moe_layer_gate_logits_per_token.pt", weights_only=True)
    freq = torch.load(tmp_out / "decoder/decode_stage_expert_freq_per_token.pt", weights_only=True)
    for t in (feat, gate, freq):
        assert t.shape == (6, 6, 128) and t.dtype == torch.float32

    assert torch.equal(feat, gate), "input feature must equal gate logits per design"
    row_sums = freq.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    seq = torch.load(tmp_out / "decoder/decode_stage_seq_id_of_token.pt", weights_only=True)
    idx = torch.load(tmp_out / "decoder/decode_stage_token_idx_in_seq.pt", weights_only=True)
    tids = torch.load(tmp_out / "decoder/decode_stage_token_ids_per_token.pt", weights_only=True)
    for t in (seq, idx, tids):
        assert t.shape == (6,) and t.dtype == torch.int64


def test_contract_files_load_and_have_consistent_shapes(tmp_out: Path):
    acc = _make_acc(n_seqs=2, tokens_per_seq=3, vocab=128)
    writer = ContractWriter(tmp_out / "decoder", stage="decoder")
    writer.write(acc)
    d = tmp_out / "decoder"

    tensors = {
        fn: torch.load(d / fn, weights_only=True)
        for fn in CONTRACT_PT_FILES
    }
    expert_selection = tensors["expert_selection.pt"]
    assert expert_selection.shape == (6, 6, 1)
    n_tokens, n_layers, per_token_expert = expert_selection.shape
    num_expert = int(expert_selection.max()) + 1

    assert num_expert == 128
    assert n_layers == NUM_SPARSE_LAYERS
    assert per_token_expert == 1
    for fn in [
        "decode_stage_moe_layer_logits_per_token.pt",
        "decode_stage_moe_layer_gate_logits_per_token.pt",
        "decode_stage_expert_freq_per_token.pt",
    ]:
        assert tensors[fn].shape == (n_tokens, n_layers, num_expert)
    for fn in [
        "decode_stage_token_ids_per_token.pt",
        "decode_stage_seq_id_of_token.pt",
        "decode_stage_token_idx_in_seq.pt",
    ]:
        assert tensors[fn].shape == (n_tokens,)


def test_metadata_records_stage(tmp_out: Path):
    acc = _make_acc()
    writer = ContractWriter(tmp_out / "encoder", stage="encoder")
    writer.write(acc, extra_metadata={"foo": "bar"})
    meta = json.loads((tmp_out / "encoder/metadata.json").read_text())
    assert meta["stage"] == "encoder"
    assert meta["num_expert"] == 128
    assert meta["num_moe_layer"] == 6
    assert meta["per_token_expert"] == 1
    assert meta["N"] == 6
    assert meta["foo"] == "bar"


def test_empty_accumulator_raises(tmp_out: Path):
    acc = TraceAccumulator()
    w = ContractWriter(tmp_out / "empty", stage="decoder")
    with pytest.raises(ValueError, match="empty trace"):
        w.write(acc)
    assert not (tmp_out / "empty").exists()


def test_incomplete_expert_coverage_raises(tmp_out: Path):
    """All argmax on expert 0 => max+1 < V; writer must not emit a self-contradicting trace dir."""
    acc = TraceAccumulator(num_experts=128)
    for s in range(1):
        for k in range(2):
            layers = [_make_layer_logits(0, 128) for _ in range(NUM_SPARSE_LAYERS)]
            acc.add_token(seq_id=s, token_idx_in_seq=k, token_id=100 + k, per_layer_logits=layers)
    w = ContractWriter(tmp_out / "bad", stage="decoder")
    with pytest.raises(ValueError, match="do not cover the full last-dim"):
        w.write(acc)
    assert not (tmp_out / "bad").exists()
