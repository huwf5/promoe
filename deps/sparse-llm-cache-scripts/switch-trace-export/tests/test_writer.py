import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

THIS_DIR = Path(__file__).resolve().parent
MODULE_DIR = THIS_DIR.parent
sys.path.insert(0, str(MODULE_DIR))
from utils import NUM_SPARSE_LAYERS, ContractWriter, TraceAccumulator  # noqa: E402

_TRAIN_UTILS_REL = Path("deps") / "sparse-llm-cache-scripts" / "train-predict-model" / "utils.py"


def resolve_train_predict_utils_path(start: Path | None = None) -> Path:
    """Find train-predict-model/utils.py: same-repo `deps/...` under an ancestor, walk upward."""
    here = (start or THIS_DIR).resolve()
    for d in (here, *here.parents):
        cand = d / _TRAIN_UTILS_REL
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"could not find train-predict-model utils at */{_TRAIN_UTILS_REL!s} starting from {here}"
    )


def _load_trace_class():
    p = resolve_train_predict_utils_path()
    spec = importlib.util.spec_from_file_location("train_predict_model_utils", p)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Trace


Trace = _load_trace_class()


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
    for fn in [
        "expert_selection.pt",
        "decode_stage_moe_layer_logits_per_token.pt",
        "decode_stage_moe_layer_gate_logits_per_token.pt",
        "decode_stage_expert_freq_per_token.pt",
        "decode_stage_token_ids_per_token.pt",
        "decode_stage_seq_id_of_token.pt",
        "decode_stage_token_idx_in_seq.pt",
        "metadata.json",
    ]:
        assert (d / fn).exists(), fn


def test_shapes_and_dtypes(tmp_out: Path):
    acc = _make_acc(n_seqs=2, tokens_per_seq=3, vocab=128)
    writer = ContractWriter(tmp_out / "decoder", stage="decoder")
    writer.write(acc)

    sel = torch.load(tmp_out / "decoder/expert_selection.pt")
    assert sel.shape == (6, 6, 1) and sel.dtype == torch.int64

    feat = torch.load(tmp_out / "decoder/decode_stage_moe_layer_logits_per_token.pt")
    gate = torch.load(tmp_out / "decoder/decode_stage_moe_layer_gate_logits_per_token.pt")
    freq = torch.load(tmp_out / "decoder/decode_stage_expert_freq_per_token.pt")
    for t in (feat, gate, freq):
        assert t.shape == (6, 6, 128) and t.dtype == torch.float32

    assert torch.equal(feat, gate), "input feature must equal gate logits per design"
    row_sums = freq.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    seq = torch.load(tmp_out / "decoder/decode_stage_seq_id_of_token.pt")
    idx = torch.load(tmp_out / "decoder/decode_stage_token_idx_in_seq.pt")
    tids = torch.load(tmp_out / "decoder/decode_stage_token_ids_per_token.pt")
    for t in (seq, idx, tids):
        assert t.shape == (6,) and t.dtype == torch.int64


def test_unpack_from_dir_works(tmp_out: Path):
    acc = _make_acc(n_seqs=2, tokens_per_seq=3, vocab=128)
    writer = ContractWriter(tmp_out / "decoder", stage="decoder")
    writer.write(acc)
    trace = Trace()
    trace.unpack_from_dir(str(tmp_out / "decoder"))
    trace.prepare_tensors()
    assert trace.num_expert == 128
    assert trace.num_moe_layer == 6
    assert trace.per_token_expert == 1


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
