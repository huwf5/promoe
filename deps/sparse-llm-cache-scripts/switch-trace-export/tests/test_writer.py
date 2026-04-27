import importlib.util
import json
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
MODULE_DIR = THIS_DIR.parent
sys.path.insert(0, str(MODULE_DIR))
from utils import NUM_SPARSE_LAYERS, ContractWriter, TraceAccumulator  # noqa: E402

# Worktree may only vendor `switch-trace-export/`; load train_predict Trace from main promoe tree.
_PROMOE_ROOT = THIS_DIR.parent.parent.parent.parent.parent.parent
TRAIN_UTILS_PATH = _PROMOE_ROOT / "deps" / "sparse-llm-cache-scripts" / "train-predict-model" / "utils.py"
_spec = importlib.util.spec_from_file_location("train_predict_model_utils", TRAIN_UTILS_PATH)
assert _spec and _spec.loader
_train_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train_utils)
Trace = _train_utils.Trace


def _make_acc(n_seqs: int = 2, tokens_per_seq: int = 3, vocab: int = 128) -> TraceAccumulator:
    acc = TraceAccumulator(num_experts=vocab)
    tok = 1000
    for s in range(n_seqs):
        for k in range(tokens_per_seq):
            layers = [
                torch.linspace(s + 0.1 * i, s + 0.1 * i + 0.5, vocab, dtype=torch.float32)
                for i in range(NUM_SPARSE_LAYERS)
            ]
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
