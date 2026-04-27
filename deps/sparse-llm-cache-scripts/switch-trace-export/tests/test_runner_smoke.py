import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

THIS_DIR = Path(__file__).resolve().parent
MODULE_DIR = THIS_DIR.parent
sys.path.insert(0, str(MODULE_DIR))
from utils import EXPECTED_NUM_EXPERTS, NUM_SPARSE_LAYERS, SwitchRunner  # noqa: E402


@pytest.fixture
def runner_cpu(switch_model_path):
    return SwitchRunner(model_path=str(switch_model_path), device="cpu", seed=42)


def test_install_hooks_finds_six_per_stage(runner_cpu):
    runner_cpu._load_model()
    runner_cpu._install_hooks()
    enc = [
        m
        for m in runner_cpu.model.modules()
        if getattr(m, "_promoe_stage", None) == "encoder"
    ]
    dec = [
        m
        for m in runner_cpu.model.modules()
        if getattr(m, "_promoe_stage", None) == "decoder"
    ]
    assert len(enc) == NUM_SPARSE_LAYERS, f"encoder sparse modules = {len(enc)}"
    assert len(dec) == NUM_SPARSE_LAYERS, f"decoder sparse modules = {len(dec)}"
    enc_ids = sorted(m._promoe_sparse_layer_id for m in enc)
    dec_ids = sorted(m._promoe_sparse_layer_id for m in dec)
    assert enc_ids == list(range(NUM_SPARSE_LAYERS))
    assert dec_ids == list(range(NUM_SPARSE_LAYERS))


def test_jitter_noise_zeroed(runner_cpu):
    runner_cpu._load_model()
    runner_cpu._install_hooks()
    from transformers.models.switch_transformers.modeling_switch_transformers import (
        SwitchTransformersTop1Router,
    )

    routers = [
        m
        for m in runner_cpu.model.modules()
        if isinstance(m, SwitchTransformersTop1Router)
    ]
    assert len(routers) == 2 * NUM_SPARSE_LAYERS
    for r in routers:
        assert r.jitter_noise == 0.0


def test_install_hooks_is_idempotent(runner_cpu):
    runner_cpu._load_model()
    runner_cpu._install_hooks()
    runner_cpu._install_hooks()
    assert len(runner_cpu._hook_handles) == 2 * NUM_SPARSE_LAYERS


def test_sparse_mlp_hook_accepts_nested_router_logits(runner_cpu):
    logits = torch.zeros((2, 3, EXPECTED_NUM_EXPERTS), dtype=torch.float32)
    module = SimpleNamespace(_promoe_stage="encoder", _promoe_sparse_layer_id=0)
    hidden_states = torch.zeros((2, 3, 4), dtype=torch.float32)

    runner_cpu._sparse_mlp_hook(module, (hidden_states,), (hidden_states, (logits, None)))

    assert runner_cpu._encoder_slot_buffer[0].shape == (2, 3, EXPECTED_NUM_EXPERTS)


def test_forward_populates_encoder_hook_buffers(runner_cpu):
    runner_cpu._load_model()
    runner_cpu._install_hooks()
    encoded = runner_cpu.tokenizer("hello", return_tensors="pt").to(runner_cpu.device)

    with torch.no_grad():
        runner_cpu.model.encoder(**encoded)

    assert runner_cpu._encoder_slot_buffer
    for layer_id, tensor in runner_cpu._encoder_slot_buffer.items():
        assert 0 <= layer_id < NUM_SPARSE_LAYERS
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape[-1] == EXPECTED_NUM_EXPERTS


def test_run_two_prompts_produces_records(switch_model_path):
    runner = SwitchRunner(model_path=str(switch_model_path), device="cpu", seed=42)
    enc_acc, dec_acc = runner.run(
        prompts=["Translate: hello", "Summarize: the quick brown fox"],
        max_new_tokens=4,
        batch_size=2,
    )
    # encoder: every prompt token recorded
    assert enc_acc.total_tokens() > 0
    assert max(enc_acc.seq_ids()) == 1
    # decoder: at most 2*4 = 8 tokens (less if EOS)
    assert 0 < dec_acc.total_tokens() <= 8
    assert max(dec_acc.seq_ids()) == 1
    # token_idx within seq starts at 0
    for s in (0, 1):
        s_idx = [i for i, ss in enumerate(dec_acc.seq_ids()) if ss == s]
        if s_idx:
            assert dec_acc.token_idx_in_seq()[s_idx[0]] == 0
