import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

THIS_DIR = Path(__file__).resolve().parent
MODULE_DIR = THIS_DIR.parent
sys.path.insert(0, str(MODULE_DIR))
from utils import EXPECTED_NUM_EXPERTS, NUM_SPARSE_LAYERS, SwitchRunner, TraceAccumulator  # noqa: E402


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


def test_run_records_generated_decoder_tokens_not_decoder_start(switch_model_path):
    prompts = ["Translate: hello", "Summarize: the quick brown fox"]
    max_new_tokens = 4
    runner = SwitchRunner(model_path=str(switch_model_path), device="cpu", seed=42)
    _, dec_acc = runner.run(prompts=prompts, max_new_tokens=max_new_tokens, batch_size=2)

    encoded = runner.tokenizer(prompts, padding="longest", return_tensors="pt")
    input_ids = encoded["input_ids"].to(runner.device)
    attn_mask = encoded["attention_mask"].to(runner.device)
    with torch.inference_mode():
        output = runner.model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            decoder_start_token_id=runner._decoder_start_token_id(),
            output_scores=False,
            return_dict_in_generate=True,
        )

    sequences = output.sequences.cpu()
    eos_id = runner.model.config.eos_token_id
    pad_id = runner.tokenizer.pad_token_id
    expected = []
    alive = [True] * len(prompts)
    for step_idx in range(sequences.shape[1] - 1):
        for batch_idx in range(len(prompts)):
            if not alive[batch_idx]:
                continue
            generated_token = int(sequences[batch_idx, step_idx + 1])
            if pad_id is not None and generated_token == pad_id:
                alive[batch_idx] = False
                continue
            expected.append(generated_token)
            if eos_id is not None and generated_token == eos_id:
                alive[batch_idx] = False

    assert dec_acc.token_ids() == expected
    assert dec_acc.token_ids()[0] != int(sequences[0, 0])


def test_drain_decoder_rejects_inconsistent_layer_steps(runner_cpu):
    runner_cpu._decoder_acc = TraceAccumulator()
    runner_cpu._decoder_step_buffer = {
        layer_id: [torch.zeros((1, 1, EXPECTED_NUM_EXPERTS), dtype=torch.float32)]
        for layer_id in range(NUM_SPARSE_LAYERS)
    }
    runner_cpu._decoder_step_buffer[0].append(
        torch.zeros((1, 1, EXPECTED_NUM_EXPERTS), dtype=torch.float32)
    )

    with pytest.raises(RuntimeError, match="decoder sparse layer step counts differ"):
        runner_cpu._drain_decoder(
            global_seq_ids=[0],
            gen_seqs=torch.tensor([[0, 10, 1]], dtype=torch.long),
            pad_id=0,
            eos_id=1,
            decoder_start=0,
        )
