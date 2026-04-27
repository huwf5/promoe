import sys
from pathlib import Path

import pytest

THIS_DIR = Path(__file__).resolve().parent
MODULE_DIR = THIS_DIR.parent
sys.path.insert(0, str(MODULE_DIR))
from utils import NUM_SPARSE_LAYERS, SwitchRunner  # noqa: E402


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
