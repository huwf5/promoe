from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "experiment/scripts/trace/export_hot_experts.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("experiment_export_hot_experts", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_output_path_uses_model_dataset_task_split_layout():
    module = _load_module()
    repo_root = Path("/tmp/repo")

    output_path = module.default_hot_expert_output_path(
        repo_root=repo_root,
        model_name="switch-base-128",
        dataset_name="mmlu",
        task_name="professional_law",
        split="test",
    )

    assert output_path == (
        repo_root
        / "experiment/traces/switch-base-128-mmlu-professional_law-test/hot_experts/switch-base-128.test.json"
    )


def test_load_prompts_prefers_pt_file(tmp_path: Path):
    module = _load_module()
    prompt_root = tmp_path / "mmlu" / "professional_law" / "test"
    prompt_root.mkdir(parents=True)
    torch.save(["q1", "q2"], prompt_root / "prompt_list.pt")
    (prompt_root / "prompt_list.txt").write_text("ignored\n", encoding="utf-8")

    prompts = module.load_prompts(prompt_root)

    assert prompts == ["q1", "q2"]


def test_load_prompts_falls_back_to_txt(tmp_path: Path):
    module = _load_module()
    prompt_root = tmp_path / "mmlu" / "professional_law" / "validation"
    prompt_root.mkdir(parents=True)
    (prompt_root / "prompt_list.txt").write_text("line1\\n\nline2\n", encoding="utf-8")

    prompts = module.load_prompts(prompt_root)

    assert prompts == ["line1\n", "line2"]


def test_infer_router_topk_uses_num_selected_experts():
    module = _load_module()

    assert module.resolve_router_topk("auto", SimpleNamespace(num_selected_experts=2)) == 2


def test_infer_router_topk_uses_second_expert_policy():
    module = _load_module()

    assert module.resolve_router_topk("auto", SimpleNamespace(second_expert_policy="all")) == 2


def test_infer_router_topk_auto_requires_explicit_config():
    module = _load_module()

    with pytest.raises(ValueError, match="Cannot infer router_topk"):
        module.resolve_router_topk("auto", SimpleNamespace())


def test_infer_router_topk_accepts_manual_positive_int():
    module = _load_module()

    assert module.resolve_router_topk("3", SimpleNamespace()) == 3


def test_router_counter_counts_actual_dispatch_mask_experts():
    module = _load_module()
    collector = module.RouterHotExpertCollector(router_topk=2)
    router_mask = torch.tensor(
        [
            [
                [0, 1, 0],
                [0, 0, 0],
                [1, 0, 1],
            ]
        ]
    )

    collector.add_router_mask("encoder.block.0.router", router_mask)

    summary = collector.usage_summary()
    info = summary["encoder.block.0.router"]
    assert info["token_total_hits"] == 3
    assert info["token_hit_freq"] == {"0": 1, "1": 1, "2": 1}
    assert info["top_token_eids"] == [0, 1, 2]


def test_resolve_torch_dtype_accepts_supported_aliases():
    module = _load_module()

    assert module.resolve_torch_dtype("float32") is torch.float32
    assert module.resolve_torch_dtype("fp16") is torch.float16
    assert module.resolve_torch_dtype("bf16") is torch.bfloat16


def test_resolve_torch_dtype_auto_reads_config_torch_dtype():
    module = _load_module()

    assert module.resolve_torch_dtype("auto", SimpleNamespace(torch_dtype="bfloat16")) is torch.bfloat16


def test_resolve_torch_dtype_auto_requires_config_field():
    module = _load_module()

    with pytest.raises(ValueError, match="Cannot infer torch dtype"):
        module.resolve_torch_dtype("auto", SimpleNamespace())


def test_decoder_start_token_id_falls_back_to_tokenizer_pad_id():
    module = _load_module()
    model = SimpleNamespace(
        generation_config=SimpleNamespace(decoder_start_token_id=None, bos_token_id=None),
        config=SimpleNamespace(decoder_start_token_id=None, bos_token_id=None),
    )
    tokenizer = SimpleNamespace(pad_token_id=0)

    assert module._decoder_start_token_id(model, tokenizer) == 0


def test_generation_kwargs_includes_decoder_start_token_id():
    module = _load_module()
    model = SimpleNamespace(
        generation_config=SimpleNamespace(decoder_start_token_id=None, bos_token_id=None),
        config=SimpleNamespace(decoder_start_token_id=None, bos_token_id=None),
    )
    tokenizer = SimpleNamespace(pad_token_id=0)

    kwargs = module._generation_kwargs(model, tokenizer, max_new_tokens=7)

    assert kwargs["decoder_start_token_id"] == 0
    assert kwargs["pad_token_id"] == 0
    assert kwargs["max_new_tokens"] == 7


def test_extract_router_mask_accepts_router_tuple_output():
    module = _load_module()
    mask = torch.zeros(2, 4, 8)
    probs = torch.ones(2, 4, 1)
    logits = torch.randn(2, 4, 8)

    extracted = module.extract_router_mask((mask, probs, logits), num_experts=8)

    assert extracted is mask


def test_extract_router_mask_rejects_plain_classifier_logits():
    module = _load_module()
    logits = torch.randn(2, 4, 8)

    extracted = module.extract_router_mask(logits, num_experts=8)

    assert extracted is None


def test_router_hook_selection_is_limited_to_router_modules():
    module = _load_module()

    assert module._looks_like_router_module("encoder.block.1.layer.1.mlp.router")
    assert not module._looks_like_router_module("encoder.block.1.layer.1.mlp")
    assert not module._looks_like_router_module("encoder.block.1.layer.1.mlp.router.classifier")


def test_build_payload_matches_expected_hot_expert_shape():
    module = _load_module()
    usage_summary = {
        "encoder.block.0.layer.1.mlp.router.classifier": {
            "token_total_hits": 5,
            "top_token_eids": [3, 1],
            "token_hit_freq": {"3": 4, "1": 1},
        },
        "decoder.block.0.layer.2.mlp.router.classifier": {
            "token_total_hits": 7,
            "top_token_eids": [6, 2],
            "token_hit_freq": {"6": 5, "2": 2},
        },
        "ignored.router": {
            "token_total_hits": 99,
            "top_token_eids": [0],
            "token_hit_freq": {"0": 99},
        },
    }
    frozen = {
        "encoder.block.0.layer.1.mlp.router.classifier": [3, 1],
        "decoder.block.0.layer.2.mlp.router.classifier": [6, 2],
    }

    payload = module.build_hot_expert_payload(
        model_name="switch-base-128",
        dataset_name="mmlu",
        task_name="professional_law",
        split="test",
        actual_samples=12,
        frozen=frozen,
        usage_summary=usage_summary,
        encoder_layers={"encoder.block.0.layer.1.mlp.router.classifier"},
        decoder_layers={"decoder.block.0.layer.2.mlp.router.classifier"},
        router_topk=2,
        selection_rule="router_mask.nonzero",
        counting_unit="token_dispatched_expert",
    )

    assert payload["model"] == "switch-base-128"
    assert payload["dataset"] == "mmlu"
    assert payload["task_name"] == "professional_law"
    assert payload["split"] == "test"
    assert payload["actual_samples"] == 12
    assert payload["router_topk"] == 2
    assert payload["selection_rule"] == "router_mask.nonzero"
    assert payload["counting_unit"] == "token_dispatched_expert"
    assert payload["frozen_hot_experts"] == frozen
    assert payload["token_hits_by_stage"] == {"decoder": 7, "encoder": 5}
    assert "ignored.router" not in payload["expert_usage_summary"]
    assert payload["expert_usage_summary"]["encoder.block.0.layer.1.mlp.router.classifier"]["top_token_counts"] == [
        {"eid": 3, "count": 4},
        {"eid": 1, "count": 1},
    ]


def test_collect_hot_experts_uses_small_demo_dynamic_padding(monkeypatch):
    module = _load_module()

    class FakeHandle:
        def __init__(self):
            self.removed = False

        def remove(self):
            self.removed = True

    class FakeTokenizer:
        pad_token_id = 0

        def __init__(self):
            self.calls = []

        def __call__(self, texts, **kwargs):
            self.calls.append({"texts": texts, **kwargs})
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }

    class FakeModel:
        def __init__(self):
            self.collector = None

        def generate(self, **_kwargs):
            self.collector.add_router_mask(
                "encoder.block.0.layer.1.mlp.router",
                torch.tensor([[[0, 1], [1, 0]]]),
            )

    fake_handle = FakeHandle()

    def fake_install_router_hooks(*, model, config, collector):
        model.collector = collector
        return [fake_handle]

    monkeypatch.setattr(module, "install_router_hooks", fake_install_router_hooks)
    tokenizer = FakeTokenizer()

    actual, _frozen, summary, encoder_layers, decoder_layers, _num_experts = module.collect_hot_experts(
        model=FakeModel(),
        tokenizer=tokenizer,
        config=SimpleNamespace(num_experts=2),
        prompts=["hello"],
        device="cpu",
        router_topk=1,
        max_new_tokens=32,
        enc_pad_to=512,
        max_samples=1,
    )

    assert actual == 1
    assert tokenizer.calls == [
        {
            "texts": ["hello"],
            "truncation": True,
            "padding": True,
            "max_length": 512,
            "return_tensors": "pt",
        }
    ]
    assert summary["encoder.block.0.layer.1.mlp.router"]["token_hit_freq"] == {"0": 1, "1": 1}
    assert encoder_layers == {"encoder.block.0.layer.1.mlp.router"}
    assert decoder_layers == set()
    assert fake_handle.removed


def test_save_hot_expert_snapshot_writes_json(tmp_path: Path):
    module = _load_module()
    save_path = tmp_path / "trace/hot_experts/switch-base-128.test.json"
    payload = {"model": "switch-base-128", "frozen_hot_experts": {"layer": [1, 2]}}

    written_path = module.save_hot_expert_snapshot(payload=payload, save_path=save_path)

    assert written_path == save_path
    assert json.loads(save_path.read_text(encoding="utf-8")) == payload


def test_export_script_does_not_depend_on_sida_or_predictor_manager():
    text = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "sida" not in text
    assert "predictor_manager" not in text
    assert "WithPredictor" not in text
