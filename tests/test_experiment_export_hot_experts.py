from __future__ import annotations

import importlib.util
import json
import sys
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


def test_infer_router_topk_defaults_missing_num_selected_experts_for_switch_transformers():
    module = _load_module()

    assert module.resolve_router_topk("auto", SimpleNamespace(model_type="switch_transformers")) == 1


def test_infer_router_topk_auto_requires_explicit_config_for_unknown_model_type():
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


def test_router_counter_counts_nllb_top2_router_probs():
    module = _load_module()
    collector = module.RouterHotExpertCollector(router_topk=2)
    router_probs = torch.tensor(
        [
            [0.7, 0.0, 0.3, 0.0],
            [0.0, 0.6, 0.0, 0.4],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    collector.add_router_mask("encoder.layers.3.ffn.router", router_probs)

    info = collector.usage_summary()["encoder.layers.3.ffn.router"]
    assert info["token_total_hits"] == 5
    assert info["token_hit_freq"] == {"2": 2, "0": 1, "1": 1, "3": 1}
    assert info["top_token_eids"] == [2, 0, 1, 3]


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


def test_parse_max_gpu_memory_applies_single_value_to_all_gpus():
    module = _load_module()

    parsed = module.parse_max_gpu_memory("70GiB", gpu_count=2)

    assert parsed == {0: "70GiB", 1: "70GiB"}


def test_parse_max_gpu_memory_accepts_indexed_values():
    module = _load_module()

    parsed = module.parse_max_gpu_memory("0:70GiB,1:68GiB", gpu_count=4)

    assert parsed == {0: "70GiB", 1: "68GiB"}


def test_build_max_memory_auto_uses_selected_gpu_with_reserve(monkeypatch):
    module = _load_module()

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 2

        @staticmethod
        def mem_get_info(index):
            free = [80 * 1024**3, 40 * 1024**3][index]
            total = free
            return free, total

    monkeypatch.setattr(module.torch, "cuda", FakeCuda)

    max_memory = module.build_max_memory(
        device_map="auto",
        device="cuda:1",
        device_map_gpus="device",
        max_gpu_memory=None,
        max_cpu_memory="256GiB",
        gpu_memory_reserve_mib=1024,
    )

    assert max_memory == {
        1: "39936MiB",
        "cpu": "256GiB",
    }


def test_build_max_memory_single_cap_uses_selected_gpu(monkeypatch):
    module = _load_module()

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 2

        @staticmethod
        def mem_get_info(index):
            raise AssertionError("explicit max_gpu_memory should not probe GPU memory")

    monkeypatch.setattr(module.torch, "cuda", FakeCuda)

    max_memory = module.build_max_memory(
        device_map="auto",
        device="cuda:1",
        device_map_gpus="device",
        max_gpu_memory="70GiB",
        max_cpu_memory="256GiB",
        gpu_memory_reserve_mib=1024,
    )

    assert max_memory == {1: "70GiB", "cpu": "256GiB"}


def test_build_max_memory_can_still_use_all_visible_gpus(monkeypatch):
    module = _load_module()

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 2

        @staticmethod
        def mem_get_info(index):
            free = [80 * 1024**3, 40 * 1024**3][index]
            total = free
            return free, total

    monkeypatch.setattr(module.torch, "cuda", FakeCuda)

    max_memory = module.build_max_memory(
        device_map="auto",
        device="cuda:1",
        device_map_gpus="all",
        max_gpu_memory=None,
        max_cpu_memory="256GiB",
        gpu_memory_reserve_mib=1024,
    )

    assert max_memory == {
        0: "80896MiB",
        1: "39936MiB",
        "cpu": "256GiB",
    }


def test_load_hf_model_uses_device_map_without_model_to(monkeypatch, tmp_path: Path):
    module = _load_module()
    calls = {}

    class FakeConfig:
        torch_dtype = "float16"
        decoder_start_token_id = 0

    class FakeTokenizer:
        pad_token_id = 0

    class FakeModel:
        config = FakeConfig()
        generation_config = SimpleNamespace(decoder_start_token_id=0, bos_token_id=None)

        def to(self, _device):
            calls["to_called"] = True
            return self

        def eval(self):
            calls["eval_called"] = True
            return self

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(path, trust_remote_code):
            calls["config"] = (path, trust_remote_code)
            return FakeConfig()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path, trust_remote_code):
            calls["tokenizer"] = (path, trust_remote_code)
            return FakeTokenizer()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls["model_path"] = path
            calls["model_kwargs"] = kwargs
            return FakeModel()

    fake_transformers = SimpleNamespace(
        AutoConfig=FakeAutoConfig,
        AutoModelForSeq2SeqLM=FakeAutoModel,
        AutoTokenizer=FakeAutoTokenizer,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    model, tokenizer, config = module.load_hf_model(
        tmp_path,
        device="cuda:0",
        dtype_name="auto",
        device_map="auto",
        max_memory={0: "70GiB", "cpu": "256GiB"},
        offload_folder=tmp_path / "offload",
    )

    assert isinstance(model, FakeModel)
    assert isinstance(tokenizer, FakeTokenizer)
    assert isinstance(config, FakeConfig)
    assert "to_called" not in calls
    assert calls["eval_called"]
    assert calls["model_kwargs"]["device_map"] == "auto"
    assert calls["model_kwargs"]["max_memory"] == {0: "70GiB", "cpu": "256GiB"}
    assert calls["model_kwargs"]["offload_folder"] == str(tmp_path / "offload")
    assert calls["model_kwargs"]["offload_state_dict"] is True
    assert calls["model_kwargs"]["low_cpu_mem_usage"] is True


def test_load_sparse_cache_model_patches_transformers_and_uses_device_map_zero(monkeypatch, tmp_path: Path):
    module = _load_module()
    calls = {}

    class FakeConfig:
        torch_dtype = "float32"
        decoder_start_token_id = 2
        bos_token_id = None
        model_type = "nllb-moe"

    class FakeTokenizer:
        pad_token_id = 1
        eos_token = "</s>"

    class FakeModel:
        config = FakeConfig()
        generation_config = SimpleNamespace(decoder_start_token_id=None, bos_token_id=None)

        def eval(self):
            calls["eval_called"] = True
            return self

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(path, trust_remote_code, local_files_only):
            calls["config"] = (path, trust_remote_code, local_files_only)
            return FakeConfig()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path, trust_remote_code, local_files_only):
            calls["tokenizer"] = (path, trust_remote_code, local_files_only)
            return FakeTokenizer()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls["model_path"] = path
            calls["model_kwargs"] = kwargs
            return FakeModel()

    def fake_hack_transformers(**kwargs):
        calls["hack_transformers"] = kwargs

    fake_sparse_llm_cache = SimpleNamespace(
        utils=SimpleNamespace(hack_transformers=fake_hack_transformers)
    )
    fake_transformers = SimpleNamespace(
        AutoConfig=FakeAutoConfig,
        AutoModelForSeq2SeqLM=FakeAutoModel,
        AutoTokenizer=FakeAutoTokenizer,
    )
    monkeypatch.setitem(sys.modules, "sparse_llm_cache", fake_sparse_llm_cache)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    model, tokenizer, config = module.load_sparse_cache_model(
        tmp_path,
        device="cuda:0",
        dtype_name="auto",
        cache_rate=0.01,
        cache_policy="lru",
        per_layer_cache=False,
    )

    assert isinstance(model, FakeModel)
    assert isinstance(tokenizer, FakeTokenizer)
    assert isinstance(config, FakeConfig)
    assert calls["hack_transformers"] == {
        "model_id": str(tmp_path),
        "cache_rate": 0.01,
        "cache_policy": "lru",
        "per_layer_cache": False,
        "num_predict_expert_per_layer": 0,
        "reorder_experts": False,
        "early_preempt": False,
        "chunk_prefetch": False,
        "predict_input_mode": "no_predict",
        "pin_memory": True,
        "enable_model_timer": False,
    }
    assert calls["model_kwargs"]["device_map"] == 0
    assert calls["model_kwargs"]["torch_dtype"] is torch.float32
    assert calls["model_kwargs"]["local_files_only"] is True
    assert calls["eval_called"]


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


def test_extract_router_mask_prefers_nllb_router_probs_for_top2_dispatch():
    module = _load_module()
    top_1_mask = torch.tensor(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ],
        dtype=torch.int64,
    )
    router_probs = torch.tensor(
        [
            [0.7, 0.0, 0.3, 0.0],
            [0.0, 0.6, 0.0, 0.4],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    extracted = module.extract_router_mask(
        (top_1_mask, router_probs),
        num_experts=4,
        model_type="nllb-moe",
    )

    assert extracted is router_probs


def test_extract_router_mask_rejects_invalid_nllb_router_probs_without_top1_fallback():
    module = _load_module()
    top_1_mask = torch.tensor(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ],
        dtype=torch.int64,
    )
    wrong_expert_count = torch.ones(3, 3, dtype=torch.float32)

    not_a_tensor = module.extract_router_mask(
        (top_1_mask, None),
        num_experts=4,
        model_type="nllb-moe",
    )
    wrong_shape = module.extract_router_mask(
        (top_1_mask, wrong_expert_count),
        num_experts=4,
        model_type="nllb-moe",
    )

    assert not_a_tensor is None
    assert wrong_shape is None


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


def test_sparse_cache_nllb_hooks_collect_encoder_and_decoder_router_probs():
    module = _load_module()

    class FakeHandle:
        def __init__(self):
            self.removed = False

        def remove(self):
            self.removed = True

    class FakeSparseMlp:
        def __init__(self, stage: str, stage_layer_id: int, block_id: int):
            self._stage = stage
            self._stage_layer_id = stage_layer_id
            self._nllb_block_id = block_id
            self.handles = []

        def register_forward_hook(self, hook):
            self.hook = hook
            handle = FakeHandle()
            self.handles.append(handle)
            return handle

    class FakeModel:
        def __init__(self):
            self.encoder_mlp = FakeSparseMlp("encoder", 0, 3)
            self.decoder_mlp = FakeSparseMlp("decoder", 0, 3)

        def named_modules(self):
            return [
                ("model.encoder.layers.3.ffn", self.encoder_mlp),
                ("model.decoder.layers.3.ffn", self.decoder_mlp),
            ]

    model = FakeModel()
    collector = module.RouterHotExpertCollector(router_topk=2)

    handles = module.install_sparse_cache_nllb_router_hooks(
        model=model,
        config=SimpleNamespace(model_type="nllb-moe", num_experts=4),
        collector=collector,
    )

    encoder_probs = torch.tensor(
        [
            [0.7, 0.0, 0.3, 0.0],
            [0.0, 0.6, 0.0, 0.4],
        ],
        dtype=torch.float32,
    )
    decoder_probs = torch.tensor(
        [
            [[0.0, 0.5, 0.5, 0.0]],
        ],
        dtype=torch.float32,
    )
    model.encoder_mlp.hook(model.encoder_mlp, (torch.zeros(1, 2, 8),), (torch.zeros(1, 2, 8), (encoder_probs,)))
    model.decoder_mlp.hook(model.decoder_mlp, (torch.zeros(1, 1, 8),), (torch.zeros(1, 1, 8), (decoder_probs,)))

    summary = collector.usage_summary()
    assert summary["encoder.layers.3.ffn.router"]["token_hit_freq"] == {"0": 1, "1": 1, "2": 1, "3": 1}
    assert summary["decoder.layers.3.ffn.router"]["token_hit_freq"] == {"1": 1, "2": 1}
    assert collector.encoder_layers() == {"encoder.layers.3.ffn.router"}
    assert collector.decoder_layers() == {"decoder.layers.3.ffn.router"}
    assert len(handles) == 2


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


def test_collect_hot_experts_sparse_cache_backend_does_not_request_router_logits(monkeypatch):
    module = _load_module()

    class FakeHandle:
        def remove(self):
            pass

    class FakeTokenizer:
        pad_token_id = 0

        def __call__(self, _texts, **_kwargs):
            return {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.tensor([[1, 1]]),
            }

    class FakeModel:
        def __init__(self):
            self.collector = None
            self.generate_kwargs = None

        def generate(self, **kwargs):
            self.generate_kwargs = kwargs
            self.collector.add_router_mask(
                "encoder.layers.3.ffn.router",
                torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
            )

    def fake_install_sparse_cache_nllb_router_hooks(*, model, config, collector):
        model.collector = collector
        return [FakeHandle()]

    monkeypatch.setattr(
        module,
        "install_sparse_cache_nllb_router_hooks",
        fake_install_sparse_cache_nllb_router_hooks,
    )
    model = FakeModel()

    module.collect_hot_experts(
        model=model,
        tokenizer=FakeTokenizer(),
        config=SimpleNamespace(model_type="nllb-moe", num_experts=2),
        prompts=["hello"],
        device="cpu",
        router_topk=2,
        max_new_tokens=1,
        enc_pad_to=512,
        max_samples=1,
        hook_backend="sparse-cache",
    )

    assert "output_router_logits" not in model.generate_kwargs


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
