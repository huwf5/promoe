from types import SimpleNamespace
import json

import pytest
import torch

from sparse_llm_cache.model_adapters.switch import SwitchAdapter
from sparse_llm_cache.utils.hooks import MoeLayerHook
from tests.test_switch_adapter import _switch_config


class RecordingPrefetchMngr:
    def __init__(self):
        self.logit_reports = []
        self.done_layers = []

    def report_moe_layer_logits(self, layer_id, logits):
        self.logit_reports.append((layer_id, logits))

    def one_moe_layer_done(self, layer_id):
        self.done_layers.append(layer_id)


class GlobalDecoderAdapter:
    def __init__(self, first_decoder_layer):
        self.first_decoder_layer = first_decoder_layer

    def should_report_moe_layer_to_predictor(self, stage, global_layer_id):
        return stage == "decoder"

    def should_report_predictor_pre_forward(self, stage, global_layer_id):
        return stage == "decoder" and global_layer_id == self.first_decoder_layer

    def predictor_input_id_before_layer(self, stage, global_layer_id):
        return global_layer_id

    def predictor_input_id_after_layer(self, stage, global_layer_id):
        return global_layer_id + 1

    def extract_moe_layer_input_for_predictor(self, *args, **kwargs):
        return args[0]

    def extract_moe_layer_output_for_predictor(self, output):
        return output


def test_moe_layer_hook_reports_first_decoder_pre_forward_global_id():
    prefetch = RecordingPrefetchMngr()
    hook = MoeLayerHook(prefetch, adapter=GlobalDecoderAdapter(first_decoder_layer=6))
    module = SimpleNamespace(_stage="decoder", _layer_id=6)
    x = torch.zeros(1, 1, 4)

    hook.pre_forward(module, x)

    assert prefetch.logit_reports == [(6, x)]


def test_moe_layer_hook_reports_decoder_post_forward_next_global_boundary():
    prefetch = RecordingPrefetchMngr()
    hook = MoeLayerHook(prefetch, adapter=GlobalDecoderAdapter(first_decoder_layer=6))
    module = SimpleNamespace(_stage="decoder", _layer_id=7)
    out = torch.ones(1, 1, 4)

    hook.post_forward(module, out)

    assert prefetch.logit_reports == [(8, out)]
    assert prefetch.done_layers == [7]


def test_moe_layer_hook_skips_encoder_predictor_report():
    prefetch = RecordingPrefetchMngr()
    hook = MoeLayerHook(prefetch, adapter=GlobalDecoderAdapter(first_decoder_layer=6))
    module = SimpleNamespace(_stage="encoder", _layer_id=1)
    x = torch.zeros(1, 1, 4)

    hook.pre_forward(module, x)
    hook.post_forward(module, x)

    assert prefetch.logit_reports == []
    assert prefetch.done_layers == []


def _write_global_predictor_dir(path, e=6, d=6, predictor_type="sep"):
    path.mkdir(exist_ok=True)
    l = e + d
    outputs = {}
    for src in range(e, l):
        outputs[str(src)] = [src, min(src + 2, l)]
    outputs[str(l)] = [e, min(e + 2, l)]
    (path / "metas.json").write_text(json.dumps({
        "schema_version": 2,
        "id_space": "global",
        "predict_stage": "decoder",
        "num_layer": l,
        "num_encoder_moe_layer": e,
        "num_decoder_moe_layer": d,
        "outputs": outputs,
    }))
    for src, (start, stop) in outputs.items():
        if predictor_type == "legacy":
            (path / f"{src}.pt").write_bytes(b"placeholder")
        else:
            for dst in range(start, stop):
                (path / f"{src}-{dst}.pt").write_bytes(b"placeholder")


def test_switch_validate_predictor_path_accepts_global_v2(tmp_path):
    adapter = SwitchAdapter(SimpleNamespace(config=_switch_config()), "google/switch-base-128")
    _write_global_predictor_dir(tmp_path)

    adapter.validate_predictor_path(str(tmp_path), num_predict_expert_per_layer=2, predictor_type="sep")


def test_switch_validate_predictor_path_rejects_local_v1(tmp_path):
    adapter = SwitchAdapter(SimpleNamespace(config=_switch_config()), "google/switch-base-128")
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "metas.json").write_text(json.dumps({"0": [0, 2], "1": [1, 3]}))
    (tmp_path / "0-0.pt").write_bytes(b"placeholder")

    with pytest.raises(ValueError, match="global"):
        adapter.validate_predictor_path(str(tmp_path), num_predict_expert_per_layer=2, predictor_type="sep")


def test_switch_validate_predictor_path_accepts_empty_global_output_range(tmp_path):
    adapter = SwitchAdapter(SimpleNamespace(config=_switch_config()), "google/switch-base-128")
    _write_global_predictor_dir(tmp_path)
    data = json.loads((tmp_path / "metas.json").read_text())
    data["outputs"]["12"] = [12, 12]
    (tmp_path / "metas.json").write_text(json.dumps(data))

    adapter.validate_predictor_path(str(tmp_path), num_predict_expert_per_layer=2, predictor_type="sep")


def test_switch_validate_predictor_path_accepts_empty_legacy_output_range_without_model_file(tmp_path):
    adapter = SwitchAdapter(SimpleNamespace(config=_switch_config()), "google/switch-base-128")
    _write_global_predictor_dir(tmp_path, predictor_type="legacy")
    data = json.loads((tmp_path / "metas.json").read_text())
    data["outputs"]["12"] = [12, 12]
    (tmp_path / "metas.json").write_text(json.dumps(data))
    (tmp_path / "12.pt").unlink()

    adapter.validate_predictor_path(str(tmp_path), num_predict_expert_per_layer=2, predictor_type="legacy")
