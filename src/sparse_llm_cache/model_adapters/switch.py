from __future__ import annotations

import json
import re
from pathlib import Path

from sparse_llm_cache.utils.filter import Filter, RegexFilter

from .base import ModelAdapter


class SwitchSparseMlpFilter(Filter):
  def __init__(self, adapter):
    self.adapter = adapter

  def __call__(self, name, *args, **kwargs):
    return self.adapter.parse_moe_layer_name(name) is not None


class SwitchAdapter(ModelAdapter):
  _EXPERT_NAME_PATTERN = (
    r".*(encoder)\.block\.(\d+)\.layer\.1\.mlp\.experts\.expert_(\d+)$"
    r"|.*(decoder)\.block\.(\d+)\.layer\.2\.mlp\.experts\.expert_(\d+)$"
  )
  _MLP_NAME_PATTERN = (
    r".*(encoder)\.block\.(\d+)\.layer\.1\.mlp$"
    r"|.*(decoder)\.block\.(\d+)\.layer\.2\.mlp$"
  )

  def __init__(self, model, model_id: str):
    super().__init__(model, model_id)
    cfg = model.config
    if getattr(cfg, "model_type", None) != "switch_transformers":
      raise ValueError("SwitchAdapter requires model.config.model_type == 'switch_transformers'")
    if int(getattr(cfg, "num_selected_experts", 0)) != 1:
      raise ValueError("SwitchAdapter currently supports Switch top-1 routing only")

    self.num_encoder_sparse_layers = int(cfg.num_sparse_encoder_layers)
    self.num_decoder_sparse_layers = int(cfg.num_sparse_decoder_layers)
    self.encoder_sparse_step = int(cfg.encoder_sparse_step)
    self.decoder_sparse_step = int(cfg.decoder_sparse_step)
    self.num_encoder_layers = int(cfg.num_layers)
    self.num_decoder_layers = int(cfg.num_decoder_layers)
    self._num_experts = int(cfg.num_experts)
    self._num_selected_experts = int(cfg.num_selected_experts)

    self.encoder_sparse_layer_ids = self._expected_sparse_layer_ids(
      self.num_encoder_layers,
      self.encoder_sparse_step,
      self.num_encoder_sparse_layers,
    )
    self.decoder_sparse_layer_ids = self._expected_sparse_layer_ids(
      self.num_decoder_layers,
      self.decoder_sparse_step,
      self.num_decoder_sparse_layers,
    )

    self._expert_filter = RegexFilter(self._EXPERT_NAME_PATTERN)
    self._moe_mlp_filter = SwitchSparseMlpFilter(self)
    self._moe_layer_filter = self._moe_mlp_filter
    self._moe_attn_filter = RegexFilter(r"^$")

  @staticmethod
  def _expected_sparse_layer_ids(num_layers: int, sparse_step: int, expected_count: int) -> list[int]:
    ids = list(range(sparse_step - 1, num_layers, sparse_step))
    if len(ids) != expected_count:
      raise ValueError(
        f"Switch sparse layer config mismatch: derived {ids}, expected {expected_count} sparse layers"
      )
    return ids

  @property
  def num_moe_layer(self) -> int:
    return self.num_encoder_sparse_layers + self.num_decoder_sparse_layers

  @property
  def num_expert_per_layer(self) -> int:
    return self._num_experts

  @property
  def num_expert_per_token(self) -> int:
    return self._num_selected_experts

  @property
  def expert_name_filter(self):
    return self._expert_filter

  @property
  def moe_mlp_name_filter(self):
    return self._moe_mlp_filter

  @property
  def moe_layer_name_filter(self):
    return self._moe_layer_filter

  @property
  def moe_attn_name_filter(self):
    return self._moe_attn_filter

  @property
  def expert_meta_parser(self):
    return self.parse_expert_meta_from_name

  @staticmethod
  def _stage_block_expert_from_match(match):
    if match.group(1) == "encoder":
      return match.group(1), int(match.group(2)), int(match.group(3))
    return match.group(4), int(match.group(5)), int(match.group(6))

  @staticmethod
  def _stage_block_from_match(match):
    if match.group(1) == "encoder":
      return match.group(1), int(match.group(2))
    return match.group(3), int(match.group(4))

  def stage_layer_id(self, stage: str, block_id: int) -> int:
    sparse_ids = self.encoder_sparse_layer_ids if stage == "encoder" else self.decoder_sparse_layer_ids
    if block_id not in sparse_ids:
      raise ValueError(f"{stage} block {block_id} is not a configured Switch sparse block")
    return sparse_ids.index(block_id)

  def global_layer_id(self, stage: str, stage_layer_id: int) -> int:
    if stage == "encoder":
      return stage_layer_id
    if stage == "decoder":
      return self.num_encoder_sparse_layers + stage_layer_id
    raise ValueError(f"invalid Switch stage {stage!r}")

  def predictor_layer_id(self, stage: str, global_layer_id: int) -> int:
    if stage != "decoder":
      raise ValueError("encoder predictor is not implemented in this phase")
    first_decoder_layer = self.num_encoder_sparse_layers
    last_decoder_layer = first_decoder_layer + self.num_decoder_sparse_layers - 1
    if global_layer_id < first_decoder_layer or global_layer_id > last_decoder_layer:
      raise ValueError(f"decoder global layer id must be in [{first_decoder_layer}, {last_decoder_layer}]")
    return global_layer_id - first_decoder_layer

  def parse_expert_meta_from_name(self, name: str) -> tuple[int, int] | None:
    match = re.match(self._EXPERT_NAME_PATTERN, name)
    if not match:
      return None
    stage, block_id, expert_id = self._stage_block_expert_from_match(match)
    stage_layer_id = self.stage_layer_id(stage, block_id)
    return self.global_layer_id(stage, stage_layer_id), expert_id

  def parse_moe_layer_name(self, name: str) -> tuple[str, int, int] | None:
    match = re.match(self._MLP_NAME_PATTERN, name)
    if not match:
      return None
    stage, block_id = self._stage_block_from_match(match)
    sparse_ids = self.encoder_sparse_layer_ids if stage == "encoder" else self.decoder_sparse_layer_ids
    if block_id not in sparse_ids:
      return None
    stage_layer_id = self.stage_layer_id(stage, block_id)
    return stage, stage_layer_id, self.global_layer_id(stage, stage_layer_id)

  def add_metadata_to_module(self, module, name: str) -> None:
    module._prefix = name
    expert_match = re.match(self._EXPERT_NAME_PATTERN, name)
    if expert_match:
      stage, block_id, expert_id = self._stage_block_expert_from_match(expert_match)
      stage_layer_id = self.stage_layer_id(stage, block_id)
      module._stage = stage
      module._stage_layer_id = stage_layer_id
      module._layer_id = self.global_layer_id(stage, stage_layer_id)
      module._expert_id = expert_id
      return

    mlp_meta = self.parse_moe_layer_name(name)
    if mlp_meta is not None:
      stage, stage_layer_id, global_layer_id = mlp_meta
      module._stage = stage
      module._stage_layer_id = stage_layer_id
      module._layer_id = global_layer_id

  def validate_predictor_path(self, predictor_model_path: str | None, num_predict_expert_per_layer: int | None) -> None:
    if not num_predict_expert_per_layer:
      return
    if predictor_model_path is None:
      raise ValueError("decoder predictor prefetch requires predictor_model_path")
    path = Path(predictor_model_path)
    if not path.exists():
      raise FileNotFoundError(f"decoder predictor path does not exist: {path}")
    metas_path = path / "metas.json"
    if not metas_path.exists():
      raise FileNotFoundError(f"decoder predictor metas.json does not exist: {metas_path}")
    with metas_path.open() as f:
      metas = json.load(f)
    required_layers = {str(i) for i in range(self.num_decoder_sparse_layers)}
    missing = sorted(required_layers - set(metas.keys()), key=int)
    if missing:
      raise ValueError(f"decoder predictor metas.json missing stage-local layers: {missing}")
    global_like_layers = {str(i) for i in range(self.num_encoder_sparse_layers, self.num_moe_layer)}
    if global_like_layers.issubset(set(metas.keys())):
      raise ValueError("decoder predictor appears to use global layer ids; expected stage-local ids 0..D-1")
    for src_layer, span in metas.items():
      src_layer_id = int(src_layer)
      if src_layer_id >= self.num_decoder_sparse_layers:
        continue
      start_layer, stop_layer = span
      for dst_layer_id in range(int(start_layer), int(stop_layer)):
        model_file = path / f"{src_layer_id}-{dst_layer_id}.pt"
        if not model_file.exists():
          raise FileNotFoundError(f"decoder predictor model file does not exist: {model_file}")

  def configure_module_meta(self, meta) -> None:
    meta.predictor_num_layer = self.num_decoder_sparse_layers
    meta.predictor_layer_offset = self.num_encoder_sparse_layers
    meta.layer_predict_replace_first_input_with_last_output = False

  def should_report_moe_layer_to_predictor(self, stage: str | None, global_layer_id: int) -> bool:
    return stage == "decoder"

  def report_layer_id_for_predictor(self, stage: str | None, global_layer_id: int) -> int:
    return self.predictor_layer_id(stage, global_layer_id)

  def extract_moe_layer_input_for_predictor(self, *args, **kwargs):
    return args[0]

  def extract_moe_layer_output_for_predictor(self, output):
    if isinstance(output, tuple) and len(output) >= 2:
      return output[0]
    return output[0]

  def should_patch_report_experts(self, module) -> bool:
    return hasattr(module, "report_experts")
