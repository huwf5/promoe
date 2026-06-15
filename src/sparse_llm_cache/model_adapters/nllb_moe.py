from __future__ import annotations

import re

from sparse_llm_cache.utils.filter import Filter, RegexFilter

from .base import ModelAdapter


class NllbMoeSparseMlpFilter(Filter):
  def __init__(self, adapter):
    self.adapter = adapter

  def __call__(self, name, *args, **kwargs):
    return self.adapter.parse_moe_layer_name(name) is not None


class NllbMoeAdapter(ModelAdapter):
  _EXPERT_NAME_PATTERN = r".*(encoder|decoder)\.layers\.(\d+)\.ffn\.experts\.expert_(\d+)$"
  _MLP_NAME_PATTERN = r".*(encoder|decoder)\.layers\.(\d+)\.ffn$"

  def __init__(self, model, model_id: str):
    super().__init__(model, model_id)
    cfg = model.config
    if getattr(cfg, "model_type", None) != "nllb-moe":
      raise ValueError("NllbMoeAdapter requires model.config.model_type == 'nllb-moe'")

    self.num_encoder_layers = int(cfg.encoder_layers)
    self.num_decoder_layers = int(cfg.decoder_layers)
    self.encoder_sparse_step = int(cfg.encoder_sparse_step)
    self.decoder_sparse_step = int(cfg.decoder_sparse_step)
    self._num_experts = int(cfg.num_experts)
    self._num_selected_experts = 2

    self.encoder_sparse_layer_ids = self._expected_sparse_layer_ids(
      self.num_encoder_layers,
      self.encoder_sparse_step,
    )
    self.decoder_sparse_layer_ids = self._expected_sparse_layer_ids(
      self.num_decoder_layers,
      self.decoder_sparse_step,
    )

    self._expert_filter = RegexFilter(self._EXPERT_NAME_PATTERN)
    self._moe_mlp_filter = NllbMoeSparseMlpFilter(self)
    self._moe_layer_filter = self._moe_mlp_filter
    self._moe_attn_filter = RegexFilter(r"^$")

  @staticmethod
  def _expected_sparse_layer_ids(num_layers: int, sparse_step: int) -> list[int]:
    if sparse_step <= 0:
      raise ValueError(f"NLLB sparse step must be positive, got {sparse_step}")
    return list(range(sparse_step - 1, num_layers, sparse_step))

  @property
  def num_encoder_sparse_layers(self) -> int:
    return len(self.encoder_sparse_layer_ids)

  @property
  def num_decoder_sparse_layers(self) -> int:
    return len(self.decoder_sparse_layer_ids)

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

  def stage_layer_id(self, stage: str, block_id: int) -> int:
    sparse_ids = self.encoder_sparse_layer_ids if stage == "encoder" else self.decoder_sparse_layer_ids
    if block_id not in sparse_ids:
      raise ValueError(f"{stage} layer {block_id} is not a configured NLLB sparse layer")
    return sparse_ids.index(block_id)

  def global_layer_id(self, stage: str, stage_layer_id: int) -> int:
    if stage == "encoder":
      return stage_layer_id
    if stage == "decoder":
      return self.num_encoder_sparse_layers + stage_layer_id
    raise ValueError(f"invalid NLLB stage {stage!r}")

  def parse_expert_meta_from_name(self, name: str) -> tuple[int, int] | None:
    match = re.match(self._EXPERT_NAME_PATTERN, name)
    if not match:
      return None
    stage = match.group(1)
    block_id = int(match.group(2))
    expert_id = int(match.group(3))
    stage_layer_id = self.stage_layer_id(stage, block_id)
    return self.global_layer_id(stage, stage_layer_id), expert_id

  def parse_moe_layer_name(self, name: str) -> tuple[str, int, int] | None:
    match = re.match(self._MLP_NAME_PATTERN, name)
    if not match:
      return None
    stage = match.group(1)
    block_id = int(match.group(2))
    sparse_ids = self.encoder_sparse_layer_ids if stage == "encoder" else self.decoder_sparse_layer_ids
    if block_id not in sparse_ids:
      return None
    stage_layer_id = self.stage_layer_id(stage, block_id)
    return stage, stage_layer_id, self.global_layer_id(stage, stage_layer_id)

  def add_metadata_to_module(self, module, name: str) -> None:
    module._prefix = name
    expert_match = re.match(self._EXPERT_NAME_PATTERN, name)
    if expert_match:
      stage = expert_match.group(1)
      block_id = int(expert_match.group(2))
      expert_id = int(expert_match.group(3))
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

  def configure_module_meta(self, meta) -> None:
    meta.num_encoder_moe_layer = self.num_encoder_sparse_layers
    meta.num_decoder_moe_layer = self.num_decoder_sparse_layers
    meta.predictor_num_layer = self.num_moe_layer

  def validate_predictor_path(
    self,
    predictor_model_path: str | None,
    num_predict_expert_per_layer: int | None,
    predictor_type: str | None = None,
  ) -> None:
    if num_predict_expert_per_layer:
      raise ValueError("NLLB MoE predictor prefetch is not implemented")

  def should_patch_report_experts(self, module) -> bool:
    return hasattr(module, "report_experts")
