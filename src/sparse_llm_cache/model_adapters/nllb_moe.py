from __future__ import annotations

import json
import re
from pathlib import Path

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

  def predictor_layer_id(self, stage: str, global_layer_id: int) -> int:
    if stage != "decoder":
      raise ValueError("encoder predictor is not implemented")
    return global_layer_id

  def should_report_moe_layer_to_predictor(self, stage: str | None, global_layer_id: int) -> bool:
    return stage == "decoder"

  def should_report_predictor_pre_forward(self, stage: str | None, global_layer_id: int) -> bool:
    return stage == "decoder" and global_layer_id == self.num_encoder_sparse_layers

  def predictor_input_id_before_layer(self, stage: str | None, global_layer_id: int) -> int:
    if stage != "decoder":
      raise ValueError("encoder predictor is not implemented")
    return global_layer_id

  def predictor_input_id_after_layer(self, stage: str | None, global_layer_id: int) -> int:
    if stage != "decoder":
      raise ValueError("encoder predictor is not implemented")
    return global_layer_id + 1

  def extract_moe_layer_output_for_predictor(self, output):
    return output[0]

  def validate_predictor_path(
    self,
    predictor_model_path: str | None,
    num_predict_expert_per_layer: int | None,
    predictor_type: str | None = None,
  ) -> None:
    if not num_predict_expert_per_layer:
      return
    if predictor_model_path is None:
      raise ValueError("NLLB decoder predictor prefetch requires predictor_model_path")
    path = Path(predictor_model_path)
    if not path.exists():
      raise FileNotFoundError(f"NLLB decoder predictor path does not exist: {path}")
    metas_path = path / "metas.json"
    if not metas_path.exists():
      raise FileNotFoundError(f"NLLB decoder predictor metas.json does not exist: {metas_path}")
    with metas_path.open() as f:
      metas = json.load(f)
    if not isinstance(metas, dict) or int(metas.get("schema_version", 1)) != 2 or metas.get("id_space") != "global":
      raise ValueError("NLLB decoder predictor requires global v2 metas.json; retrain predictor with global ids")
    outputs = metas.get("outputs")
    if not isinstance(outputs, dict):
      raise ValueError("global v2 predictor metas.json missing outputs")

    first_decoder = self.num_encoder_sparse_layers
    num_layer = self.num_moe_layer
    required_sources = {str(i) for i in range(first_decoder, num_layer + 1)}
    missing = sorted(required_sources - set(outputs.keys()), key=int)
    if missing:
      raise ValueError(f"global predictor metas.json missing decoder source layers: {missing}")

    use_legacy_files = predictor_type == "legacy"
    for src_layer, span in outputs.items():
      src_layer_id = int(src_layer)
      if src_layer_id < first_decoder or src_layer_id > num_layer:
        raise ValueError(
          f"predictor source id {src_layer_id} outside decoder global boundary range "
          f"[{first_decoder}, {num_layer}]"
        )
      if not isinstance(span, (list, tuple)) or len(span) != 2:
        raise ValueError(f"global predictor output range for source {src_layer_id} must be [start, stop]")
      start_layer, stop_layer = int(span[0]), int(span[1])
      if start_layer < first_decoder or start_layer > stop_layer or stop_layer > num_layer:
        raise ValueError(
          f"predictor output range [{start_layer}, {stop_layer}) outside decoder global range "
          f"[{first_decoder}, {num_layer})"
        )
      if start_layer == num_layer and start_layer != stop_layer:
        raise ValueError(
          f"predictor output range [{start_layer}, {stop_layer}) outside decoder global range "
          f"[{first_decoder}, {num_layer})"
        )
      if use_legacy_files:
        if start_layer == stop_layer:
          continue
        model_file = path / f"{src_layer_id}.pt"
        if not model_file.exists():
          raise FileNotFoundError(f"NLLB decoder predictor model file does not exist: {model_file}")
        continue
      for dst_layer_id in range(start_layer, stop_layer):
        model_file = path / f"{src_layer_id}-{dst_layer_id}.pt"
        if not model_file.exists():
          raise FileNotFoundError(f"NLLB decoder predictor model file does not exist: {model_file}")

  def erpp_encoder_prefetch_module(self):
    encoder = None
    if hasattr(self.model, "get_encoder"):
      encoder = self.model.get_encoder()
    if encoder is None:
      encoder = getattr(getattr(self.model, "model", None), "encoder", None)
    if encoder is None:
      encoder = getattr(self.model, "encoder", None)
    encoder_layers = getattr(encoder, "layers", None)
    if encoder_layers is None or len(encoder_layers) == 0:
      raise ValueError("NLLB ERPP encoder prefetch requires model.get_encoder().layers[0]")
    return encoder_layers[0]

  def should_patch_report_experts(self, module) -> bool:
    return hasattr(module, "report_experts")
