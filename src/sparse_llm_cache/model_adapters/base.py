from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sparse_llm_cache.utils.common_metas import auto_infer_model_metas


@dataclass(frozen=True)
class ExpertInfo:
  prefix: str
  stage: str | None
  stage_layer_id: int
  global_layer_id: int
  expert_id: int


class ModelAdapter:
  def __init__(self, model, model_id: str):
    self.model = model
    self.model_id = model_id

  @property
  def num_moe_layer(self) -> int:
    raise NotImplementedError

  @property
  def num_expert_per_layer(self) -> int:
    raise NotImplementedError

  @property
  def num_expert_per_token(self) -> int:
    raise NotImplementedError

  @property
  def expert_name_filter(self):
    raise NotImplementedError

  @property
  def moe_mlp_name_filter(self):
    raise NotImplementedError

  @property
  def moe_layer_name_filter(self):
    raise NotImplementedError

  @property
  def moe_attn_name_filter(self):
    raise NotImplementedError

  @property
  def expert_meta_parser(self) -> Callable[[str], tuple[int, int] | None]:
    raise NotImplementedError

  def add_metadata_to_module(self, module, name: str) -> None:
    module._prefix = name
    expert_meta = self.expert_meta_parser(name)
    try:
      module._layer_id = int(expert_meta[0])
      module._expert_id = int(expert_meta[1])
    except TypeError:
      pass

  def configure_module_meta(self, meta) -> None:
    meta.num_encoder_moe_layer = 0
    meta.num_decoder_moe_layer = self.num_moe_layer

  def validate_predictor_path(
    self,
    predictor_model_path: str | None,
    num_predict_expert_per_layer: int | None,
    predictor_type: str | None = None,
  ) -> None:
    return None

  def should_report_moe_layer_to_predictor(self, stage: str | None, global_layer_id: int) -> bool:
    return True

  def report_layer_id_for_predictor(self, stage: str | None, global_layer_id: int) -> int:
    return global_layer_id

  def extract_moe_layer_input_for_predictor(self, *args, **kwargs):
    return args[0]

  def extract_moe_layer_output_for_predictor(self, output):
    return output[0]

  def should_patch_report_experts(self, module) -> bool:
    return True


class DefaultModelAdapter(ModelAdapter):
  def __init__(self, model, model_id: str):
    super().__init__(model, model_id)
    self._metas = auto_infer_model_metas(model_id, return_dict=False)

  @property
  def num_moe_layer(self) -> int:
    return self._metas.num_moe_layer

  @property
  def num_expert_per_layer(self) -> int:
    return self._metas.num_expert_per_layer

  @property
  def num_expert_per_token(self) -> int:
    return self._metas.num_expert_per_token

  @property
  def expert_name_filter(self):
    return self._metas.expert_name_filter

  @property
  def moe_mlp_name_filter(self):
    return self._metas.moe_mlp_name_filter

  @property
  def moe_layer_name_filter(self):
    return self._metas.moe_layer_name_filter

  @property
  def moe_attn_name_filter(self):
    return self._metas.moe_attn_name_filter

  @property
  def expert_meta_parser(self):
    return self._metas.expert_meta_parser
