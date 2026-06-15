from __future__ import annotations

from .base import DefaultModelAdapter, ModelAdapter
from .nllb_moe import NllbMoeAdapter
from .switch import SwitchAdapter


def get_model_adapter(model, model_id: str | None = None) -> ModelAdapter:
  resolved_model_id = model.config._name_or_path if model_id is None else model_id
  if getattr(model.config, "model_type", None) == "switch_transformers":
    return SwitchAdapter(model, resolved_model_id)
  if getattr(model.config, "model_type", None) == "nllb-moe":
    return NllbMoeAdapter(model, resolved_model_id)
  return DefaultModelAdapter(model, resolved_model_id)
