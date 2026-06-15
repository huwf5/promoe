from __future__ import annotations

from pathlib import Path


def is_switch_model_id(value: str) -> bool:
  lowered = value.lower()
  return "switch-" in lowered or lowered.endswith("switch-base-128")


def is_nllb_moe_model_id(value: str) -> bool:
  return "nllb-moe" in value.lower()


def resolve_transformers_module_model_path(value: str, repo_root: str | Path) -> str:
  path = Path(value)
  if path.is_absolute() or "/" not in value:
    return value

  vendor, model_name = value.split("/", 1)
  local_path = (
    Path(repo_root)
    / "deps"
    / "sparse-llm-cache-scripts"
    / "huggingface-modules"
    / "modules"
    / "transformers_modules"
    / vendor
    / model_name
  )
  if local_path.exists():
    return str(local_path)
  return value
