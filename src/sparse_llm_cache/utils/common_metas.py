from dataclasses import dataclass, asdict
import re

from .filter import RegexFilter

@dataclass
class ModelMetas:
  num_moe_layer: int
  num_expert_per_layer: int
  expert_meta_parser: callable
  expert_name_filter: callable
  moe_layer_name_filter: callable

  @staticmethod
  def build_deepseek_moe(return_dict = True):
    def parse_moe_layer_id(name):
      match = re.match(r'.*layers\.(\d+).*', name)
      if not match:
        return None
      return int(match.group(1)) - 1 # first layer is not moe layer
    def parse_expert_id(name):
      match = re.match(r'.*layers\.\d+\.mlp\.experts\.(\d+).*$', name)
      if not match:
        return None
      return int(match.group(1))

    def parse_expert_meta_from_name(name):
      return (parse_moe_layer_id(name), parse_expert_id(name))
    ret = ModelMetas(
      num_moe_layer         = 27,
      num_expert_per_layer  = 64,
      expert_meta_parser    = parse_expert_meta_from_name,
      expert_name_filter    = RegexFilter(r'.*layers\.(\d+)\.mlp\.experts\.(\d+)$'),
      moe_layer_name_filter = RegexFilter(r'.*layers\.([1-9]\d*)\.mlp$')
    )
    return asdict(ret) if return_dict else ret

  @staticmethod
  def build_qwen_moe(return_dict = True):
    def parse_moe_layer_id(name):
      match = re.match(r'.*layers\.(\d+).*', name)
      if not match:
        return None
      return int(match.group(1))
    def parse_expert_id(name):
      match = re.match(r'.*layers\.\d+\.mlp\.experts\.(\d+).*$', name)
      if not match:
        return None
      return int(match.group(1))

    def parse_expert_meta_from_name(name):
      return (parse_moe_layer_id(name), parse_expert_id(name))
    ret = ModelMetas(
      num_moe_layer         = 24,
      num_expert_per_layer  = 60,
      expert_meta_parser    = parse_expert_meta_from_name,
      expert_name_filter    = RegexFilter(r'.*layers\.(\d+)\.mlp\.experts\.(\d+)$'),
      moe_layer_name_filter = RegexFilter(r'.*layers\.([1-9]\d*)\.mlp$')
    )
    return asdict(ret) if return_dict else ret


predefined_metas = {
  'deepseek-ai/deepseek-moe-16b-chat' : ModelMetas.build_deepseek_moe,
  'Qwen/Qwen1.5-MoE-A2.7B-Chat'       : ModelMetas.build_qwen_moe,
}

def auto_infer_model_metas(model_id, return_dict = True):
  if model_id in predefined_metas:
    return predefined_metas[model_id](return_dict)
  raise ValueError(f"model {model_id} is not supported")