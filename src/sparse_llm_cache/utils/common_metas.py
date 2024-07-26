from dataclasses import dataclass, asdict
import re

from .filter import RegexFilter

@dataclass
class ModelMetas:
  num_moe_layer: int
  num_expert_per_layer: int
  num_expert_per_token: int
  expert_meta_parser: callable
  expert_name_filter: callable
  moe_attn_name_filter: callable
  moe_mlp_name_filter: callable
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
      num_expert_per_token  = 6,
      expert_meta_parser    = parse_expert_meta_from_name,
      expert_name_filter    = RegexFilter(r'.*layers\.(\d+)\.mlp\.experts\.(\d+)$'),
      moe_attn_name_filter  = RegexFilter(r'.*layers\.([1-9]\d*)\.self_attn$'),
      moe_mlp_name_filter   = RegexFilter(r'.*layers\.([1-9]\d*)\.mlp$'),
      moe_layer_name_filter = RegexFilter(r'.*layers\.([1-9]\d*)$')
    )
    return asdict(ret) if return_dict else ret
  def build_deepseek_moe_simulate(return_dict = True):
    def parse_moe_layer_id(name):
      match = re.match(r'.*layers\.(\d+).*', name)
      if not match:
        return None
      return int(match.group(1))
    def parse_expert_id(name):
      match = re.match(r'.*layers\.\d+\.experts\.(\d+).*$', name)
      if not match:
        return None
      return int(match.group(1))

    def parse_expert_meta_from_name(name):
      return (parse_moe_layer_id(name), parse_expert_id(name))
    ret = ModelMetas(
      num_moe_layer         = 27,
      num_expert_per_layer  = 64,
      num_expert_per_token  = 6,
      expert_meta_parser    = parse_expert_meta_from_name,
      expert_name_filter    = RegexFilter(r'.*layers\.(\d+)\.experts\.(\d+)$'),
      moe_attn_name_filter  = RegexFilter(r'.*layers\.([1-9]\d*)\.self_attn$'),
      moe_mlp_name_filter   = RegexFilter(r'.*layers\.(\d+).mlp$'),
      moe_layer_name_filter = RegexFilter(r'.*layers\.(\d+)$')
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
      num_expert_per_token  = 4,
      expert_meta_parser    = parse_expert_meta_from_name,
      expert_name_filter    = RegexFilter(r'.*layers\.(\d+)\.mlp\.experts\.(\d+)$'),
      moe_attn_name_filter  = RegexFilter(r'.*layers\.(\d+)\.self_attn$'),
      moe_mlp_name_filter   = RegexFilter(r'.*layers\.(\d+)\.mlp$'),
      moe_layer_name_filter = RegexFilter(r'.*layers\.(\d+)$')
    )
    return asdict(ret) if return_dict else ret

  @staticmethod
  def build_mixtral(return_dict = True):
    def parse_moe_layer_id(name):
      match = re.match(r'.*layers\.(\d+).*', name)
      if not match:
        return None
      return int(match.group(1))
    def parse_expert_id(name):
      match = re.match(r'.*layers\.\d+\.block_sparse_moe\.experts\.(\d+).*$', name)
      if not match:
        return None
      return int(match.group(1))

    def parse_expert_meta_from_name(name):
      return (parse_moe_layer_id(name), parse_expert_id(name))
    ret = ModelMetas(
      num_moe_layer         = 32,
      num_expert_per_layer  = 8,
      num_expert_per_token  = 2,
      expert_meta_parser    = parse_expert_meta_from_name,
      expert_name_filter    = RegexFilter(r'.*layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)$'),
      moe_attn_name_filter  = RegexFilter(r'.*layers\.(\d+)\.self_attn$'),
      moe_mlp_name_filter   = RegexFilter(r'.*layers\.(\d+)\.block_sparse_moe$'),
      moe_layer_name_filter = RegexFilter(r'.*layers\.(\d+)$')
    )
    return asdict(ret) if return_dict else ret


predefined_metas = {
  'deepseek-ai/deepseek-moe-16b-chat'          : ModelMetas.build_deepseek_moe,
  'deepseek-ai/deepseek-moe-16b-chat-simulate' : ModelMetas.build_deepseek_moe_simulate,
  'Qwen/Qwen1.5-MoE-A2.7B-Chat'                : ModelMetas.build_qwen_moe,
  'mistralai/Mixtral-8x7B-Instruct-v0.1'       : ModelMetas.build_mixtral,
  'TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ'   : ModelMetas.build_mixtral,
}

def auto_infer_model_metas(model_id, return_dict = True):
  if model_id in predefined_metas:
    return predefined_metas[model_id](return_dict)
  raise ValueError(f"model {model_id} is not supported")