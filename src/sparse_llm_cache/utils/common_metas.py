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
  def build_deepseek_moe():
    def parse_moe_layer_id(name):
      match = re.match(r'.*layers\.(\d+).*', name)
      if not match:
        return None
      return int(match.group(1)) - 1
    def parse_expert_id(name):
      match = re.match(r'.*layers\.\d+\.mlp\.experts\.(\d+).*$', name)
      if not match:
        return None
      return int(match.group(1))

    def parse_expert_meta_from_name(name):
      return (parse_moe_layer_id(name), parse_expert_id(name))
      # match = re.match(r'.*layers\.(\d+)(\.mlp\.experts\.(\d+)(\..*)?)?', name)
      # layer_id = None
      # expert_id = None
      # if match:
      #   layer_id = int(match.group(1)) - 1
      # if match and match.group(3):
      #   expert_id = int(match.group(3))
      # return (layer_id, expert_id)
    return asdict(ModelMetas(
      num_moe_layer         = 27,
      num_expert_per_layer  = 64,
      expert_meta_parser    = parse_expert_meta_from_name,
      expert_name_filter    = RegexFilter(r'.*layers\.(\d+)\.mlp\.experts\.(\d+)$'),
      moe_layer_name_filter = RegexFilter(r'.*layers\.([1-9]\d*)\.mlp$')
    ))