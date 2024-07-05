# %%
import argparse

class CustomBooleanAction(argparse.Action):
  def __init__(self, option_strings, dest, nargs=None, **kwargs):
    if nargs is not None:
      raise ValueError("nargs not allowed")
    super().__init__(option_strings, dest, **kwargs)
  def __call__(self, parser, namespace, values, option_string=None):
    values = str(values).lower()
    if values in ['true', '1', 'on']:
      setattr(namespace, self.dest, True)
    elif values in ['false', '0', 'off']:
      setattr(namespace, self.dest, False)
    else:
      raise ValueError("invalid boolean value {}".format(self.values))

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--num_predict_expert_per_layer", type=int)
  parser.add_argument("--cache_rate", type=float)
  parser.add_argument("--cache_len", type=int, default=None)
  parser.add_argument("--max_prefetch_layer_distance", type=int, default=-1)
  parser.add_argument("--promote_hit_in_prefetch", action=CustomBooleanAction, default=True)
  parser.add_argument("--early_preempt", action=CustomBooleanAction, default=True)
  parser.add_argument("--cache_policy", type=str, choices=["lru", "fifo", "nn", "min"], default="lru")
  parser.add_argument("--cache_trace_path", type=str, default=None)
  parser.add_argument("--predictor_model_path", type=str, default=None)
  parser.add_argument("--predict_input_mode", type=str, choices=["one_token", "decode_cumsum", "last_use_distance", "weighted_decode_cumsum", "first_moe_attn_input_logits", "moe_attn_input_logits"], default='one_token')
  # parser.add_argument("--predict_mode", type=str, choices=["entire_token", "layer_window"], default='one_token')
  # layer_predict_interval
  # layer_predict_window
  parser.add_argument("--layer_predict_interval", type=int, default=None)
  parser.add_argument("--layer_predict_window",   type=int, default=None)

  parser.add_argument(        "--per_layer_cache", action=CustomBooleanAction, default=True)
  parser.add_argument( "--enable_per_layer_cache", action="store_true",  dest="per_layer_cache", default=True)
  parser.add_argument("--disable_per_layer_cache", action="store_false", dest="per_layer_cache", default=True)

  parser.add_argument(        "--reorder_experts", action=CustomBooleanAction, default=True)
  parser.add_argument( "--enable_reorder_experts", action="store_true",   dest="reorder_experts", default=True)
  parser.add_argument("--disable_reorder_experts", action="store_false",  dest="reorder_experts", default=True)

  parser.add_argument(        "--trace_event", action=CustomBooleanAction, default=False)
  parser.add_argument( "--enable_trace_event", action="store_true",   dest="trace_event", default=False)
  parser.add_argument("--disable_trace_event", action="store_false",  dest="trace_event", default=False)

  parser.add_argument(        "--module_trace_event", action=CustomBooleanAction, default=False)
  parser.add_argument( "--enable_module_trace_event", action="store_true",   dest="module_trace_event", default=False)
  parser.add_argument("--disable_module_trace_event", action="store_false",  dest="module_trace_event", default=False)
  args = parser.parse_args()

  return vars(args)
