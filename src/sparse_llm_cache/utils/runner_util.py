# %%
import argparse
import re

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

def prepare_argparser(parser = None):
  if parser is None:
    parser = argparse.ArgumentParser()
  parser._negative_number_matcher = re.compile(r'^-\d+(?:,-?\d+)*$|^-\d+(?:\.\d*)?$|^-\.\d+$')
  parser.add_argument("--model_id",                     type=str)
  parser.add_argument("--model_revision",               type=str)

  parser.add_argument("--cache_rate",                   type=float)
  # parser.add_argument("--cache_len",                    type=int, default=None)
  parser.add_argument("--num_predict_expert_per_layer", type=int)
  parser.add_argument("--early_preempt",   action=CustomBooleanAction)
  parser.add_argument("--chunk_prefetch",  action=CustomBooleanAction)
  parser.add_argument("--reorder_experts", action=CustomBooleanAction)

  parser.add_argument("--predict_input_mode",      type=str, choices=["one_token", "decode_cumsum", "last_use_distance", "weighted_decode_cumsum", "first_moe_attn_input_logits", "moe_attn_input_logits", "moe_layer_logits"])
  parser.add_argument("--predictor_type",          type=str, choices=["legacy", "sep"])

  parser.add_argument("--predictor_model_path",         type=str)
  parser.add_argument("--layer_predict_interval",       type=int)
  parser.add_argument("--layer_predict_max_window",     type=int)
  parser.add_argument("--layer_predict_use_last_output", action=CustomBooleanAction, dest='layer_predict_replace_first_input_with_last_output')

  parser.add_argument("--limit_layer_0_window",      type=int)
  parser.add_argument("--limit_layer_0_num_predict", type=int)

  parser.add_argument("--max_prefetch_layer_distance",  type=int)
  parser.add_argument("--cache_only",              action=CustomBooleanAction)
  parser.add_argument("--per_layer_cache",         action=CustomBooleanAction)
  parser.add_argument("--promote_hit_in_prefetch", action=CustomBooleanAction)
  parser.add_argument("--cache_policy",            type=str, choices=["lru", "fifo", "nn", "min", "static-1", "static-2", "scheduler_aware"])
  parser.add_argument("--initial_cache_policy",           type=str, choices=["manual", "hot_expert", "hot_encoder_coverage", "hot_encoder_balanced_coverage"])
  parser.add_argument("--initial_layer_budgets",          type=str) # manual only
  parser.add_argument("--initial_hot_expert_file",        type=str) # hot_expert/hot_encoder_coverage/hot_encoder_balanced_coverage only
  parser.add_argument("--enable_decoder_warmup_overlap",  action=CustomBooleanAction)

  parser.add_argument("--enable_erpp_encoder_prefetch", action=CustomBooleanAction)
  parser.add_argument("--erpp_encoder_model_path")
  parser.add_argument("--erpp_encoder_budgets")
  parser.add_argument("--erpp_encoder_layers")
  parser.add_argument("--enable_erpp_encoder_jit_refill", action=CustomBooleanAction)
  parser.add_argument("--erpp_encoder_jit_refill_window", type=int)
  parser.add_argument("--erpp_encoder_jit_refill_floor_mode", choices=["avg", "fixed"])
  parser.add_argument("--erpp_encoder_jit_refill_floor_value", type=int)
  parser.add_argument("--erpp_encoder_jit_refill_low_watermark_ratio", type=float)
  parser.add_argument("--erpp_encoder_jit_refill_layers")
  parser.add_argument("--erpp_encoder_jit_refill_per_idle", type=int)
  parser.add_argument("--enable_erpp_encoder_jit_topk_cover", action=CustomBooleanAction)

  parser.add_argument("--trace_event",        action=CustomBooleanAction)
  parser.add_argument("--module_trace_event", action=CustomBooleanAction)
  parser.add_argument("--cache_trace_path",   type=str)
  parser.add_argument("--gpu_mem_limit_gb",   type=float, default=None)

  parser.add_argument("--max_num_batch",                type=int, default=20)
  parser.add_argument("--max_new_tokens",               type=int, default=128)
  parser.add_argument("--batch_size",                   type=int, default=1)
  parser.add_argument("--dataset",                      type=str, default='chatgpt-prompts-small')
  return parser

def parse_args(args = None, parser = None):
  if parser is None:
    parser = prepare_argparser()
  args = parser.parse_args(args)

  return vars(args)
