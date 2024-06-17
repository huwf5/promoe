# %%
import argparse

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--num_predict_expert_per_layer", type=int)
  parser.add_argument("--cache_rate", type=float)
  parser.add_argument("--cache_len", type=int, default=None)
  parser.add_argument("--max_prefetch_layer_distance", type=int, default=-1)
  parser.add_argument("--per_layer_cache", type=bool, default=True)
  parser.add_argument("--enable_per_layer_cache",  action="store_true",  dest="per_layer_cache", default=True)
  parser.add_argument("--disable_per_layer_cache", action="store_false", dest="per_layer_cache", default=True)
  parser.add_argument("--cache_policy", type=str, choices=["lru", "fifo"], default="lru")
  parser.add_argument("--reorder_experts", type=bool, default=True)
  parser.add_argument("--enable_reorder_experts",   action="store_true",   dest="reorder_experts", default=True)
  parser.add_argument("--disable_reorder_experts",  action="store_false",  dest="reorder_experts", default=True)
  args = parser.parse_args()
  return vars(args)
