from dataclasses import dataclass
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
# os.environ['SPARSE_CACHE_LOG_LEVEL'] = 'TRACE'
# os.environ['SPARSE_CACHE_LOG_LEVEL'] = 'DEBUG'
# os.environ['SPARSE_CACHE_ENABLE_TRACE'] = '1'
os.environ['HUGGINGFACE_OFFLINE'] = "1"

import torch

import sparse_llm_cache
import time

from sparse_llm_cache.utils.runner_util import parse_args
cache_configs = parse_args()
for k, v in cache_configs.items(): print(k,v)

print("loading model...")

@dataclass
class SimulateConfig:
  _name_or_path : str = None


class SimulateExpert(torch.nn.Module):
  def __init__(self):
    super().__init__()
    # self.fc = torch.nn.Linear(1, 1)
    self.fc = torch.nn.Linear(0, 0, bias=False)

  def forward(self, inputs):
    return inputs

class SimulateMoeLayer(torch.nn.Module):
  def __init__(self, num_experts):
    super().__init__()
    self.num_experts = num_experts
    self.experts = torch.nn.ModuleList([SimulateExpert() for _ in range(num_experts)])
  def report_experts(self, experts):
    return experts
  def forward(self, inputs):
    expert_ids = inputs
    expert_ids = self.report_experts(expert_ids)
    for eid in expert_ids:
      expert = self.experts[eid]
      expert(eid)
    return inputs

class SimulateModel(torch.nn.Module):
  def __init__(self, num_layers, per_layer_experts):
    super().__init__()
    self.config = SimulateConfig(_name_or_path = "deepseek-ai/deepseek-moe-16b-chat-simulate")
    self.num_layers = num_layers
    self.per_layer_experts = per_layer_experts
    self.layers = torch.nn.ModuleList([SimulateMoeLayer(per_layer_experts) for _ in range(num_layers)])
    self.trace = None

  def forward(self, inputs):
    experts_per_layer = inputs
    for i, layer in enumerate(self.layers):
      layer(torch.asarray(experts_per_layer[i], dtype=torch.int64))
    return inputs
  
  def generate(self, seq_id, max_new_tokens = None):
    prompt_len = self.trace[str(seq_id)]['prompt_len']
    rply_len = len(self.trace[str(seq_id)]['0']) - prompt_len
    if max_new_tokens:
      rply_len = min(rply_len, max_new_tokens)

    experts_per_layer = []

    for decoder_moe_layer_id in range(self.num_layers):
      dedup_expert_in_cur_layer = []
      for prompt_token_idx in range(prompt_len):
        dedup_expert_in_cur_layer += self.trace[str(seq_id)][str(decoder_moe_layer_id)][str(prompt_token_idx)]
      dedup_expert_in_cur_layer = list(set(dedup_expert_in_cur_layer))
      dedup_expert_in_cur_layer.sort()
      experts_per_layer.append(dedup_expert_in_cur_layer)

    self.forward(experts_per_layer)

    for rply_token_idx in range(prompt_len, prompt_len + rply_len):
      experts_per_layer = []
      for decoder_moe_layer_id in range(self.num_layers):
        experts_per_layer.append(self.trace[str(seq_id)][str(decoder_moe_layer_id)][str(rply_token_idx)])
      self.forward(experts_per_layer)

model = SimulateModel(27, 64)

fname = "/nvme/songxiaoniu/moe/moe-traces/deepseek-moe-sharegpt-0412.pickle"
import pickle
with open(fname, "rb") as f:
  trace = pickle.load(f)
  trace : dict
model.trace = trace

print("loading model...done")

prefetch_mngr = sparse_llm_cache.utils.inject_model(
  model,
  **cache_configs,
  pin_memory = True,
  enable_model_timer=True,
)

model.to('cuda')

def gen_long(seq_id, max_new_tokens=None):
  prefetch_mngr.cache.set_cur_seq(seq_id)
  outputs = model.generate(seq_id, max_new_tokens=max_new_tokens)

for seq_id in range(20):
  start_time = time.time()
  try:
    gen_long(seq_id, max_new_tokens=None)
  except Exception as e:
    print(f"error at seq {seq_id}")
    print(str(e))
    print(e.with_traceback())
  print(time.time() - start_time)
