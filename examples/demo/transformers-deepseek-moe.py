import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
# os.environ['SPARSE_CACHE_LOG_LEVEL'] = 'TRACE'
# os.environ['SPARSE_CACHE_LOG_LEVEL'] = 'DEBUG'
# os.environ['SPARSE_CACHE_ENABLE_TRACE'] = '1'

import os
os.environ['HUGGINGFACE_OFFLINE'] = "1"
from transformers.utils import logging
logging.disable_progress_bar()

print("loading model...")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = "deepseek-ai/deepseek-moe-16b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
  model_id,
  torch_dtype=torch.float16,
  # use_flash_attention_2=True,
  local_files_only=True,
  device_map='cpu',
  trust_remote_code=True
)
print("loading model...done")

import sparse_llm_cache

cache_configs = {
  "num_predict_expert_per_layer" : 6,
  "cache_rate"                   : 0.2,
  "per_layer_cache"              : True,
  "cache_policy"                 : 'lru',
}

sparse_llm_cache.utils.inject_model(
  model,
  **cache_configs,
  pin_memory = True,
  # enable_timing = True,
  enable_timing = False,
)

model.to('cuda')

def gen_long(text, do_print=False, max_new_tokens=1000):
  inputs = tokenizer(text, return_tensors="pt").input_ids.to(f"cuda")
  outputs = model.generate(inputs, max_new_tokens=max_new_tokens)
  output_str = tokenizer.batch_decode(outputs)
  if do_print:
    print(output_str)
  return inputs[0].nelement()
import time

text = "can you design a referral system similar on how dropbox did? I need a technical overview on how it should work, instead of free space we use the generic term \"credits\" where users can get more credits for every 3 friends they recommend."
output_len = 665
# text, output_len = tokenized_dataset[10]
for _ in range(3):
  start_time = time.time()
  gen_long(text, max_new_tokens=20, do_print=True)
  print(time.time() - start_time)

with open('traces.json', 'w') as f:
  print(sparse_llm_cache.cpp_worker.dump_trace_event_collector_singleton(), file=f)
