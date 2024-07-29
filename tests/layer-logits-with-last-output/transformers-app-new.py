import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
# os.environ['SPARSE_CACHE_LOG_LEVEL'] = 'DEBUG'
# os.environ['SPARSE_CACHE_LOG_LEVEL'] = 'TRACE'
os.environ['HF_HUB_OFFLINE'] = "1"
os.environ['HUGGINGFACE_OFFLINE'] = "1"

from transformers.utils import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sparse_llm_cache
import time

from sparse_llm_cache.utils.runner_util import parse_args
cache_configs = parse_args()
for k, v in cache_configs.items(): print(k,v)

sparse_llm_cache.utils.hack_transformers(**cache_configs, pin_memory=True, enable_model_timer=True)

print("loading model...")
load_time_start = time.time()
logging.disable_progress_bar()
model_id = cache_configs['model_id']
torch_dtype = None if 'Mixtral' in model_id else 'auto'
print("dtype is", torch_dtype)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
  model_id,
  # torch_dtype=torch.float16,
  # torch_dtype=torch.bfloat16,
  torch_dtype=torch_dtype,
  # use_flash_attention_2=True,
  local_files_only=True,
  device_map=0,
  trust_remote_code=True
)
print("loading model...done", time.time() - load_time_start)

def gen_long(text, do_print=False, max_new_tokens=1000):
  inputs = tokenizer(text, return_tensors="pt").input_ids.to(f"cuda")
  outputs = model.generate(inputs, max_new_tokens=max_new_tokens)
  output_str = tokenizer.batch_decode(outputs)
  if do_print:
    print(output_str)
  return inputs[0].nelement()

text = "can you design a referral system similar on how dropbox did? I need a technical overview on how it should work, instead of free space we use the generic term \"credits\" where users can get more credits for every 3 friends they recommend."
output_len = 665
# text, output_len = tokenized_dataset[10]

from torch.profiler import profile, record_function, ProfilerActivity

_global_profiler : profile = None
# profile_fname = "nllb-transformer-trace-stack-short.json"

def create_profile(record_shapes=True,profile_memory=True,with_stack=True, with_cpu=True, with_cuda=True):
  global _global_profiler
  activities = []
  if with_cpu:
    activities.append(ProfilerActivity.CPU)
  if with_cuda:
    activities.append(ProfilerActivity.CUDA)
  _global_profiler = profile(
    activities=activities,
    with_stack=with_stack,
    record_shapes=record_shapes,
    profile_memory=profile_memory,
  )

def start_profile() :
  global _global_profiler
  _global_profiler.__enter__()

def stop_profile():
  global _global_profiler
  _global_profiler.__exit__(None, None, None)

def export_trace(fname):
  _global_profiler.export_chrome_trace(fname)

# create_profile(record_shapes=False, profile_memory=False, with_stack=False, with_cpu=False, with_cuda=True)
# create_profile(record_shapes=False, profile_memory=False, with_stack=True, with_cpu=True, with_cuda=True)
# start_profile()

for _ in range(3):
  start_time = time.time()
  gen_long(text, max_new_tokens=20, do_print=True)
  print(time.time() - start_time)

# stop_profile()
# export_trace('trace-cuda.json')