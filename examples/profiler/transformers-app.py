import os

os.environ['HF_HUB_OFFLINE'] = "1"
os.environ['HUGGINGFACE_OFFLINE'] = "1"

from transformers.utils import logging
import torch
torch.cuda.set_device(0)
from transformers import AutoModelForCausalLM, AutoTokenizer

# import sparse_llm_cache
import time

# from sparse_llm_cache.utils.runner_util import parse_args
# cache_configs = parse_args()
# for k, v in cache_configs.items(): print(k,v)

# sparse_llm_cache.utils.hack_transformers(**cache_configs, pin_memory=True, enable_model_timer=True)

print("loading model...")
load_time_start = time.time()
logging.disable_progress_bar()
# model_id = cache_configs['model_id']

model_id = "deepseek-ai/deepseek-moe-16b-chat"
torch_dtype = 'auto'
if 'Mixtral' in model_id or 'GPTQ' in model_id:
  torch_dtype = None
print("dtype is", torch_dtype)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
  model_id,
  # torch_dtype=torch.float16,
  torch_dtype=torch_dtype,
  local_files_only=True,
  device_map=0,
  trust_remote_code=True,
  # revision=cache_configs['model_revision'],
)
if 'Mixtral' in model_id:
  import auto_gptq
  model = auto_gptq.exllama_set_max_input_length(model, 7200)
print("loading model...done", time.time() - load_time_start)

def gen_batch(text_list, do_print=False, max_new_tokens=100):
  inputs = tokenizer(text_list, return_tensors="pt", padding=True).to(f"cuda") # input_ids, attention_mask
  input_len = inputs['input_ids'].shape[1]
  outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
  output_len = outputs.shape[1] - input_len
  outputs = outputs[:, input_len:]
  output_len = outputs.shape[1]
  output_str = tokenizer.batch_decode(outputs)
  if do_print:
    print(text_list, output_str, flush=True)
  return input_len, output_len

# dataset_path = f'/code/sparse-llm-cache-scripts/dataset/{cache_configs["dataset"]}/prompt_list.pt'
# print(dataset_path)
# prompts = torch.load(dataset_path)

# from torch.utils.data import Dataset
# class StringListDataset(Dataset):
#   def __init__(self, string_list):
#     self.string_list = string_list
#   def __len__(self):
#     return len(self.string_list)
#   def __getitem__(self, idx):
#     return self.string_list[idx]
# ds = StringListDataset(prompts)
# dl = torch.utils.data.DataLoader(ds, batch_size=cache_configs['batch_size'], shuffle=False)



from torch.profiler import profile, record_function, ProfilerActivity
# profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, with_stack=True)
# profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True)
# profiler, profile_fname = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_modules=True), "nllb-transformer-trace-module.json"
# profiler, profile_fname = profile(activities=[ProfilerActivity.CUDA], with_stack=True), "nllb-transformer-trace-cuda-only-stack.json"

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



prompts = ["Introduce yourself"]

input_len, output_len = gen_batch(prompts, max_new_tokens=10, do_print=True)
input_len, output_len = gen_batch(prompts, max_new_tokens=10, do_print=True)
input_len, output_len = gen_batch(prompts, max_new_tokens=10, do_print=True)

create_profile(record_shapes=False, profile_memory=False, with_stack=True, with_cpu=True, with_cuda=True)
start_profile()
input_len, output_len = gen_batch(prompts, max_new_tokens=10, do_print=True)
stop_profile()
export_trace('trace-cuda.json')