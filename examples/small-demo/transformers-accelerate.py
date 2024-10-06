import os

os.environ['HF_HUB_OFFLINE'] = "1"
os.environ['HUGGINGFACE_OFFLINE'] = "1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"

from transformers.utils import logging
from transformers.generation.utils import TimeProfiler, recursive_attach
import torch
torch.cuda.set_device(0)
from transformers import AutoModelForCausalLM, AutoTokenizer

import sparse_llm_cache
import time

from sparse_llm_cache.utils.runner_util import parse_args
cache_configs = parse_args()
for k, v in cache_configs.items(): print(k,v)

print("loading model...")
load_time_start = time.time()
logging.disable_progress_bar()
model_id = cache_configs['model_id']
torch_dtype = 'auto'
if 'GPTQ' in model_id:
  torch_dtype = None
print("dtype is", torch_dtype)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
  model_id,
  # torch_dtype=torch.float16,
  torch_dtype=torch_dtype,
  local_files_only=True,
  device_map='auto',
  trust_remote_code=True,
  revision=cache_configs['model_revision'],
)
print("loading model...done", time.time() - load_time_start)

time_profiler = TimeProfiler()
recursive_attach(model, time_profiler, '_time_profiler')

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

dataset_path = f'/code/sparse-llm-cache-scripts/dataset/{cache_configs["dataset"]}/prompt_list.pt'
print(dataset_path)
prompts = torch.load(dataset_path)

from torch.utils.data import Dataset
class StringListDataset(Dataset):
  def __init__(self, string_list):
    self.string_list = string_list
  def __len__(self):
    return len(self.string_list)
  def __getitem__(self, idx):
    return self.string_list[idx]
ds = StringListDataset(prompts)
dl = torch.utils.data.DataLoader(ds, batch_size=cache_configs['batch_size'], shuffle=False)

for seq_id,text_list in enumerate(dl):
  if seq_id > cache_configs['max_num_batch']:
    print("max_num_batch reached")
    break
  start_time = time.time()
  input_len, output_len = gen_batch(text_list, max_new_tokens=128, do_print=True)
  print(input_len, output_len, time.time() - start_time, flush=True)

time_profiler.log()
sparse_llm_cache.cpp_worker.log_gpu_mem_info()