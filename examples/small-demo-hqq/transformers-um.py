import os

os.environ['HF_HUB_OFFLINE'] = "1"
os.environ['HUGGINGFACE_OFFLINE'] = "1"

import json
import time
import torch
from transformers.utils import logging
logging.disable_progress_bar()
from transformers.generation.utils import TimeProfiler, recursive_attach
torch.cuda.set_device(0)
from transformers import AutoModelForCausalLM, AutoTokenizer

import sparse_llm_cache
from sparse_llm_cache.utils import repo_folder_name

from hqq.models.hf.base import AutoHQQHFModel
from hqq.core.quantize import HQQLinear, HQQBackend
from hqq.utils.patching import prepare_for_inference
import argparse
from accelerate import dispatch_model, infer_auto_device_map

def prepare_args():
  from sparse_llm_cache.utils.runner_util import parse_args, prepare_argparser
  parser = prepare_argparser()
  parser.add_argument('--save_dir', type=str, help='By default, it will be inferred from model_id at <save_dir_base>/<repo_folder_name(model_id)>')
  parser.add_argument('--save_dir_base', type=str, default='/code/hqq-quant-model', help='Base directory to save the quantized model')
  # parser.add_argument('--dtype', type=str, default='auto', help='Compute dtype. Default is auto infer from model config.json')
  # parser.add_argument('--backend', type=str, default='aten', choices=['none', 'torch', 'torchao', 'marlin', 'bitblas', 'aten'], help='Backend for inference. Default is aten. Currently aten is the fastest backend.')
  cache_configs = parse_args(parser=parser)
  return cache_configs

def load_model(cache_configs):
  model_id = cache_configs['model_id']
  save_dir = cache_configs['save_dir']
  if save_dir is None:
    save_dir = os.path.join(cache_configs['save_dir_base'], repo_folder_name(model_id))
  # if cache_configs['backend'] == 'aten':
  #   HQQLinear.set_backend(HQQBackend.ATEN)
  HQQLinear.set_backend(HQQBackend.ATEN)

  load_time_start = time.time()
  tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
  tokenizer.pad_token = tokenizer.eos_token

  with open(os.path.join(save_dir, 'config.json'), 'r') as f:
    dtype = json.load(f)['torch_dtype']
  dtype = getattr(torch, dtype)

  print(f"Loading model {model_id} with dtype {dtype} and backend aten")
  model = AutoHQQHFModel.from_quantized(save_dir, trust_remote_code=True, device='cpu', compute_dtype=dtype)
  print("Loading model...done", time.time() - load_time_start)

  sparse_llm_cache.utils.inject_model_um(model, model_id)
  model.to('cuda')

  time_profiler = TimeProfiler()
  recursive_attach(model, time_profiler, '_time_profiler')

  model.eval()

  return model, tokenizer, time_profiler

def gen_batch(model, tokenizer, text_list, do_print=False, max_new_tokens=100):
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

def load_prompt_list(cache_configs):
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
  return dl

def main(cache_configs):
  model, tokenizer, time_profiler = load_model(cache_configs)
  dl = load_prompt_list(cache_configs)

  for seq_id,text_list in enumerate(dl):
    if seq_id >= cache_configs['max_num_batch']:
      print("max_num_batch reached")
      break
    start_time = time.time()
    input_len, output_len = gen_batch(model, tokenizer, text_list, max_new_tokens=128, do_print=True)
    print(input_len, output_len, time.time() - start_time, flush=True)

  time_profiler.log()
  sparse_llm_cache.cpp_worker.log_gpu_mem_info()

if __name__ == "__main__":
  cache_configs = prepare_args()
  main(cache_configs)
