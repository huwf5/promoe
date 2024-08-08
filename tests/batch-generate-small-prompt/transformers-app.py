import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# os.environ['SPARSE_CACHE_LOG_LEVEL'] = 'TRACE'
# os.environ['SPARSE_CACHE_ENABLE_TRACE'] = '1'
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
  # use_flash_attention_2=True,
  local_files_only=True,
  device_map=0,
  trust_remote_code=True,
  revision=cache_configs['model_revision'],
)
if 'Mixtral' in model_id:
  import auto_gptq
  model = auto_gptq.exllama_set_max_input_length(model, 7200)
print("loading model...done", time.time() - load_time_start)

def gen_long(text, do_print=False, max_new_tokens=100):
  inputs = tokenizer(text, return_tensors="pt").input_ids.to(f"cuda")
  outputs = model.generate(inputs, max_new_tokens=max_new_tokens)
  output_str = tokenizer.batch_decode(outputs)
  if do_print:
    print(output_str, flush=True)
  return inputs[0].nelement()

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


# import json
# # dataset_path = "/nvme/sxn/moe/datasets/vllm-benchmark/shareGPT/ShareGPT_V3_unfiltered_cleaned_split.json"
# # dataset_path = "/nvme/sxn/moe/datasets/vllm-benchmark/shareGPT/small-dataset.json"
# dataset_path = "/code/moe/datasets/vllm-benchmark/shareGPT/small-dataset.json"
# with open(dataset_path) as f:
#   dataset = json.load(f)

# # # Filter out the conversations with less than 2 turns.
# # dataset = [data for data in dataset if len(data["conversations"]) >= 2]
# # # Only keep the first two turns of each conversation.
# # dataset = [(data["conversations"][0]["value"],
# #             data["conversations"][1]["value"]) for data in dataset]

# # Tokenize the prompts and completions.
# prompts = [prompt for prompt, _ in dataset]
# # completions = [completion for _, completion in dataset]
# # completion_token_ids = tokenizer(completions).input_ids
# # tokenized_dataset = []
# # for i in range(len(dataset)):
# #   output_len = len(completion_token_ids[i])
# #   tokenized_dataset.append((prompts[i], output_len))

dataset_path = f'/code/moe/datasets/{cache_configs["dataset"]}/prompt_list.pt'
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
  input_len, output_len = gen_batch(text_list, max_new_tokens=20, do_print=True)
  print(input_len, output_len, time.time() - start_time, flush=True)