import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
# os.environ['SPARSE_CACHE_LOG_LEVEL'] = 'TRACE'
# os.environ['SPARSE_CACHE_ENABLE_TRACE'] = '1'
os.environ['HUGGINGFACE_OFFLINE'] = "1"

from transformers.utils import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sparse_llm_cache
import time

from sparse_llm_cache.utils.runner_util import parse_args
cache_configs = parse_args()
for k, v in cache_configs.items(): print(k,v)

print("loading model...")
logging.disable_progress_bar()
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


sparse_llm_cache.utils.inject_model(
  model,
  **cache_configs,
  pin_memory = True,
  enable_model_timer=True,
)

model.to('cuda')

def gen_long(text, do_print=False, max_new_tokens=1000):
  inputs = tokenizer(text, return_tensors="pt").input_ids.to(f"cuda")
  outputs = model.generate(inputs, max_new_tokens=max_new_tokens)
  output_str = tokenizer.batch_decode(outputs)
  if do_print:
    print(output_str)
  return inputs[0].nelement()


# %%
import json
# dataset_path = "/nvme/songxiaoniu/moe/datasets/vllm-benchmark/shareGPT/ShareGPT_V3_unfiltered_cleaned_split.json"
dataset_path = "/nvme/songxiaoniu/moe/datasets/vllm-benchmark/shareGPT/small-dataset.json"
with open(dataset_path) as f:
  dataset = json.load(f)

# # Filter out the conversations with less than 2 turns.
# dataset = [data for data in dataset if len(data["conversations"]) >= 2]
# # Only keep the first two turns of each conversation.
# dataset = [(data["conversations"][0]["value"],
#             data["conversations"][1]["value"]) for data in dataset]

# Tokenize the prompts and completions.
prompts = [prompt for prompt, _ in dataset]
completions = [completion for _, completion in dataset]
completion_token_ids = tokenizer(completions).input_ids
tokenized_dataset = []
for i in range(len(dataset)):
  output_len = len(completion_token_ids[i])
  tokenized_dataset.append((prompts[i], output_len))


for seq_id in range(len(tokenized_dataset)):
  start_time = time.time()
  try:
    text, output_len = tokenized_dataset[seq_id]
    prompt_len = gen_long(text, max_new_tokens=min(output_len, 500), do_print=False)
    # prompt_len = gen_long(text, max_new_tokens=output_len, do_print=False)
  except Exception as e:
    print(f"error at seq {seq_id}")
    print(str(e))
  print(time.time() - start_time)
