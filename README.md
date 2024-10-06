# MoE Cache

## Install

Please refer to [./install.md](./install.md)

## Prepare the model (transformers)

```bash
# Download model from huggingface. It will be stored at ~/.cache/huggingface/hub
huggingface-cli download deepseek-ai/deepseek-moe-16b-chat
huggingface-cli download Qwen/Qwen1.5-MoE-A2.7B-Chat
# Patch model implementation to use our cache
cp -r /code/sparse-llm-cache-scripts/huggingface-modules/modules /root/.cache/huggingface/
# optionally download tokenizers for two large model
huggingface-cli download Qwen/Qwen2-57B-A14B-Instruct tokenizer.json tokenizer_config.json vocab.json special_tokens_map.json
huggingface-cli download mistralai/Mixtral-8x7B-Instruct-v0.1 special_tokens_map.json tokenizer.json tokenizer.model tokenizer_config.json
```

## Prepare the model (llama.cpp)

```bash
# convert huggingface safetensors to gguf
python3 /code/llama.cpp/convert_hf_to_gguf.py --outtype f16 --outfile <path-to-gguf-file> <path-to-input-huggingface-model-snapshot>
##### for example
mkdir -p /code/huggingface-gguf/DeepSeek-V2-Lite-Chat/f16
python3 /code/llama.cpp/convert_hf_to_gguf.py --outtype f16 --outfile /code/huggingface-gguf/DeepSeek-V2-Lite-Chat/f16/main.gguf /code/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7

# convert gguf to quantized version
/code/llama.cpp/build/bin/llama-quantize <path-to-input-gguf> <path-to-output-gguf> <quantization-method>
##### for example
mkdir -p /code/huggingface-gguf/DeepSeek-V2-Lite-Chat/q4_k_m
cd /code/huggingface-gguf/DeepSeek-V2-Lite-Chat
/app/build/bin/llama-quantize ./f16/main.gguf ./q4_k_m/main.gguf q4_k_m
```

## Prepare dataset

```bash
cd /code/sparse-llm-cache-scripts/dataset/chatgpt-prompts-small
bash ./get.sh
python3 to_prompt_list.py
```

## Run (llama.cpp)

```bash
# for example
/code/llama.cpp/build/bin/llama-parallel --model /code/huggingface-gguf/deepseek-moe-16b-chat/q4_k_m/main.gguf \
  --file /code/sparse-llm-cache-scripts/dataset/chatgpt-prompts-small/prompt_list.txt \
  --delay-escape  --sequences 3 --ctx-size 512 --gpu-layers 100 --predict 128 --no-cont-batching \
  --moe_cache 1 --num_predict 6 --moe_cache_rate 0.375 --reorder_experts True --early_preempt True  \
  --pred_model_path /code/moe/moe-predict-models/models--deepseek-ai--deepseek-moe-16b-chat/moe-layer-logits 
```

## Run (transformers)

### Run small example

```bash
cd /code/sparse-llm-cache/examples/small-demo
# run baseline
make run-base
# run ours
make run-ours
```

### Run full eval

```bash
cd /code/sparse-llm-cache/examples/full-eval
# run both baseline and ours under various cache rate. This could take hours.
python runner.py run
# parse results to output.csv
python runner.py parse
```

Figures can be plotted using `/code/sparse-llm-cache/examples/full-eval/plot.ipynb`