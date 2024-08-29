# MoE Cache

## Prepare the model

```bash
# Download model from huggingface. It will be stored at ~/.cache/huggingface/hub
huggingface-cli download deepseek-ai/deepseek-moe-16b-chat
huggingface-cli download Qwen/Qwen1.5-MoE-A2.7B-Chat
# Patch model implementation to use our cache
cp -r /code/sparse-llm-cache-scripts/huggingface-modules/modules /root/.cache/huggingface/
```

## Prepare dataset

```bash
cd /code/sparse-llm-cache-scripts/dataset/chatgpt-prompts-small
bash ./get.sh
python3 to_prompt_list.py
```

## Run small example

```bash
cd /code/sparse-llm-cache/examples/small-demo
# run baseline
make run-base
# run ours
make run-ours
```

## Run full eval

```bash
cd /code/sparse-llm-cache/examples/full-eval
# run both baseline and ours under various cache rate. This could take hours.
python runner.py run
# parse results to output.csv
python runner.py parse
```

Figures can be plotted using `/code/sparse-llm-cache/examples/full-eval/plot.ipynb`