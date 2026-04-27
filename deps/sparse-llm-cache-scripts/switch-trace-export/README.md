# switch-trace-export

将 HuggingFace **Switch Transformers**（如 `switch-base-128`）在推理时的 **router 行为**导出为与 `train_predict_model.py` **数据流等价**的契约目录：`encoder/` 与 `decoder/` 各自一套 `.pt` 文件，可直接作为 `--logits_path` 输入训练 predictor，无需改训练脚本。

设计背景与契约约定见仓库内文档：[Switch Trace Export 设计](../../../docs/superpowers/specs/2026-04-24-switch-trace-export-design.md)。

## 依赖与环境

- 需要已安装的 PyTorch、Transformers，以及本地 Switch 模型目录（含 `config.json`）。
- 本仓库默认 prompt 与模型路径相对于 `deps/sparse-llm-cache-scripts/`（与 `trace.sh` 同级约定）。
- Shell 包装器使用 `python3`；若系统 `python3` 不是目标环境，请先 `conda activate` 后再运行，或直接用下方「直接调用 CLI」一节中的解释器路径。

## 一键运行（带契约校验）

在 `deps/sparse-llm-cache-scripts/` 下：

```bash
chmod +x switch_trace.sh   # 首次克隆后如需要
./switch_trace.sh
```

当本目录下**已存在**包装器默认的模型与 prompt 路径（见下表 `MODEL_PATH`、`PROMPT_FILE`）时，可直接按上面命令运行。若当前 checkout 未包含 `dataset/chatgpt-prompts/` 等默认资源，请显式设置 `PROMPT_FILE`（及按需设置 `MODEL_PATH`），例如：`PROMPT_FILE=/path/to/prompt_list.txt ./switch_trace.sh`。

默认会调用 `switch_trace_export.py` 并附带 **`--verify`**（契约层 Verifier）。可通过环境变量覆盖：

| 变量 | 含义 | 默认 |
|------|------|------|
| `MODEL_PATH` | 本地 HF Switch 模型目录 | `.../sparse-llm-cache-scripts/huggingface-modules/.../switch-base-128` |
| `PROMPT_FILE` | 每行一条 prompt 的文本文件 | `.../dataset/chatgpt-prompts/prompt_list.txt` |
| `OUTPUT_DIR` | 输出根目录 | `.../deps/moe-traces/switch-base-128-chatgpt-prompts` |
| `MAX_NEW_TOKENS` | 解码最大新 token 数 | `64` |
| `BATCH_SIZE` | 批大小 | `8` |
| `DEVICE` | 如 `cuda:0` / `cpu` | `cuda:0` |
| `SEED` | 随机种子 | `42` |

## 直接调用 CLI

```bash
python3 switch-trace-export/switch_trace_export.py \
  --model-path /path/to/switch-base-128 \
  --prompt-file /path/to/prompt_list.txt \
  --output-dir /path/to/out \
  --max-new-tokens 64 \
  --batch-size 8 \
  --device cuda:0 \
  --seed 42 \
  --verify
```

可选标志：

- **`--verify`**：写入后对 `encoder/` 与 `decoder/` 运行契约 Verifier。
- **`--smoke-train`**：在两侧子目录上各跑一次 `train-predict-model/train_predict_model.py` 端到端冒烟训练（需该脚本存在且依赖齐全）；用于确认产出可被训练脚本消费。

## 输出目录结构

在 `--output-dir` 下会生成：

- **`encoder/`** — encoder 阶段稀疏层 router 契约（与 `Trace.unpack_from_dir` 约定一致的文件名与形状）。
- **`decoder/`** — decoder 自回归阶段契约。
- **`run_log.json`** — 运行摘要（如 `seed`、`batch_size`、`N_enc`、`N_dec`、`elapsed_sec` 等）。

具体张量命名与轴含义以设计文档与本文「输出目录结构」、以及写出到磁盘上的文件名为准。若你使用的仓库快照里存在 `train-predict-model/TRACE_DATA_FORMAT.md`，可一并参考。

## 用 `train_predict_model.py` 训练 predictor

对 **encoder** 或 **decoder** 分别指定对应子目录为 `--logits_path`（不要混用两阶段数据于一次训练）。示例（路径请按实际修改）：

```bash
python3 train-predict-model/train_predict_model.py \
  --logits_path /path/to/out/encoder \
  --predict_model_path /path/to/predict_encoder \
  --predict_output freq \
  --model_type split \
  --window 1 \
  --hidden_size 16 \
  --n_layer 1 \
  --batch_size 16 \
  --lr 0.001 \
  --threshold 1.0 \
  --threshold_window 1 \
  --model_index 0
```

decoder 将 `--logits_path` 改为 `.../decoder` 即可。超参需与任务规模匹配；`--smoke-train` 使用的就是与 CLI 内置冒烟一致的一组较小超参。

## 测试

在 `switch-trace-export/` 目录：

```bash
python -m pytest tests/ -q
```

需要本地 Switch 权重时，模型应位于 `sparse-llm-cache-scripts/huggingface-modules/.../switch-base-128`，或设置：

- **`PROMOE_SWITCH_TRACE_MODEL_PATH`**：指向本地模型目录；
- 或 **`PROMOE_SWITCH_TRACE_ALLOW_REMOTE_MODEL=1`**：允许从 Hugging Face 拉取（需网络与缓存）。

较慢的端到端训练冒烟测试可带 `-m slow` 筛选。
