# Experiment Trace Scripts

This directory contains trace/export utilities used by the experiment layout.

## Encoder ERPP Trace Training Notes

The encoder ERPP trace is generated through:

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python \
  experiment/scripts/trace/run_encoder_erpp_trace.py \
  --model-path experiment/models/google/switch-base-128 \
  --dataset mmlu \
  --task-name professional_law \
  --device cuda:0
```

The wrapper calls the existing exporter:

```text
performance_predictor/encoder/ERPP/implement/trace/export_erpp_encoder_trace.py
```

Default input:

```text
experiment/datasets/<dataset>/<task>/test/prompt_list.txt
experiment/datasets/<dataset>/<task>/validation/prompt_list.txt
```

Default output:

```text
experiment/traces/<model>-<dataset>-<task>/encoder_erpp_trace/
```

The `test` split is used as the training split, and `validation` is used as the validation split.

## Runtime Input Policy

The wrapper follows the small-demo single-sample inference shape:

```text
--padding longest
--batch-size 1
--max-input-tokens 512
```

With `batch_size=1`, `padding=longest` means the model sees only the prompt's real token length, capped by `max_input_tokens`. It does not pad every sample to 512 before the encoder/router forward pass.

This is different from:

```text
--padding max_length
```

`max_length` padding would add padding tokens before model forward. Those padding positions would pass through encoder hidden states and router logic, which is not the desired small-demo behavior.

## Storage Padding

Samples in one split can have different real token lengths. To save them as one tensor, the exporter pads only when writing files.

Example:

```text
sample 0 real length = 151
sample 1 real length = 132
split storage length = 151
```

`sample 1` is padded from 132 to 151 only at save time. The extra positions are zero-filled storage padding, not model-forward tokens.

This storage padding applies to files with a token dimension:

```text
input_ids.pt          [S, T]
attention_mask.pt     [S, T]
layer0_attn_out.pt    [S, T, H]
router_logits.pt      [S, L, T, E]
expert_selection.pt   [S, L, T, K]
router_probs.pt       [S, L, T, E]
expert_weights.pt     [S, L, T, K]
```

Files without token padding:

```text
seq_ids.pt
prompt_texts.jsonl
metadata.json
run_log.json
```

## File Meanings

`input_ids.pt`

Token ids after tokenizer truncation. Storage-padded positions are `0`.

`attention_mask.pt`

The source of truth for valid tokens:

```text
1 = real token
0 = storage padding
```

`layer0_attn_out.pt`

Encoder layer-0 attention output hidden states. This is the predictor input feature used by ERPP-style encoder predictors.

`router_logits.pt`

Router logits for encoder MoE layers:

```text
[S, L, T, E]
```

`expert_selection.pt`

Raw ERPP exporter expert index tensor:

```text
[S, L, T, K]
```

Do not treat storage-padded positions as valid labels. They are present only because tensors must share one storage length.

`router_probs.pt`

Softmax of `router_logits.pt`.

`expert_weights.pt`

Router probabilities gathered at `expert_selection.pt`.

`seq_ids.pt`

Original sample ids inside the split.

`prompt_texts.jsonl`

Prompt text records aligned with `seq_ids.pt`.

## Training Rules

Always use `attention_mask.pt` to filter storage padding before computing loss.

For token-level features:

```python
attention_mask = torch.load("train/attention_mask.pt")  # [S, T]
layer0 = torch.load("train/layer0_attn_out.pt")         # [S, T, H]

valid_tokens = attention_mask.bool()                    # [S, T]
x = layer0[valid_tokens]                                # [N, H]
```

For router-layer labels:

```python
attention_mask = torch.load("train/attention_mask.pt")      # [S, T]
expert_selection = torch.load("train/expert_selection.pt")  # [S, L, T, K]

target = expert_selection[..., 0]                           # [S, L, T]
valid = attention_mask.bool()[:, None, :]                   # [S, 1, T]
valid = valid.expand_as(target)                             # [S, L, T]

target_valid = target[valid]                                # [N]
```

For cross-entropy over experts:

```python
router_pred = model(layer0_attn_out)  # expected [S, L, T, E]
target = expert_selection[..., 0]     # [S, L, T]
valid = attention_mask.bool()[:, None, :].expand_as(target)

loss = torch.nn.functional.cross_entropy(
    router_pred[valid],  # [N, E]
    target[valid],       # [N]
)
```

Never compute loss over `attention_mask == 0` positions.

## Common Mistakes

Do not assume all positions in `[S, T]` are valid.

`T` is the split storage length, not every sample's real length.

Do not train on padded `expert_selection == 0` positions.

Storage padding fills zeros, so padded label positions may look like expert 0. They are not valid labels.

Do not confuse trace storage dtype with model forward dtype.

`--storage-dtype` controls saved floating tensors. `--model-torch-dtype` controls model loading dtype. By default the wrapper keeps model config precision.

Do not use `--padding max_length` unless you intentionally want padding tokens to participate in model forward.

For small-demo-style single prompt inference, use the wrapper defaults.
