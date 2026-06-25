from __future__ import annotations

import math
import os
import torch
from torch.utils.data import DataLoader
import sys
import utils
from utils import Trace
try:
  from transformers.utils import logging
except ModuleNotFoundError:
  class _NoopTransformersLogging:
    @staticmethod
    def disable_progress_bar():
      return None

  logging = _NoopTransformersLogging()
import torch
import gc
import argparse
import json
from types import SimpleNamespace
from tqdm import tqdm
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES']='0' 
os.environ['HF_HUB_OFFLINE'] = "1" 
os.environ['HUGGINGFACE_OFFLINE'] = "1" 
# device = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


torch.set_printoptions(threshold=10000,linewidth= 300000) 

sys.path.append('/mnt/huwf5/promoe/deps/sparse-llm-cache-scripts/train-predict-model')

logging.disable_progress_bar()

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--logits_path",            type=str)
  parser.add_argument("--val_logits_path",        type=str,   default=None)
  parser.add_argument("--predict_model_path",     type=str)
  parser.add_argument("--train_log_path",         type=str,   default=None)
  parser.add_argument("--hidden_size",            type=int,   default=1024)
  parser.add_argument("--window_begin",           type=int,   default=0)
  parser.add_argument("--window",                 type=int,   default=6)
  parser.add_argument("--batch_size",             type=int,   default=8192)
  parser.add_argument("--eval_batch_size",        type=int,   default=256)
  parser.add_argument("--n_layer",                type=int,   default=2)
  parser.add_argument("--lr",                     type=float, default=0.001)
  parser.add_argument("--dropout",                type=float, default=0.5)
  parser.add_argument("--threshold",              type=float, default=0.01)
  parser.add_argument("--threshold_window",       type=int,   default=20)
  parser.add_argument("--model_index", nargs='+', type=int,   default=None)
  parser.add_argument("--predict_output",         type=str, choices=["gate", "freq"], default="freq")
  parser.add_argument("--predict_input",          type=str, choices=["moe-layer-logits", "token-id"], default="moe-layer-logits")
  parser.add_argument("--input_norm_method",      type=str, choices=["max1", "std", "replace"], default="max1")
  parser.add_argument("--loss_func",              type=str, choices=["l1", "smooth-l1"], default="l1")
  parser.add_argument("--print_loss",             default=False, action='store_true')
  parser.add_argument("--model_type",             type=str, choices=["single", "split"], default="split")
  parser.add_argument("--id_space",              type=str, choices=["local", "global"], default=None)
  parser.add_argument("--predict_stage",         type=str, choices=["decoder"], default=None)
  parser.add_argument("--num_encoder_moe_layer", type=int,   default=None)
  parser.add_argument("--num_decoder_moe_layer", type=int,   default=None)
  args = parser.parse_args()

  if args.train_log_path == None:
    args.train_log_path = args.predict_model_path + "/train_log"

  assert os.path.abspath(os.path.normpath(args.train_log_path)) != os.path.abspath(os.path.normpath(args.predict_model_path)), "train_log_path and predict_model_path cannot be the same."

  # convert args to dict
  metas = vars(args)

  return args, metas


def _apply_trace_layout_config(trace: utils.Trace, trace_config) -> None:
  if trace_config is None:
    return

  for key in ("id_space", "predict_stage", "num_encoder_moe_layer", "num_decoder_moe_layer"):
    value = getattr(trace_config, key, None)
    if value is not None and key not in trace.trace_metas:
      trace.trace_metas[key] = value

  if trace.trace_metas.get("id_space") == "global":
    if "num_layer" not in trace.trace_metas:
      trace.trace_metas["num_layer"] = int(trace.expert_selection.shape[1])
    if "schema_version" not in trace.trace_metas:
      trace.trace_metas["schema_version"] = 2
    if "predict_stage" not in trace.trace_metas:
      trace.trace_metas["predict_stage"] = "decoder"

    missing = [
      key
      for key in ("num_encoder_moe_layer", "num_decoder_moe_layer")
      if key not in trace.trace_metas
    ]
    if missing:
      raise ValueError(
        "global trace layout requires trace_metas.json or CLI values for "
        + ", ".join(missing)
      )

    e = int(trace.trace_metas["num_encoder_moe_layer"])
    l = int(trace.trace_metas["num_layer"])
    trace.trace_metas.setdefault("decoder_global_layer_start", e)
    trace.trace_metas.setdefault("decoder_global_layer_stop", l)


def prepare_trace(logits_path: str, trace_config=None) -> utils.Trace:
  print("Loading trace...")
  trace = Trace()
  trace.unpack_from_dir(logits_path)
  _apply_trace_layout_config(trace, trace_config)

  trace.num_expert = int(torch.max(trace.expert_selection)) + 1
  trace.num_moe_layer = list(trace.expert_selection.shape)[1]
  trace.per_token_expert = list(trace.expert_selection.shape)[2]

  trace.prepare_tensors()

  print("Loding trace done!")
  return trace

def save_log_metas(trace: Trace, train_log_path, metas):
  metas['num_expert']       = trace.num_expert
  metas['num_moe_layer']    = trace.num_moe_layer
  metas['per_token_expert'] = trace.per_token_expert

  with open(f"{train_log_path}/metas.json", "w") as json_file:
    json.dump(metas, json_file, indent=4)

def _global_decoder_trace(trace) -> bool:
  return getattr(trace, 'id_space', 'local') == 'global' and getattr(trace, 'predict_stage', 'decoder') == 'decoder'

def _boundary_layer_id(trace) -> int:
  boundary_id = getattr(trace, 'num_layer', None)
  if boundary_id is None:
    boundary_id = trace.num_moe_layer
  return boundary_id

def predictor_source_ids(trace):
  if _global_decoder_trace(trace):
    return list(range(trace.num_encoder_moe_layer, trace.num_layer + 1))
  return list(range(trace.num_moe_layer + 1))

def predictor_output_range(trace, train_config, input_layer: int) -> tuple[int, int]:
  if _global_decoder_trace(trace):
    start = trace.num_encoder_moe_layer if input_layer == trace.num_layer else input_layer
    stop = min(start + train_config.window, trace.num_layer)
    if start < trace.num_encoder_moe_layer or stop > trace.num_layer:
      raise ValueError(
        f"global decoder output range [{start}, {stop}) is outside "
        f"decoder target range [{trace.num_encoder_moe_layer}, {trace.num_layer})"
      )
    return start, stop

  output_layer_mod = input_layer if input_layer < trace.num_moe_layer else 0
  return output_layer_mod, min(output_layer_mod + train_config.window, trace.num_moe_layer)

def iter_split_training_pairs(trace, train_config):
  boundary_id = _boundary_layer_id(trace)
  for src in predictor_source_ids(trace):
    if src not in train_config.model_index:
      continue
    start, stop = predictor_output_range(trace, train_config, src)
    for dst in range(start + train_config.window_begin, stop):
      token_distance = 1 if src == boundary_id else 0
      yield src, dst, token_distance

def _routing_metric_counts(predict_freq: torch.Tensor, correct_freq: torch.Tensor):
  if predict_freq.shape != correct_freq.shape:
    predict_freq = predict_freq.reshape_as(correct_freq)

  num_expert = correct_freq.shape[-1]
  actual_idx = correct_freq.argmax(dim=-1)
  _, predict_idx = predict_freq.sort(dim=-1, descending=True)
  matches = predict_idx.eq(actual_idx.unsqueeze(-1))
  total = actual_idx.numel()
  rank = matches.to(torch.int64).argmax(dim=-1).to(torch.float64) + 1

  return {
    "total": total,
    "top1_correct": int(matches[..., :1].any(dim=-1).sum().item()),
    "recall_at_2_correct": int(matches[..., :min(2, num_expert)].any(dim=-1).sum().item()),
    "recall_at_3_correct": int(matches[..., :min(3, num_expert)].any(dim=-1).sum().item()),
    "mrr_sum": float((1.0 / rank).sum().item()),
  }

def _merge_routing_metric_counts(total_counts, batch_counts):
  if total_counts is None:
    return dict(batch_counts)
  for key, value in batch_counts.items():
    total_counts[key] += value
  return total_counts

def _routing_metrics_from_counts(counts):
  total = counts["total"]
  if total == 0:
    return {
      "top1_acc": math.nan,
      "recall_at_2": math.nan,
      "recall_at_3": math.nan,
      "mrr": math.nan,
    }
  return {
    "top1_acc": counts["top1_correct"] / total,
    "recall_at_2": counts["recall_at_2_correct"] / total,
    "recall_at_3": counts["recall_at_3_correct"] / total,
    "mrr": counts["mrr_sum"] / total,
  }

def compute_routing_metrics(predict_freq: torch.Tensor, correct_freq: torch.Tensor):
  return _routing_metrics_from_counts(_routing_metric_counts(predict_freq, correct_freq))

def train_loop(model_ctx : utils.ModelContext, train_loader: DataLoader, test_loader: DataLoader, train_config, max_epochs=2000):
  train_loss_list = []
  test_loss_list  = []
  test_metrics_list = []

  for epoch in range(max_epochs):
    model_ctx.model.train()
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}", leave=False)

    loss_list = []
    for inputs, labels, _ in progress_bar:
      _, loss = model_ctx.model_forward_with_loss_and_optimize(inputs, labels)
      loss_list.append(loss)

    train_loss = torch.asarray(loss_list).mean().item()
    train_loss_list.append(train_loss)
    
    model_ctx.model.eval()
    loss_list = []
    metric_counts = None
    with torch.no_grad():
      for inputs, labels, _ in test_loader:
        outputs, loss = model_ctx.model_forward_with_loss(inputs, labels)
        loss_list.append(loss)
        metric_counts = _merge_routing_metric_counts(
          metric_counts,
          _routing_metric_counts(outputs.reshape_as(labels), labels),
        )
    
    test_loss = torch.asarray(loss_list).mean().item()
    test_loss_list.append(test_loss)
    test_metrics = _routing_metrics_from_counts(metric_counts)
    test_metrics_list.append(test_metrics)

    loss_delta = math.nan

    if len(test_loss_list) > train_config.threshold_window * 2:
      test_loss_tensor = torch.asarray(test_loss_list)
      loss_part_1 = test_loss_tensor[-2*train_config.threshold_window:-train_config.threshold_window].mean()
      loss_part_2 = test_loss_tensor[-train_config.threshold_window:].mean()
      loss_delta = (loss_part_1 - loss_part_2).abs() / loss_part_1

    if train_config.print_loss:
      print(
        f"Epoch [{epoch+1:4d}/{max_epochs}], "
        f"Train loss: {train_loss:.6f}, Test loss: {test_loss:.6f}, "
        f"Loss delta: {loss_delta:.6f}, "
        f"Val top1: {test_metrics['top1_acc']:.4f}, "
        f"Val r@2: {test_metrics['recall_at_2']:.4f}, "
        f"Val r@3: {test_metrics['recall_at_3']:.4f}, "
        f"Val mrr: {test_metrics['mrr']:.4f}"
      )

    if math.isnan(loss_delta) == False and loss_delta < train_config.threshold:
      break
  
  return train_loss_list, test_loss_list, test_metrics_list

def save_accuracy_logs(trace: Trace, net: utils.ModelContext, dataset, idx_list, logs_file_name: str, layer_list = None, eval_batch_size: int = 2048):
  # whole_output = utils.custom_norm_no_scale(dataset.orig_labels)
  whole_output = dataset.orig_labels.detach().cpu()
  num_local_moe_layer = dataset.orig_labels.shape[1] # numseq, numlayer, numexpert
  ideal_records = torch.zeros((len(idx_list), num_local_moe_layer, trace.num_expert + 1))
  predict_records = torch.zeros((len(idx_list), num_local_moe_layer, trace.num_expert + 1))

  idx_list = idx_list.detach().cpu()
  correct_freq = whole_output[idx_list]
  predict_freq_list = []
  model_device = next(net.model.parameters(), torch.empty(0, device=dataset.input.device)).device

  with torch.no_grad():
    net.model.eval()
    for start in range(0, len(idx_list), eval_batch_size):
      batch_idx = idx_list[start:start + eval_batch_size]
      inputs = dataset.input[batch_idx]
      if inputs.device != model_device:
        inputs = inputs.to(model_device)
      batch_predict_freq = net.model_forward(inputs).reshape((len(batch_idx), num_local_moe_layer, trace.num_expert))
      predict_freq_list.append(batch_predict_freq.detach().cpu())

  predict_freq = torch.cat(predict_freq_list, dim=0)

  _, _predict_idx = predict_freq.sort(dim = -1, descending=True)
  _, _correct_idx = correct_freq.sort(dim = -1, descending=True)
  correct_freq = torch.zeros_like(correct_freq)
  correct_freq.scatter_(2, _correct_idx[:,:,:trace.per_token_expert], 1/trace.per_token_expert)
  predict_records[:,:,1:] = torch.gather(correct_freq, 2, _predict_idx).cumsum(dim=-1)
  ideal_records[:,:,1:] = torch.gather(correct_freq, 2, _correct_idx).cumsum(dim=-1)

  ideal_records = ideal_records.mean(0)
  predict_records = predict_records.mean(0)

  if layer_list == None:
    layer_list = list(range(0, num_local_moe_layer, 1))

  x = [i/trace.num_expert for i in range(trace.num_expert + 1)]

  accuracy_df = pd.DataFrame({
    "x": x,
    "avg": predict_records[layer_list].mean(0),
    "ideal": ideal_records[layer_list].mean(0),
  })

  for layer in layer_list:
    accuracy_df[f"{layer}"] = predict_records[layer]

  accuracy_df.to_csv(logs_file_name, index=False)

def save_loss_logs(train_loss_list, test_loss_list, logs_file_name, test_metrics_list = None):
  loss_data = {
    'train_loss': train_loss_list,
    "test_loss":  test_loss_list,
  }
  if test_metrics_list is not None:
    loss_data["val_top1_acc"] = [metrics["top1_acc"] for metrics in test_metrics_list]
    loss_data["val_recall_at_2"] = [metrics["recall_at_2"] for metrics in test_metrics_list]
    loss_data["val_recall_at_3"] = [metrics["recall_at_3"] for metrics in test_metrics_list]
    loss_data["val_mrr"] = [metrics["mrr"] for metrics in test_metrics_list]
  loss_df = pd.DataFrame(loss_data)

  loss_df.to_csv(logs_file_name, index=False)

def train_single_model(trace: utils.Trace, train_config, val_trace: utils.Trace | None = None) -> None:
  boundary_id = _boundary_layer_id(trace)
  if boundary_id not in train_config.model_index:
    return
  return train_one_model(trace, train_config, boundary_id, token_distance=1, val_trace=val_trace)


def _select_input_feature(trace: utils.Trace, input_layer: int) -> torch.Tensor:
  feature = trace.decode_stage_moe_layer_logits_per_token
  if input_layer < 0 or input_layer >= feature.shape[1]:
    raise ValueError(
      f"input layer {input_layer} is not available in "
      f"decode_stage_moe_layer_logits_per_token with shape {tuple(feature.shape)}. "
      "Hidden-aligned traces should provide L+1 feature layers so input_layer == num_moe_layer "
      "can train cross-token predictors."
    )
  return feature[:, input_layer:input_layer+1].to(torch.float32)


def _align_cross_token_dataset(input: torch.Tensor, output: torch.Tensor, meta_data: torch.Tensor, token_distance: int):
  if token_distance <= 0:
    return input, output, meta_data

  seq_ids = meta_data[:, 0].tolist()
  token_indices = meta_data[:, 1].tolist()
  row_by_position = {
    (int(seq_id), int(token_idx)): row
    for row, (seq_id, token_idx) in enumerate(zip(seq_ids, token_indices))
  }

  input_rows = []
  output_rows = []
  for row, (seq_id, token_idx) in enumerate(zip(seq_ids, token_indices)):
    target_row = row_by_position.get((int(seq_id), int(token_idx) + token_distance))
    if target_row is None:
      continue
    input_rows.append(row)
    output_rows.append(target_row)

  if not input_rows:
    empty = torch.empty((0,), dtype=torch.long, device=input.device)
    return input[empty], output[empty], meta_data[empty]

  input_index = torch.tensor(input_rows, dtype=torch.long, device=input.device)
  output_index = torch.tensor(output_rows, dtype=torch.long, device=output.device)
  meta_index = torch.tensor(output_rows, dtype=torch.long, device=meta_data.device)
  return input[input_index], output[output_index], meta_data[meta_index]


def _build_dataset_from_trace(
  trace: utils.Trace,
  train_config,
  input_layer: int,
  token_distance: int,
  target_layer: int | None = None,
):
  input = _select_input_feature(trace, input_layer)

  if target_layer is None:
    output_slice_begin, output_slice_end = predictor_output_range(trace, train_config, input_layer)
  else:
    output_slice_begin = target_layer
    output_slice_end = target_layer + 1

  if train_config.predict_output == "gate":
    output = trace.decode_stage_moe_layer_gate_logits_per_token[:, output_slice_begin:output_slice_end].to(torch.float32)
    if train_config.input_norm_method == "max1":
      output = utils.custom_norm_to_max_1(output)
    elif train_config.input_norm_method == "std":
      output = utils.custom_norm_std(output)
    elif train_config.input_norm_method == "replace":
      output = utils.custom_norm_fake_gate(output)
    else:
      raise ValueError(f"Unknow input_norm_method: {train_config.input_norm_method}")
  elif train_config.predict_output == "freq":
    output = trace.decode_stage_expert_freq_per_token[:, output_slice_begin:output_slice_end].to(torch.float32)
  else:
    raise ValueError(f"Unknow predict_output: {train_config.predict_output}")

  meta_data = torch.cat([trace.decode_stage_seq_id_of_token.reshape((-1, 1)), trace.decode_stage_token_idx_in_seq.reshape((-1, 1))], dim=1)

  input, output, meta_data = _align_cross_token_dataset(input, output, meta_data, token_distance)

  return utils.CustomDataset(input, output, meta_data)


def train_one_model(trace: utils.Trace, train_config, i : int, token_distance : int = 0, val_trace: utils.Trace | None = None) -> None:
  gc.collect()
  torch.cuda.empty_cache()

  print(f"Training the model that use moe layer input logits to predict next {train_config.window} layer...")

  train_dataset = _build_dataset_from_trace(trace, train_config, i, token_distance)
  if val_trace is None:
    train_dataset, test_dataset = train_dataset.split(0.9)
  else:
    test_dataset = _build_dataset_from_trace(val_trace, train_config, i, token_distance)

  # Create model
  if train_config.loss_func == "smooth-l1":
    loss_func = torch.nn.SmoothL1Loss
  elif train_config.loss_func == "l1":
    loss_func = torch.nn.L1Loss
  else:
    raise ValueError(f"Unknow loss_func: {train_config.loss_func}")
  net = utils.ModelContext(utils.SimpleNN(train_dataset[0][0].nelement(), train_config.hidden_size, train_dataset[0][1].nelement(), n_layer=train_config.n_layer, dropout=train_config.dropout), loss_func, torch.optim.Adam, lr=train_config.lr)
  net.model.to(device)

  # Train the model
  train_loader, test_loader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=True), DataLoader(test_dataset, batch_size=train_config.batch_size)
  train_dataset.to(device)
  test_dataset.to(device)

  print(f"Training model {i}....")
  train_loss_list, test_loss_list, test_metrics_list = train_loop(net, train_loader, test_loader, train_config)
  print(f"Training model {i} done!")

  # Save logs
  save_loss_logs(train_loss_list, test_loss_list, f"{train_config.train_log_path}/{i}_loss.csv", test_metrics_list)
  save_accuracy_logs(trace, net, train_dataset, torch.randperm(len(train_dataset)), f"{train_config.train_log_path}/{i}_train_acc.csv", eval_batch_size=train_config.eval_batch_size)
  save_accuracy_logs(trace, net, test_dataset, torch.randperm(len(test_dataset)), f"{train_config.train_log_path}/{i}_test_acc.csv", eval_batch_size=train_config.eval_batch_size)

  # Save parameter
  scripted_model = torch.jit.script(net.model)
  scripted_model.save(f"{train_config.predict_model_path}/{i}.pt")

  print("Training multi models done!")

def train_single_target_layer_model(trace: utils.Trace, train_config, input_layer : int, target_layer : int, token_distance : int = 0, val_trace: utils.Trace | None = None) -> None:
  gc.collect()
  torch.cuda.empty_cache()

  print(f"Training the model that use moe layer {input_layer} input logits to predict layer {target_layer}'s {train_config.predict_output}...")

  train_dataset = _build_dataset_from_trace(trace, train_config, input_layer, token_distance, target_layer=target_layer)
  if val_trace is None:
    train_dataset, test_dataset = train_dataset.split(0.9)
  else:
    test_dataset = _build_dataset_from_trace(val_trace, train_config, input_layer, token_distance, target_layer=target_layer)
  if train_config.loss_func == "smooth-l1":
    loss_func = torch.nn.SmoothL1Loss
  elif train_config.loss_func == "l1":
    loss_func = torch.nn.L1Loss
  else:
    raise ValueError(f"Unknow loss_func: {train_config.loss_func}")

  # Create model
  net = utils.ModelContext(utils.SimpleNN(train_dataset[0][0].nelement(), train_config.hidden_size, train_dataset[0][1].nelement(), n_layer=train_config.n_layer, dropout=train_config.dropout), loss_func, torch.optim.Adam, lr=train_config.lr)
  # net = utils.ModelContext(utils.SimpleNN(train_dataset[0][0].nelement(), train_config.hidden_size, train_dataset[0][1].nelement(), n_layer=train_config.n_layer, dropout=train_config.dropout), loss_func, torch.optim.SGD, lr=train_config.lr)
  net.model.to(device)

  # Train the model
  train_loader, test_loader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=True), DataLoader(test_dataset, batch_size=train_config.batch_size)
  train_dataset.to(device)
  test_dataset.to(device)

  print(f"Training model {input_layer}-{target_layer}....")
  train_loss_list, test_loss_list, test_metrics_list = train_loop(net, train_loader, test_loader, train_config)
  print(f"Training model {input_layer}-{target_layer} done!")

  # Save logs
  save_loss_logs(train_loss_list, test_loss_list, f"{train_config.train_log_path}/{input_layer}-{target_layer}_loss.csv", test_metrics_list)
  save_accuracy_logs(trace, net, train_dataset, torch.randperm(len(train_dataset)), f"{train_config.train_log_path}/{input_layer}-{target_layer}_train_acc.csv", eval_batch_size=train_config.eval_batch_size)
  save_accuracy_logs(trace, net, test_dataset, torch.randperm(len(test_dataset)), f"{train_config.train_log_path}/{input_layer}-{target_layer}_test_acc.csv", eval_batch_size=train_config.eval_batch_size)

  # Save parameter
  scripted_model = torch.jit.script(net.model)
  scripted_model.save(f"{train_config.predict_model_path}/{input_layer}-{target_layer}.pt")

  print("Training multi models done!")

def _build_token_id_dataset_from_trace(
  trace: utils.Trace,
  train_config,
  input_layer: int,
  token_distance: int,
):
  input = trace.decode_stage_token_ids_per_token
  output_slice_begin, output_slice_end = predictor_output_range(trace, train_config, input_layer)

  if train_config.predict_output == "gate":
    raise NotImplementedError("token-id training only supports predict_output=freq")
  elif train_config.predict_output == "freq":
    output = trace.decode_stage_expert_freq_per_token[:, output_slice_begin:output_slice_end].to(torch.float32)
  else:
    raise ValueError(f"Unknow predict_output: {train_config.predict_output}")

  meta_data = torch.cat([trace.decode_stage_seq_id_of_token.reshape((-1, 1)), trace.decode_stage_token_idx_in_seq.reshape((-1, 1))], dim=1)
  input, output, meta_data = _align_cross_token_dataset(input, output, meta_data, token_distance)
  return utils.CustomDataset(input, output, meta_data)


def train_token_id_model(trace: utils.Trace, train_config, i : int = 0, token_distance : int = 0, val_trace: utils.Trace | None = None) -> None:
  gc.collect()
  torch.cuda.empty_cache()

  class TokenIdModel(torch.nn.Module):
    def __init__(self, token_id_to_expert_freq):
      super(TokenIdModel, self).__init__()
      self.register_buffer("token_id_to_expert_freq", token_id_to_expert_freq)

    def forward(self, x):
      return self.token_id_to_expert_freq[x.long()]

  class TokenIdModelContext:
    def __init__(self, model):
      self.model = model

    def model_forward(self, inputs):
      with torch.no_grad():
        return self.model(inputs.reshape(-1))

  print(f"Training the model that use token id to predict next {train_config.window} layer...")

  train_dataset = _build_token_id_dataset_from_trace(trace, train_config, i, token_distance)
  if val_trace is None:
    train_dataset, test_dataset = train_dataset.split(0.9)
  else:
    test_dataset = _build_token_id_dataset_from_trace(val_trace, train_config, i, token_distance)

  # Create model

  n_vocab = trace.token_ids.max().item() + 1
  output_size = train_dataset[0][1].nelement()
  expert_freq_for_each_token = torch.zeros((n_vocab, output_size))
  for entry_idx in tqdm(range(len(train_dataset))):
    token_id = train_dataset[entry_idx][0]
    expert_freq_for_each_token[token_id] += train_dataset[entry_idx][1].reshape(-1)

  # Create model
  net = TokenIdModelContext(TokenIdModel(expert_freq_for_each_token))

  save_accuracy_logs(trace, net, train_dataset, torch.randperm(len(train_dataset)), f"{train_config.train_log_path}/{i}_train_acc.csv", eval_batch_size=train_config.eval_batch_size)
  save_accuracy_logs(trace, net, test_dataset,  torch.randperm(len(test_dataset)),  f"{train_config.train_log_path}/{i}_test_acc.csv", eval_batch_size=train_config.eval_batch_size)

  # Save parameter
  scripted_model = torch.jit.script(net.model)
  scripted_model.save(f"{train_config.predict_model_path}/{i}.pt")

  print("Training multi models done!")

def train_multi_models(trace: utils.Trace, train_config, val_trace: utils.Trace | None = None) -> None:
  boundary_id = _boundary_layer_id(trace)
  for i in train_config.model_index:
    if i == boundary_id:
      continue
    train_one_model(trace, train_config, i, val_trace=val_trace)

def prepare_metas(trace: utils.Trace, predict_model_path: str, window: int, model_index=None) -> None:
  if _global_decoder_trace(trace):
    cfg = SimpleNamespace(window=window, window_begin=0)
    outputs = {}
    source_ids = predictor_source_ids(trace)
    if model_index is not None:
      model_index_set = set(model_index)
      source_ids = [src for src in source_ids if src in model_index_set]
    for src in source_ids:
      outputs[str(src)] = list(predictor_output_range(trace, cfg, src))

    json_data = {
      "schema_version": 2,
      "id_space": "global",
      "num_layer": trace.num_layer,
      "num_encoder_moe_layer": trace.num_encoder_moe_layer,
      "num_decoder_moe_layer": trace.num_decoder_moe_layer,
      "predict_stage": getattr(trace, "predict_stage", "decoder"),
      "predictor_input_start_layer": trace.num_encoder_moe_layer,
      "predictor_input_stop_layer": trace.num_layer + 1,
      "outputs": outputs,
    }
  else:
    json_data = {}
    source_ids = list(range(trace.num_moe_layer + 1))
    if model_index is not None:
      model_index_set = set(model_index)
      source_ids = [src for src in source_ids if src in model_index_set]
    for i in source_ids:
      if i < trace.num_moe_layer:
        json_data[str(i)] = [i, min(i + window, trace.num_moe_layer)]
      else:
        json_data[str(trace.num_moe_layer)] = [0, min(window, trace.num_moe_layer)]

  with open(f"{predict_model_path}/metas.json", 'w', encoding='utf-8') as json_file:
    json.dump(json_data, json_file, ensure_ascii=False, indent=2)
  
  return 

if __name__ == "__main__":
  train_config, config_metas = parse_args()
  os.system(f'mkdir -p {train_config.predict_model_path}')
  os.system(f'mkdir -p {train_config.train_log_path}')

  trace = prepare_trace(train_config.logits_path, train_config)
  val_trace = None
  if train_config.val_logits_path is not None:
    print("Loading validation trace...")
    val_trace = prepare_trace(train_config.val_logits_path, train_config)
    if trace.num_moe_layer != val_trace.num_moe_layer or trace.num_expert != val_trace.num_expert:
      raise ValueError(
        "Training trace and validation trace mismatch: "
        f"(train num_moe_layer={trace.num_moe_layer}, num_expert={trace.num_expert}) vs "
        f"(val num_moe_layer={val_trace.num_moe_layer}, num_expert={val_trace.num_expert})"
      )

  if train_config.window == -1:
    train_config.window = trace.num_moe_layer

  if train_config.model_index == None:
    train_config.model_index = predictor_source_ids(trace)
    config_metas['model_index'] = train_config.model_index

  save_log_metas(trace, train_config.train_log_path, config_metas)

  print("Training models...")
  if train_config.predict_input == "token-id":
    boundary_id = _boundary_layer_id(trace)
    for input_layer in train_config.model_index:
      token_distance = 1 if input_layer == boundary_id else 0
      train_token_id_model(trace, train_config, input_layer, token_distance=token_distance, val_trace=val_trace)
  elif train_config.predict_input == "moe-layer-logits":
    if train_config.model_type == "split":
      for input_layer, target_layer, token_distance in iter_split_training_pairs(trace, train_config):
        train_single_target_layer_model(
          trace,
          train_config,
          input_layer,
          target_layer,
          token_distance,
          val_trace=val_trace,
        )
    elif train_config.model_type == "single":
      train_single_model(
        trace, 
        train_config,
        val_trace=val_trace
      )
      train_multi_models(
        trace, 
        train_config,
        val_trace=val_trace
      )
    else:
      raise ValueError(f"Unknow model_type: {train_config.model_type}")
  else:
    raise ValueError(f"Unknow predict_input: {train_config.predict_input}")

  print("Training models done!")

  print("Writing metas.json...")
  prepare_metas(trace, train_config.predict_model_path, train_config.window, model_index=train_config.model_index)
  print("Writing metas.json done!")
