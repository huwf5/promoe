import math
import os
import torch
from torch.utils.data import DataLoader
import sys
import utils
from utils import Trace
from transformers.utils import logging
import torch
import gc
import argparse
import json
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
  parser.add_argument("--predict_model_path",     type=str)
  parser.add_argument("--train_log_path",         type=str,   default=None)
  parser.add_argument("--hidden_size",            type=int,   default=1024)
  parser.add_argument("--window_begin",           type=int,   default=0)
  parser.add_argument("--window",                 type=int,   default=6)
  parser.add_argument("--batch_size",             type=int,   default=8192)
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
  args = parser.parse_args()

  if args.train_log_path == None:
    args.train_log_path = args.predict_model_path + "/train_log"

  assert os.path.abspath(os.path.normpath(args.train_log_path)) != os.path.abspath(os.path.normpath(args.predict_model_path)), "train_log_path and predict_model_path cannot be the same."

  # convert args to dict
  metas = vars(args)

  return args, metas


def prepare_trace(logits_path: str) -> utils.Trace:
  print("Loading trace...")
  trace = Trace()
  trace.unpack_from_dir(logits_path)

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

def train_loop(model_ctx : utils.ModelContext, train_loader: DataLoader, test_loader: DataLoader, train_config, max_epochs=2000):
  train_loss_list = []
  test_loss_list  = []

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
    with torch.no_grad():
      for inputs, labels, _ in test_loader:
        _, loss = model_ctx.model_forward_with_loss(inputs, labels)
        loss_list.append(loss)
    
    test_loss = torch.asarray(loss_list).mean().item()
    test_loss_list.append(test_loss)

    loss_delta = math.nan

    if len(test_loss_list) > train_config.threshold_window * 2:
      test_loss_tensor = torch.asarray(test_loss_list)
      loss_part_1 = test_loss_tensor[-2*train_config.threshold_window:-train_config.threshold_window].mean()
      loss_part_2 = test_loss_tensor[-train_config.threshold_window:].mean()
      loss_delta = (loss_part_1 - loss_part_2).abs() / loss_part_1

    if train_config.print_loss:
      print(f"Epoch [{epoch+1:4d}/{max_epochs}], Test loss: {test_loss:.6f}, Loss delta: {loss_delta:.6f}")

    if math.isnan(loss_delta) == False and loss_delta < train_config.threshold:
      break
  
  return train_loss_list, test_loss_list

def save_accuracy_logs(trace: Trace, net: utils.ModelContext, dataset, idx_list, logs_file_name: str, layer_list = None):
  # whole_output = utils.custom_norm_no_scale(dataset.orig_labels)
  whole_output = dataset.orig_labels
  num_local_moe_layer = dataset.orig_labels.shape[1] # numseq, numlayer, numexpert
  ideal_records = torch.zeros((len(idx_list), num_local_moe_layer, trace.num_expert + 1))
  predict_records = torch.zeros((len(idx_list), num_local_moe_layer, trace.num_expert + 1))

  correct_freq = whole_output[idx_list]
  predict_freq : torch.Tensor
  with torch.no_grad():
    net.model.eval()
    predict_freq = net.model_forward(dataset[idx_list][0]).reshape((len(idx_list), num_local_moe_layer, trace.num_expert))

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

def save_loss_logs(train_loss_list, test_loss_list, logs_file_name):
  loss_df = pd.DataFrame({
    'train_loss': train_loss_list,
    "test_loss":  test_loss_list,
  })

  loss_df.to_csv(logs_file_name, index=False)

def train_single_model(trace: utils.Trace, train_config) -> None:
  if trace.num_moe_layer not in train_config.model_index:
    return
  return train_one_model(trace, train_config, trace.num_moe_layer, token_distance=1)


def train_one_model(trace: utils.Trace, train_config, i : int, token_distance : int = 0) -> None:
  gc.collect()
  torch.cuda.empty_cache()

  print(f"Training the model that use moe layer input logits to predict next {train_config.window} layer...")

  # Prepare for dataset
  input = trace.decode_stage_moe_layer_logits_per_token[:, i:i+1].to(torch.float32)

  output_layer_mod = i if i < trace.num_moe_layer else 0

  if train_config.predict_output == "gate":
    output = trace.decode_stage_moe_layer_gate_logits_per_token[:, output_layer_mod:min(output_layer_mod+train_config.window, trace.num_moe_layer)].to(torch.float32)
    if train_config.input_norm_method == "max1":
      output = utils.custom_norm_to_max_1(output)
    elif train_config.input_norm_method == "std":
      output = utils.custom_norm_std(output)
    elif train_config.input_norm_method == "replace":
      output = utils.custom_norm_fake_gate(output)
    else:
      raise ValueError(f"Unknow input_norm_method: {train_config.input_norm_method}") 
  elif train_config.predict_output == "freq":
    output = trace.decode_stage_expert_freq_per_token[:, output_layer_mod:min(output_layer_mod+train_config.window, trace.num_moe_layer)].to(torch.float32)
  else:
    raise ValueError(f"Unknow predict_output: {train_config.predict_output}")

  meta_data = torch.cat([trace.decode_stage_seq_id_of_token.reshape((-1, 1)), trace.decode_stage_token_idx_in_seq.reshape((-1, 1))], dim=1)

  if token_distance > 0:
    input     = input     [trace.decode_stage_token_idx_in_seq_flip > token_distance]
    output    = output    [trace.decode_stage_token_idx_in_seq      > token_distance]
    meta_data = meta_data [trace.decode_stage_token_idx_in_seq      > token_distance]

  dataset = utils.CustomDataset(
    input,
    output,
    meta_data,
  )
  train_dataset, test_dataset = dataset.split(0.9)

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
  train_loss_list, test_loss_list = train_loop(net, train_loader, test_loader, train_config)
  print(f"Training model {i} done!")

  # Save logs
  save_loss_logs(train_loss_list, test_loss_list, f"{train_config.train_log_path}/{i}_loss.csv")
  save_accuracy_logs(trace, net, train_dataset, torch.randperm(len(train_dataset)), f"{train_config.train_log_path}/{i}_train_acc.csv")
  save_accuracy_logs(trace, net, test_dataset, torch.randperm(len(test_dataset)), f"{train_config.train_log_path}/{i}_test_acc.csv")

  # Save parameter
  scripted_model = torch.jit.script(net.model)
  scripted_model.save(f"{train_config.predict_model_path}/{i}.pt")

  print("Training multi models done!")

def train_single_target_layer_model(trace: utils.Trace, train_config, input_layer : int, target_layer : int, token_distance : int = 0) -> None:
  gc.collect()
  torch.cuda.empty_cache()

  print(f"Training the model that use moe layer {input_layer} input logits to predict layer {target_layer}'s {train_config.predict_output}...")

  # Prepare for dataset
  input = trace.decode_stage_moe_layer_logits_per_token[:, input_layer:input_layer+1].to(torch.float32)

  if train_config.predict_output == "gate":
    output = trace.decode_stage_moe_layer_gate_logits_per_token[:, target_layer:target_layer+1].to(torch.float32)
    if train_config.input_norm_method == "max1":
      output = utils.custom_norm_to_max_1(output)
    elif train_config.input_norm_method == "std":
      output = utils.custom_norm_std(output)
    elif train_config.input_norm_method == "replace":
      output = utils.custom_norm_fake_gate(output)
    else:
      raise ValueError(f"Unknow input_norm_method: {train_config.input_norm_method}")
  elif train_config.predict_output == "freq":
    output = trace.decode_stage_expert_freq_per_token[:, target_layer:target_layer+1].to(torch.float32)
  else:
    raise ValueError(f"Unknow predict_output: {train_config.predict_output}")

  meta_data = torch.cat([trace.decode_stage_seq_id_of_token.reshape((-1, 1)), trace.decode_stage_token_idx_in_seq.reshape((-1, 1))], dim=1)

  if token_distance > 0:
    input     = input     [trace.decode_stage_token_idx_in_seq_flip > token_distance]
    output    = output    [trace.decode_stage_token_idx_in_seq      > token_distance]
    meta_data = meta_data [trace.decode_stage_token_idx_in_seq      > token_distance]

  dataset = utils.CustomDataset(
    input,
    output,
    meta_data,
  )
  train_dataset, test_dataset = dataset.split(0.9)
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
  train_loss_list, test_loss_list = train_loop(net, train_loader, test_loader, train_config)
  print(f"Training model {input_layer}-{target_layer} done!")

  # Save logs
  save_loss_logs(train_loss_list, test_loss_list, f"{train_config.train_log_path}/{input_layer}-{target_layer}_loss.csv")
  save_accuracy_logs(trace, net, train_dataset, torch.randperm(len(train_dataset)), f"{train_config.train_log_path}/{input_layer}-{target_layer}_train_acc.csv")
  save_accuracy_logs(trace, net, test_dataset, torch.randperm(len(test_dataset)), f"{train_config.train_log_path}/{input_layer}-{target_layer}_test_acc.csv")

  # Save parameter
  scripted_model = torch.jit.script(net.model)
  scripted_model.save(f"{train_config.predict_model_path}/{input_layer}-{target_layer}.pt")

  print("Training multi models done!")

def train_token_id_model(trace: utils.Trace, train_config, i : int = 0, token_distance : int = 0) -> None:
  gc.collect()
  torch.cuda.empty_cache()

  class TokenIdModel(torch.nn.Module):
    def __init__(self, token_id_to_expert_freq):
      super(TokenIdModel, self).__init__()
      self.token_id_to_expert_freq = token_id_to_expert_freq

    def forward(self, x):
      return self.token_id_to_expert_freq[x]

  print(f"Training the model that use token id to predict next {train_config.window} layer...")

  # Prepare for dataset
  input = trace.decode_stage_token_ids_per_token

  output_layer_mod = i if i < trace.num_moe_layer else 0

  if train_config.predict_output == "gate":
    assert False, "Not implemented"
    output = trace.decode_stage_moe_layer_gate_logits_per_token[:, output_layer_mod:min(output_layer_mod+train_config.window, trace.num_moe_layer)].to(torch.float32)
    output = utils.custom_norm_to_max_1(output)
  elif train_config.predict_output == "freq":
    output = trace.decode_stage_expert_freq_per_token[:, output_layer_mod:min(output_layer_mod+train_config.window, trace.num_moe_layer)].to(torch.float32)
  else:
    raise ValueError(f"Unknow predict_output: {train_config.predict_output}")

  meta_data = torch.cat([trace.decode_stage_seq_id_of_token.reshape((-1, 1)), trace.decode_stage_token_idx_in_seq.reshape((-1, 1))], dim=1)

  if token_distance > 0:
    input     = input     [trace.decode_stage_token_idx_in_seq_flip > token_distance]
    output    = output    [trace.decode_stage_token_idx_in_seq      > token_distance]
    meta_data = meta_data [trace.decode_stage_token_idx_in_seq      > token_distance]

  dataset = utils.CustomDataset(
    input,
    output,
    meta_data,
  )
  train_dataset, test_dataset = dataset.split(0.9)

  # Create model

  n_vocab = trace.token_ids.max().item() + 1
  expert_freq_for_each_token = torch.zeros((n_vocab, trace.num_moe_layer * trace.num_expert))
  for entry_idx in tqdm(range(len(train_dataset))):
    expert_freq_for_each_token[train_dataset[entry_idx][0]] += train_dataset[entry_idx][1]

  # Create model
  net = utils.ModelContext(TokenIdModel(expert_freq_for_each_token))

  save_accuracy_logs(trace, net, train_dataset, torch.randperm(len(train_dataset)), f"{train_config.train_log_path}/{i}_train_acc.csv")
  save_accuracy_logs(trace, net, test_dataset,  torch.randperm(len(test_dataset)),  f"{train_config.train_log_path}/{i}_test_acc.csv")

  # Save parameter
  scripted_model = torch.jit.script(net.model)
  scripted_model.save(f"{train_config.predict_model_path}/{i}.pt")

  print("Training multi models done!")

def train_multi_models(trace: utils.Trace, train_config) -> None:
  model_index = [i for i in train_config.model_index if i != trace.num_moe_layer]
  for i in train_config.model_index:
    train_one_model(trace, train_config, i)

def prepare_metas(trace: utils.Trace, predict_model_path: str, window: int) -> None:
  json_data = {}
  for i in range(trace.num_moe_layer):
    json_data[str(i)] = [i, min(i + window, trace.num_moe_layer)]
  
  json_data[str(trace.num_moe_layer)] = [0, min(window, trace.num_moe_layer)]

  with open(f"{predict_model_path}/metas.json", 'w', encoding='utf-8') as json_file:
    json.dump(json_data, json_file, ensure_ascii=False)
  
  return 

if __name__ == "__main__":
  train_config, config_metas = parse_args()
  os.system(f'mkdir -p {train_config.predict_model_path}')
  os.system(f'mkdir -p {train_config.train_log_path}')

  trace = prepare_trace(train_config.logits_path)

  if train_config.window == -1:
    train_config.window = trace.num_moe_layer

  if train_config.model_index == None:
    train_config.model_index = [i for i in range(trace.num_moe_layer + 1)]
    config_metas['model_index'] = train_config.model_index

  save_log_metas(trace, train_config.train_log_path, config_metas)

  print("Training models...")
  if train_config.predict_input == "token-id":
    train_token_id_model(trace, train_config)
  elif train_config.predict_input == "moe-layer-logits":
    if train_config.model_type == "split":
      for layer_distance in range(train_config.window_begin, train_config.window):
        for l in range(trace.num_moe_layer):
          if l not in train_config.model_index:
            continue
          if l + layer_distance >= trace.num_moe_layer:
            continue
          train_single_target_layer_model(trace, train_config,l,l+layer_distance,0)
        if trace.num_moe_layer in train_config.model_index:
          train_single_target_layer_model(trace, train_config, trace.num_moe_layer, layer_distance, 1)
    elif train_config.model_type == "single":
      train_single_model(
        trace, 
        train_config
      )
      train_multi_models(
        trace, 
        train_config
      )
    else:
      raise ValueError(f"Unknow model_type: {train_config.model_type}")
  else:
    raise ValueError(f"Unknow predict_input: {train_config.predict_input}")

  print("Training models done!")

  print("Writing metas.json...")
  prepare_metas(trace, train_config.predict_model_path, train_config.window)
  print("Writing metas.json done!")
